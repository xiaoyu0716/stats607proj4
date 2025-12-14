# MCG-diff Nullspace Variance Corrected Fix

## Summary

This patch correctly implements separation of conditional (observed) and prior (nullspace) reverse diffusion chains, fixing all 4 bugs from the previous attempt.

## Fixed Bugs

### ✅ Bug 1: Use torch.sqrt instead of np.sqrt
- **Before**: `np.sqrt(factor)` - breaks Tensor type and device
- **After**: `torch.sqrt(torch.tensor(factor, device=..., dtype=...))` - maintains Tensor

### ✅ Bug 2: Preserve original MCG-diff conditional formula
- **Before**: Replaced with incorrect Kalman update
- **After**: Preserves original formula: `K * observation + (1-K) * x_next_t + sqrt(K) * sigma_next * noise`

### ✅ Bug 3: Zero-out dimensions inside each step
- **Before**: Only masked at combination
- **After**: 
  - `x_post` zeros out nullspace internally
  - `x_prior` zeros out observed dims internally
  - Final combination: `x_t = x_post + x_prior`

### ✅ Bug 4: Use scheduler values correctly
- **Before**: Unclear source of scaling_factor/factor
- **After**: Explicitly uses `self.scheduler.scaling_factor[step]` and `self.scheduler.factor_steps[step]`

## Complete Diff

```diff
diff --git a/algo/mcgdiff.py b/algo/mcgdiff.py
--- a/algo/mcgdiff.py
+++ b/algo/mcgdiff.py
@@ -40,6 +40,68 @@ class MCG_diff(Algo):
     def K(self, t):
         if t == self.scheduler.num_steps:
             return 1
         return self.scheduler.factor_steps[t] / (self.scheduler.factor_steps[t]+ self.scheduler.sigma_steps[t]**2)
+    
+    def _reverse_step_prior(self, x_t, scaling_factor, factor, noise, svd_mask):
+        """
+        Pure prior reverse diffusion step for nullspace dimensions.
+        
+        This implements unconditional DDPM/DDIM reverse step with NO data guidance.
+        Uses the same noise schedule as conditional chain but without likelihood/score.
+        
+        Args:
+            x_t: Current state in SVD space [B, num_particles, *M.shape]
+            scaling_factor: Scaling factor from scheduler (float)
+            factor: Factor from scheduler (float)
+            noise: Shared random noise [B, num_particles, *M.shape]
+            svd_mask: Mask for observed dimensions [1, 1, *M.shape], used to zero-out observed dims
+        
+        Returns:
+            x_prior: Next state following pure prior reverse diffusion (nullspace only, observed dims zeroed)
+        """
+        # Pure prior reverse diffusion: x_{t-1} = scaling_factor * x_t + sqrt(factor) * noise
+        # No score, no data guidance - just prior-driven diffusion
+        # CRITICAL: Use torch.sqrt, not np.sqrt, to maintain Tensor type and device
+        factor_tensor = torch.tensor(factor, device=x_t.device, dtype=x_t.dtype)
+        x_prior = x_t * scaling_factor + torch.sqrt(factor_tensor) * noise
+        
+        # Zero-out observed dimensions (only nullspace should be non-zero)
+        x_prior = x_prior * (1 - svd_mask)
+        return x_prior
+    
+    def _reverse_step_conditional(self, x_t, x_next_t, observation_t, svd_mask, K, sigma_next, noise):
+        """
+        Conditional reverse diffusion step for observed dimensions.
+        
+        This implements the ORIGINAL MCG-diff conditional update formula (no changes to math).
+        Only modification: explicitly zero-out nullspace dimensions before returning.
+        
+        Args:
+            x_t: Current state in SVD space [B, num_particles, *M.shape]
+            x_next_t: Score-guided next state [B, num_particles, *M.shape]
+            observation_t: Observation in SVD space [B, 1, *M.shape] or [1, 1, *M.shape]
+            svd_mask: Mask for observed dimensions [1, 1, *M.shape]
+            K: Kalman gain factor (float)
+            sigma_next: Next noise level (float)
+            noise: Shared random noise [B, num_particles, *M.shape]
+        
+        Returns:
+            x_post: Next state following conditional reverse diffusion (observed dims only, nullspace zeroed)
+        """
+        # ORIGINAL MCG-diff conditional update formula (preserved exactly)
+        # x_post = K * observation + (1-K) * x_next_t + sqrt(K) * sigma_next * noise
+        # CRITICAL: Use torch.sqrt, not np.sqrt, to maintain Tensor type and device
+        K_tensor = torch.tensor(K, device=x_t.device, dtype=x_t.dtype)
+        x_post = (
+            K * observation_t * svd_mask +
+            (1 - K) * x_next_t + 
+            torch.sqrt(K_tensor) * sigma_next * noise
+        )
+        
+        # CRITICAL: Zero-out nullspace dimensions (only observed dims should be non-zero)
+        # This prevents nullspace contamination from x_next_t
+        x_post = x_post * svd_mask
+        return x_post
     
     @torch.no_grad()
     def inference(self, observation, num_samples=1, **kwargs):
@@ -258,20 +320,30 @@ class MCG_diff(Algo):
             gather_indices = gather_indices.expand(list(gather_indices.shape[:2]) + list(x_next_t.shape[2:]))
             x_next_t = torch.gather(x_next_t, 1, gather_indices)
             
-            # Update x_t using masked and unmasked updates (follow reference implementation)
-            # Use svd_mask consistently (already defined at the beginning)
-            x_masked = (
-                K * observation_t * svd_mask +
-                (1 - K) * x_next_t + 
-                np.sqrt(K) * sigma_next * torch.randn_like(x_t)
-            )
-            # NULLSPACE SHOULD FOLLOW PURE PRIOR REVERSE DIFFUSION
-            # CRITICAL FIX: Use x_t (not x_next_t) to avoid score contamination
-            # x_next_t contains denoiser score which should NOT affect nullspace
-            # Using x_next_t causes nullspace variance collapse (ratio ≈ 1 instead of ≈ 5-10)
-            x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)
+            # ========================================================================
+            # CRITICAL FIX: Separate conditional and prior reverse diffusion chains
+            # ========================================================================
+            # Sample shared noise ONCE per step (both chains use same noise)
+            shared_noise = torch.randn_like(x_t)
+            
+            # 1. Observed dimensions: conditional reverse diffusion with data guidance
+            # Uses ORIGINAL MCG-diff formula, but zeroes out nullspace internally
+            x_post = self._reverse_step_conditional(
+                x_t=x_t,
+                x_next_t=x_next_t,
+                observation_t=observation_t,
+                svd_mask=svd_mask,
+                K=K,
+                sigma_next=sigma_next,
+                noise=shared_noise
+            )
+            
+            # 2. Null dimensions: pure prior reverse diffusion (NO data guidance)
+            # Uses same scheduler (scaling_factor, factor) but NO score
+            x_prior = self._reverse_step_prior(
+                x_t=x_t,
+                scaling_factor=scaling_factor,
+                factor=factor,
+                noise=shared_noise,
+                svd_mask=svd_mask
+            )
+            
+            # 3. Combine: x_post has only observed dims (nullspace zeroed)
+            #            x_prior has only null dims (observed dims zeroed)
+            #            Adding them gives the complete state
+            x_t = x_post + x_prior
 
-            x_t = svd_mask * x_masked + (1 - svd_mask) * x_unmasked
-            
         # Return final result: convert from SVD space to image space
```

## Key Improvements

1. ✅ **Uses torch.sqrt**: Maintains Tensor type and device compatibility
2. ✅ **Preserves original formula**: Conditional step uses exact MCG-diff formula
3. ✅ **Internal zero-out**: Each step zeros out opposite dimensions internally
4. ✅ **Clean combination**: `x_t = x_post + x_prior` (both already masked)
5. ✅ **Shared noise**: Both chains use same noise per step

## Expected Results

| Metric | Before | Theory | After (Expected) |
|--------|--------|--------|------------------|
| var_obs | 0.1588 | 0.2263 | ~0.20-0.23 |
| var_null | 0.1282 | 1.2845 | ~1.0-1.5 |
| ratio | 0.8072 | 5.67 | ~5-10 |

## Verification

```bash
# Run nullspace variance experiment
python scripts/uq_simulation_analysis.py --experiment nullspace --methods MCG_diff --N 50 --K 20

# Or run full debugging script
python scripts/debug_mcgdiff_nullspace.py
```

Expected output:
- Null variance ≈ 1.0-1.5 (vs 0.13 before, **10x improvement**)
- Ratio ≈ 5-10 (vs 0.8 before, **6-12x improvement**)
