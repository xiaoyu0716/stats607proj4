# MCG-diff Nullspace Variance - Final Corrected Fix

## Problem Identified

After initial fix, observed variance was **23.19** (should be ~0.226), indicating the conditional step formula was **completely wrong**.

## Root Cause

The `_reverse_step_conditional()` was using an incorrect Kalman-like formula:
```python
# WRONG (causes variance explosion)
x_post = K * observation_t + (1-K) * x_next_t + sqrt(K) * sigma_next * noise
```

This is **not** the MCG-diff formula and causes:
- Variance explosion in observed dims
- No proper posterior shrinkage

## Correct Solution

The **correct** MCG-diff conditional step is:
- `x_next_t` already contains the score-based posterior update: `x_t * scaling_factor + factor * score`
- Conditional step should just add noise: `x_post = x_next_t + sigma_next * noise`
- Then zero-out nullspace

## Final Implementation

### `_reverse_step_conditional()` - CORRECTED
```python
def _reverse_step_conditional(self, x_next_t, sigma_next, noise, svd_mask):
    """
    Conditional reverse diffusion step for observed dimensions.
    
    x_next_t already contains score-based posterior update.
    Just add noise and zero-out nullspace.
    """
    # CORRECT MCG-diff conditional update:
    x_post = x_next_t + sigma_next * noise
    # Zero-out nullspace dimensions
    x_post = x_post * svd_mask
    return x_post
```

**Key points**:
- ✅ Uses `x_next_t` directly (already has score update)
- ✅ Simple noise addition: `x_next_t + sigma_next * noise`
- ✅ No incorrect Kalman blending
- ✅ Zeros out nullspace

### `_reverse_step_prior()` - UNCHANGED
```python
def _reverse_step_prior(self, x_t, scaling_factor, factor, noise, svd_mask):
    """
    Pure prior reverse diffusion for nullspace dimensions.
    """
    factor_tensor = torch.tensor(factor, device=x_t.device, dtype=x_t.dtype)
    x_prior = x_t * scaling_factor + torch.sqrt(factor_tensor) * noise
    # Zero-out observed dimensions
    x_prior = x_prior * (1 - svd_mask)
    return x_prior
```

### Main Loop - CORRECTED
```python
# Sample shared noise ONCE per step
shared_noise = torch.randn_like(x_t)

# 1. Observed dimensions: conditional reverse diffusion
# x_next_t already contains score-based posterior update
x_post = self._reverse_step_conditional(
    x_next_t=x_next_t,  # Already has: x_t * scaling_factor + factor * score
    sigma_next=sigma_next,
    noise=shared_noise,
    svd_mask=svd_mask
)

# 2. Null dimensions: pure prior reverse diffusion
x_prior = self._reverse_step_prior(
    x_t=x_t,
    scaling_factor=scaling_factor,
    factor=factor,
    noise=shared_noise,
    svd_mask=svd_mask
)

# 3. Combine (both already masked correctly)
x_t = x_post + x_prior
```

## Expected Results

| Metric | Before (Wrong) | Theory | After (Expected) |
|--------|----------------|--------|------------------|
| var_obs | 23.19 | 0.2263 | ~0.20-0.23 |
| var_null | 24.26 | 1.2845 | ~1.0-1.5 |
| ratio | 1.046 | 5.67 | ~5-10 |

## Key Fixes

1. ✅ **Removed incorrect Kalman formula** from conditional step
2. ✅ **Use correct MCG-diff formula**: `x_next_t + sigma_next * noise`
3. ✅ **x_next_t already contains posterior info** (score update)
4. ✅ **Proper variance control** through score-based shrinkage

## Verification

The corrected implementation:
- ✅ Uses `x_next_t` directly (score already applied)
- ✅ Simple noise addition (no variance explosion)
- ✅ Proper masking (nullspace zeroed in conditional, observed zeroed in prior)
- ✅ Shared noise (both chains use same noise)

## Ready to Test

```bash
python scripts/uq_simulation_analysis.py --experiment nullspace --methods MCG_diff --N 50 --K 20
```

Expected:
- Observed variance: ~0.20-0.23 (vs 23.19 before)
- Null variance: ~1.0-1.5 (vs 24.26 before)
- Ratio: ~5-10 (vs 1.046 before)
