# MCG-diff Nullspace Variance - Final Corrected Patch

## ✅ All 4 Bugs Fixed

### Bug 1: ✅ Fixed - Use torch.sqrt instead of np.sqrt
- **Before**: `np.sqrt(factor)` - breaks Tensor type and device
- **After**: `torch.sqrt(torch.tensor(factor, device=..., dtype=...))` - maintains Tensor

### Bug 2: ✅ Fixed - Preserve original MCG-diff conditional formula
- **Before**: Incorrect Kalman update formula
- **After**: Original MCG-diff formula preserved: `K * observation + (1-K) * x_next_t + sqrt(K) * sigma_next * noise`

### Bug 3: ✅ Fixed - Zero-out dimensions inside each step
- **Before**: Only masked at combination
- **After**: 
  - `x_post` zeros out nullspace internally: `x_post = x_post * svd_mask`
  - `x_prior` zeros out observed dims internally: `x_prior = x_prior * (1 - svd_mask)`
  - Final combination: `x_t = x_post + x_prior` (both already correctly masked)

### Bug 4: ✅ Fixed - Use scheduler values correctly
- **Before**: Unclear source
- **After**: Explicitly uses `self.scheduler.scaling_factor[step]` and `self.scheduler.factor_steps[step]`

## Implementation Details

### `_reverse_step_prior()` Method
```python
def _reverse_step_prior(self, x_t, scaling_factor, factor, noise, svd_mask):
    # Pure prior reverse diffusion: x_{t-1} = scaling_factor * x_t + sqrt(factor) * noise
    factor_tensor = torch.tensor(factor, device=x_t.device, dtype=x_t.dtype)
    x_prior = x_t * scaling_factor + torch.sqrt(factor_tensor) * noise
    # Zero-out observed dimensions
    x_prior = x_prior * (1 - svd_mask)
    return x_prior
```

**Key points**:
- Uses `torch.sqrt` (not `np.sqrt`)
- Uses scheduler's `scaling_factor` and `factor`
- Zeros out observed dims internally
- NO score, NO data guidance

### `_reverse_step_conditional()` Method
```python
def _reverse_step_conditional(self, x_t, x_next_t, observation_t, svd_mask, K, sigma_next, noise):
    # ORIGINAL MCG-diff formula (preserved exactly)
    K_tensor = torch.tensor(K, device=x_t.device, dtype=x_t.dtype)
    x_post = (
        K * observation_t * svd_mask +
        (1 - K) * x_next_t + 
        torch.sqrt(K_tensor) * sigma_next * noise
    )
    # Zero-out nullspace dimensions
    x_post = x_post * svd_mask
    return x_post
```

**Key points**:
- Uses `torch.sqrt` (not `np.sqrt`)
- Preserves original MCG-diff conditional formula
- Zeros out nullspace internally
- Uses data guidance (observation_t)

### Main Loop Update
```python
# Sample shared noise ONCE per step
shared_noise = torch.randn_like(x_t)

# 1. Observed dimensions: conditional reverse diffusion
x_post = self._reverse_step_conditional(...)

# 2. Null dimensions: pure prior reverse diffusion
x_prior = self._reverse_step_prior(...)

# 3. Combine (both already masked correctly)
x_t = x_post + x_prior
```

## Verification Results

✅ **Logic test passed**:
- Prior step uses torch.sqrt, zeros observed dims
- Conditional step uses torch.sqrt, preserves original formula, zeros nullspace
- Combination works correctly
- Masking verification: both steps correctly zero opposite dimensions

## Expected Results

| Metric | Before | Theory | After (Expected) |
|--------|--------|--------|------------------|
| var_obs | 0.1588 | 0.2263 | ~0.20-0.23 |
| var_null | 0.1282 | 1.2845 | ~1.0-1.5 |
| ratio | 0.8072 | 5.67 | ~5-10 |

## Files Modified

- ✅ `algo/mcgdiff.py` - Applied all fixes
- ✅ All `np.sqrt` replaced with `torch.sqrt`
- ✅ Original conditional formula preserved
- ✅ Internal zero-out implemented
- ✅ Scheduler values used correctly

## Ready to Test

The patch is complete and ready for testing. Run:

```bash
python scripts/uq_simulation_analysis.py --experiment nullspace --methods MCG_diff --N 50 --K 20
```

Expected: null variance ratio ≈ 5-10 (vs 0.8 before)
