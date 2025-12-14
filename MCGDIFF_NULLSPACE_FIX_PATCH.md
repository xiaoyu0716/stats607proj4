# MCG-diff Nullspace Variance Fix Patch

## 问题诊断总结

**理论后验方差**（SVD 坐标）:
- Observed dims variance: **0.2263**
- Null dims variance: **1.2845**
- Ratio: **5.67**

**实际 MCG-diff 输出**:
- Observed dims variance: **0.1588** ✅ (接近理论值)
- Null dims variance: **0.1282** ❌ (比理论值小 **10倍**)
- Ratio: **0.8072** ❌ (比理论值小 **7倍**)

**根因**: Nullspace 被错误地使用了 `x_next_t`（包含 score），导致被 denoiser 收缩。

---

## 完整 Patch 代码

### Patch 1: 修复 nullspace diffusion（最关键）

**位置**: `algo/mcgdiff.py` 第 219 行

**修改前**:
```python
x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)
# pure prior reverse diffusion, not guided by likelihood or score
# x_unmasked = x_t + np.sqrt(factor) * torch.randn_like(x_t)
```

**修改后**:
```python
# NULLSPACE SHOULD FOLLOW PURE PRIOR REVERSE DIFFUSION
# Do NOT use x_next_t (which contains score) - this causes nullspace collapse
# Use x_t instead to maintain pure prior diffusion in nullspace
x_unmasked = x_t + np.sqrt(factor) * torch.randn_like(x_t)
```

**原因**:
- `x_next_t = x_t * scaling_factor + factor * score` 包含 denoiser score
- Score 在 image-space 会影响所有维度，包括 nullspace
- Nullspace 应该完全忽略 score，只遵循 prior reverse diffusion
- 使用 `x_t` 而不是 `x_next_t` 确保 nullspace 是 pure diffusion

---

### Patch 2: 可选 - 确保 score 不直接作用到 nullspace

**位置**: `algo/mcgdiff.py` 第 142 行（score 计算后）

**修改前**:
```python
score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
x_next_t = x_t * scaling_factor + factor * score
```

**修改后**:
```python
score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
# Note: score is computed in SVD space, but x_next_t update applies to all dims
# This is OK because we will mask it out in x_unmasked update
x_next_t = x_t * scaling_factor + factor * score
```

**说明**: 
- 实际上，由于我们在 `x_unmasked` 中使用 `x_t` 而不是 `x_next_t`，score 已经不会影响 nullspace
- 这个 patch 是可选的，主要是为了代码清晰度

---

## 完整修改后的代码段

```python
# ... (前面的代码保持不变) ...

for step in pbar:
    sigma, sigma_next, factor, scaling_factor, scaling_step = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step], self.scheduler.scaling_steps[step]
    x = self.forward_op.V(x_t.flatten(0,1))

    denoised_t = []
    for i in range(0, x.shape[0], MAX_BATCH_SIZE):
        denoised_t.append(self.forward_op.Vt(self.net(x[i:i+MAX_BATCH_SIZE]/scaling_step, torch.as_tensor(sigma).to(x.device))).view(-1, self.num_particles, *self.forward_op.M.shape))
    denoised_t = torch.cat(denoised_t, dim=0)
    score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
    x_next_t = x_t * scaling_factor + factor * score
    
    # ... (log_prob 和 resampling 代码保持不变) ...
    
    K = self.K(step+1)
    gather_indices = indices.unsqueeze(-1)
    for _ in range(len(self.forward_op.M.shape) - 1):
        gather_indices = gather_indices.unsqueeze(-1)
    gather_indices = gather_indices.expand(list(gather_indices.shape[:2]) + list(x_next_t.shape[2:]))
    x_next_t = torch.gather(x_next_t, 1, gather_indices)
    
    # Update x_t using masked and unmasked updates
    # Observed dims: use data-guided update (x_masked)
    x_masked = (
        K * observation_t * svd_mask +
        (1 - K) * x_next_t + 
        np.sqrt(K) * sigma_next * torch.randn_like(x_t)
    )
    
    # Nullspace dims: use PURE PRIOR REVERSE DIFFUSION
    # CRITICAL FIX: Use x_t (not x_next_t) to avoid score contamination
    # x_next_t contains denoiser score which should NOT affect nullspace
    x_unmasked = x_t + np.sqrt(factor) * torch.randn_like(x_t)
    
    # Combine: observed dims use x_masked, null dims use x_unmasked
    x_t = svd_mask * x_masked + (1 - svd_mask) * x_unmasked
```

---

## 预期效果

应用 patch 后，预期结果：

| 指标 | 当前 | 理论 | Patch 后预期 |
|------|------|------|--------------|
| var_obs | 0.1588 | 0.2263 | ~0.20-0.23 |
| var_null | 0.1282 | 1.2845 | ~1.0-1.5 |
| ratio | 0.8072 | 5.67 | ~5-10 |

**关键改进**:
- ✅ Null variance 从 0.13 → 1.0+ (提升 **7-10倍**)
- ✅ Ratio 从 0.8 → 5-10 (提升 **6-12倍**)
- ✅ Observed variance 保持接近理论值

---

## 应用 Patch

直接修改 `algo/mcgdiff.py` 第 219 行：

```python
# 将这行：
x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)

# 改为：
x_unmasked = x_t + np.sqrt(factor) * torch.randn_like(x_t)
```

然后重新运行实验验证效果。

---

## 验证命令

```bash
# 运行 nullspace variance 实验
python scripts/uq_simulation_analysis.py --experiment nullspace --methods MCG_diff --N 50 --K 20

# 或运行完整调试脚本
python scripts/debug_mcgdiff_nullspace.py
```

预期看到：
- Null variance ≈ 1.0-1.5
- Ratio ≈ 5-10
