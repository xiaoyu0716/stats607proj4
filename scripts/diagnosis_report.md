# UQ Pipeline 诊断报告

## 诊断结果总结

### ✅ 正常的部分
1. **SVD 一致性**：dataset['S'] 和 forward_op.S_vec 完全一致
2. **Nullspace 存在**：MRI-like A 有 7 个 nullspace 维度（indices 9-15）
3. **MCG-diff 样本方差**：样本之间有合理的方差（max variance ≈ 47.8）

### ⚠️ 发现的问题

#### 问题 1: DPS 均值严重低估（70.76% 相对误差）

**症状**：
- 单次重建视觉效果好
- 但后验均值比真实值小 30-70%
- Coverage 只有 0.0025（目标 0.95）

**可能原因**：
1. **Likelihood gradient 缩放不正确**：
   - `forward_op.gradient` 返回 `A^T @ (y - Ax)`（缺少 `1/sigma_noise^2`）
   - DPS 中的归一化 `0.5 / sqrt(loss_scale)` 可能不正确
   - 应该使用 `1 / sigma_noise^2` 来缩放

2. **Guidance scale 使用方式**：
   - 当前：`drift = ... + self.scale * beta * grad_ll`
   - 可能需要：`drift = ... + self.scale * grad_ll`（不乘以 beta）

3. **VP-SDE 的 dt 缩放**：
   - 当前：`x = x + drift * dt`
   - 可能需要检查 dt 的符号和大小

#### 问题 2: Nullspace 方差比例太低（1.03，期望 > 2-3）

**症状**：
- Nullspace variance / Observed variance ≈ 1.03
- 期望应该是 2-10 或更大

**可能原因**：
1. **样本多样性不足**：
   - 虽然 MCG-diff 样本有方差，但可能不够多样化
   - 需要检查是否所有样本都收敛到相似区域

2. **SVD 投影可能有问题**：
   - 使用 `Z = X @ V` 投影到 SVD 空间
   - 需要验证投影是否正确

3. **Nullspace 没有被正确探索**：
   - 后验采样可能没有充分探索 nullspace 方向
   - 需要增加采样多样性

## 修复建议

### 修复 1: DPS Likelihood Gradient 缩放

**当前代码**（第 97 行）：
```python
grad_ll = grad_ll * 0.5 / torch.sqrt(loss_scale + 1e-8)
```

**问题**：
- `forward_op.gradient` 返回 `A^T @ (y - Ax)`，缺少 `1/sigma_noise^2`
- 归一化使用 `sqrt(loss_scale)` 可能不正确

**建议修复**：
```python
# Option 1: 使用 noise_std 缩放（推荐）
grad_ll = grad_ll / (self.forward_op.noise_std ** 2)
# 然后应用较小的归一化
grad_ll = grad_ll * 0.5 / torch.sqrt(loss_scale + 1e-8)

# Option 2: 直接使用官方 DPS 的归一化
grad_ll = grad_ll * 0.5 / torch.sqrt(loss_scale + 1e-8)
# 但需要确保 forward_op.gradient 已经包含了正确的缩放
```

### 修复 2: DPS Guidance Scale 使用

**当前代码**（第 100 行）：
```python
drift = -0.5 * beta * x - beta * eps + self.scale * beta * grad_ll
```

**建议**：检查是否需要 `beta` 因子：
```python
# 可能需要：
drift = -0.5 * beta * x - beta * eps + self.scale * grad_ll
# 或者：
drift = -0.5 * beta * x - beta * eps + self.scale * beta * grad_ll * dt
```

### 修复 3: 增加 Nullspace 探索

**建议**：
1. 增加 MCG-diff 的 `num_particles`（当前 100，可以尝试 200-500）
2. 确保每次 inference 调用使用不同的随机种子
3. 检查 resampling 是否正常工作

### 修复 4: 验证 SVD 投影

**当前代码**（第 613 行）：
```python
Z = X @ V  # (N*K, 16)
```

**验证**：确保 V 是正确的（V^T 在 SVD 中）：
```python
# 验证：A ≈ U @ diag(S) @ V^T
A_reconstructed = U @ np.diag(S) @ V.T
assert np.allclose(A, A_reconstructed, atol=1e-5)
```

## 下一步行动

1. 修复 DPS likelihood gradient 缩放
2. 调整 DPS guidance scale 使用方式
3. 增加 MCG-diff num_particles 并验证样本多样性
4. 重新运行诊断脚本验证修复效果
