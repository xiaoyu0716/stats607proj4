#!/usr/bin/env python3
"""
解释为什么 step 中的 ratio 是几千，而最后输出的 ratio 只有 ~1.14

关键区别：
1. Step 中的 ratio：基于单次 inference 内部的所有 particles（用于 guidance）
2. 最后的 ratio：基于多次独立 inference 的结果（每次返回一个样本）
"""

print("="*80)
print("Ratio 计算差异分析")
print("="*80)

print("\n1. Step 中的 ratio（几千）")
print("-" * 80)
print("""
计算位置：mcgdiff.py 第 270-282 行
计算方式：
  - x_t[0]: [num_particles, 1, 1, 4, 4] - 所有内部粒子
  - x_particles = x_t[0].view(num_particles, -1)  # [P, 16]
  - x_obs = x_particles[:, obs_idx]  # [P, n_obs]
  - x_null = x_particles[:, null_idx]  # [P, n_null]
  - var_obs = x_obs.var(dim=0).mean()  # 所有 particles 在 observed dims 的方差
  - var_null = x_null.var(dim=0).mean()  # 所有 particles 在 null dims 的方差
  - ratio = var_null / var_obs

含义：
  - 这是单次 inference 内部，所有 particles 之间的方差
  - 在 diffusion 过程中，nullspace 的 particles 会发散（纯 prior diffusion）
  - 所以 var_null 很大（~200），var_obs 很小（~0.02），ratio 几千

为什么这么大？
  - Nullspace 使用纯 prior reverse diffusion: x_{t+1} = x_t + sqrt(factor) * noise
  - 每步都加噪声，particles 会逐渐发散
  - 100 步后，nullspace particles 之间的差异会累积到很大
""")

print("\n2. 最后的 output ratio（~1.14）")
print("-" * 80)
print("""
计算位置：uq_simulation_analysis.py 第 628-664 行
计算方式：
  - 对每个 observation，调用 K 次 inference()
  - 每次 inference() 返回：x_t[0, 0:1] - 只取第一个粒子
  - samples: [N, K, 16] - N 个 observations，每个有 K 个独立样本
  - Z = samples @ V  # 投影到 SVD 空间
  - var_j = mean(Z[:, :, j].var(axis=1))  # 对每个 dim j，计算 K 个样本的方差，然后平均
  - var_observed_mean = mean(var_per_singular_dim[observed_mask])
  - var_null_mean = mean(var_per_singular_dim[null_mask])
  - ratio = var_null_mean / var_observed_mean

含义：
  - 这是多次独立 inference 调用，每次返回一个样本，基于 K 个独立样本的方差
  - 反映的是后验分布的真实方差

为什么只有 ~1.14？
  - 虽然内部 particles 在 nullspace 发散很大
  - 但每次 inference() 只返回第一个粒子
  - 如果多次 inference() 的结果太相似（都被 score 压缩到同一个区域）
  - 那么 K 个独立样本之间的方差就会很小
""")

print("\n3. 问题诊断")
print("-" * 80)
print("""
现象：
  - Step 中：nullspace particles 发散很大（var_null ~200）
  - 最后：nullspace 样本方差很小（var_null ~93，接近 var_obs ~82）

可能原因：
  1. 虽然内部 particles 发散，但最终返回的样本（第一个粒子）可能被 score 压缩
  2. 多次独立 inference() 的结果太相似，说明算法有系统性偏差
  3. Resampling 可能过度集中了 particles，导致最终样本缺乏多样性

检查方法：
  - 查看最终返回的样本是否真的来自 nullspace 高方差区域
  - 检查多次 inference() 的结果是否太相似
  - 检查 resampling 是否过度集中了 particles
""")

print("\n4. 建议的修复方向")
print("-" * 80)
print("""
1. 检查最终样本的选择：
   - 当前：x_t[0, 0:1] - 总是返回第一个粒子
   - 建议：随机选择一个粒子，或者返回所有 particles 的统计量

2. 检查 resampling 的影响：
   - Resampling 可能过度集中了 particles
   - 导致最终样本缺乏多样性

3. 检查 scheduler 的 factor：
   - factor 可能过大，导致 nullspace 方差爆炸
   - 但最终样本可能没有保持这种方差
""")

print("="*80)
