#!/usr/bin/env python3
"""
分析 MCG-diff resampling 对 nullspace 多样性的影响

关键问题：
1. Resampling 基于 observed dims 的 likelihood
2. 但 resampling 会复制整个粒子（包括 nullspace 部分）
3. 这可能导致 nullspace 的多样性被破坏
"""

print("="*80)
print("MCG-diff Resampling 对 Nullspace 多样性的影响分析")
print("="*80)

print("\n1. Resampling 的工作原理")
print("-" * 80)
print("""
Resampling 过程（mcgdiff.py 第 178-228 行）：

1. 计算 log probability（基于 observed dims）：
   log_prob = -||(observation_t - x_next_t) * svd_mask||^2 / (2 * (sigma_next^2 + factor))
   
   注意：只考虑 observed dims（svd_mask），nullspace dims 不参与计算

2. 转换为概率并 resample：
   prob = exp(log_prob) / sum(exp(log_prob))
   indices = multinomial(prob, num_particles, replacement=True)
   
3. 复制粒子：
   x_next_t = gather(x_next_t, indices)
   
   关键：这里复制的是整个粒子，包括 nullspace 部分！
""")

print("\n2. 问题分析")
print("-" * 80)
print("""
现象：
  - Step 中：nullspace particles 发散很大（var_null ~200）
  - 最后：nullspace 样本方差很小（var_null ~93）

可能原因：

1. Resampling 过度集中：
   - 如果某些粒子的 observed dims 与 observation 非常匹配
   - 这些粒子的 likelihood 会很高，被频繁复制
   - 虽然 resampling 只基于 observed dims，但会复制整个粒子
   - 如果这些"好"粒子的 nullspace 部分恰好相似，那么所有粒子的 nullspace 都会相似

2. Nullspace 更新被 score 污染（已修复）：
   - 之前使用 x_next_t（包含 score），现在使用 x_t（纯 prior）
   - 但 resampling 仍然可能破坏多样性

3. 最终样本选择：
   - 如果总是返回第一个粒子，而 resampling 把所有粒子都集中到同一区域
   - 那么多次 inference() 的结果会太相似
""")

print("\n3. 为什么 Step 中的 ratio 很大，但最后的 ratio 很小？")
print("-" * 80)
print("""
Step 中的 ratio（几千）：
  - 计算的是所有 particles 之间的方差
  - 在 diffusion 过程中，nullspace 的 particles 会发散（纯 prior diffusion）
  - 所以 var_null 很大（~200），var_obs 很小（~0.02）

最后的 ratio（~1.14）：
  - 计算的是 K 个独立 inference() 调用的结果之间的方差
  - 每次 inference() 返回一个粒子
  - 如果 resampling 把所有粒子都集中到同一区域，那么：
    * 单次 inference 内部，particles 可能仍然有差异（因为 nullspace 更新是随机的）
    * 但多次 inference() 的结果会太相似（因为 resampling 总是选择相似的粒子）

关键问题：
  - Resampling 会复制整个粒子，包括 nullspace 部分
  - 如果 resampling 过度集中，nullspace 的多样性会被破坏
  - 虽然 nullspace 更新是随机的，但 resampling 会"重置"这种随机性
""")

print("\n4. 解决方案")
print("-" * 80)
print("""
1. 随机选择粒子（已实现）：
   - 不要总是返回第一个粒子
   - 随机选择一个粒子，保持多样性

2. 检查 resampling 的影响：
   - 添加诊断代码，检查 resampling 后 particles 的多样性
   - 如果 resampling 过度集中，考虑调整 resampling 策略

3. 考虑 nullspace 的多样性：
   - Resampling 只基于 observed dims，但会复制整个粒子
   - 可能需要考虑 nullspace 的多样性，避免过度集中
""")

print("="*80)
