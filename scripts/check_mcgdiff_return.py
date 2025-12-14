#!/usr/bin/env python3
"""
检查 MCG-diff 返回的是单个粒子还是多个粒子
"""

print("="*80)
print("MCG-diff 返回逻辑分析")
print("="*80)

print("\n1. 之前提供的代码（用户问题中的代码）")
print("-" * 80)
print("""
代码：
  return self.forward_op.V(x_t.squeeze(0))

分析：
  - x_t 的形状：[num_samples, num_particles, *M.shape]
    - 例如：[1, 500, 1, 1, 4, 4]（如果 num_samples=1, num_particles=500）
  
  - x_t.squeeze(0)：
    - 移除第 0 维（batch 维度）
    - 结果：[num_particles, *M.shape]
    - 例如：[500, 1, 1, 4, 4]
  
  - V(x_t.squeeze(0))：
    - 转换到 image 空间
    - 返回：[num_particles, 1, 4, 4]
    - 例如：[500, 1, 4, 4]

结论：✅ 返回所有 num_particles 个粒子
""")

print("\n2. 当前文件中的实现（用户修改后的）")
print("-" * 80)
print("""
代码（第 416-421 行）：
  pid = torch.randint(0, self.num_particles, (1,), device=device)
  x_final_svd = x_t[0, pid:pid+1].view(1, 1, 4, 4)
  x_img = self.forward_op.V(x_final_svd)
  return x_img

分析：
  - x_t 的形状：[1, num_particles, 1, 1, 4, 4]
  - pid：随机选择一个粒子索引（0 到 num_particles-1）
  - x_t[0, pid:pid+1]：选择第 pid 个粒子，形状 [1, 1, 1, 4, 4]
  - view(1, 1, 4, 4)：reshape 为 [1, 1, 4, 4]
  - V(x_final_svd)：转换到 image 空间
  - 返回：[1, 1, 4, 4]

结论：✅ 只返回一个粒子（随机选择的）
""")

print("\n3. 两种实现的对比")
print("-" * 80)
print("""
之前提供的代码：
  - 返回：所有 num_particles 个粒子
  - 形状：[num_particles, 1, 4, 4]
  - 问题：
    * 不符合 MCG-diff 的设计（应该返回一个样本）
    * 如果 UQ 脚本调用 inference() K 次，每次都会得到 num_particles 个样本
    * 这会导致样本数量不对（K * num_particles 而不是 K）

当前实现（修改后）：
  - 返回：一个粒子（随机选择）
  - 形状：[1, 1, 4, 4]
  - 优点：
    * 符合 MCG-diff 的设计
    * 多次 inference() 调用 = 多次独立采样
    * 样本数量正确（K 次调用 = K 个样本）
""")

print("\n4. 为什么应该返回一个粒子？")
print("-" * 80)
print("""
MCG-diff 的设计理念：
  - Particles 是内部 Monte Carlo 对象，用于 guidance
  - 每个 inference() 调用应该返回一个后验样本
  - 多次 inference() 调用 = 多次独立采样

如果返回所有粒子：
  - 混淆了"内部 particles"和"后验样本"的概念
  - 导致样本数量不对
  - 可能重复计算方差（如果 UQ 脚本没有正确处理）

正确做法：
  - 每次 inference() 返回一个粒子（随机选择或第一个）
  - 多次 inference() 调用得到多个独立样本
  - 用这些独立样本计算后验方差
""")

print("="*80)
