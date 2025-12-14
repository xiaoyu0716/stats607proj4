#!/usr/bin/env python3
"""
MCG-diff 代码逻辑详解

分析用户提供的 MCG-diff 实现，解释每个步骤的作用和潜在问题
"""

print("="*80)
print("MCG-diff 代码逻辑详解")
print("="*80)

print("\n1. 初始化阶段")
print("-" * 80)
print("""
代码：
  observation_t = self.forward_op.Ut(observation).unsqueeze(1).repeat(1, self.num_particles, 1) * (self.forward_op.M / self.forward_op.S)
  z = torch.randn(num_samples, self.num_particles, *self.forward_op.M.shape, device=device)
  x_t = self.scheduler.sigma_max * z * (1 - self.forward_op.M) + self.forward_op.M * self.scheduler.sigma_max * self.K(0) * z

逻辑：
  1. observation_t: 将观测 y 投影到 SVD 空间，并扩展到所有粒子
     - Ut(observation): 投影到 SVD 空间
     - unsqueeze(1).repeat(1, num_particles, 1): 扩展到 [B, P, ...]
     - * (M / S): 对 observed dims 除以 S，null dims 保持 0
  
  2. x_t 初始化：
     - Null dims: sigma_max * z (纯噪声)
     - Observed dims: sigma_max * K(0) * z (K(0)=1，所以也是 sigma_max * z)
     - 实际上所有 dims 都用 sigma_max * z 初始化
""")

print("\n2. Diffusion Loop - Score 计算")
print("-" * 80)
print("""
代码：
  x = self.forward_op.V(x_t.flatten(0,1))  # SVD space → image space
  denoised_t = Vt(net(x / scaling_step, sigma))  # image → denoised → SVD space
  score = (denoised_t - x_t / scaling_step) / sigma^2 / scaling_step
  x_next_t = x_t * scaling_factor + factor * score

逻辑：
  1. 将 x_t 从 SVD 空间转换到 image 空间（用于网络输入）
  2. 网络预测 denoised（在 image 空间）
  3. 转换回 SVD 空间，计算 score
  4. 使用 score 更新：x_next_t = x_t * scaling_factor + factor * score
  
  注意：x_next_t 包含了 score guidance，会同时影响 observed 和 null dims
""")

print("\n3. Resampling 步骤")
print("-" * 80)
print("""
代码：
  log_prob = -||(observation_t - x_next_t) * M||^2 / (2 * (sigma_next^2 + factor))
           + ||(observation_t - x_t) * M||^2 / (2 * sigma^2)
  indices = multinomial(exp(log_prob), num_particles, replacement=True)
  x_next_t = gather(x_next_t, indices)

逻辑：
  1. 计算 log probability（只基于 observed dims，使用 M mask）
  2. 使用 multinomial 采样选择粒子
  3. 根据 indices 复制粒子（包括 nullspace 部分）
  
  关键问题：
    - Resampling 只基于 observed dims 的 likelihood
    - 但会复制整个粒子（包括 nullspace 部分）
    - 这可能导致 nullspace 的多样性被破坏
""")

print("\n4. 更新步骤（关键！）")
print("-" * 80)
print("""
代码：
  x_masked = K * observation_t + (1 - K) * x_next_t + sqrt(K) * sigma_next * randn_like(x_t)
  x_unmasked = x_next_t + sqrt(factor) * torch.randn_like(x_t)
  x_t = M * x_masked + (1 - M) * x_unmasked

逻辑：
  1. Observed dims (x_masked):
     - 使用 observation_t 和 x_next_t 的混合
     - 添加噪声 sqrt(K) * sigma_next * noise
     - 这是 conditional reverse diffusion
  
  2. Null dims (x_unmasked):
     - 使用 x_next_t + sqrt(factor) * noise
     - ⚠️ 问题：x_next_t 包含了 score！
     - 这意味着 nullspace 仍然被 score 污染了
  
  3. 组合：
     - Observed dims 用 x_masked
     - Null dims 用 x_unmasked
""")

print("\n5. 返回结果")
print("-" * 80)
print("""
代码：
  return self.forward_op.V(x_t.squeeze(0))

逻辑：
  - 返回所有粒子（squeeze(0) 移除 batch 维度）
  - 形状：[num_particles, 1, 4, 4]
  - 注意：这里返回所有粒子，不是只返回一个！
""")

print("\n6. 关键问题分析")
print("-" * 80)
print("""
问题 1: Nullspace 更新使用了 x_next_t
  - x_unmasked = x_next_t + sqrt(factor) * noise
  - x_next_t 包含了 score，会污染 nullspace
  - 应该使用：x_unmasked = x_t + sqrt(factor) * noise

问题 2: 返回所有粒子
  - return self.forward_op.V(x_t.squeeze(0))
  - 返回所有 num_particles 个粒子
  - 这不符合 MCG-diff 的设计（应该返回一个样本）

问题 3: Resampling 可能破坏多样性
  - Resampling 只基于 observed dims，但会复制整个粒子
  - 可能导致 nullspace 的多样性被破坏
""")

print("\n7. 与正确实现的对比")
print("-" * 80)
print("""
正确实现应该：
  1. Nullspace 使用纯 prior reverse diffusion:
     x_unmasked = x_t + sqrt(factor) * noise  # 使用 x_t，不是 x_next_t
  
  2. 返回一个粒子:
     particle_idx = randint(0, num_particles)
     return V(x_t[0, particle_idx:particle_idx+1])
  
  3. 多次 inference() 调用 = 多次独立采样
""")

print("="*80)
