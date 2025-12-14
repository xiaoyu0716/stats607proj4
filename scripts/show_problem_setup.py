#!/usr/bin/env python3
"""
展示当前问题设置中的：
1. Prior x (先验分布)
2. A (前向算子矩阵)
3. y (观测值)
4. true x (真实值)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("当前问题设置详细说明")
print("="*80)

# 生成数据集（MRI-like A）
dataset = generate_dataset(
    A_type='mri_like',
    N=1,
    noise_std=0.5,
    seed=0,
    A_seed=1234
)

forward_op = dataset['problem']
x0_true = dataset['x0'][0]  # [16] 真实值
y_obs = dataset['y'][0]      # [16] 观测值
A = dataset['A']             # [16, 16] 前向算子
S = dataset['S']              # [16] 奇异值

print("\n" + "="*80)
print("1. PRIOR X (先验分布)")
print("="*80)

print("\n先验分布类型: 16维混合高斯分布 (Mixture of Gaussians, MoG)")
print("\n结构:")
print("  - 前8维 (dim 0-7): MoG，2个分量")
print("    * 分量1均值: [0, 0, 0, 0, 0, 0, 0, -2.0]")
print("    * 分量2均值: [0, 0, 0, 0, 0, 0, 0, +2.0]")
print("    * 协方差: Toeplitz结构，rho=0.8")
print("    * 权重: 每个分量 0.5")
print("  - 后8维 (dim 8-15): 弱高斯先验")
print("    * 均值: [0, 0, 0, 0, 0, 0, 0, 0]")
print("    * 协方差: 对角矩阵，方差=5.0")

# 显示先验协方差矩阵的结构
print("\n先验协方差矩阵结构 (Sigma_prior):")
Sigma_prior = forward_op.Sigma_prior
print(f"  形状: {Sigma_prior.shape}")
print(f"  前8x8块 (Toeplitz, rho=0.8):")
print(f"    {Sigma_prior[:8, :8]}")
print(f"  后8x8块 (对角，方差=5.0):")
print(f"    {Sigma_prior[8:, 8:]}")
print(f"  对角线元素: {torch.diag(Sigma_prior)}")

# 显示先验均值
print("\n先验均值 (means):")
means = forward_op.means
print(f"  分量1: {means[0]}")
print(f"  分量2: {means[1]}")
print(f"  权重: {forward_op.weights}")

print("\n" + "="*80)
print("2. A (前向算子矩阵)")
print("="*80)

print(f"\nA矩阵形状: {A.shape}")
print(f"A矩阵类型: MRI-like (A = M @ F)")
print(f"  - F: 16x16 类Fourier矩阵 (正交)")
print(f"  - M: 对角mask矩阵，只有9个1，7个0")
print(f"  - A_obs_dim: {forward_op.A_obs_dim} (实际观测维度)")

print(f"\nA矩阵 (前10行，所有列):")
print(A[:10, :])

print(f"\nA矩阵的SVD分解:")
print(f"  奇异值 S: {S}")
print(f"  S的最小值: {S.min():.6e}")
print(f"  S的最大值: {S.max():.6e}")
print(f"  S的唯一值: {np.unique(S)}")

# 分析observed和null空间
threshold = 0.1
observed_dims = np.where(S > threshold)[0]
null_dims = np.where(S <= threshold)[0]
print(f"\n  观测维度 (S > {threshold}): {observed_dims.tolist()} (共{len(observed_dims)}个)")
print(f"  Null空间维度 (S <= {threshold}): {null_dims.tolist()} (共{len(null_dims)}个)")

# 显示A的秩
rank = np.sum(S > 1e-10)
print(f"\n  A的数值秩: {rank} (理论上应该是9)")

print("\n" + "="*80)
print("3. TRUE X (真实值)")
print("="*80)

print(f"\n真实值 x0_true 形状: {x0_true.shape}")
print(f"真实值 x0_true:")
print(f"  {x0_true}")

print(f"\n真实值统计:")
print(f"  最小值: {x0_true.min():.4f}")
print(f"  最大值: {x0_true.max():.4f}")
print(f"  均值: {x0_true.mean():.4f}")
print(f"  标准差: {x0_true.std():.4f}")
print(f"  L2范数: {np.linalg.norm(x0_true):.4f}")

print(f"\n真实值分块:")
print(f"  前8维 (MoG部分): {x0_true[:8]}")
print(f"  后8维 (弱先验部分): {x0_true[8:]}")

print("\n" + "="*80)
print("4. Y (观测值)")
print("="*80)

print(f"\n观测值 y 形状: {y_obs.shape}")
print(f"观测值 y:")
print(f"  {y_obs}")

print(f"\n观测值统计:")
print(f"  最小值: {y_obs.min():.4f}")
print(f"  最大值: {y_obs.max():.4f}")
print(f"  均值: {y_obs.mean():.4f}")
print(f"  标准差: {y_obs.std():.4f}")
print(f"  L2范数: {np.linalg.norm(y_obs):.4f}")

# 计算 y = A @ x + noise
y_computed = A @ x0_true
noise = y_obs - y_computed
print(f"\n观测模型验证:")
print(f"  y = A @ x + noise")
print(f"  y_computed = A @ x0_true: {y_computed}")
print(f"  noise = y - y_computed: {noise}")
print(f"  noise标准差: {noise.std():.4f} (期望: {forward_op.noise_std})")
print(f"  noise L2范数: {np.linalg.norm(noise):.4f}")

# 显示观测值在observed和null空间的投影
print(f"\n观测值在SVD空间的投影:")
U = dataset.get('U', None)
if U is not None:
    U_tensor = torch.from_numpy(U)
    y_obs_tensor = torch.from_numpy(y_obs).float()
    y_svd = U_tensor.T @ y_obs_tensor  # Ut @ y
    print(f"  y_svd = Ut @ y: {y_svd}")
    print(f"  y_svd在observed dims: {y_svd[observed_dims]}")
    print(f"  y_svd在null dims: {y_svd[null_dims]}")

print("\n" + "="*80)
print("5. 问题总结")
print("="*80)

print(f"\n问题类型: 线性逆问题 y = A @ x + noise")
print(f"  - 维度: 16维")
print(f"  - A类型: MRI-like (rank-deficient)")
print(f"  - A的秩: {rank} (observed dims: {len(observed_dims)}, null dims: {len(null_dims)})")
print(f"  - 噪声标准差: {forward_op.noise_std}")
print(f"  - 先验: 16D MoG (前8维MoG，后8维弱高斯)")

print(f"\n关键特性:")
print(f"  - Null空间存在: 是 (7个维度)")
print(f"  - Null空间中的不确定性: 理论上应该很大")
print(f"  - Observed空间中的不确定性: 主要由观测噪声决定")

print("\n" + "="*80)
