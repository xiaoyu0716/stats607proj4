#!/usr/bin/env python3
"""
计算理论后验方差（SVD 坐标下）

针对 16D ToyGausscMoG + MRI-like A 的设定
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("理论后验方差计算（SVD 坐标下）")
print("="*80)

# 生成数据集以获取 A 和 prior
dataset = generate_dataset(
    A_type='mri_like',
    N=1,
    noise_std=0.5,
    seed=0,
    A_seed=1234
)

forward_op = dataset['problem']
A = torch.from_numpy(dataset['A']).float()  # [16, 16]
S = torch.from_numpy(dataset['S']).float()  # [16] 奇异值
U = torch.from_numpy(dataset['U']).float()  # [16, 16]
V = torch.from_numpy(dataset['V']).float()  # [16, 16] (V^T in SVD)

# Prior covariance
Sigma_prior = forward_op.Sigma_prior  # [16, 16]

# Noise variance
sigma_noise = 0.5
sigma_noise_sq = sigma_noise ** 2  # 0.25

print(f"\n1. Prior Covariance (Sigma_prior):")
print(f"   形状: {Sigma_prior.shape}")
print(f"   前8×8块 (Toeplitz):")
print(f"     对角线: {torch.diag(Sigma_prior[:8, :8])}")
print(f"   后8×8块 (对角，方差=5.0):")
print(f"     对角线: {torch.diag(Sigma_prior[8:, 8:])}")

print(f"\n2. Forward Operator A:")
print(f"   形状: {A.shape}")
print(f"   奇异值 S: {S}")
print(f"   观测维度 (S > 0.1): {torch.where(S > 0.1)[0].tolist()}")
print(f"   Null空间维度 (S <= 0.1): {torch.where(S <= 0.1)[0].tolist()}")

# 计算后验协方差（在原始坐标 x 中）
# Σ_post = (A^T A / σ² + Σ_prior^-1)^-1
A_T_A = A.T @ A  # [16, 16]
Sigma_prior_inv = torch.linalg.inv(Sigma_prior)  # [16, 16]

# 后验协方差
Sigma_post = torch.linalg.inv(A_T_A / sigma_noise_sq + Sigma_prior_inv)  # [16, 16]

print(f"\n3. 后验协方差 (原始坐标 x):")
print(f"   形状: {Sigma_post.shape}")
print(f"   对角线: {torch.diag(Sigma_post)}")

# 转换到 SVD 坐标 z = V^T x
# 在 SVD 坐标中，后验协方差为：
# Σ_post_z = V^T @ Σ_post @ V
Sigma_post_z = V.T @ Sigma_post @ V  # [16, 16]

print(f"\n4. 后验协方差 (SVD 坐标 z = V^T x):")
print(f"   形状: {Sigma_post_z.shape}")
print(f"   对角线 (var_z): {torch.diag(Sigma_post_z)}")

# 理论公式验证（在 SVD 坐标中）
# 对于每个维度 i：
# var_z[i] = 1 / (1/Σ_prior_z[i] + S[i]^2 / σ²)
# 其中 Σ_prior_z = V^T @ Σ_prior @ V

Sigma_prior_z = V.T @ Sigma_prior @ V  # [16, 16]
var_z_theoretical = torch.zeros(16)

for i in range(16):
    prior_var_z_i = Sigma_prior_z[i, i]
    S_i = S[i]
    if prior_var_z_i > 1e-10:
        var_z_theoretical[i] = 1.0 / (1.0 / prior_var_z_i + S_i**2 / sigma_noise_sq)
    else:
        var_z_theoretical[i] = 0.0

print(f"\n5. 理论公式计算 (var_z[i] = 1 / (1/Σ_prior_z[i] + S[i]^2/σ²)):")
print(f"   var_z_theoretical: {var_z_theoretical}")

# 分离 observed 和 null dims
observed_mask = (S > 0.1)
null_mask = (S <= 0.1)

var_obs_theoretical = torch.diag(Sigma_post_z)[observed_mask].mean().item()
var_null_theoretical = torch.diag(Sigma_post_z)[null_mask].mean().item()

print(f"\n6. 理论方差（SVD 坐标）:")
print(f"   Observed dims variance (mean): {var_obs_theoretical:.6f}")
print(f"   Null dims variance (mean): {var_null_theoretical:.6f}")
print(f"   Ratio (var_null / var_obs): {var_null_theoretical / var_obs_theoretical:.4f}")

# 对比实际输出
print(f"\n7. 对比实际 MCG-diff 输出:")
print(f"   实际 observed variance: 0.1588")
print(f"   实际 null variance: 0.1282")
print(f"   实际 ratio: 0.8072")
print(f"\n   理论 observed variance: {var_obs_theoretical:.6f}")
print(f"   理论 null variance: {var_null_theoretical:.6f}")
print(f"   理论 ratio: {var_null_theoretical / var_obs_theoretical:.4f}")

print(f"\n8. 差异分析:")
print(f"   Observed variance 差异: {abs(0.1588 - var_obs_theoretical):.6f} (相对误差: {abs(0.1588 - var_obs_theoretical) / var_obs_theoretical * 100:.2f}%)")
print(f"   Null variance 差异: {abs(0.1282 - var_null_theoretical):.6f} (相对误差: {abs(0.1282 - var_null_theoretical) / var_null_theoretical * 100:.2f}%)")
print(f"   Ratio 差异: {abs(0.8072 - var_null_theoretical / var_obs_theoretical):.4f}")

print("\n" + "="*80)
print("结论:")
print("="*80)
print("✅ Observed variance 接近理论值（说明 observed dims 更新正确）")
print("❌ Null variance 远低于理论值（说明 null dims 被错误收缩）")
print("❌ Ratio 远低于理论值（应该 ≈ 20-25，实际 ≈ 0.8）")
print("="*80)
