#!/usr/bin/env python3
"""
检查 MCG-diff 中的 observed 和 null 维度是否与 SVD 维度对齐

关键检查：
1. 理论后验方差在 SVD 空间中的维度索引
2. MCG-diff 中 svd_mask 对应的维度索引
3. 验证两者是否一致
4. 检查 SVD 变换（V, Ut, Vt）的维度映射
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset
# Import compute_theoretical_posterior_variance functions if needed
# from scripts.compute_theoretical_posterior_variance import *

print("="*80)
print("SVD 维度对齐检查")
print("="*80)

# 生成数据集
dataset = generate_dataset(
    A_type='mri_like',
    N=1,
    noise_std=0.5,
    seed=0,
    A_seed=1234
)

forward_op = dataset['problem']
A = torch.from_numpy(dataset['A']).float()  # [16, 16]
S_vec = torch.from_numpy(dataset['S']).float()  # [16] 奇异值向量
U = torch.from_numpy(dataset['U']).float()  # [16, 16]
V = torch.from_numpy(dataset['V']).float()  # [16, 16] (V^T in SVD)

# Prior covariance
Sigma_prior = forward_op.Sigma_prior  # [16, 16]

# Noise variance
sigma_noise = 0.5
sigma_noise_sq = sigma_noise ** 2

print("\n" + "="*80)
print("1. 理论后验方差计算（SVD 坐标）")
print("="*80)

# 计算后验协方差（原始坐标 x）
A_T_A = A.T @ A
Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
Sigma_post = torch.linalg.inv(A_T_A / sigma_noise_sq + Sigma_prior_inv)  # [16, 16]

# 转换到 SVD 坐标 z = V^T x
# 注意：V 是 Vt^T，所以 V^T = Vt
# 在代码中，_Vt_matrix 是 Vt，所以 V = _Vt_matrix.T
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - 这是 Vt
V_matrix = Vt_matrix.T  # [16, 16] - 这是 V

# 在 SVD 坐标中的后验协方差
Sigma_post_z = V_matrix.T @ Sigma_post @ V_matrix  # [16, 16]
var_z_theoretical = torch.diag(Sigma_post_z)  # [16]

print(f"\n理论后验方差（SVD 坐标，16维向量）:")
print(f"  var_z_theoretical: {var_z_theoretical}")

# 将 var_z 从 16维向量 reshape 到 [1, 1, 4, 4]
var_z_4d = var_z_theoretical.reshape(1, 1, 4, 4)
print(f"\n理论后验方差（4D 格式 [1, 1, 4, 4]）:")
print(f"  var_z_4d shape: {var_z_4d.shape}")
print(f"  var_z_4d:\n{var_z_4d.squeeze()}")

print("\n" + "="*80)
print("2. MCG-diff 中的 svd_mask 定义")
print("="*80)

# MCG-diff 中如何定义 svd_mask
S_img = forward_op.S  # [1, 1, 4, 4] - SVD 奇异值（4D 格式）
svd_mask_4d = (S_img > 0.1).float()  # [1, 1, 4, 4] - MCG-diff 使用的 mask

print(f"\nS_img (forward_op.S):")
print(f"  Shape: {S_img.shape}")
print(f"  Values:\n{S_img.squeeze()}")

print(f"\nsvd_mask_4d (S > 0.1):")
print(f"  Shape: {svd_mask_4d.shape}")
print(f"  Values:\n{svd_mask_4d.squeeze()}")
print(f"  Observed dims count: {svd_mask_4d.sum().item()}")
print(f"  Null dims count: {(1 - svd_mask_4d).sum().item()}")

# 将 S_vec 从 16维 reshape 到 [1, 1, 4, 4] 看看是否一致
S_vec_reshaped = S_vec.reshape(1, 1, 4, 4)
print(f"\nS_vec (16维) reshape 到 [1, 1, 4, 4]:")
print(f"  Values:\n{S_vec_reshaped.squeeze()}")

print("\n" + "="*80)
print("3. 维度索引对齐检查")
print("="*80)

# 方法1：基于 S_vec（16维向量）的索引
observed_indices_vec = torch.where(S_vec > 0.1)[0].tolist()
null_indices_vec = torch.where(S_vec <= 0.1)[0].tolist()

print(f"\n基于 S_vec (16维向量) 的索引:")
print(f"  Observed dims indices: {observed_indices_vec}")
print(f"  Null dims indices: {null_indices_vec}")

# 方法2：基于 S_img（4D）的索引
S_img_flat = S_img.flatten()  # [16]
observed_indices_4d = torch.where(S_img_flat > 0.1)[0].tolist()
null_indices_4d = torch.where(S_img_flat <= 0.1)[0].tolist()

print(f"\n基于 S_img (4D flatten) 的索引:")
print(f"  Observed dims indices: {observed_indices_4d}")
print(f"  Null dims indices: {null_indices_4d}")

# 检查是否一致
if observed_indices_vec == observed_indices_4d:
    print(f"\n✅ Observed indices 一致")
else:
    print(f"\n❌ Observed indices 不一致!")
    print(f"  S_vec: {observed_indices_vec}")
    print(f"  S_img: {observed_indices_4d}")

if null_indices_vec == null_indices_4d:
    print(f"✅ Null indices 一致")
else:
    print(f"❌ Null indices 不一致!")
    print(f"  S_vec: {null_indices_vec}")
    print(f"  S_img: {null_indices_4d}")

print("\n" + "="*80)
print("4. 理论方差 vs MCG-diff mask 对齐")
print("="*80)

# 检查理论方差在 observed/null 维度上的值
var_obs_theoretical = var_z_theoretical[observed_indices_vec].mean().item()
var_null_theoretical = var_z_theoretical[null_indices_vec].mean().item()

print(f"\n理论方差（基于 S_vec 索引）:")
print(f"  Observed dims variance (mean): {var_obs_theoretical:.6f}")
print(f"  Null dims variance (mean): {var_null_theoretical:.6f}")
print(f"  Ratio: {var_null_theoretical / var_obs_theoretical:.4f}")

# 检查理论方差在 4D 格式中的对应位置
var_z_4d_flat = var_z_4d.flatten()  # [16]
var_obs_4d = var_z_4d_flat[observed_indices_4d].mean().item()
var_null_4d = var_z_4d_flat[null_indices_4d].mean().item()

print(f"\n理论方差（基于 S_img 索引）:")
print(f"  Observed dims variance (mean): {var_obs_4d:.6f}")
print(f"  Null dims variance (mean): {var_null_4d:.6f}")
print(f"  Ratio: {var_null_4d / var_obs_4d:.4f}")

# 检查每个维度的对应关系
print(f"\n逐维度检查（前10个维度）:")
print(f"{'Index':<8} {'S_vec':<12} {'S_img_flat':<12} {'var_z':<12} {'Mask':<8} {'Type':<10}")
print("-" * 70)
for i in range(min(10, len(S_vec))):
    s_vec_val = S_vec[i].item()
    s_img_val = S_img_flat[i].item()
    var_val = var_z_theoretical[i].item()
    mask_val = svd_mask_4d.flatten()[i].item()
    dim_type = "Observed" if mask_val > 0.5 else "Null"
    print(f"{i:<8} {s_vec_val:<12.6e} {s_img_val:<12.6e} {var_val:<12.6f} {mask_val:<8.1f} {dim_type:<10}")

print("\n" + "="*80)
print("5. SVD 变换维度映射检查")
print("="*80)

# 检查 V 变换是否正确
# 创建一个测试向量：在 SVD 空间的第 i 个维度为 1，其他为 0
print(f"\n测试 SVD 变换（V 和 Vt）:")

# 测试：在 SVD 空间创建一个单位向量
test_svd = torch.zeros(1, 1, 4, 4)
test_svd[0, 0, 0, 0] = 1.0  # 第0个位置

# 转换到 image 空间
test_img = forward_op.V(test_svd)  # [1, 1, 4, 4]

# 转换回 SVD 空间
test_svd_back = forward_op.Vt(test_img)  # [1, 1, 4, 4]

print(f"  原始 SVD: {test_svd.flatten()[:5]}")
print(f"  转换到 image 再转回: {test_svd_back.flatten()[:5]}")
print(f"  是否一致: {torch.allclose(test_svd, test_svd_back, atol=1e-5)}")

# 检查 V @ Vt 是否等于单位矩阵
V_matrix = forward_op._Vt_matrix.T  # V
Vt_matrix = forward_op._Vt_matrix   # Vt
V_Vt = V_matrix @ Vt_matrix  # 应该是单位矩阵
print(f"\n  V @ Vt 是否接近单位矩阵: {torch.allclose(V_Vt, torch.eye(16), atol=1e-5)}")
print(f"  V @ Vt 的最大偏离: {(V_Vt - torch.eye(16)).abs().max().item():.6e}")

print("\n" + "="*80)
print("6. MCG-diff 实际使用的维度索引")
print("="*80)

# 模拟 MCG-diff 中的处理流程
# 1. 获取 S_img 和 svd_mask
S_img_mcg = forward_op.S.to('cpu')  # [1, 1, 4, 4]
svd_mask_mcg = (S_img_mcg > 0.1).float()  # [1, 1, 4, 4]

# 2. 展平查看索引
svd_mask_flat = svd_mask_mcg.flatten()  # [16]
observed_indices_mcg = torch.where(svd_mask_flat > 0.5)[0].tolist()
null_indices_mcg = torch.where(svd_mask_flat <= 0.5)[0].tolist()

print(f"\nMCG-diff 实际使用的索引:")
print(f"  Observed dims indices: {observed_indices_mcg}")
print(f"  Null dims indices: {null_indices_mcg}")

# 检查与理论索引是否一致
if set(observed_indices_vec) == set(observed_indices_mcg):
    print(f"  ✅ Observed indices 与理论一致")
else:
    print(f"  ❌ Observed indices 与理论不一致!")
    print(f"    理论: {observed_indices_vec}")
    print(f"    MCG:  {observed_indices_mcg}")

if set(null_indices_vec) == set(null_indices_mcg):
    print(f"  ✅ Null indices 与理论一致")
else:
    print(f"  ❌ Null indices 与理论不一致!")
    print(f"    理论: {null_indices_vec}")
    print(f"    MCG:  {null_indices_mcg}")

print("\n" + "="*80)
print("7. 维度映射可视化")
print("="*80)

# 创建一个映射表
print(f"\n{'SVD Index':<12} {'S Value':<12} {'Mask':<8} {'Var (Theory)':<15} {'Type':<10}")
print("-" * 70)
for i in range(16):
    s_val = S_vec[i].item()
    mask_val = svd_mask_flat[i].item()
    var_val = var_z_theoretical[i].item()
    dim_type = "Observed" if mask_val > 0.5 else "Null"
    print(f"{i:<12} {s_val:<12.6e} {mask_val:<8.1f} {var_val:<15.6f} {dim_type:<10}")

# 检查 4D reshape 的顺序
print(f"\n4D reshape 顺序检查:")
print(f"  S_vec[0:4]: {S_vec[0:4]}")
print(f"  S_img[0, 0, 0, :]: {S_img[0, 0, 0, :]}")
print(f"  是否一致: {torch.allclose(S_vec[0:4], S_img[0, 0, 0, :], atol=1e-6)}")

print("\n" + "="*80)
print("总结")
print("="*80)

# 最终检查
all_aligned = (
    set(observed_indices_vec) == set(observed_indices_mcg) and
    set(null_indices_vec) == set(null_indices_mcg) and
    torch.allclose(S_vec, S_img.flatten(), atol=1e-6)
)

if all_aligned:
    print("✅ 维度对齐检查通过：")
    print("  - S_vec 和 S_img 一致")
    print("  - Observed indices 一致")
    print("  - Null indices 一致")
    print("  - 理论方差索引与 MCG-diff mask 索引对应")
else:
    print("❌ 维度对齐检查失败：")
    if not torch.allclose(S_vec, S_img.flatten(), atol=1e-6):
        print("  - S_vec 和 S_img 不一致")
    if set(observed_indices_vec) != set(observed_indices_mcg):
        print("  - Observed indices 不一致")
    if set(null_indices_vec) != set(null_indices_mcg):
        print("  - Null indices 不一致")

print("="*80)
