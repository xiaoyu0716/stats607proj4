#!/usr/bin/env python3
"""
检查 MCG-diff 实际输出的方差是否与理论后验方差在对应维度上对齐

关键检查：
1. 计算理论后验方差（SVD 坐标，16维）
2. 运行 MCG-diff 获取实际输出的方差（SVD 坐标，16维）
3. 逐维度对比，找出哪些维度偏差最大
4. 检查是否是维度映射问题还是算法问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset, load_model_and_algorithm

print("="*80)
print("MCG-diff 方差对齐检查（理论 vs 实际）")
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
print("1. 计算理论后验方差（SVD 坐标）")
print("="*80)

# 计算后验协方差（原始坐标 x）
A_T_A = A.T @ A
Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
Sigma_post = torch.linalg.inv(A_T_A / sigma_noise_sq + Sigma_prior_inv)  # [16, 16]

# 转换到 SVD 坐标 z = V^T x
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - 这是 Vt
V_matrix = Vt_matrix.T  # [16, 16] - 这是 V

# 在 SVD 坐标中的后验协方差
Sigma_post_z = V_matrix.T @ Sigma_post @ V_matrix  # [16, 16]
var_z_theoretical = torch.diag(Sigma_post_z)  # [16]

# 分离 observed 和 null
observed_indices = torch.where(S_vec > 0.1)[0].tolist()
null_indices = torch.where(S_vec <= 0.1)[0].tolist()

var_obs_theoretical = var_z_theoretical[observed_indices].mean().item()
var_null_theoretical = var_z_theoretical[null_indices].mean().item()

print(f"理论方差（SVD 坐标）:")
print(f"  Observed dims (indices {observed_indices}): mean = {var_obs_theoretical:.6f}")
print(f"  Null dims (indices {null_indices}): mean = {var_null_theoretical:.6f}")
print(f"  Ratio: {var_null_theoretical / var_obs_theoretical:.4f}")

print("\n" + "="*80)
print("2. 运行 MCG-diff 获取实际方差")
print("="*80)

# 加载模型和算法
net, algo, _ = load_model_and_algorithm('MCG_diff', forward_op)

# 获取一个观测
observation_np = dataset['y'][0:1]  # [1, 16] numpy
observation_img = forward_op._vec_to_img(torch.from_numpy(observation_np))  # [1, 1, 4, 4]

# 生成 K 个后验样本
K = 20
print(f"生成 K={K} 个后验样本...")
samples_list = []
for k in range(K):
    sample_k = algo.inference(observation_img, num_samples=1)
    samples_list.append(sample_k)

samples_stack = torch.stack(samples_list, dim=0)  # [K, 1, 4, 4]

# 转换到 SVD 空间进行分析
# 方法1：直接使用 Vt 转换
samples_vec = forward_op._img_to_vec(samples_stack)  # [K, 16] - image space
samples_svd = samples_vec @ Vt_matrix  # [K, 16] - SVD space

# 计算方差
var_z_actual = samples_svd.var(dim=0)  # [16] - 每个 SVD 维度的方差

var_obs_actual = var_z_actual[observed_indices].mean().item()
var_null_actual = var_z_actual[null_indices].mean().item()

print(f"实际方差（MCG-diff 输出，SVD 坐标）:")
print(f"  Observed dims (indices {observed_indices}): mean = {var_obs_actual:.6f}")
print(f"  Null dims (indices {null_indices}): mean = {var_null_actual:.6f}")
print(f"  Ratio: {var_null_actual / var_obs_actual:.4f}")

print("\n" + "="*80)
print("3. 逐维度对比")
print("="*80)

print(f"\n{'Index':<8} {'S Value':<12} {'Type':<10} {'Var (Theory)':<15} {'Var (Actual)':<15} {'Ratio':<10} {'Error %':<10}")
print("-" * 85)
for i in range(16):
    s_val = S_vec[i].item()
    var_theory = var_z_theoretical[i].item()
    var_actual = var_z_actual[i].item()
    dim_type = "Observed" if i in observed_indices else "Null"
    ratio = var_actual / var_theory if var_theory > 1e-10 else 0.0
    error_pct = abs(var_actual - var_theory) / var_theory * 100 if var_theory > 1e-10 else 0.0
    print(f"{i:<8} {s_val:<12.6e} {dim_type:<10} {var_theory:<15.6f} {var_actual:<15.6f} {ratio:<10.4f} {error_pct:<10.2f}")

print("\n" + "="*80)
print("4. 汇总统计")
print("="*80)

print(f"\nObserved dimensions:")
print(f"  理论均值: {var_obs_theoretical:.6f}")
print(f"  实际均值: {var_obs_actual:.6f}")
print(f"  相对误差: {abs(var_obs_actual - var_obs_theoretical) / var_obs_theoretical * 100:.2f}%")
print(f"  比率 (实际/理论): {var_obs_actual / var_obs_theoretical:.4f}")

print(f"\nNull dimensions:")
print(f"  理论均值: {var_null_theoretical:.6f}")
print(f"  实际均值: {var_null_actual:.6f}")
print(f"  相对误差: {abs(var_null_actual - var_null_theoretical) / var_null_theoretical * 100:.2f}%")
print(f"  比率 (实际/理论): {var_null_actual / var_null_theoretical:.4f}")

print(f"\nRatio:")
print(f"  理论: {var_null_theoretical / var_obs_theoretical:.4f}")
print(f"  实际: {var_null_actual / var_obs_actual:.4f}")
print(f"  相对误差: {abs((var_null_actual / var_obs_actual) - (var_null_theoretical / var_obs_theoretical)) / (var_null_theoretical / var_obs_theoretical) * 100:.2f}%")

print("\n" + "="*80)
print("5. 维度对齐验证")
print("="*80)

# 检查维度索引是否一致
S_img = forward_op.S  # [1, 1, 4, 4]
svd_mask_4d = (S_img > 0.1).float()
svd_mask_flat = svd_mask_4d.flatten()  # [16]
observed_indices_mcg = torch.where(svd_mask_flat > 0.5)[0].tolist()
null_indices_mcg = torch.where(svd_mask_flat <= 0.5)[0].tolist()

if set(observed_indices) == set(observed_indices_mcg):
    print("✅ Observed indices 对齐")
else:
    print("❌ Observed indices 不对齐!")
    print(f"  理论: {observed_indices}")
    print(f"  MCG:  {observed_indices_mcg}")

if set(null_indices) == set(null_indices_mcg):
    print("✅ Null indices 对齐")
else:
    print("❌ Null indices 不对齐!")
    print(f"  理论: {null_indices}")
    print(f"  MCG:  {null_indices_mcg}")

print("\n" + "="*80)
print("结论")
print("="*80)

if set(observed_indices) == set(observed_indices_mcg) and set(null_indices) == set(null_indices_mcg):
    print("✅ 维度索引对齐：MCG-diff 使用的 observed/null 维度与理论一致")
    print("\n如果方差仍然不匹配，可能的原因：")
    print("  1. Diffusion prior 本身 under-dispersed")
    print("  2. Resampling 导致方差收缩")
    print("  3. MCG-diff 更新公式的问题")
else:
    print("❌ 维度索引不对齐：这是导致方差不匹配的根本原因！")

print("="*80)
