#!/usr/bin/env python3
"""
MCG-diff 综合测试脚本：检查三个可疑点

1. 用 posterior covariance + V 算"理论 SVD var"，确认 obs/null index 跟 svd_mask 一致
2. 测 unconditional diffusion prior 的方差，看是不是本来就远小于 true prior
3. 在"完全一样的 code path 下"复现原 MoG toy（用 diagonal mask）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.uq_simulation_analysis import generate_dataset, load_model_and_algorithm
from algo.unconditional import UnconditionalDiffusionSampler

print("="*80)
print("MCG-diff 综合测试：检查三个可疑点")
print("="*80)

# ============================================================================
# Test 1: 理论 SVD 方差 vs svd_mask 对齐检查
# ============================================================================

print("\n" + "="*80)
print("Test 1: 理论 SVD 方差 vs svd_mask 对齐检查")
print("="*80)

# 生成 MRI-like 数据集
dataset = generate_dataset(
    A_type='mri_like',
    N=1,
    noise_std=0.5,
    seed=0,
    A_seed=1234
)

forward_op = dataset['problem']
A = torch.from_numpy(dataset['A']).float()  # [16, 16]
S_vec = torch.from_numpy(dataset['S']).float()  # [16]
V = torch.from_numpy(dataset['V']).float()  # [16, 16] (V^T in SVD)

# Prior covariance
Sigma_prior = forward_op.Sigma_prior  # [16, 16]

# Noise variance
sigma_noise = 0.5
sigma_noise_sq = sigma_noise ** 2

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

# MCG-diff 使用的 svd_mask
S_img = forward_op.S  # [1, 1, 4, 4]
svd_mask_4d = (S_img > 0.1).float()  # [1, 1, 4, 4]
svd_mask_flat = svd_mask_4d.flatten()  # [16]

# 将 var_z_theoretical reshape 到 4D 格式
var_z_4d = var_z_theoretical.reshape(1, 1, 4, 4)

# 分离 observed 和 null
observed_indices = torch.where(S_vec > 0.1)[0].tolist()
null_indices = torch.where(S_vec <= 0.1)[0].tolist()

var_obs_theoretical = var_z_theoretical[observed_indices].mean().item()
var_null_theoretical = var_z_theoretical[null_indices].mean().item()

print(f"\n理论后验方差（SVD 坐标）:")
print(f"  Observed dims (indices {observed_indices}): mean = {var_obs_theoretical:.6f}")
print(f"  Null dims (indices {null_indices}): mean = {var_null_theoretical:.6f}")
print(f"  Ratio: {var_null_theoretical / var_obs_theoretical:.4f}")

# 检查 4D reshape 后的对应关系
print(f"\n4D reshape 检查:")
print(f"  var_z_theoretical (16D):\n{var_z_theoretical}")
print(f"  var_z_4d (4D):\n{var_z_4d.squeeze()}")
print(f"  svd_mask_4d (4D):\n{svd_mask_4d.squeeze()}")

# 检查 observed 位置的理论方差
var_obs_4d = (var_z_4d * svd_mask_4d).sum() / svd_mask_4d.sum()
var_null_4d = (var_z_4d * (1 - svd_mask_4d)).sum() / (1 - svd_mask_4d).sum()

print(f"\n基于 4D mask 的理论方差:")
print(f"  Observed (mask=1): mean = {var_obs_4d.item():.6f}")
print(f"  Null (mask=0): mean = {var_null_4d.item():.6f}")
print(f"  Ratio: {var_null_4d.item() / var_obs_4d.item():.4f}")

# 验证一致性
if abs(var_obs_4d.item() - var_obs_theoretical) < 0.01:
    print(f"\n✅ Test 1 通过：理论方差在 16D 和 4D 格式下一致")
else:
    print(f"\n❌ Test 1 失败：理论方差不一致")
    print(f"  16D: {var_obs_theoretical:.6f}, 4D: {var_obs_4d.item():.6f}")

# ============================================================================
# Test 2: Unconditional Diffusion Prior 方差检查
# ============================================================================

print("\n" + "="*80)
print("Test 2: Unconditional Diffusion Prior 方差检查")
print("="*80)

# 加载模型和算法
net, _, algo_config = load_model_and_algorithm('MCG_diff', forward_op)

# 创建无条件采样器
unconditional_sampler = UnconditionalDiffusionSampler(
    net=net,
    forward_op=forward_op,
    diffusion_scheduler_config=algo_config['scheduler_config'],
    sde=False  # 使用 ODE 模式
)

# 生成 K 个无条件样本
K = 200
print(f"\n生成 K={K} 个无条件 diffusion prior 样本...")
samples_list = []
for k in range(K):
    # 创建一个 dummy observation（不会被使用）
    dummy_obs = torch.zeros(1, 1, 4, 4, device=forward_op.device)
    sample_k = unconditional_sampler.inference(dummy_obs, num_samples=1, verbose=False)
    samples_list.append(sample_k.cpu())

samples_stack = torch.stack(samples_list, dim=0)  # [K, 1, 4, 4]

# 转换到向量空间
samples_vec = forward_op._img_to_vec(samples_stack)  # [K, 16]

# 计算每个维度的方差
var_prior_per_dim = samples_vec.var(dim=0)  # [16]
var_prior_mean = var_prior_per_dim.mean().item()

print(f"\nUnconditional diffusion prior 方差:")
print(f"  每个维度的方差: {var_prior_per_dim}")
print(f"  平均方差: {var_prior_mean:.6f}")
print(f"  最小方差: {var_prior_per_dim.min().item():.6f}")
print(f"  最大方差: {var_prior_per_dim.max().item():.6f}")

# 对比理论 prior 方差
# 理论 prior: 前8维 Toeplitz (对角线≈1), 后8维 对角 (方差=5.0)
# 平均理论方差 ≈ (8*1 + 8*5) / 16 = 3.0
theoretical_prior_var_mean = (8 * 1.0 + 8 * 5.0) / 16.0

print(f"\n理论 prior 方差（平均）: {theoretical_prior_var_mean:.6f}")
print(f"实际 diffusion prior 方差（平均）: {var_prior_mean:.6f}")
print(f"相对误差: {abs(var_prior_mean - theoretical_prior_var_mean) / theoretical_prior_var_mean * 100:.2f}%")

if var_prior_mean < theoretical_prior_var_mean * 0.5:
    print(f"\n❌ Test 2 失败：Diffusion prior 严重 under-dispersed")
    print(f"  实际方差 ({var_prior_mean:.6f}) 远小于理论值 ({theoretical_prior_var_mean:.6f})")
    print(f"  这会导致 MCG-diff 无法恢复正确的 nullspace 方差")
else:
    print(f"\n✅ Test 2 通过：Diffusion prior 方差合理")

# 检查 SVD 坐标下的 prior 方差
samples_svd = samples_vec @ Vt_matrix  # [K, 16] - SVD space
var_prior_svd = samples_svd.var(dim=0)  # [16]

var_prior_obs = var_prior_svd[observed_indices].mean().item()
var_prior_null = var_prior_svd[null_indices].mean().item()

print(f"\nUnconditional prior 方差（SVD 坐标）:")
print(f"  Observed dims: mean = {var_prior_obs:.6f}")
print(f"  Null dims: mean = {var_prior_null:.6f}")
print(f"  Ratio: {var_prior_null / var_prior_obs:.4f}")

# ============================================================================
# Test 3: 用 Diagonal Mask 复现原 MoG toy
# ============================================================================

print("\n" + "="*80)
print("Test 3: 用 Diagonal Mask 复现原 MoG toy")
print("="*80)

# 创建一个简单的 diagonal mask A
# 前 8 个维度 observed，后 8 个维度 null
A_diagonal = torch.zeros(16, 16)
for i in range(8):
    A_diagonal[i, i] = 1.0  # 前 8 维 observed

# 创建 diagonal mask 的 problem
# 我们需要手动创建一个简单的 forward_op
print(f"\n创建 diagonal mask A (前8维 observed, 后8维 null)...")

# 使用 Identity A 类型，但我们需要一个简单的测试
# 实际上，我们可以直接测试 A=I 的情况
dataset_identity = generate_dataset(
    A_type='identity',
    N=1,
    noise_std=0.5,
    seed=0,
    A_seed=1234
)

forward_op_identity = dataset_identity['problem']

# 加载 MCG-diff 用于 A=I
net_identity, algo_identity, _ = load_model_and_algorithm('MCG_diff', forward_op_identity)

# 获取一个观测
observation_np = dataset_identity['y'][0:1]  # [1, 16] numpy
observation_img = forward_op_identity._vec_to_img(torch.from_numpy(observation_np))  # [1, 1, 4, 4]

# 生成 K 个后验样本
K_test = 20
print(f"生成 K={K_test} 个后验样本（A=I）...")
samples_list_identity = []
for k in range(K_test):
    sample_k = algo_identity.inference(observation_img, num_samples=1)
    samples_list_identity.append(sample_k)

samples_stack_identity = torch.stack(samples_list_identity, dim=0)  # [K, 1, 4, 4]

# 转换到向量空间
samples_vec_identity = forward_op_identity._img_to_vec(samples_stack_identity)  # [K, 16]

# 计算方差
var_identity_per_dim = samples_vec_identity.var(dim=0)  # [16]
var_identity_mean = var_identity_per_dim.mean().item()

print(f"\nMCG-diff 输出方差（A=I）:")
print(f"  每个维度的方差: {var_identity_per_dim}")
print(f"  平均方差: {var_identity_mean:.6f}")

# 对于 A=I，理论后验方差应该是 noise_std^2 = 0.25
theoretical_var_identity = 0.25

print(f"\n理论后验方差（A=I）: {theoretical_var_identity:.6f}")
print(f"实际方差（A=I）: {var_identity_mean:.6f}")
print(f"相对误差: {abs(var_identity_mean - theoretical_var_identity) / theoretical_var_identity * 100:.2f}%")

if abs(var_identity_mean - theoretical_var_identity) / theoretical_var_identity < 0.2:
    print(f"\n✅ Test 3 通过：A=I 时 MCG-diff 方差接近理论值")
else:
    print(f"\n❌ Test 3 失败：A=I 时 MCG-diff 方差偏离理论值")
    print(f"  可能原因：算法实现问题或 diffusion prior under-dispersion")

# ============================================================================
# 总结报告
# ============================================================================

print("\n" + "="*80)
print("总结报告")
print("="*80)

print(f"\nTest 1 (理论 SVD 方差对齐):")
if abs(var_obs_4d.item() - var_obs_theoretical) < 0.01:
    print(f"  ✅ 通过：理论方差在 16D 和 4D 格式下一致")
else:
    print(f"  ❌ 失败：理论方差不一致")

print(f"\nTest 2 (Unconditional diffusion prior 方差):")
if var_prior_mean >= theoretical_prior_var_mean * 0.5:
    print(f"  ✅ 通过：Diffusion prior 方差合理 ({var_prior_mean:.6f} vs {theoretical_prior_var_mean:.6f})")
else:
    print(f"  ❌ 失败：Diffusion prior 严重 under-dispersed ({var_prior_mean:.6f} vs {theoretical_prior_var_mean:.6f})")
    print(f"      → 这会导致 MCG-diff 无法恢复正确的 nullspace 方差")

print(f"\nTest 3 (A=I 复现):")
if abs(var_identity_mean - theoretical_var_identity) / theoretical_var_identity < 0.2:
    print(f"  ✅ 通过：A=I 时 MCG-diff 方差接近理论值")
else:
    print(f"  ❌ 失败：A=I 时 MCG-diff 方差偏离理论值")

print("\n" + "="*80)
print("诊断建议")
print("="*80)

if var_prior_mean < theoretical_prior_var_mean * 0.5:
    print("\n🔍 主要问题：Diffusion prior under-dispersion")
    print("  建议：")
    print("    1. 检查 diffusion model 的训练数据分布")
    print("    2. 检查 scheduler 的 sigma_max/sigma_min 设置")
    print("    3. 考虑使用更合适的 prior 模型")
elif abs(var_identity_mean - theoretical_var_identity) / theoretical_var_identity >= 0.2:
    print("\n🔍 主要问题：MCG-diff 算法实现问题")
    print("  建议：")
    print("    1. 检查 nullspace 更新公式（应该使用 x_t 而不是 x_next_t）")
    print("    2. 检查 resampling 策略（可能过度收缩方差）")
    print("    3. 检查 SVD 变换的正确性")
else:
    print("\n✅ 所有测试通过，问题可能在其他地方")
    print("  建议：")
    print("    1. 检查 MRI-like A 的 SVD 结构")
    print("    2. 检查 resampling 在复杂问题上的表现")

print("="*80)
