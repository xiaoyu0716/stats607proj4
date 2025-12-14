#!/usr/bin/env python3
"""
验证 SVD 维度对应关系：检查 V 矩阵的列向量是否真的对应正确的 SVD 维度

关键问题：
1. SVD 分解：A = U @ diag(S) @ V^T
2. V 的列向量 V[:, i] 对应 S[i] 的奇异方向
3. 但是这些奇异方向是原始空间维度的线性组合
4. 需要验证：在 SVD 空间中，维度 i 是否真的对应 S[i] 和 V[:, i]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("SVD 维度对应关系验证")
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
S_vec = torch.from_numpy(dataset['S']).float()  # [16]
U = torch.from_numpy(dataset['U']).float()  # [16, 16]
V = torch.from_numpy(dataset['V']).float()  # [16, 16] (V^T in SVD)

Vt_matrix = forward_op._Vt_matrix  # [16, 16] - V^T
V_matrix = Vt_matrix.T  # [16, 16] - V

print("\n1. 验证 SVD 分解的正确性")
print("-" * 80)
# 重建 A
A_reconstructed = U @ torch.diag(S_vec) @ Vt_matrix
reconstruction_error = (A - A_reconstructed).abs().max().item()
print(f"A 重建误差: {reconstruction_error:.6e}")
if reconstruction_error < 1e-5:
    print("✅ SVD 分解正确")
else:
    print("❌ SVD 分解有误！")

print("\n2. 验证 V 矩阵的列向量对应关系")
print("-" * 80)
print("测试：在 SVD 空间的第 i 维为 1，转换到原始空间后，应该等于 V[:, i]")

for i in [0, 8, 9, 15]:  # 测试几个关键维度
    # 在 SVD 空间创建单位向量
    z_svd = torch.zeros(16)
    z_svd[i] = 1.0
    
    # 转换到原始空间：x = V @ z
    x_original = V_matrix @ z_svd  # [16]
    
    # 应该等于 V[:, i]
    v_col_i = V_matrix[:, i]  # [16]
    
    match = torch.allclose(x_original, v_col_i, atol=1e-5)
    print(f"  维度 {i:2d}: {'✅' if match else '❌'} x = V @ e_i 是否等于 V[:, {i}]")
    if not match:
        print(f"    差异: {(x_original - v_col_i).abs().max().item():.6e}")

print("\n3. 验证 Vt 变换（原始空间 -> SVD 空间）")
print("-" * 80)
print("测试：在原始空间的单位向量，转换到 SVD 空间后，应该等于 V^T 的对应行")

for i in [0, 8, 9, 15]:  # 测试几个关键维度
    # 在原始空间创建单位向量
    x_original = torch.zeros(16)
    x_original[i] = 1.0
    
    # 转换到 SVD 空间：z = V^T @ x
    z_svd = Vt_matrix @ x_original  # [16]
    
    # 应该等于 Vt[i, :]（V^T 的第 i 行）
    vt_row_i = Vt_matrix[i, :]  # [16]
    
    match = torch.allclose(z_svd, vt_row_i, atol=1e-5)
    print(f"  维度 {i:2d}: {'✅' if match else '❌'} z = V^T @ e_i 是否等于 V^T[{i}, :]")
    if not match:
        print(f"    差异: {(z_svd - vt_row_i).abs().max().item():.6e}")

print("\n4. 验证 forward_op.V 和 Vt 的对应关系")
print("-" * 80)
# 测试 forward_op 的 V 和 Vt 方法
test_svd_4d = torch.zeros(1, 1, 4, 4)
test_svd_4d[0, 0, 0, 0] = 1.0  # 第 0 个位置

# 使用 forward_op.V
test_img = forward_op.V(test_svd_4d)  # [1, 1, 4, 4]
test_img_vec = forward_op._img_to_vec(test_img)  # [1, 16]

# 手动计算：应该等于 V[:, 0]
test_svd_vec = torch.zeros(16)
test_svd_vec[0] = 1.0
expected_img_vec = (V_matrix @ test_svd_vec).unsqueeze(0)  # [1, 16]

match = torch.allclose(test_img_vec, expected_img_vec, atol=1e-5)
print(f"forward_op.V 是否正确: {'✅' if match else '❌'}")
if not match:
    print(f"  差异: {(test_img_vec - expected_img_vec).abs().max().item():.6e}")

# 测试 forward_op.Vt
test_img_4d = torch.zeros(1, 1, 4, 4)
test_img_4d[0, 0, 0, 0] = 1.0  # 第 0 个位置

test_svd_back = forward_op.Vt(test_img_4d)  # [1, 1, 4, 4]
test_svd_back_vec = forward_op._img_to_vec(test_svd_back)  # [1, 16]

# 手动计算：应该等于 V^T[0, :]
test_img_vec_manual = torch.zeros(16)
test_img_vec_manual[0] = 1.0
expected_svd_vec = (Vt_matrix @ test_img_vec_manual).unsqueeze(0)  # [1, 16]

match = torch.allclose(test_svd_back_vec, expected_svd_vec, atol=1e-5)
print(f"forward_op.Vt 是否正确: {'✅' if match else '❌'}")
if not match:
    print(f"  差异: {(test_svd_back_vec - expected_svd_vec).abs().max().item():.6e}")

print("\n5. 关键检查：SVD 空间中的维度索引是否对应正确的奇异值")
print("-" * 80)
print("""
在 SVD 空间中：
  - 维度 i 对应奇异值 S[i]
  - 维度 i 的奇异方向是 V[:, i]
  - 如果 S[i] > 0.1，则维度 i 是 observed
  - 如果 S[i] <= 0.1，则维度 i 是 null

MCG-diff 中：
  - svd_mask[i] = 1 if S[i] > 0.1 else 0
  - 使用 svd_mask 来区分 observed 和 null 维度

需要验证：
  - MCG-diff 使用的 svd_mask 索引是否和理论计算的索引一致
  - V 矩阵的列向量顺序是否和 S 的顺序一致
""")

# 检查 S 的顺序
print(f"\nS 的顺序（前10个）:")
for i in range(10):
    print(f"  S[{i}] = {S_vec[i]:.6e}, {'Observed' if S_vec[i] > 0.1 else 'Null'}")

# 检查 V 矩阵的列向量
print(f"\nV 矩阵的列向量（前3个）:")
for i in range(3):
    v_col = V_matrix[:, i]
    print(f"  V[:, {i}] 的范数: {v_col.norm().item():.6f}")
    print(f"  V[:, {i}] 的前5个元素: {v_col[:5]}")

print("\n6. 验证：在 SVD 空间中，observed 和 null 维度的实际含义")
print("-" * 80)
print("""
理论：
  - Observed dims (S > 0.1): 这些维度被观测约束，后验方差应该较小
  - Null dims (S <= 0.1): 这些维度不被观测约束，后验方差应该较大（接近 prior）

如果维度映射错误：
  - 可能会把 observed 维度当作 null 维度处理
  - 或者把 null 维度当作 observed 维度处理
  - 这会导致方差估计错误
""")

# 创建一个测试：在 null 维度添加噪声，应该不影响观测
print("\n测试：在 null 维度添加噪声")
test_x_original = torch.randn(16)
test_y = A @ test_x_original  # 观测

# 在 SVD 空间
test_z_svd = Vt_matrix @ test_x_original  # [16]

# 在 null 维度添加噪声
test_z_svd_perturbed = test_z_svd.clone()
null_indices = torch.where(S_vec <= 0.1)[0]
test_z_svd_perturbed[null_indices] += torch.randn(len(null_indices)) * 10.0  # 大噪声

# 转换回原始空间
test_x_perturbed = V_matrix @ test_z_svd_perturbed  # [16]

# 检查观测是否改变
test_y_perturbed = A @ test_x_perturbed
obs_change = (test_y - test_y_perturbed).abs().max().item()

print(f"  在 null 维度添加噪声后，观测的变化: {obs_change:.6e}")
if obs_change < 1e-3:
    print("  ✅ Null 维度确实不影响观测（维度映射正确）")
else:
    print("  ❌ Null 维度影响了观测（维度映射可能错误！）")

print("="*80)
