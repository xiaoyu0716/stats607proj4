#!/usr/bin/env python3
"""
修复 forward_op.V 和 Vt 的实现

问题：
  - forward_op.V: x_vec @ Vt^T 计算的是 V 的行，而不是列
  - forward_op.Vt: x_vec @ Vt 计算的是 V^T 的行，而不是列

正确实现：
  - V: 应该计算 V @ z，即 V 的列
  - Vt: 应该计算 V^T @ x，即 V^T 的列

在 PyTorch 中：
  - 如果 x_vec 是 [B, 16]（行向量），那么：
    * V @ x_vec 会报错（维度不匹配）
    * x_vec @ V 计算的是 V 的行
    * 要计算 V 的列，需要：x_vec @ V.T 或者 (V @ x_vec.T).T
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("修复 forward_op.V 和 Vt 的实现")
print("="*80)

# 生成数据集
dataset = generate_dataset('mri_like', 1, 0.5, 0, 1234)
forward_op = dataset['problem']
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - V^T
V_matrix = Vt_matrix.T  # [16, 16] - V

print("\n1. 测试当前的实现")
print("-" * 80)
# 在 SVD 空间创建测试向量
z_svd = torch.zeros(16)
z_svd[0] = 1.0

# 正确的计算：V @ z
expected = V_matrix @ z_svd  # [16]

# 当前的实现：z @ Vt^T
z_4d = z_svd.reshape(1, 1, 4, 4)
z_vec = forward_op._img_to_vec(z_4d)  # [1, 16]
current = z_vec @ Vt_matrix.T  # [1, 16]

print(f"正确结果 (V @ z): {expected[:8]}")
print(f"当前实现 (z @ Vt^T): {current[0, :8]}")
print(f"是否一致: {torch.allclose(expected, current[0], atol=1e-5)}")

print("\n2. 修复方案")
print("-" * 80)
print("""
正确的实现应该是：
  - V: x_img = V @ z_svd
    * 如果 z_vec 是 [B, 16]（行向量），需要转置
    * 正确：x_img = (V_matrix @ z_vec.T).T 或者 x_img = z_vec @ V_matrix.T
    * 等等，z_vec @ V_matrix.T 就是当前实现，但为什么不对？
    
  - Vt: z_svd = V^T @ x_img
    * 正确：z_svd = (Vt_matrix @ x_vec.T).T 或者 z_svd = x_vec @ Vt_matrix
    * 当前实现是 x_vec @ Vt_matrix，这应该是正确的

让我重新检查...
""")

# 重新测试
print("\n3. 重新测试 V 变换")
print("-" * 80)
z_svd = torch.zeros(16)
z_svd[0] = 1.0

# 方法1：直接计算 V @ z (列向量)
expected1 = V_matrix @ z_svd  # [16]

# 方法2：z @ V^T (行向量)
z_row = z_svd.unsqueeze(0)  # [1, 16]
result2 = z_row @ V_matrix.T  # [1, 16] @ [16, 16] = [1, 16]

# 方法3：z @ V (行向量)
result3 = z_row @ V_matrix  # [1, 16] @ [16, 16] = [1, 16]

print(f"V @ z (列向量): {expected1[:8]}")
print(f"z @ V^T (行向量): {result2[0, :8]}")
print(f"z @ V (行向量): {result3[0, :8]}")
print(f"z @ V^T 是否等于 V @ z: {torch.allclose(expected1, result2[0], atol=1e-5)}")
print(f"z @ V 是否等于 V @ z: {torch.allclose(expected1, result3[0], atol=1e-5)}")

# 检查 V 和 V^T 的关系
print(f"\nV_matrix[0, :] (V 的第 0 行): {V_matrix[0, :][:8]}")
print(f"V_matrix[:, 0] (V 的第 0 列): {V_matrix[:, 0][:8]}")
print(f"Vt_matrix[0, :] (V^T 的第 0 行 = V 的第 0 列): {Vt_matrix[0, :][:8]}")
print(f"Vt_matrix[:, 0] (V^T 的第 0 列 = V 的第 0 行): {Vt_matrix[:, 0][:8]}")

print("\n4. 关键发现")
print("-" * 80)
print("""
如果 z_vec 是 [1, 16]（行向量），那么：
  - z_vec @ V 计算的是 V 的行（第 0 行）
  - z_vec @ V^T 计算的是 V^T 的行（第 0 行）= V 的第 0 列

所以：
  - forward_op.V: z_vec @ Vt^T = z_vec @ V，这应该是正确的
  - 但是测试结果显示不对，可能是因为 Vt_matrix 的定义有问题

让我检查 Vt_matrix 是否真的是 V^T...
""")

# 验证 Vt_matrix 是否真的是 V^T
A = torch.from_numpy(dataset['A']).float()
U, S, Vt = torch.linalg.svd(A, full_matrices=True)
print(f"\n直接 SVD 得到的 Vt[0, :]: {Vt[0, :][:8]}")
print(f"forward_op._Vt_matrix[0, :]: {forward_op._Vt_matrix[0, :][:8]}")
print(f"是否一致: {torch.allclose(Vt, forward_op._Vt_matrix, atol=1e-5)}")

print("="*80)
