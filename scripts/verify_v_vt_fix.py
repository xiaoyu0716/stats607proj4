#!/usr/bin/env python3
"""
验证 forward_op.V 和 Vt 的修复是否正确

修复内容：
1. forward_op.V: 从 x_vec @ Vt_matrix.T 改为 x_vec @ Vt_matrix
2. forward_op.Vt: 从 x_vec @ Vt_matrix 改为 x_vec @ Vt_matrix.T
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("验证 forward_op.V 和 Vt 的修复")
print("="*80)

# 生成数据集
dataset = generate_dataset('mri_like', 1, 0.5, 0, 1234)
forward_op = dataset['problem']
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - V^T
V_matrix = Vt_matrix.T  # [16, 16] - V

print("\n1. 测试 forward_op.V (SVD 空间 -> Image 空间)")
print("-" * 80)
# 在 SVD 空间创建测试向量
z_svd = torch.zeros(16)
z_svd[0] = 1.0

# 正确的计算：V @ z
expected_v = V_matrix @ z_svd

# 使用 forward_op.V
z_4d = z_svd.reshape(1, 1, 4, 4)
result_v = forward_op.V(z_4d)
result_v_vec = forward_op._img_to_vec(result_v).squeeze()

print(f"正确结果 (V @ z): {expected_v[:8]}")
print(f"forward_op.V:     {result_v_vec[:8]}")
print(f"是否一致: {torch.allclose(expected_v, result_v_vec, atol=1e-5)}")

print("\n2. 测试 forward_op.Vt (Image 空间 -> SVD 空间)")
print("-" * 80)
# 在 Image 空间创建测试向量
x_img = torch.zeros(16)
x_img[0] = 1.0

# 正确的计算：V^T @ x
expected_vt = Vt_matrix @ x_img

# 使用 forward_op.Vt
x_4d = x_img.reshape(1, 1, 4, 4)
result_vt = forward_op.Vt(x_4d)
result_vt_vec = forward_op._img_to_vec(result_vt).squeeze()

print(f"正确结果 (V^T @ x): {expected_vt[:8]}")
print(f"forward_op.Vt:     {result_vt_vec[:8]}")
print(f"是否一致: {torch.allclose(expected_vt, result_vt_vec, atol=1e-5)}")

print("\n3. 测试 Round-trip (SVD -> Image -> SVD)")
print("-" * 80)
# 在 SVD 空间创建随机向量
z_svd_test = torch.randn(16)

# SVD -> Image
z_4d_test = z_svd_test.reshape(1, 1, 4, 4)
x_4d_test = forward_op.V(z_4d_test)

# Image -> SVD
z_4d_back = forward_op.Vt(x_4d_test)
z_svd_back = forward_op._img_to_vec(z_4d_back).squeeze()

print(f"原始 SVD 向量: {z_svd_test[:8]}")
print(f"Round-trip 后:  {z_svd_back[:8]}")
print(f"是否一致: {torch.allclose(z_svd_test, z_svd_back, atol=1e-5)}")

print("\n4. 关键发现")
print("-" * 80)
print("""
修复前的问题：
  - forward_op.V: x_vec @ Vt_matrix.T 计算的是 V 的行，而不是列
  - forward_op.Vt: x_vec @ Vt_matrix 计算的是 V^T 的行，而不是列
  
  这导致 SVD 空间和 Image 空间之间的转换不正确，可能造成：
  - observed/null 维度的更新应用到错误的维度上
  - 即使 svd_mask 的索引是对的，但实际更新的维度可能是错的

修复后：
  - forward_op.V: x_vec @ Vt_matrix 正确计算 V @ z
  - forward_op.Vt: x_vec @ Vt_matrix.T 正确计算 V^T @ x
  
  现在 SVD 空间和 Image 空间之间的转换是正确的。
""")

print("="*80)
