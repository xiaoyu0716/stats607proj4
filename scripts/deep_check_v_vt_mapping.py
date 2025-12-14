#!/usr/bin/env python3
"""
深度检查 V 和 Vt 的映射关系

关键问题：reshape 的顺序是否和矩阵乘法的顺序一致？
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("深度检查 V 和 Vt 的映射关系")
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
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - V^T
V_matrix = Vt_matrix.T  # [16, 16] - V
S_vec = torch.from_numpy(dataset['S']).float()  # [16]

print("\n1. 检查 reshape 顺序")
print("-" * 80)
# 创建一个测试：SVD 空间的第 i 维
for i in [0, 1, 4, 8, 9, 15]:
    test_svd_vec = torch.zeros(16)
    test_svd_vec[i] = 1.0
    
    # Reshape 到 4D
    test_svd_4d = test_svd_vec.reshape(1, 1, 4, 4)
    
    # 检查哪个位置是 1
    pos_4d = torch.where(test_svd_4d.squeeze() == 1.0)
    print(f"  SVD 维度 {i:2d} -> 4D 位置: {pos_4d}")
    
    # 检查 flatten 回去是否一致
    test_back = test_svd_4d.flatten()
    match = torch.allclose(test_svd_vec, test_back)
    print(f"    Reshape 一致性: {'✅' if match else '❌'}")

print("\n2. 检查 forward_op.V 的完整流程")
print("-" * 80)
# 测试：SVD 空间的第 0 维
test_svd_vec = torch.zeros(16)
test_svd_vec[0] = 1.0

print(f"输入（SVD 空间，第 0 维为 1）: {test_svd_vec}")

# 方法1：直接使用 V 矩阵
expected_img_vec = V_matrix @ test_svd_vec
print(f"直接计算 (V @ z): {expected_img_vec}")

# 方法2：使用 forward_op.V（完整流程）
test_svd_4d = test_svd_vec.reshape(1, 1, 4, 4)
print(f"Reshape 到 4D:\n{test_svd_4d.squeeze()}")

# forward_op.V 内部：
# 1. _img_to_vec: [1, 1, 4, 4] -> [1, 16]
x_vec_internal = forward_op._img_to_vec(test_svd_4d)  # [1, 16]
print(f"forward_op._img_to_vec: {x_vec_internal.squeeze()}")

# 2. x_vec @ Vt^T
x_img_vec = x_vec_internal @ forward_op._Vt_matrix.T  # [1, 16] @ [16, 16] = [1, 16]
print(f"x_vec @ Vt^T: {x_img_vec.squeeze()}")

# 3. _vec_to_img: [1, 16] -> [1, 1, 4, 4]
result_4d = forward_op._vec_to_img(x_img_vec)
print(f"forward_op._vec_to_img:\n{result_4d.squeeze()}")

# 最终结果
result_vec = forward_op._img_to_vec(result_4d).squeeze()
print(f"最终结果: {result_vec}")

match = torch.allclose(expected_img_vec, result_vec, atol=1e-5)
print(f"\n是否一致: {'✅' if match else '❌'}")
if not match:
    print(f"差异: {(expected_img_vec - result_vec).abs().max().item():.6e}")
    print(f"差异向量: {expected_img_vec - result_vec}")

print("\n3. 关键发现：检查 _img_to_vec 和 _vec_to_img 的对应关系")
print("-" * 80)
# 测试：创建一个已知的 4D 张量
test_4d = torch.zeros(1, 1, 4, 4)
test_4d[0, 0, 0, 0] = 1.0  # 位置 (0, 0)

# 转换为向量
test_vec = forward_op._img_to_vec(test_4d)  # [1, 16]
print(f"4D 位置 (0,0) -> 向量索引: {torch.where(test_vec.squeeze() == 1.0)[0].item()}")

# 转换回去
test_4d_back = forward_op._vec_to_img(test_vec)
print(f"向量索引 0 -> 4D 位置: {torch.where(test_4d_back.squeeze() == 1.0)}")

# 测试其他位置
for row in range(4):
    for col in range(4):
        test_4d = torch.zeros(1, 1, 4, 4)
        test_4d[0, 0, row, col] = 1.0
        test_vec = forward_op._img_to_vec(test_4d).squeeze()
        vec_idx = torch.where(test_vec == 1.0)[0].item()
        expected_idx = row * 4 + col
        match = (vec_idx == expected_idx)
        if not match:
            print(f"  位置 ({row}, {col}): 期望索引 {expected_idx}, 实际索引 {vec_idx} {'❌' if not match else '✅'}")

print("\n4. 检查 V 矩阵乘法的顺序")
print("-" * 80)
print("""
关键问题：
  - forward_op.V 实现: x_vec @ Vt^T
  - 这等价于: x_vec @ V
  - 但是，x_vec 是 reshape 后的结果
  - 需要确保 reshape 的顺序和矩阵乘法的顺序一致

如果 reshape 顺序是 row-major (C-style):
  - [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> [1,1,4,4]
  - 那么 x_vec[i] 对应 SVD 空间的第 i 维
  - 所以 x_vec @ V 应该正确

但是，如果 V 矩阵的列向量顺序和 S 的顺序不一致，就会有问题。
""")

# 检查 V 矩阵的列向量是否对应 S 的顺序
print("\n检查 V 矩阵的列向量顺序:")
for i in range(5):
    v_col = V_matrix[:, i]
    print(f"  V[:, {i}] 的前3个元素: {v_col[:3]}, 范数: {v_col.norm().item():.6f}")

print("\n5. 验证：完整的 round-trip")
print("-" * 80)
# 在 SVD 空间创建一个测试向量
test_svd_vec = torch.randn(16)

# 转换到 image 空间（使用 forward_op.V）
test_svd_4d = test_svd_vec.reshape(1, 1, 4, 4)
test_img_4d = forward_op.V(test_svd_4d)
test_img_vec = forward_op._img_to_vec(test_img_4d).squeeze()

# 转换回 SVD 空间（使用 forward_op.Vt）
test_img_4d_back = forward_op._vec_to_img(test_img_vec.unsqueeze(0))
test_svd_4d_back = forward_op.Vt(test_img_4d_back)
test_svd_vec_back = forward_op._img_to_vec(test_svd_4d_back).squeeze()

# 应该和原始一致
match = torch.allclose(test_svd_vec, test_svd_vec_back, atol=1e-5)
print(f"Round-trip 一致性: {'✅' if match else '❌'}")
if not match:
    print(f"  差异: {(test_svd_vec - test_svd_vec_back).abs().max().item():.6e}")

# 直接计算 round-trip
test_img_vec_direct = V_matrix @ test_svd_vec
test_svd_vec_back_direct = Vt_matrix @ test_img_vec_direct
match_direct = torch.allclose(test_svd_vec, test_svd_vec_back_direct, atol=1e-5)
print(f"直接计算 round-trip: {'✅' if match_direct else '❌'}")

print("="*80)
