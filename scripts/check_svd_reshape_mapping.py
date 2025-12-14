#!/usr/bin/env python3
"""
检查 SVD 维度映射：确认 reshape 顺序是否正确

关键问题：
1. SVD 分解后的 S 是 [16] 向量，按奇异值大小排序（从大到小）
2. reshape 到 [1, 1, 4, 4] 后，索引对应关系是什么？
3. MCG-diff 中使用的 svd_mask 是否和理论计算的索引一致？
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.uq_simulation_analysis import generate_dataset

print("="*80)
print("SVD Reshape 映射检查")
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
S_vec = torch.from_numpy(dataset['S']).float()  # [16] - 奇异值向量（按大小排序）

print("\n1. SVD 分解后的奇异值（16维向量）")
print("-" * 80)
print(f"S_vec (16D): {S_vec}")
print(f"S_vec shape: {S_vec.shape}")
print(f"Observed dims (S > 0.1): {torch.where(S_vec > 0.1)[0].tolist()}")
print(f"Null dims (S <= 0.1): {torch.where(S_vec <= 0.1)[0].tolist()}")

print("\n2. forward_op.S (4D 格式)")
print("-" * 80)
S_img = forward_op.S  # [1, 1, 4, 4]
print(f"S_img shape: {S_img.shape}")
print(f"S_img:\n{S_img.squeeze()}")

# 检查 reshape 的对应关系
S_img_flat = S_img.flatten()  # [16]
print(f"\nS_img.flatten(): {S_img_flat}")

print("\n3. Reshape 对应关系检查")
print("-" * 80)
print("检查 S_vec 和 S_img.flatten() 是否一致：")
if torch.allclose(S_vec, S_img_flat, atol=1e-6):
    print("✅ S_vec 和 S_img.flatten() 一致")
    print("   Reshape 顺序：row-major (C-style)")
    print("   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> [1,1,4,4]")
    print("   对应关系：")
    print("   S_vec[i] <-> S_img[0, 0, i//4, i%4]")
else:
    print("❌ S_vec 和 S_img.flatten() 不一致！")
    print(f"   差异: {(S_vec - S_img_flat).abs().max().item():.6e}")
    print("   这可能导致维度映射错误！")

print("\n4. MCG-diff 中使用的 svd_mask")
print("-" * 80)
# MCG-diff 中的逻辑
S_flat_mcg = forward_op.S.view(1, 1, 1, 1, 16)  # [1,1,1,1,16]
svd_mask_mcg = (S_flat_mcg > 0.1).float()  # [1,1,1,1,16]
svd_mask_flat = svd_mask_mcg.flatten()  # [16]

print(f"S_flat_mcg shape: {S_flat_mcg.shape}")
print(f"svd_mask_mcg shape: {svd_mask_mcg.shape}")
print(f"svd_mask_flat: {svd_mask_flat}")

# 检查索引
obs_idx_mcg = (svd_mask_flat > 0.5).nonzero(as_tuple=True)[0].tolist()
null_idx_mcg = (svd_mask_flat <= 0.5).nonzero(as_tuple=True)[0].tolist()

print(f"\nMCG-diff 使用的索引:")
print(f"  Observed dims: {obs_idx_mcg}")
print(f"  Null dims: {null_idx_mcg}")

# 理论索引
obs_idx_theory = torch.where(S_vec > 0.1)[0].tolist()
null_idx_theory = torch.where(S_vec <= 0.1)[0].tolist()

print(f"\n理论索引（基于 S_vec）:")
print(f"  Observed dims: {obs_idx_theory}")
print(f"  Null dims: {null_idx_theory}")

if set(obs_idx_mcg) == set(obs_idx_theory):
    print("\n✅ 索引一致：MCG-diff 使用的索引与理论一致")
else:
    print("\n❌ 索引不一致！")
    print(f"  MCG-diff: {obs_idx_mcg}")
    print(f"  理论: {obs_idx_theory}")

print("\n5. 检查 V 矩阵的列向量顺序")
print("-" * 80)
Vt_matrix = forward_op._Vt_matrix  # [16, 16] - V^T
V_matrix = Vt_matrix.T  # [16, 16] - V

print(f"V_matrix shape: {V_matrix.shape}")
print(f"V_matrix 的列向量对应 SVD 的维度顺序")
print(f"V[:, i] 对应 S_vec[i] 的奇异方向")

# 检查 V 矩阵是否正交
V_Vt = V_matrix @ Vt_matrix
is_orthogonal = torch.allclose(V_Vt, torch.eye(16), atol=1e-5)
print(f"\nV @ Vt 是否接近单位矩阵: {is_orthogonal}")
if not is_orthogonal:
    print(f"  最大偏离: {(V_Vt - torch.eye(16)).abs().max().item():.6e}")

print("\n6. 关键问题：SVD 维度顺序 vs Reshape 顺序")
print("-" * 80)
print("""
SVD 分解：
  A = U @ diag(S) @ V^T
  
  其中：
  - S[0] 是最大的奇异值
  - S[15] 是最小的奇异值
  - V[:, 0] 对应 S[0] 的奇异方向
  - V[:, 15] 对应 S[15] 的奇异方向

Reshape 顺序：
  S_vec: [16] -> S_img: [1, 1, 4, 4]
  使用 row-major (C-style):
    S_vec[i] -> S_img[0, 0, i//4, i%4]
  
  例如：
    S_vec[0] -> S_img[0, 0, 0, 0]
    S_vec[1] -> S_img[0, 0, 0, 1]
    S_vec[4] -> S_img[0, 0, 1, 0]
    S_vec[15] -> S_img[0, 0, 3, 3]

关键问题：
  - SVD 的维度顺序是按照奇异值大小排序的（从大到小）
  - Reshape 到 [1,1,4,4] 后，索引的对应关系保持不变
  - 所以 S_vec[i] 和 S_img.flatten()[i] 应该一致
""")

print("\n7. 验证：创建一个测试向量")
print("-" * 80)
# 创建一个测试向量：在 SVD 空间的第 i 个维度为 1
test_idx = 0  # 测试第 0 个维度
test_svd_vec = torch.zeros(16)
test_svd_vec[test_idx] = 1.0

# Reshape 到 4D
test_svd_4d = test_svd_vec.reshape(1, 1, 4, 4)
print(f"测试向量（SVD 空间，第 {test_idx} 维为 1）:")
print(f"  SVD 向量: {test_svd_vec}")
print(f"  4D 格式:\n{test_svd_4d.squeeze()}")

# 转换到 image 空间
test_img = forward_op.V(test_svd_4d)  # [1, 1, 4, 4]
test_img_vec = forward_op._img_to_vec(test_img)  # [1, 16]

# 转换回 SVD 空间
test_img_back = forward_op._vec_to_img(test_img_vec)  # [1, 1, 4, 4]
test_svd_back = forward_op.Vt(test_img_back)  # [1, 1, 4, 4]
test_svd_back_vec = test_svd_back.flatten()  # [16]

print(f"\n转换到 image 空间再转回:")
print(f"  SVD 向量: {test_svd_back_vec}")
print(f"  是否一致: {torch.allclose(test_svd_vec, test_svd_back_vec, atol=1e-5)}")

print("="*80)
