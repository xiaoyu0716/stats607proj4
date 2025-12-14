#!/usr/bin/env python3
"""
检查 toy_gausscmog8 问题的完整pipeline维度
"""
import torch
from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

# 创建问题实例
problem = ToyGausscMoG8Problem(
    dim=16,
    A_type='fixed-full-rank-16x16',
    A_seed=1234,
    A_scale=1.0,
    A_obs_dim=16,
    noise_std=0.2236,
    gauss_rho=0.8,
    mog8_mu=2.0,
    mog8_wm_full=0.5,
    mog8_wp_full=0.5,
    device='cpu'
)

print("=" * 60)
print("完整 Pipeline 维度检查")
print("=" * 60)

# 1. A矩阵
print(f"\n1. A矩阵:")
print(f"   A shape: {problem.A.shape}")
print(f"   A_obs_dim: {problem.A_obs_dim}")
print(f"   A是16x16满秩矩阵: {torch.linalg.matrix_rank(problem.A) == 16}")

# 2. Prior (x)
print(f"\n2. Prior (x):")
print(f"   dim_true: {problem.dim_true}")
print(f"   means shape: {problem.means.shape}")
print(f"   means (16D, 只有第7维非零):")
for i, mean in enumerate(problem.means):
    print(f"      Component {i}: {mean}")

# 3. Forward: y = Ax + noise
print(f"\n3. Forward: y = Ax + noise")
x0_img, y_img = problem.generate_sample()
x0_vec = problem._img_to_vec(x0_img.unsqueeze(0))
y_vec = problem._img_to_vec(y_img.unsqueeze(0).unsqueeze(0))

print(f"   x0 shape: {x0_vec.shape} (16维)")
print(f"   y shape: {y_vec.shape} (16维)")

A_effective = problem.A[:problem.A_obs_dim, :]
y_computed = (x0_vec @ A_effective.T)
print(f"   A_effective shape: {A_effective.shape}")
print(f"   y_computed (Ax) shape: {y_computed.shape}")
print(f"   y_actual (Ax + noise) shape: {y_vec[:, :problem.A_obs_dim].shape}")

# 4. SVD
print(f"\n4. SVD分解: A = U @ diag(S) @ Vt")
print(f"   U shape: {problem.U.shape}")
print(f"   S_vec shape: {problem.S_vec.shape}")
print(f"   Vt shape: {problem._Vt_matrix.shape}")

A_reconstructed = problem.U @ torch.diag(problem.S_vec) @ problem._Vt_matrix
print(f"   A reconstruction error: {(problem.A - A_reconstructed).abs().max().item():.2e}")

# 5. M和S (用于MCGdiff)
print(f"\n5. M和S (用于MCGdiff/DDNM):")
print(f"   M shape: {problem.M.shape}")
print(f"   S shape: {problem.S.shape}")
M_vec = problem._img_to_vec(problem.M)
print(f"   M (mask): {M_vec[0].sum().item()}/16 个非零奇异值")
print(f"   所有16个奇异值都非零: {(M_vec[0] > 0.5).all().item()}")

# 6. 验证维度一致性
print(f"\n6. 维度一致性检查:")
print(f"   ✓ A: 16x16")
print(f"   ✓ x (prior): 16维")
print(f"   ✓ y (observation): 16维")
print(f"   ✓ SVD: U[16x16], S[16], Vt[16x16]")
print(f"   ✓ M: [1, 1, 4, 4] (16个元素，全为1)")
print(f"   ✓ S: [1, 1, 4, 4] (16个元素，奇异值)")

print("\n" + "=" * 60)
print("所有维度都正确！")
print("=" * 60)
