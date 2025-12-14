#!/usr/bin/env python3
"""
重新生成16维的prior训练数据

这个脚本会生成真正的16维prior数据：
- 前8维: MoG (std约1.0)
- 后8维: 弱高斯先验 (std约2.24 = sqrt(5.0))
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 手动构建16D prior（避免创建完整的problem，因为可能有SVD bug）
dim_true = 16
gauss_rho = 0.8
mog8_mu = 2.0
mog8_wm_full = 0.5
mog8_wp_full = 0.5

print("=" * 80)
print("生成16维prior训练数据")
print("=" * 80)

# Prior means
means = torch.zeros(2, dim_true)  # [2, 16]
means[0, 7] = -mog8_mu  # -2.0
means[1, 7] = +mog8_mu  # +2.0

# Prior weights
w = torch.tensor([mog8_wm_full, mog8_wp_full])
weights = w / w.sum()

# Prior covariance
print("\n构建16D prior协方差矩阵...")
idx_8 = torch.arange(8)
absdiff_8 = (idx_8[:, None] - idx_8[None, :]).abs()
Sigma0_8x8 = (gauss_rho ** absdiff_8)  # [8, 8] - Toeplitz结构

weak_variance = 5.0
Sigma0_16x16 = torch.zeros(16, 16, dtype=torch.float32)
Sigma0_16x16[:8, :8] = Sigma0_8x8  # 前8x8块: MoG协方差
Sigma0_16x16[8:, 8:] = torch.eye(8) * weak_variance  # 后8x8块: 弱先验，方差=5.0

L = torch.linalg.cholesky(Sigma0_16x16)

print(f"Sigma_prior形状: {Sigma0_16x16.shape}")
print(f"前8x8块对角线: {torch.diag(Sigma0_16x16[:8, :8])}")
print(f"后8x8块对角线: {torch.diag(Sigma0_16x16[8:, 8:])} (应该是5.0)")

# 采样16维prior
N = 50000
print(f"\n采样 {N} 个16维prior样本...")
comp = torch.multinomial(weights, N, True)
means_samples = means[comp]  # [N, 16]
z = torch.randn(N, dim_true) @ L.T  # [N, 16]
v_16d = means_samples + z  # [N, 16]

# 验证采样结果
print(f"\n验证采样结果:")
print(f"  样本形状: {v_16d.shape}")
print(f"  前8维std: {v_16d[:, :8].std().item():.3f} (应该约1.0)")
print(f"  后8维std: {v_16d[:, 8:].std().item():.3f} (应该约2.24 = sqrt(5.0))")

# Reshape为图像格式 [N, 1, 4, 4]
x_img = v_16d.reshape(N, 1, 4, 4)

# 保存
output_file = "toy_gausscmog8_prior.pt"
torch.save(x_img.cpu(), output_file)
print(f"\n✓ 保存到 {output_file}")
print(f"  形状: {x_img.shape} (图像格式)")
print(f"  实际是16维向量，reshape为1x4x4图像")

# 验证保存的数据
loaded = torch.load(output_file)
loaded_vec = loaded.view(-1, 16)
print(f"\n验证保存的数据:")
print(f"  后8维std: {loaded_vec[:, 8:].std().item():.3f} (应该约2.24)")

print("\n" + "=" * 80)
print("数据生成完成！")
print("=" * 80)
