#!/usr/bin/env python3
"""
训练16维toy_gausscmog8的diffusion prior模型

这个脚本会：
1. 生成16维的prior训练数据（使用ToyGausscMoG8Problem.sample_prior）
2. 训练一个16维的diffusion模型（ToyDiffusionMLP, dim=16）
3. 保存模型到 toy_gausscmog8_diffusion.pt

关键点：
- Prior是16维的：前8维是MoG，后8维是弱高斯先验（方差=5.0）
- 模型输入输出都是16维（reshape为1x4x4图像格式）
- 训练时使用图像格式 [N, 1, 4, 4]
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.toy_mlp_diffusion import ToyDiffusionMLP
from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="toy_gausscmog8")
def train(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 80)
    print("训练16维toy_gausscmog8的diffusion prior模型")
    print("=" * 80)

    # ============================================================
    # 步骤1: 加载或生成训练数据（16维prior）
    # ============================================================
    print("\n[步骤1] 加载16维prior训练数据...")
    
    prior_file = "toy_gausscmog8_prior.pt"
    
    # 检查是否存在训练数据，如果不存在或后8维std不对，则重新生成
    need_regenerate = False
    if os.path.exists(prior_file):
        existing_data = torch.load(prior_file, map_location='cpu')
        existing_vec = existing_data.view(-1, 16)
        back8_std = existing_vec[:, 8:].std().item()
        print(f"  发现现有数据: {existing_data.shape}")
        print(f"  后8维std: {back8_std:.3f}")
        if back8_std < 2.0:  # 如果后8维std太小，说明不是16D prior
            print(f"  ⚠️  现有数据的后8维std只有{back8_std:.3f}，不是2.24，需要重新生成！")
            need_regenerate = True
        else:
            print(f"  ✓ 现有数据正确（后8维std≈2.24）")
    else:
        print(f"  数据文件不存在，需要生成")
        need_regenerate = True
    
    if need_regenerate:
        print(f"\n  重新生成16维prior数据...")
        # 直接使用regenerate_prior_data.py的逻辑，避免创建完整的problem
        dim_true = 16
        means = torch.zeros(2, dim_true)
        means[0, 7] = -cfg.prior.mog8_mu
        means[1, 7] = +cfg.prior.mog8_mu
        w = torch.tensor([cfg.prior.mog8_wm_full, cfg.prior.mog8_wp_full])
        weights = w / w.sum()
        
        # Prior covariance
        idx_8 = torch.arange(8)
        absdiff_8 = (idx_8[:, None] - idx_8[None, :]).abs()
        Sigma0_8x8 = (cfg.prior.gauss_rho ** absdiff_8)
        weak_variance = 5.0
        Sigma0_16x16 = torch.zeros(16, 16, dtype=torch.float32)
        Sigma0_16x16[:8, :8] = Sigma0_8x8
        Sigma0_16x16[8:, 8:] = torch.eye(8) * weak_variance
        L = torch.linalg.cholesky(Sigma0_16x16)
        
        # 采样
        N = 50000
        comp = torch.multinomial(weights, N, True)
        means_samples = means[comp]
        z = torch.randn(N, dim_true) @ L.T
        v_16d = means_samples + z
        x0_img = v_16d.reshape(N, 1, 4, 4)
        
        torch.save(x0_img.cpu(), prior_file)
        print(f"  ✓ 生成并保存到 {prior_file}")
    
    # 加载训练数据
    x0 = torch.load(prior_file, map_location=device)
    x0 = x0.float()
    N = x0.shape[0]
    
    # 验证数据
    x0_vec = x0.view(N, -1)  # [N, 16]
    print(f"\n  训练数据验证:")
    print(f"    形状: {x0.shape} (图像格式 [N, 1, 4, 4])")
    print(f"    向量形状: {x0_vec.shape} (16维向量)")
    print(f"    前8维std: {x0_vec[:, :8].std().item():.3f} (应该约1.0)")
    print(f"    后8维std: {x0_vec[:, 8:].std().item():.3f} (应该约2.24 = sqrt(5.0))")
    assert x0_vec.shape[1] == 16, f"训练数据维度错误！应该是16维，但得到{x0_vec.shape[1]}维"
    assert x0_vec[:, 8:].std().item() > 2.0, f"后8维std太小！应该是2.24，但得到{x0_vec[:, 8:].std().item():.3f}"
    print(f"  ✓ 训练数据验证通过（16维，后8维std≈2.24）")

    # ============================================================
    # 步骤2: 构建模型（16维）
    # ============================================================
    print("\n[步骤2] 构建16维diffusion模型...")
    model = hydra.utils.instantiate(cfg.model).to(device)  # ToyDiffusionMLP(dim=16)
    print(f"  模型类型: {type(model).__name__}")
    print(f"  模型维度: {model.dim} (应该是16)")
    print(f"  模型输入: [batch, 16] 或 [batch, 1, 4, 4]")
    print(f"  模型输出: [batch, 16] 或 [batch, 1, 4, 4]")
    
    # 验证模型结构
    test_input = torch.randn(2, 1, 4, 4, device=device)
    test_sigma = torch.tensor(1.0, device=device)
    test_output = model(test_input, test_sigma)
    print(f"  测试输入形状: {test_input.shape}")
    print(f"  测试输出形状: {test_output.shape}")
    assert test_output.shape == test_input.shape, "模型输入输出形状不匹配！"
    print(f"  ✓ 模型结构正确")

    # ============================================================
    # 步骤3: 训练模型
    # ============================================================
    print("\n[步骤3] 开始训练...")
    opt = Adam(model.parameters(), lr=cfg.training.lr)
    
    x0 = x0_img.to(device)  # [N, 1, 4, 4]
    N = x0.shape[0]

    for step in range(cfg.training.total_steps):
        idx = torch.randint(0, N, (cfg.training.batch_size,))
        x = x0[idx]  # [batch_size, 1, 4, 4] - 16维图像格式

        # Sample sigma (VP schedule)
        t = torch.rand(cfg.training.batch_size, device=device)
        sigma = cfg.noise_schedule.sigma_min * (cfg.noise_schedule.sigma_max / cfg.noise_schedule.sigma_min) ** t

        noise = torch.randn_like(x)  # [batch_size, 1, 4, 4]
        x_t = x + sigma.view(-1, 1, 1, 1) * noise

        # 模型输入: x_t [batch, 1, 4, 4] (16维图像格式)
        # 模型内部会reshape为 [batch, 16] 进行处理
        eps_pred = model(x_t, sigma)  # [batch, 1, 4, 4]
        
        loss = ((eps_pred - noise)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"  step {step}/{cfg.training.total_steps} | loss={loss.item():.6f}")

    # ============================================================
    # 步骤4: 保存模型
    # ============================================================
    print("\n[步骤4] 保存模型...")
    saved_dict = {'ema': model.state_dict(), 'net': model.state_dict()}
    model_file = "toy_gausscmog8_diffusion.pt"
    torch.save(saved_dict, model_file)
    print(f"  ✓ 保存模型到 {model_file}")
    
    # 验证保存的模型
    checkpoint = torch.load(model_file, map_location='cpu')
    if 'net' in checkpoint:
        model_state = checkpoint['net']
        if 'net.4.weight' in model_state:
            output_dim = model_state['net.4.weight'].shape[0]
            print(f"  验证: 保存的模型输出维度 = {output_dim} (应该是16)")
            assert output_dim == 16, f"模型维度错误！应该是16，但得到{output_dim}"
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"\n模型文件: {model_file}")
    print(f"训练数据: {prior_file}")
    print(f"模型维度: 16维")
    print(f"Prior分布: 16D MoG (前8维MoG，后8维弱高斯先验)")

if __name__ == "__main__":
    train()
