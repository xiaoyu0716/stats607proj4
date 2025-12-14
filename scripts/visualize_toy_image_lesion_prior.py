"""
Script to generate and visualize 100 samples from toy_image_lesion prior.
10% with lesion, 90% without lesion.
Updated for new implementation with organ template and smooth texture.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from inverse_problems.toy_image_lesion import ToyImageLesionProblem

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize problem with recommended parameters
    # Note: The new implementation uses different parameter names
    # For now, we'll use the old interface but with updated parameters
    problem = ToyImageLesionProblem(
        blur_sigma=1.0,
        noise_std=0.03,
        tau=0.06,                 # 纹理强度更小更平滑 (was 0.2)
        lesion_prior_weight=0.10, # 10% 病灶
        lesion_amplitude=0.60,    # 病灶更亮 (was 0.25)
        lesion_radius=2,          # 病灶半径
        device=device
    )
    
    # Generate 100 samples (exact proportion: 10% lesion, 90% normal)
    n_samples = 100
    print(f"Generating {n_samples} samples from prior...")
    result = problem.sample_prior(n_samples, exact_proportion=True)
    
    # Check if new implementation returns (samples, labels) or just samples
    if isinstance(result, tuple):
        samples, labels = result
        has_labels = True
    else:
        samples = result
        has_labels = False
        # Fallback: detect lesion by comparing with mu_1
        mu_1 = problem.mu_1.cpu().numpy()
        mu_0 = problem.mu_0.cpu().numpy()
        labels = []
        for i in range(n_samples):
            sample = samples[i, 0].cpu().numpy()
            dist_to_mu1 = np.mean((sample - mu_1)**2)
            dist_to_mu0 = np.mean((sample - mu_0)**2)
            labels.append(bool(dist_to_mu1 < dist_to_mu0))  # Convert to Python bool
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Count how many have lesion
    if has_labels:
        n_lesion = labels.sum().item()
    else:
        n_lesion = sum(labels)
    n_normal = n_samples - n_lesion
    print(f"Generated samples: {n_lesion} with lesion, {n_normal} without lesion")
    
    # Create output directory
    output_dir = "exps/prior_samples/toy_image_lesion"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize all samples in a grid
    # Create a 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle(f'Toy Image Lesion Prior Samples (10×10 grid)\n{n_lesion} with lesion, {n_normal} without lesion', 
                 fontsize=16, fontweight='bold')
    
    for i in range(n_samples):
        row = i // 10
        col = i % 10
        ax = axes[row, col]
        
        sample = samples[i, 0].cpu().numpy()
        if has_labels:
            has_lesion = labels[i].item() == 1
        else:
            has_lesion = bool(labels[i])
        
        # Use grayscale colormap with [0,1] range
        # Upsample for better visualization (16x16 -> 64x64)
        sample_upsampled = zoom(sample, 4, order=1)
        
        im = ax.imshow(sample_upsampled, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        ax.axis('off')
        
        # Add label: L for lesion, N for normal
        label = 'L' if has_lesion else 'N'
        ax.text(0.05, 0.95, label, transform=ax.transAxes, 
                fontsize=10, fontweight='bold',
                color='white' if has_lesion else 'black',
                bbox=dict(boxstyle='round', facecolor='red' if has_lesion else 'gray', alpha=0.7))
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'prior_samples_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization to {grid_path}")
    plt.close()
    
    # Also save individual images (all samples, or first 20 as examples)
    individual_dir = os.path.join(output_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    # Compute global min/max for consistent scale across all images
    all_samples_np = samples[:, 0].cpu().numpy()
    global_vmin = all_samples_np.min()
    global_vmax = all_samples_np.max()
    # Add small margin
    margin = (global_vmax - global_vmin) * 0.05
    global_vmin = global_vmin - margin
    global_vmax = global_vmax + margin
    
    # Save all samples, not just first 20
    num_to_save = n_samples  # Change to min(20, n_samples) if you only want first 20
    for i in range(num_to_save):
        sample = samples[i, 0].cpu().numpy()
        if has_labels:
            has_lesion = labels[i].item() == 1
        else:
            has_lesion = bool(labels[i])
        
        # Upsample for better visualization
        sample_upsampled = zoom(sample, 4, order=1)
        
        # Use global scale for consistent visualization across all images
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(sample_upsampled, cmap='gray', vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
        ax.set_title(f'Sample {i+1}: {"Lesion" if has_lesion else "Normal"}', 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        img_path = os.path.join(individual_dir, f'sample_{i+1:03d}_{"lesion" if has_lesion else "normal"}.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved first 20 individual images to {individual_dir}/")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total samples: {n_samples}")
    print(f"Samples with lesion: {n_lesion} ({100*n_lesion/n_samples:.1f}%)")
    print(f"Samples without lesion: {n_normal} ({100*n_normal/n_samples:.1f}%)")
    
    # Value ranges
    all_values = samples[:, 0].cpu().numpy().flatten()
    if has_labels:
        lesion_mask = labels.cpu().numpy() == 1
    else:
        # Convert to numpy array without copy keyword to avoid deprecation warning
        lesion_mask = np.asarray(labels, dtype=bool)
    lesion_values = samples[lesion_mask, 0].cpu().numpy().flatten() if n_lesion > 0 else np.array([])
    normal_values = samples[~lesion_mask, 0].cpu().numpy().flatten() if n_normal > 0 else np.array([])
    
    print(f"\nValue ranges:")
    print(f"  All samples: [{all_values.min():.3f}, {all_values.max():.3f}], mean={all_values.mean():.3f}, std={all_values.std():.3f}")
    if len(lesion_values) > 0:
        print(f"  Lesion samples: [{lesion_values.min():.3f}, {lesion_values.max():.3f}], mean={lesion_values.mean():.3f}, std={lesion_values.std():.3f}")
    if len(normal_values) > 0:
        print(f"  Normal samples: [{normal_values.min():.3f}, {normal_values.max():.3f}], mean={normal_values.mean():.3f}, std={normal_values.std():.3f}")

if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
from torch import nn
import math

def _gaussian_kernel1d(sigma, ksize=None, device='cpu', dtype=torch.float32):
    if ksize is None:
        ksize = int(2*math.ceil(3*sigma)+1)
    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize-1)/2
    g = torch.exp(-0.5*(x/sigma)**2)
    g = g / g.sum()
    return g.view(1,1,-1), ksize

def _gaussian_blur(img, sigma):
    # img: (B,1,H,W)
    if sigma <= 0:
        return img
    g1d, k = _gaussian_kernel1d(sigma, device=img.device, dtype=img.dtype)
    pad = k//2
    # separable conv
    img = F.pad(img, (0,0,pad,pad), mode='reflect')
    img = F.conv2d(img, g1d.unsqueeze(3), groups=1)
    img = F.pad(img, (pad,pad,0,0), mode='reflect')
    img = F.conv2d(img, g1d.unsqueeze(2), groups=1)
    return img

class ToyImageLesionProblem(nn.Module):
    def __init__(self,
                 size=16,
                 blur_sigma=1.0,          # (若做前向算子时用)
                 noise_std=0.03,          # (若做测量噪声时用)
                 tau=0.12,                # prior 纹理强度（越小越平滑）
                 lesion_prior_weight=0.10,# 先验混合权重 π
                 lesion_amplitude=0.40,   # 病灶亮度
                 lesion_radius=3,         # 病灶半径（像素）
                 texture_sigma=1.0,       # 先验纹理的平滑 σ
                 seed=0,
                 device='cpu'):
        super().__init__()
        self.H = self.W = size
        self.device = torch.device(device)
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.rng = g

        self.pi = float(lesion_prior_weight)
        self.tau = float(tau)
        self.texture_sigma = float(texture_sigma)

        # -------- organ template (mu0) --------
        yy, xx = torch.meshgrid(torch.arange(self.H, device=self.device),
                                torch.arange(self.W, device=self.device), indexing='ij')
        cx, cy = (self.W-1)/2, (self.H-1)/2
        ax, ay = 0.70*self.W/2, 0.55*self.H/2    # 椭圆半轴
        ell = ((xx-cx)/ax)**2 + ((yy-cy)/ay)**2 <= 1.0

        base = 0.02
        organ = 0.3
        mu0 = base*torch.ones(self.H, self.W, device=self.device)
        mu0[ell] = organ

        # 轻微平滑一下边缘
        mu0 = _gaussian_blur(mu0.view(1,1,self.H,self.W), 0.6).view(self.H, self.W)

        # -------- lesion template (delta) --------
        # 让中心落在器官内部
        coords = torch.stack(torch.where(ell), dim=1)
        idx = torch.randint(0, coords.shape[0], (1,), generator=self.rng, device=self.device).item()
        cy_l, cx_l = coords[idx].tolist()

        yyf = yy.float(); xxf = xx.float()
        sigma_lesion = float(lesion_radius)
        bump = torch.exp(-((xxf-cx_l)**2 + (yyf-cy_l)**2)/(2*sigma_lesion**2))
        bump = bump / bump.max()
        bump = bump * float(lesion_amplitude)
        bump = bump * ell.float()  # 限制在器官内
        bump = _gaussian_blur(bump.view(1,1,self.H,self.W), 1.0).view(self.H, self.W)

        mu1 = (mu0 + bump).clamp(0.0, 1.0)

        self.register_buffer('mu_0', mu0)
        self.register_buffer('mu_1', mu1)
        self.blur_sigma = float(blur_sigma)
        self.noise_std = float(noise_std)

    @torch.no_grad()
    def sample_prior(self, n, exact_proportion=False):
        """
        返回：
          samples: (n,1,H,W) in [0,1]
          labels:  (n,) 0=normal, 1=lesion
        """
        device = self.device
        n1 = int(round(self.pi * n)) if exact_proportion else torch.distributions.Binomial(total_count=n, probs=self.pi).sample().int().item()
        n1 = min(max(n1, 0), n)
        n0 = n - n1

        # 构造平滑纹理噪声
        def draw_texture(k):
            z = torch.randn(k, 1, self.H, self.W, device=device, generator=self.rng)
            z = _gaussian_blur(z, self.texture_sigma)   # 平滑
            z = z * self.tau
            return z

        x0 = self.mu_0.view(1,1,self.H,self.W).repeat(n0,1,1,1) + draw_texture(n0)
        x1 = self.mu_1.view(1,1,self.H,self.W).repeat(n1,1,1,1) + draw_texture(n1)

        samples = torch.cat([x0, x1], dim=0).clamp(0,1)
        labels = torch.cat([torch.zeros(n0, dtype=torch.long, device=device),
                            torch.ones(n1, dtype=torch.long, device=device)], dim=0)

        # 打乱次序
        perm = torch.randperm(n, device=device, generator=self.rng)
        samples = samples[perm]
        labels = labels[perm]
        return samples, labels