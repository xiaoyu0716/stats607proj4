"""
Experiment 1: PSNR vs Uncertainty Mismatch
Using existing toy_gausscmog8 problem (8D).

Demonstrates that multiple "posterior feasible" x have vastly different PSNR,
showing that using PSNR/accuracy to judge "correct/incorrect" is misleading.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import math

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

def compute_psnr(x_pred, x_true, data_range=None):
    """Compute PSNR between predicted and true vectors."""
    if isinstance(x_pred, torch.Tensor):
        x_pred = x_pred.cpu().numpy()
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.cpu().numpy()
    
    mse = np.mean((x_pred - x_true) ** 2)
    if mse == 0:
        return float('inf')
    
    if data_range is None:
        data_range = np.max(np.abs(x_true)) - np.min(np.abs(x_true))
        if data_range == 0:
            data_range = 1.0
    
    psnr = 20 * math.log10(data_range / math.sqrt(mse))
    return psnr

def compute_posterior_log_density(x, problem, y, sigma_y, A_effective, n_true=8):
    """
    Compute log p(x|y) for the toy_gausscmog8 problem.
    
    Args:
        x: Input vector [B, n_true] or [n_true]
        y: Observation [m] or [B, m]
        A_effective: Effective forward operator [m, n_true]
        n_true: True dimension (8, before padding)
    """
    if x.dim() > 1:
        x_vec = x.view(x.shape[0], -1)
        # Take only first n_true dimensions if padded
        if x_vec.shape[1] > n_true:
            x_vec = x_vec[:, :n_true]
    else:
        # Handle 1D input
        if x.shape[0] > n_true:
            x_vec = x[:n_true].unsqueeze(0)
        else:
            x_vec = x.unsqueeze(0)
    
    if y.dim() > 1:
        y_vec = y.view(y.shape[0], -1)
    else:
        y_vec = y.unsqueeze(0)
    
    m = A_effective.shape[0]
    y_effective = y_vec[:, :m] if y_vec.shape[1] >= m else y_vec
    
    # Data term: ||Ax - y||² / σ_y²
    Ax = (A_effective @ x_vec.T).T  # [B, m]
    data_term = torch.sum((Ax - y_effective) ** 2, dim=1) / (sigma_y ** 2)
    
    # Prior term: For mixture of Gaussians, use weighted sum
    # log p(x) = log(sum_k w_k * N(x | μ_k, Σ_k))
    # x_vec is [B, n_true], means are [2, n_true], Sigma0 is [n_true, n_true]
    Sigma0_inv = torch.linalg.inv(problem.Sigma_prior)
    log_pdfs = []
    for mu_k in problem.means:
        diff = x_vec - mu_k.unsqueeze(0)  # [B, n_true]
        # log N(x | μ_k, Σ_k) = -0.5 * (x-μ_k)^T Σ_k^{-1} (x-μ_k) + const
        quad = torch.sum((diff @ Sigma0_inv) * diff, dim=1)  # [B]
        log_pdf = -0.5 * quad
        log_pdfs.append(log_pdf)
    
    log_pdfs = torch.stack(log_pdfs, dim=1)  # [B, K]
    log_weights = torch.log(problem.weights).unsqueeze(0)  # [1, K]
    log_prior = torch.logsumexp(log_weights + log_pdfs, dim=1)  # [B]
    
    log_density = -0.5 * data_term + log_prior
    return log_density

def experiment_with_existing_problem():
    """Run experiment using existing toy_gausscmog8 problem."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Note: dim must be 16 (for 1x4x4 image), but true dimension is 8
    # Use A_obs_dim=4 to create underdetermined case (4 observations, 8 unknowns)
    m = 4  # Observation dimension
    n_true = 8  # True dimension (before padding)
    n = 16  # Padded dimension (for 1x4x4 image)
    
    problem = ToyGausscMoG8Problem(
        dim=n,  # Must be 16 for 1x4x4 image
        A_type='random-gaussian',
        A_seed=1234,
        A_scale=1.0,
        noise_std=0.2236,
        gauss_rho=0.8,
        mog8_mu=2.0,
        mog8_wm_full=0.5,
        mog8_wp_full=0.5,
        A_obs_dim=m,  # Underdetermined: 4 observations, 8 unknowns
        device=device
    )
    
    # Extract effective A (first m rows, first n_true columns)
    # problem.A is 16x16 (padded), we need the effective part
    # The class implementation pads with zeros, so top-left block is what we want
    A_effective = problem.A[:m, :n_true]  # [4, 8]
    
    print(f"Problem: n_true={n_true}, n_padded={n}, m={m}, underdetermined ratio: {m/n_true:.2f}")
    print(f"A_effective shape: {A_effective.shape}, rank: {torch.linalg.matrix_rank(A_effective)}")
    
    # Generate true x* and observation y
    x_true_img, y_full = problem.generate_sample()
    # x_true_img is [1, 4, 4] (16D padded), extract first 8 dimensions
    x_true = x_true_img.view(-1)[:n_true]  # [n_true] = [8]
    # y_full is also [1, 4, 4], but we only use first m dimensions
    y = y_full.view(-1)[:m]  # [m] = [4]
    
    sigma_y = problem.noise_std
    
    print(f"x_true range: [{x_true.min().item():.3f}, {x_true.max().item():.3f}]")
    print(f"y range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # Compute null space of A_effective (m x n_true = 4 x 8)
    A_np = A_effective.cpu().numpy()
    null_space_basis = null_space(A_np)  # Returns (n_true, n_true-m) = (8, 4) matrix
    null_space_basis = torch.from_numpy(null_space_basis).float().to(device)
    print(f"Null space dimension: {null_space_basis.shape[1]} (should be {n_true - m})")
    
    # Generate x(λ) = x* + λv for different λ
    lambda_range = np.linspace(-3.0, 3.0, 100)
    v = null_space_basis[:, 0]
    v = v / torch.norm(v)
    
    psnr_values = []
    log_density_values = []
    lambda_values = []
    
    print("\nComputing PSNR and log density for different λ...")
    for lam in lambda_range:
        x_lambda = x_true + lam * v
        
        psnr = compute_psnr(x_lambda, x_true)
        psnr_values.append(psnr)
        
        log_dens = compute_posterior_log_density(
            x_lambda.unsqueeze(0), problem, y.unsqueeze(0), sigma_y, A_effective, n_true
        ).item()
        log_density_values.append(log_dens)
        lambda_values.append(lam)
    
    # Sample from multiple null space directions
    num_directions = min(10, null_space_basis.shape[1])
    num_samples_per_direction = 20
    
    scatter_psnr = []
    scatter_log_density = []
    
    print(f"\nSampling from {num_directions} null space directions...")
    for i in range(num_directions):
        v_i = null_space_basis[:, i]
        v_i = v_i / torch.norm(v_i)
        lambdas = torch.linspace(-2.0, 2.0, num_samples_per_direction, device=device)
        
        for lam in lambdas:
            x_sample = x_true + lam * v_i
            psnr = compute_psnr(x_sample, x_true)
            log_dens = compute_posterior_log_density(
                x_sample.unsqueeze(0), problem, y.unsqueeze(0), sigma_y, A_effective, n_true
            ).item()
            scatter_psnr.append(psnr)
            scatter_log_density.append(log_dens)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: PSNR vs λ
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(lambda_values, psnr_values, 'b-', linewidth=2, label='PSNR')
    line2 = ax1_twin.plot(lambda_values, log_density_values, 'r--', linewidth=2, label='Log Density')
    
    ax1.set_xlabel('λ (null space coefficient)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, color='b')
    ax1_twin.set_ylabel('Posterior Log Density', fontsize=12, color='r')
    ax1.set_title(f'PSNR vs λ (n_true={n_true}, m={m})\nSame y, different x(λ) = x* + λv', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(scatter_log_density, scatter_psnr, alpha=0.6, s=50, c='purple', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Posterior Log Density', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title(f'Log Density vs PSNR (n_true={n_true}, m={m})\nMultiple null space directions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "exps/experiments/psnr_uncertainty_mismatch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'psnr_uncertainty_mismatch_toy8d_n{n_true}_m{m}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"PSNR range: [{min(psnr_values):.2f}, {max(psnr_values):.2f}] dB")
    print(f"PSNR std: {np.std(psnr_values):.2f} dB")
    print(f"Log density range: [{min(log_density_values):.2f}, {max(log_density_values):.2f}]")
    print(f"Log density std: {np.std(log_density_values):.2f}")
    
    # Find examples of mismatch
    print("\n=== Examples of Mismatch ===")
    scatter_log_density_arr = np.array(scatter_log_density)
    scatter_psnr_arr = np.array(scatter_psnr)
    
    sorted_indices = np.argsort(scatter_log_density_arr)
    mid_idx = len(sorted_indices) // 2
    similar_density_indices = sorted_indices[mid_idx-5:mid_idx+5]
    
    similar_density_psnr = scatter_psnr_arr[similar_density_indices]
    psnr_range_in_similar = similar_density_psnr.max() - similar_density_psnr.min()
    
    print(f"Points with similar log density (middle 10 points):")
    print(f"  Log density range: {scatter_log_density_arr[similar_density_indices].min():.2f} to {scatter_log_density_arr[similar_density_indices].max():.2f}")
    print(f"  PSNR range: {similar_density_psnr.min():.2f} to {similar_density_psnr.max():.2f} dB")
    print(f"  PSNR variation: {psnr_range_in_similar:.2f} dB")
    
    if psnr_range_in_similar > 10:
        print(f"\n✓ SUCCESS: Found {len(similar_density_indices)} points with similar posterior quality")
        print(f"  but PSNR varies by {psnr_range_in_similar:.2f} dB!")
        print(f"  This demonstrates that PSNR is misleading for underdetermined problems.")
    
    plt.show()
    
    return {
        'lambda_values': lambda_values,
        'psnr_values': psnr_values,
        'log_density_values': log_density_values,
        'scatter_psnr': scatter_psnr,
        'scatter_log_density': scatter_log_density
    }

if __name__ == "__main__":
    experiment_with_existing_problem()
