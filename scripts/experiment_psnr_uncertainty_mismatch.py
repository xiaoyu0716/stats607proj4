"""
Experiment 1: PSNR vs Uncertainty Mismatch
Demonstrates that multiple "posterior feasible" x have vastly different PSNR,
showing that using PSNR/accuracy to judge "correct/incorrect" is misleading.

For 8D toy problem.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import math

def compute_psnr(x_pred, x_true, data_range=1.0):
    """Compute PSNR between predicted and true images."""
    mse = torch.mean((x_pred - x_true) ** 2).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(data_range / math.sqrt(mse))
    return psnr

def compute_posterior_log_density(x, A, y, mu_prior, Sigma_prior_inv, sigma_y):
    """
    Compute log p(x|y) for Gaussian prior and linear forward operator.
    
    For y = Ax + ε, ε ~ N(0, σ_y² I)
    Prior: x ~ N(μ_prior, Σ_prior)
    
    log p(x|y) ∝ -0.5 * (||Ax - y||²/σ_y² + (x - μ_prior)^T Σ_prior^{-1} (x - μ_prior))
    """
    # Data term: ||Ax - y||² / σ_y²
    if x.dim() > 2:
        x_vec = x.view(x.shape[0], -1)
    else:
        x_vec = x
    if y.dim() > 2:
        y_vec = y.view(y.shape[0], -1)
    else:
        y_vec = y
    
    Ax = (A @ x_vec.T).T
    data_term = torch.sum((Ax - y_vec) ** 2, dim=1) / (sigma_y ** 2)
    
    # Prior term: (x - μ_prior)^T Σ_prior^{-1} (x - μ_prior)
    diff = x_vec - mu_prior.unsqueeze(0) if mu_prior.dim() == 1 else x_vec - mu_prior
    # Compute: diff^T @ Sigma_prior_inv @ diff for each sample
    prior_term = torch.sum((diff @ Sigma_prior_inv) * diff, dim=1)
    
    log_density = -0.5 * (data_term + prior_term)
    return log_density

def experiment_psnr_uncertainty_mismatch(n=8, m=None, seed=42, use_existing_problem=False):
    """
    Run experiment showing PSNR vs uncertainty mismatch.
    
    Args:
        n: Input dimension (default 8)
        m: Observation dimension (default floor(n/2))
        seed: Random seed
    """
    if m is None:
        m = n // 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Experiment: n={n}, m={m}, underdetermined ratio: {m/n:.2f}")
    
    if use_existing_problem:
        # Use existing toy_gausscmog8 problem
        from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem
        problem = ToyGausscMoG8Problem(
            dim=n,
            A_type='random-gaussian',
            A_seed=seed,
            A_scale=1.0,
            noise_std=0.2236,
            gauss_rho=0.8,
            mog8_mu=2.0,
            mog8_wm_full=0.5,
            mog8_wp_full=0.5,
            device=device
        )
        A = problem.A
        # Extract effective A (first m rows if A is n×n)
        if A.shape[0] == A.shape[1] and m < n:
            # For underdetermined case, use first m rows
            A = A[:m, :]
        else:
            m = A.shape[0]
        
        # Get prior parameters
        mu_prior = torch.zeros(n, device=device)  # Simplified: use zero mean
        Sigma_prior = problem.Sigma0  # Use AR(1) covariance
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        
        # Generate true x* and observation y
        x_true, y = problem.generate_sample()
        x_true = x_true.view(-1)  # Flatten to [n]
        y = y.view(-1)[:m]  # Take first m dimensions
        sigma_y = problem.noise_std
    else:
        # 1. Generate random Gaussian A and orthogonalize columns
        A = torch.randn(m, n, device=device)
        # QR decomposition to ensure rank(A) = m
        Q, R = torch.linalg.qr(A.T)
        A = Q[:, :m].T  # A is now m×n with orthonormal rows
        print(f"A shape: {A.shape}, rank: {torch.linalg.matrix_rank(A)}")
        
        # 2. Prior: x ~ N(μ=0, Σ=I)
        mu_prior = torch.zeros(n, device=device)
        Sigma_prior = torch.eye(n, device=device)
        Sigma_prior_inv = torch.eye(n, device=device)  # Since Σ = I
        
        # 3. Generate true x* and observation y
        x_true = torch.randn(n, device=device)  # Sample from prior
        sigma_y = 0.1  # Observation noise std
        noise = sigma_y * torch.randn(m, device=device)
        y = A @ x_true + noise
    
    print(f"x_true range: [{x_true.min().item():.3f}, {x_true.max().item():.3f}]")
    print(f"y range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # 4. Compute null space of A
    A_np = A.cpu().numpy()
    null_space_basis = null_space(A_np)  # Returns (n, n-m) matrix
    null_space_basis = torch.from_numpy(null_space_basis).float().to(device)
    print(f"Null space dimension: {null_space_basis.shape[1]}")
    
    # 5. Generate x(λ) = x* + λv for different λ and null space directions
    lambda_range = np.linspace(-3.0, 3.0, 100)
    
    # Use first null space direction
    v = null_space_basis[:, 0]
    v = v / torch.norm(v)  # Normalize
    
    psnr_values = []
    log_density_values = []
    lambda_values = []
    
    print("\nComputing PSNR and log density for different λ...")
    for lam in lambda_range:
        x_lambda = x_true + lam * v
        
        # Compute PSNR
        psnr = compute_psnr(x_lambda.unsqueeze(0), x_true.unsqueeze(0))
        psnr_values.append(psnr)
        
        # Compute posterior log density
        log_dens = compute_posterior_log_density(
            x_lambda.unsqueeze(0), A, y.unsqueeze(0),
            mu_prior, Sigma_prior_inv, sigma_y
        ).item()
        log_density_values.append(log_dens)
        
        lambda_values.append(lam)
    
    # 6. Sample from multiple null space directions
    num_directions = min(10, null_space_basis.shape[1])
    num_samples_per_direction = 20
    
    scatter_psnr = []
    scatter_log_density = []
    
    print(f"\nSampling from {num_directions} null space directions...")
    for i in range(num_directions):
        v_i = null_space_basis[:, i]
        v_i = v_i / torch.norm(v_i)
        
        # Sample λ from a range
        lambdas = torch.linspace(-2.0, 2.0, num_samples_per_direction, device=device)
        
        for lam in lambdas:
            x_sample = x_true + lam * v_i
            
            # Project back to feasible set if needed (for noisy case)
            # For now, we use x_sample directly since Ax_sample ≈ Ax_true (in null space)
            
            psnr = compute_psnr(x_sample.unsqueeze(0), x_true.unsqueeze(0))
            log_dens = compute_posterior_log_density(
                x_sample.unsqueeze(0), A, y.unsqueeze(0),
                mu_prior, Sigma_prior_inv, sigma_y
            ).item()
            
            scatter_psnr.append(psnr)
            scatter_log_density.append(log_dens)
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: PSNR vs λ curve with log density
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(lambda_values, psnr_values, 'b-', linewidth=2, label='PSNR')
    line2 = ax1_twin.plot(lambda_values, log_density_values, 'r--', linewidth=2, label='Log Density')
    
    ax1.set_xlabel('λ (null space coefficient)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, color='b')
    ax1_twin.set_ylabel('Posterior Log Density', fontsize=12, color='r')
    ax1.set_title(f'PSNR vs λ (n={n}, m={m})\nSame y, different x(λ) = x* + λv', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Scatter plot - Log Density vs PSNR
    ax2 = axes[1]
    scatter = ax2.scatter(scatter_log_density, scatter_psnr, alpha=0.6, s=50, c='purple', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Posterior Log Density', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title(f'Log Density vs PSNR (n={n}, m={m})\nMultiple null space directions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation for key points
    # Find points with similar log density but different PSNR
    scatter_log_density_arr = np.array(scatter_log_density)
    scatter_psnr_arr = np.array(scatter_psnr)
    
    # Find range of log densities
    log_dens_range = scatter_log_density_arr.max() - scatter_log_density_arr.min()
    log_dens_bin_size = log_dens_range / 5
    
    # For each bin, show PSNR range
    for i in range(5):
        bin_min = scatter_log_density_arr.min() + i * log_dens_bin_size
        bin_max = bin_min + log_dens_bin_size
        mask = (scatter_log_density_arr >= bin_min) & (scatter_log_density_arr < bin_max)
        if mask.sum() > 0:
            psnr_in_bin = scatter_psnr_arr[mask]
            psnr_range = psnr_in_bin.max() - psnr_in_bin.min()
            if psnr_range > 5:  # Significant PSNR variation
                ax2.text(bin_min + log_dens_bin_size/2, psnr_in_bin.mean(),
                        f'PSNR range: {psnr_range:.1f}dB', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "exps/experiments/psnr_uncertainty_mismatch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'psnr_uncertainty_mismatch_n{n}_m{m}.png')
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
    
    # Find points with similar log density but different PSNR
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
    # Run for n=8, m=4
    result = experiment_psnr_uncertainty_mismatch(n=8, m=4, seed=42)
    
    # Optionally run for other dimensions
    # experiment_psnr_uncertainty_mismatch(n=4, m=2, seed=42)
    # experiment_psnr_uncertainty_mismatch(n=16, m=8, seed=42)
