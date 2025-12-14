"""
Experiment 1: PSNR vs Uncertainty Mismatch (V2 - Improved Visualization)
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
import matplotlib.gridspec as gridspec
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
    """Compute log p(x|y) for the toy_gausscmog8 problem."""
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
    
    # Prior term: Mixture of Gaussians
    Sigma0_inv = torch.linalg.inv(problem.Sigma_prior)
    log_pdfs = []
    for mu_k in problem.means:
        diff = x_vec - mu_k.unsqueeze(0)  # [B, n_true]
        quad = torch.sum((diff @ Sigma0_inv) * diff, dim=1)  # [B]
        log_pdf = -0.5 * quad
        log_pdfs.append(log_pdf)
    
    log_pdfs = torch.stack(log_pdfs, dim=1)  # [B, K]
    log_weights = torch.log(problem.weights).unsqueeze(0)  # [1, K]
    log_prior = torch.logsumexp(log_weights + log_pdfs, dim=1)  # [B]
    
    log_density = -0.5 * data_term + log_prior
    return log_density

def get_credible_interval(lambdas, log_densities, threshold=0.95):
    """Find the credible interval for lambda based on log densities."""
    sort_idx = np.argsort(lambdas)
    lambdas_sorted = lambdas[sort_idx]
    log_densities_sorted = log_densities[sort_idx]
    
    # Avoid underflow
    log_densities_stable = log_densities_sorted - np.max(log_densities_sorted)
    probs = np.exp(log_densities_stable)
    
    # Trapezoidal integration
    areas = 0.5 * (probs[1:] + probs[:-1]) * (lambdas_sorted[1:] - lambdas_sorted[:-1])
    total_area = np.sum(areas)
    if total_area == 0:
        return lambdas_sorted[0], lambdas_sorted[-1], np.ones_like(lambdas, dtype=bool), probs
        
    pdf = probs / total_area
    
    # CDF
    cdf = np.cumsum(areas) / total_area
    cdf = np.insert(cdf, 0, 0.0)
    
    # Find interval [2.5%, 97.5%]
    lower_p = (1 - threshold) / 2
    upper_p = 1 - lower_p
    
    lambda_lower = np.interp(lower_p, cdf, lambdas_sorted)
    lambda_upper = np.interp(upper_p, cdf, lambdas_sorted)
    
    # Indices within interval (mapped back to original order)
    mask = (lambdas >= lambda_lower) & (lambdas <= lambda_upper)
    
    return lambda_lower, lambda_upper, mask, pdf

def experiment_with_improved_viz():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup Problem
    m = 4
    n_true = 8
    n = 16
    
    problem = ToyGausscMoG8Problem(
        dim=n,
        A_type='random-gaussian',
        A_seed=1234,
        A_scale=1.0,
        noise_std=0.2236,
        gauss_rho=0.8,
        mog8_mu=2.0,
        mog8_wm_full=0.5,
        mog8_wp_full=0.5,
        A_obs_dim=m,
        device=device
    )
    
    A_effective = problem.A[:m, :n_true]
    x_true_img, y_full = problem.generate_sample()
    x_true = x_true_img.view(-1)[:n_true]
    y = y_full.view(-1)[:m]
    sigma_y = problem.noise_std
    
    # Null space
    A_np = A_effective.cpu().numpy()
    null_space_basis = null_space(A_np)
    null_space_basis = torch.from_numpy(null_space_basis).float().to(device)
    print(f"Null space dim: {null_space_basis.shape[1]}")
    
    # Prepare Visualization
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    # --- Left Side: Analysis of ONE direction ---
    ax_dens = plt.subplot(gs[0, 0])
    ax_psnr = plt.subplot(gs[1, 0], sharex=ax_dens)
    
    # Use first null space direction
    v = null_space_basis[:, 0]
    v = v / torch.norm(v)
    
    # Scan lambda
    lambda_range = np.linspace(-4.0, 4.0, 200)
    psnr_vals = []
    log_dens_vals = []
    x_vals = []
    
    for lam in lambda_range:
        x_lambda = x_true + lam * v
        x_vals.append(x_lambda)
        psnr_vals.append(compute_psnr(x_lambda, x_true))
        log_dens_vals.append(compute_posterior_log_density(
            x_lambda.unsqueeze(0), problem, y.unsqueeze(0), sigma_y, A_effective, n_true
        ).item())
        
    psnr_vals = np.array(psnr_vals)
    log_dens_vals = np.array(log_dens_vals)
    x_vals = torch.stack(x_vals)
    
    # Find Credible Interval
    l_low, l_high, mask_cred, pdf = get_credible_interval(lambda_range, log_dens_vals)
    psnr_in_cred = psnr_vals[mask_cred]
    
    # Plot Density (Top Left)
    ax_dens.plot(lambda_range, pdf, 'k-', linewidth=2, label='Posterior PDF')
    # Highlight credible band
    ax_dens.axvspan(l_low, l_high, color='gray', alpha=0.2, label='95% Credible Interval')
    ax_dens.set_ylabel('Posterior Density', fontsize=12)
    ax_dens.set_title('Posterior density over $\lambda$ along null-space ray', fontsize=13, fontweight='bold')
    ax_dens.legend(loc='upper right')
    ax_dens.grid(True, alpha=0.3)
    
    # Plot PSNR (Bottom Left)
    ax_psnr.plot(lambda_range, psnr_vals, 'b-', linewidth=2)
    # Highlight credible band
    ax_psnr.axvspan(l_low, l_high, color='gray', alpha=0.2)
    
    # Annotate PSNR range in band
    if len(psnr_in_cred) > 0:
        psnr_min = psnr_in_cred.min()
        psnr_max = psnr_in_cred.max()
        ax_psnr.text(0.5, 0.1, f'PSNR Range in Band:\n[{psnr_min:.1f}, {psnr_max:.1f}] dB', 
                     transform=ax_psnr.transAxes, ha='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=12, color='b')
    ax_psnr.set_xlabel('$\lambda$ (distance from $x^*$ along $v$)', fontsize=12)
    ax_psnr.set_title('PSNR variation within credible interval', fontsize=13, fontweight='bold')
    ax_psnr.grid(True, alpha=0.3)
    
    # Select 3 Anchor Points (A, B, C) inside credible interval
    indices_in_cred = np.where(mask_cred)[0]
    if len(indices_in_cred) >= 3:
        # Pick min PSNR, max PSNR, and one random
        idx_min_psnr = indices_in_cred[np.argmin(psnr_vals[indices_in_cred])]
        idx_max_psnr = indices_in_cred[np.argmax(psnr_vals[indices_in_cred])]
        # Pick one with high density
        idx_high_dens = indices_in_cred[np.argmax(log_dens_vals[indices_in_cred])]
        
        anchors = [
            ('A', idx_max_psnr, 'g'),  # Best PSNR
            ('B', idx_high_dens, 'orange'), # High Prob
            ('C', idx_min_psnr, 'r')   # Worst PSNR
        ]
        
        # Sort anchors by lambda for plotting order
        anchors.sort(key=lambda x: lambda_range[x[1]])
        
        for label, idx, color in anchors:
            lam = lambda_range[idx]
            p = psnr_vals[idx]
            d = pdf[idx]
            
            # Mark on plots
            ax_dens.plot(lam, d, 'o', color=color, markersize=8)
            ax_dens.annotate(label, (lam, d), xytext=(0, 10), textcoords='offset points', 
                             ha='center', color=color, fontweight='bold')
            
            ax_psnr.plot(lam, p, 'o', color=color, markersize=8)
            ax_psnr.annotate(f"{label}\n{p:.1f}dB", (lam, p), xytext=(0, -25), textcoords='offset points', 
                             ha='center', color=color, fontweight='bold')
    
    # --- Right Side: Boxplots for Multiple Directions ---
    ax_box = plt.subplot(gs[:, 1])
    
    num_directions = min(10, null_space_basis.shape[1])
    psnr_distributions = []
    labels = []
    
    print(f"Sampling {num_directions} directions for boxplot...")
    
    for i in range(num_directions):
        v_i = null_space_basis[:, i]
        v_i = v_i / torch.norm(v_i)
        
        # Scan lambda for this direction
        psnr_dir = []
        log_dens_dir = []
        
        for lam in lambda_range:
            x_l = x_true + lam * v_i
            psnr_dir.append(compute_psnr(x_l, x_true))
            log_dens_dir.append(compute_posterior_log_density(
                x_l.unsqueeze(0), problem, y.unsqueeze(0), sigma_y, A_effective, n_true
            ).item())
            
        psnr_dir = np.array(psnr_dir)
        log_dens_dir = np.array(log_dens_dir)
        
        # Find credible interval for this direction
        _, _, mask_i, _ = get_credible_interval(lambda_range, log_dens_dir)
        
        if mask_i.sum() > 0:
            psnr_distributions.append(psnr_dir[mask_i])
            labels.append(f'Dir {i+1}')
    
    # Boxplot
    ax_box.boxplot(psnr_distributions, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red'))
    ax_box.set_ylabel('PSNR (dB)', fontsize=12)
    ax_box.set_xlabel('Null-space Directions', fontsize=12)
    ax_box.set_title('PSNR Range of Posterior-Plausible Solutions\n(Across different null-space directions)', 
                     fontsize=13, fontweight='bold')
    ax_box.grid(True, axis='y', alpha=0.3)
    
    # Add Inset Plots for A, B, C
    if len(indices_in_cred) >= 3:
        pos_list = [[0.05, 0.5, 0.15, 0.25], [0.42, 0.5, 0.15, 0.25], [0.8, 0.5, 0.15, 0.25]]
        for i, (label, idx, color) in enumerate(anchors):
            if i < 3:
                rect = pos_list[i]
                ax_inset = ax_dens.inset_axes(rect)
                
                x_vec_np = x_vals[idx].cpu().numpy()
                # Bar plot for 8D vector
                ax_inset.bar(range(8), x_vec_np, color=color, alpha=0.7)
                ax_inset.set_title(f"Sol {label}", fontsize=9, color=color, fontweight='bold')
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.spines['top'].set_visible(False)
                ax_inset.spines['right'].set_visible(False)
                ax_inset.spines['left'].set_visible(False)
                ax_inset.spines['bottom'].set_visible(False)
                ax_inset.axhline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    
    # Save figure
    output_dir = "exps/experiments/psnr_uncertainty_mismatch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'psnr_uncertainty_mismatch_v2_toy8d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    # Save data
    data_path = os.path.join(output_dir, 'experiment_data.pt')
    torch.save({
        'y': y,
        'x_true': x_true,
        'A_effective': A_effective,
        'anchors': {label: x_vals[idx] for label, idx, _ in anchors} if len(indices_in_cred) >= 3 else {},
        'psnr_distributions': psnr_distributions,
        'lambda_range': lambda_range,
        'pdf': pdf,
        'credible_mask': mask_cred
    }, data_path)
    print(f"Saved data to {data_path}")

if __name__ == "__main__":
    experiment_with_improved_viz()
