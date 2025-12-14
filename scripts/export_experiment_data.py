"""
Export detailed data for PSNR vs Uncertainty Mismatch experiment.
Reproduces the exact setting (seed 1234) and exports:
1. A, y, x_true
2. Multiple x(lambda) samples along null-space
3. Corresponding PSNR and log-density values
4. Visualization of the results
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import math
import pandas as pd

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

def compute_psnr(x_pred, x_true):
    mse = np.mean((x_pred - x_true) ** 2)
    if mse == 0: return float('inf')
    data_range = np.max(np.abs(x_true)) - np.min(np.abs(x_true))
    if data_range == 0: data_range = 1.0
    return 20 * math.log10(data_range / math.sqrt(mse))

def compute_log_density(x, problem, y, sigma_y, A_effective, n_true=8):
    if x.dim() == 1: x = x.unsqueeze(0)
    if x.shape[1] > n_true: x = x[:, :n_true]
    
    # Data term
    if y.dim() == 1: y = y.unsqueeze(0)
    m = A_effective.shape[0]
    y_eff = y[:, :m]
    Ax = (A_effective @ x.T).T
    data_term = torch.sum((Ax - y_eff) ** 2, dim=1) / (sigma_y ** 2)
    
    # Prior term
    Sigma0_inv = torch.linalg.inv(problem.Sigma_prior)
    log_pdfs = []
    for mu_k in problem.means:
        diff = x - mu_k.unsqueeze(0)
        quad = torch.sum((diff @ Sigma0_inv) * diff, dim=1)
        log_pdfs.append(-0.5 * quad)
    
    log_pdfs = torch.stack(log_pdfs, dim=1)
    log_weights = torch.log(problem.weights).unsqueeze(0)
    log_prior = torch.logsumexp(log_weights + log_pdfs, dim=1)
    
    return (-0.5 * data_term + log_prior).item()

def visualize_results(lambdas, psnrs, log_densities, output_path):
    """Create a simple visualization of PSNR and Log Density vs Lambda."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Lambda (Null-space coefficient)')
    ax1.set_ylabel('PSNR (dB)', color=color)
    
    # Handle infinite PSNR for plotting (replace with a slightly higher value than max)
    valid_psnrs = [p if p != float('inf') else np.nan for p in psnrs]
    max_val = np.nanmax(valid_psnrs) if not all(np.isnan(valid_psnrs)) else 0
    plot_psnrs = [p if p != float('inf') else max_val + 5 for p in psnrs]
    
    ax1.plot(lambdas, plot_psnrs, color=color, marker='o', label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Log Posterior Density', color=color)
    ax2.plot(lambdas, log_densities, color=color, linestyle='--', marker='x', label='Log Density')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PSNR vs Log Posterior Density along Null-space Ray')
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")

def export_data():
    device = 'cpu'
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    
    # --- 1. Setup Problem ---
    m, n_true, n_padded = 4, 8, 16
    problem = ToyGausscMoG8Problem(
        dim=n_padded, A_type='random-gaussian', A_seed=1234,
        A_scale=1.0, noise_std=0.2236, gauss_rho=0.8,
        mog8_mu=2.0, mog8_wm_full=0.5, mog8_wp_full=0.5,
        A_obs_dim=m, device=device
    )
    
    A = problem.A[:m, :n_true]
    x_true_img, y_full = problem.generate_sample()
    x_true = x_true_img.view(-1)[:n_true]
    y = y_full.view(-1)[:m]
    
    # --- 2. Null Space ---
    A_np = A.numpy()
    ns = null_space(A_np)
    v = torch.from_numpy(ns[:, 0]).float()
    v = v / torch.norm(v)
    
    # --- 3. Generate Samples ---
    lambdas = np.linspace(-2.5, 2.5, 21).tolist()
    
    print("="*80)
    print(f"1. SYSTEM INFO")
    print(f"A (Forward Operator, {m}x{n_true}):\n{A.numpy()}")
    print(f"\ny (Observation, {m} dim):\n{y.numpy()}")
    print(f"\nx_true (Ground Truth, {n_true} dim):\n{x_true.numpy()}")
    
    results = []
    psnrs_for_plot = []
    log_dens_for_plot = []
    
    for lam in lambdas:
        x_sample = x_true + lam * v
        psnr = compute_psnr(x_sample.numpy(), x_true.numpy())
        log_prob = compute_log_density(x_sample, problem, y, problem.noise_std, A, n_true)
        
        psnrs_for_plot.append(psnr)
        log_dens_for_plot.append(log_prob)
        
        results.append({
            'lambda': float(lam),
            'psnr': float(psnr),
            'log_density': float(log_prob),
            'x': x_sample.numpy().tolist()
        })

    # --- 4. Save Outputs ---
    save_dir = "exps/experiments/psnr_uncertainty_mismatch"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save PT
    pt_path = os.path.join(save_dir, "exported_data_details.pt")
    torch.save({
        'A': A, 'y': y, 'x_true': x_true, 'v': v,
        'samples': results
    }, pt_path)
    print(f"\nPT Data saved to {pt_path}")
    
    # Save JSON
    json_path = os.path.join(save_dir, "exported_data_details.json")
    json_data = {
        'A': A.numpy().tolist(),
        'y': y.numpy().tolist(),
        'x_true': x_true.numpy().tolist(),
        'v': v.numpy().tolist(),
        'samples': results
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON Data saved to {json_path}")
    
    # Save Visualization
    viz_path = os.path.join(save_dir, "exported_data_viz.png")
    visualize_results(lambdas, psnrs_for_plot, log_dens_for_plot, viz_path)

if __name__ == "__main__":
    export_data()
