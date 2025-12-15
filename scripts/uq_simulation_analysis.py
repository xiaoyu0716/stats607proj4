#!/usr/bin/env python3
"""
Uncertainty Quantification (UQ) Simulation Analysis for Diffusion-based Inverse Algorithms

This script performs systematic UQ evaluation for multiple algorithms:
- MCG_diff, PnPDM, DPS, DAPS (posterior sampling methods)
- DDRM, DDNM, DiffPIR, ReDiff (point-estimate methods)

Main experiment:
1. Coverage analysis for A=I (95% credible interval coverage)
"""

import sys
import os
from pathlib import Path

# Add project root to path FIRST, before any imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pickle
from tqdm import tqdm
# itertools not needed for coverage experiment
import pandas as pd
import json

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem


# ============================================================================
# 0. Configuration and Constants
# ============================================================================

SUPPORTED_METHODS = {
    'MCG_diff': {'supports_sampling': True, 'config_file': 'configs/algorithm/mcgdiff_toy.yaml'},
    'PnPDM': {'supports_sampling': True, 'config_file': 'configs/algorithm/pnpdm_toy.yaml'},
    'DPS': {'supports_sampling': True, 'config_file': 'configs/algorithm/dps_toy.yaml'},
    'DAPS': {'supports_sampling': True, 'config_file': 'configs/algorithm/daps_toy.yaml'},
    'DDRM': {'supports_sampling': True, 'config_file': 'configs/algorithm/ddrm_toy.yaml'},
    'DDNM': {'supports_sampling': True, 'config_file': 'configs/algorithm/ddnm_toy.yaml'},
    'DiffPIR': {'supports_sampling': True, 'config_file': 'configs/algorithm/diffpir_toy.yaml'},
    'ReDiff': {'supports_sampling': True, 'config_file': 'configs/algorithm/reddiff_toy.yaml'},
}

DEFAULT_NOISE_STD = 0.5
DEFAULT_PRIOR_PARAMS = {
    'gauss_rho': 0.8,
    'mog8_mu': 2.0,
    'mog8_wm_full': 0.5,
    'mog8_wp_full': 0.5,
}


# ============================================================================
# 1. Data Generation Module
# ============================================================================

def generate_dataset(
    A_type: str,
    N: int,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = 0,
    A_seed: int = 1234,
) -> Dict:
    """
    Generate a dataset of (x0, y) pairs for the toy 16D problem.
    
    Args:
        A_type: 'identity' or 'mri_like'
        N: Number of test samples
        noise_std: Observation noise standard deviation
        seed: Random seed for data generation
        A_seed: Random seed for A matrix generation (for MRI-like)
    
    Returns:
        Dictionary with:
            'x0': (N, 16) true latent vectors
            'y': (N, 16) observations
            'A': (16, 16) forward operator matrix
            'U': (16, 16) or None, SVD U matrix
            'S': (16,) or None, singular values
            'V': (16, 16) or None, SVD V^T matrix
    """
    torch.manual_seed(seed)
    device = 'cpu'
    
    # Create problem instance for sampling
    problem = ToyGausscMoG8Problem(
        dim=16,
        A_type='Identity' if A_type == 'identity' else 'mri-like',
        A_seed=A_seed,
        A_scale=1.0,
        A_obs_dim=16,
        noise_std=noise_std,
        **DEFAULT_PRIOR_PARAMS,
        device=device
    )
    
    # Generate x0 samples from prior
    x0_img = problem.sample_prior(N)  # (N, 1, 4, 4)
    x0_vec = problem._img_to_vec(x0_img)  # (N, 16)
    
    # Generate observations
    y_img = problem.forward(x0_img)  # (N, 1, 4, 4)
    y_vec = problem._img_to_vec(y_img)  # (N, 16)
    
    # Get A matrix
    A = problem.A  # (16, 16)
    
    # SVD decomposition (for MRI-like case)
    U, S, V = None, None, None
    if A_type == 'mri_like':
        # Perform SVD: A = U @ diag(S) @ V^T
        U_full, S_full, Vt_full = torch.linalg.svd(A, full_matrices=True)
        U = U_full  # (16, 16)
        S = S_full  # (16,)
        V = Vt_full.T  # (16, 16), V^T in SVD
    elif A_type == 'identity':
        # For identity, U=V=I, S=1
        U = torch.eye(16, device=device)
        S = torch.ones(16, device=device)
        V = torch.eye(16, device=device)
    
    return {
        'x0': x0_vec.cpu().numpy(),  # (N, 16)
        'y': y_vec.cpu().numpy(),    # (N, 16)
        'A': A.cpu().numpy(),        # (16, 16)
        'U': U.cpu().numpy() if U is not None else None,
        'S': S.cpu().numpy() if S is not None else None,
        'V': V.cpu().numpy() if V is not None else None,
        'problem': problem,  # Keep problem instance for later use
    }


# ============================================================================
# 2. Method Execution Module
# ============================================================================

def load_model_and_algorithm(
    method_name: str,
    problem: ToyGausscMoG8Problem,
    config_overrides: Optional[Dict] = None,
    device: str = 'cpu',
) -> Tuple:
    """
    Load the diffusion model and initialize the algorithm.
    
    Returns:
        (net, algo, algo_config) tuple
    """
    # Load model
    ckpt_path = Path(__file__).parent.parent / 'toy_gausscmog8_diffusion.pt'
    from models.toy_mlp_diffusion import ToyDiffusionMLP
    net = ToyDiffusionMLP(dim=16, hidden=128)
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        elif 'net' in ckpt.keys():
            net.load_state_dict(ckpt['net'])
        else:
            # Direct state dict
            net.load_state_dict(ckpt)
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {ckpt_path}: {e}")
        print("Using randomly initialized model (for testing only)")
    
    net = net.to(device)
    # Keep net in eval mode, but gradients will still flow if input has requires_grad
    net.eval()
    
    # Load algorithm config
    config_path = Path(__file__).parent.parent / SUPPORTED_METHODS[method_name]['config_file']
    config = OmegaConf.load(config_path)
    # Handle different config formats
    if 'method' in config:
        algo_config = config.method
    elif '_target_' in config:
        algo_config = config
    else:
        # Try to find the method config
        for key in config.keys():
            if key != 'name' and isinstance(config[key], dict) and '_target_' in config[key]:
                algo_config = config[key]
                break
        else:
            raise ValueError(f"Could not find method config in {config_path}")
    
    # Apply overrides
    if config_overrides:
        # Handle nested parameters (e.g., 'lgvd_config.lr' -> {'lgvd_config': {'lr': value}})
        nested_overrides = {}
        for key, value in config_overrides.items():
            if '.' in key:
                # Split nested key and create nested dict
                parts = key.split('.')
                current = nested_overrides
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                nested_overrides[key] = value
        algo_config = OmegaConf.merge(algo_config, OmegaConf.create(nested_overrides))
    
    # Instantiate algorithm
    algo = instantiate(algo_config, forward_op=problem, net=net)
    
    return net, algo, algo_config


def run_method_on_dataset(
    method_name: str,
    dataset: Dict,
    K: int = 100,
    config_overrides: Optional[Dict] = None,
    device: str = 'cpu',
    verbose: bool = False,
) -> Dict:
    """
    Run a method on a dataset and collect posterior samples or point estimates.
    
    Args:
        method_name: Name of the method
        dataset: Output from generate_dataset
        K: Number of posterior samples (for sampling methods)
        config_overrides: Optional config overrides
        device: Device to run on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with:
            'samples': (N, K, 16) or None - posterior samples
            'mean': (N, 16) - posterior mean or point estimate
            'meta': dict with method info
    """
    problem = dataset['problem']
    N = dataset['x0'].shape[0]
    y_vec = torch.from_numpy(dataset['y']).to(device)  # (N, 16)
    
    # Convert y to image format
    y_img = problem._vec_to_img(y_vec)  # (N, 1, 4, 4)
    
    # Load model and algorithm
    net, algo, algo_config = load_model_and_algorithm(method_name, problem, config_overrides, device)
    
    supports_sampling = SUPPORTED_METHODS[method_name]['supports_sampling']
    
    if supports_sampling:
        # Collect K samples for each observation
        samples_list = []
        for n in tqdm(range(N), desc=f"Running {method_name}", disable=not verbose):
            observation = y_img[n:n+1]  # (1, 1, 4, 4)
            
            # Special handling for MCG_diff
            if method_name == 'MCG_diff':
                # MCG_diff can return either:
                # 1. All particles: [num_particles, 1, 4, 4] - for testing
                # 2. Single sample: [1, 1, 4, 4] - standard behavior
                with torch.no_grad():
                    recon = algo.inference(observation, num_samples=1)
                    # Check if returned all particles or single sample
                    if recon.shape[0] > 1:
                        # Returned all particles: [num_particles, 1, 4, 4]
                        # Use all particles as K samples (or sample K from them)
                        num_returned = recon.shape[0]
                        
                        # Convert to vector format first
                        recon_vec_all = problem._img_to_vec(recon)  # (num_particles, 16)
                        
                        # Convert to numpy for indexing
                        if isinstance(recon_vec_all, torch.Tensor):
                            recon_vec_all = recon_vec_all.cpu().numpy()
                        
                        # Select K samples from all particles
                        if num_returned >= K:
                            # Randomly select K particles
                            indices = np.random.choice(num_returned, K, replace=False)
                            recon_vec = recon_vec_all[indices]  # (K, 16)
                        else:
                            # Use all particles (less than K)
                            recon_vec = recon_vec_all  # (num_returned, 16)
                            # Pad with last sample if needed (not ideal, but ensures shape consistency)
                            if recon_vec.shape[0] < K:
                                padding = np.repeat(recon_vec[-1:], K - recon_vec.shape[0], axis=0)
                                recon_vec = np.concatenate([recon_vec, padding], axis=0)
                        
                        # Ensure exactly K samples
                        recon_vec = recon_vec[:K]  # (K, 16)
                        samples_list.append(recon_vec)  # (K, 16)
                    else:
                        # Returned single sample: [1, 1, 4, 4] - standard behavior
                        # Call inference() K times to get K independent samples
                        sample_batch = []
                        for k in range(K):
                            with torch.no_grad():
                                recon_k = algo.inference(observation, num_samples=1)
                                # recon_k shape: [1, 1, 4, 4] - single posterior sample
                                # Handle both image and vector formats
                                if recon_k.dim() == 4:
                                    recon_vec = problem._img_to_vec(recon_k)  # (1, 16)
                                elif recon_k.dim() == 2:
                                    recon_vec = recon_k  # (1, 16) or (16,)
                                else:
                                    raise ValueError(f"Unexpected recon shape: {recon_k.shape}")
                                # Squeeze batch dimension if needed
                                if recon_vec.shape[0] == 1:
                                    recon_vec = recon_vec.squeeze(0)  # (1, 16) -> (16,)
                                sample_batch.append(recon_vec.cpu().numpy())
                        samples_list.append(np.stack(sample_batch, axis=0))  # (K, 16)
            else:
                # For other methods, try to get K samples at once
                try:
                    # DPS and DAPS need gradients, so don't use torch.no_grad()
                    # But they return detached tensors, so it's safe to call without no_grad
                        recon = algo.inference(observation, num_samples=K)
                        # Check if we got K samples or just one
                        if recon.shape[0] == K:
                            # Convert to vector format: handle both (K, 1, 4, 4) and (K, 16) cases
                            if recon.dim() == 4:
                                # Image format: (K, 1, 4, 4) -> (K, 16)
                                recon_vec = problem._img_to_vec(recon)
                            elif recon.dim() == 2:
                                # Already vector format: (K, 16)
                                recon_vec = recon
                            else:
                                raise ValueError(f"Unexpected recon shape: {recon.shape}")
                            # Ensure shape is (K, 16), squeeze if needed
                            if recon_vec.dim() == 3 and recon_vec.shape[1] == 1:
                                recon_vec = recon_vec.squeeze(1)  # (K, 1, 16) -> (K, 16)
                            samples_list.append(recon_vec.cpu().numpy())
                        else:
                            # Fallback: run K times
                            sample_batch = []
                            for k in range(K):
                                    recon_single = algo.inference(observation, num_samples=1)
                                    # Handle both image and vector formats
                                    if recon_single.dim() == 4:
                                        recon_vec = problem._img_to_vec(recon_single)  # (1, 16)
                                    elif recon_single.dim() == 2:
                                        recon_vec = recon_single
                                    else:
                                        raise ValueError(f"Unexpected recon_single shape: {recon_single.shape}")
                                    # Squeeze batch dimension if needed
                                    if recon_vec.shape[0] == 1:
                                        recon_vec = recon_vec.squeeze(0)  # (1, 16) -> (16,)
                                    sample_batch.append(recon_vec.cpu().numpy())
                            samples_list.append(np.stack(sample_batch, axis=0))
                except Exception as e:
                    # Fallback: run K times
                    sample_batch = []
                    for k in range(K):
                            recon = algo.inference(observation, num_samples=1)
                            # Handle both image and vector formats
                            if recon.dim() == 4:
                                recon_vec = problem._img_to_vec(recon)  # (1, 16)
                            elif recon.dim() == 2:
                                recon_vec = recon
                            else:
                                raise ValueError(f"Unexpected recon shape: {recon.shape}")
                            # Squeeze batch dimension if needed
                            if recon_vec.shape[0] == 1:
                                recon_vec = recon_vec.squeeze(0)  # (1, 16) -> (16,)
                            sample_batch.append(recon_vec.cpu().numpy())
                    samples_list.append(np.stack(sample_batch, axis=0))
        
        samples = np.stack(samples_list, axis=0)  # (N, K, 16)
        mean = samples.mean(axis=1)  # (N, 16)
    else:
        # Point estimate methods - run once per observation
        samples = None
        mean_list = []
        for n in tqdm(range(N), desc=f"Running {method_name}", disable=not verbose):
            observation = y_img[n:n+1]  # (1, 1, 4, 4)
            with torch.no_grad():
                recon = algo.inference(observation, num_samples=1)
                # Handle both image and vector formats
                if recon.dim() == 4:
                    recon_vec = problem._img_to_vec(recon)  # (1, 16)
                elif recon.dim() == 2:
                    recon_vec = recon
                else:
                    raise ValueError(f"Unexpected recon shape: {recon.shape}")
                # Squeeze batch dimension if needed
                if recon_vec.shape[0] == 1:
                    recon_vec = recon_vec.squeeze(0)  # (1, 16) -> (16,)
                mean_list.append(recon_vec.cpu().numpy())
        mean = np.stack(mean_list, axis=0)  # (N, 16)
    
    # Extract config parameters (flatten nested dicts)
    def flatten_config(cfg, prefix=''):
        """Flatten nested config dict to string representation"""
        params = {}
        # Convert OmegaConf to dict if needed
        if isinstance(cfg, OmegaConf):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if k == '_target_':
                    continue
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, OmegaConf)):
                    if isinstance(v, OmegaConf):
                        v = OmegaConf.to_container(v, resolve=True)
                    params.update(flatten_config(v, new_key))
                else:
                    params[new_key] = str(v)
        return params
    
    config_params = flatten_config(algo_config)
    
    return {
        'samples': samples,
        'mean': mean,
        'meta': {
            'method_name': method_name,
            'supports_sampling': supports_sampling,
            'K': K if supports_sampling else None,
            'config_params': config_params,
        }
    }


# ============================================================================
# 3. Experiment Modules
# ============================================================================

def experiment_coverage_identity(
    methods: List[str],
    N: int = 200,
    K: int = 100,
    noise_std: float = DEFAULT_NOISE_STD,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Experiment 1: Coverage analysis for A=I.
    
    Compute per-dimension coverage of 95% credible intervals.
    
    Returns:
        Dictionary with coverage results for each method
    """
    # Generate dataset
    dataset = generate_dataset('identity', N, noise_std)
    
    coverage_results = {}
    all_results_data = []  # Store all results for CSV
    
    for method_name in methods:
        if not SUPPORTED_METHODS[method_name]['supports_sampling']:
            continue
        
        result = run_method_on_dataset(method_name, dataset, K, device=device, verbose=False)
        
        if result['samples'] is None:
            continue
        
        samples = result['samples']  # (N, K, 16)
        x0 = dataset['x0']  # (N, 16)
        
        # Compute posterior statistics
        mu = samples.mean(axis=1)  # (N, 16)
        std = samples.std(axis=1, ddof=1)  # (N, 16)
        
        # Compute 95% CI coverage for each dimension
        lower = mu - 1.96 * std
        upper = mu + 1.96 * std
        
        per_dim_coverage = []
        for i in range(16):
            cov_i = np.mean((x0[:, i] >= lower[:, i]) & (x0[:, i] <= upper[:, i]))
            per_dim_coverage.append(cov_i)
        
        per_dim_coverage = np.array(per_dim_coverage)
        global_coverage = per_dim_coverage.mean()
        
        coverage_results[method_name] = {
            'per_dim_coverage': per_dim_coverage,
            'global_coverage': global_coverage,
        }
        
        print(f"{method_name} finished. Coverage: {global_coverage:.4f} (target: 0.95)")
        
        # Prepare data for CSV: save each run's results
        config_params = result['meta']['config_params']
        
        # For each sample (n=0 to N-1), save results
        for n in range(N):
            row_data = {
                'method': method_name,
                'sample_idx': n,
                'global_coverage': global_coverage,
            }
            
            # Add config parameters
            for param_key, param_value in config_params.items():
                row_data[f'param_{param_key}'] = param_value
            
            # Add true x0 values
            for dim in range(16):
                row_data[f'true_x0_dim{dim}'] = x0[n, dim]
            
            # Add mean reconstruction
            for dim in range(16):
                row_data[f'mean_recon_dim{dim}'] = mu[n, dim]
            
            # Add std (uncertainty)
            for dim in range(16):
                row_data[f'std_dim{dim}'] = std[n, dim]
            
            # Add per-dimension coverage (overall proportion for this dimension)
            for dim in range(16):
                row_data[f'coverage_dim{dim}'] = per_dim_coverage[dim]
            
            # Add whether each dimension of this sample is within CI (1 if in, 0 if out)
            for dim in range(16):
                in_ci = 1 if (x0[n, dim] >= lower[n, dim] and x0[n, dim] <= upper[n, dim]) else 0
                row_data[f'in_ci_dim{dim}'] = in_ci
            
            # Add CI bounds for each dimension
            for dim in range(16):
                row_data[f'ci_lower_dim{dim}'] = lower[n, dim]
                row_data[f'ci_upper_dim{dim}'] = upper[n, dim]
            
            # Add all K samples for this observation
            for k in range(K):
                for dim in range(16):
                    row_data[f'sample_{k}_dim{dim}'] = samples[n, k, dim]
            
            all_results_data.append(row_data)
    
    # Save to CSV (always save, even if only one method)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if all_results_data:
            # Save detailed results CSV (all methods combined)
            df_detailed = pd.DataFrame(all_results_data)
            csv_path = output_dir / 'coverage_results_detailed.csv'
            df_detailed.to_csv(csv_path, index=False)
            
            # Save individual CSV file for each method
            saved_method_files = []
            for method_name in coverage_results.keys():
                method_data = [r for r in all_results_data if r['method'] == method_name]
                if method_data:
                    df_method = pd.DataFrame(method_data)
                    method_csv_path = output_dir / f'coverage_{method_name}_detailed.csv'
                    df_method.to_csv(method_csv_path, index=False)
                    saved_method_files.append(method_csv_path.name)
            
            # Save summary CSV (one row per method)
            summary_data = []
            for method_name, results in coverage_results.items():
                summary_row = {
                    'method': method_name,
                    'global_coverage': results['global_coverage'],
                    'target_coverage': 0.95,
                    'coverage_error': abs(results['global_coverage'] - 0.95),
                }
                # Add per-dimension coverage
                for dim in range(16):
                    summary_row[f'coverage_dim{dim}'] = results['per_dim_coverage'][dim]
                
                # Get config params from first row of this method
                method_rows = [r for r in all_results_data if r['method'] == method_name]
                if method_rows:
                    for param_key, param_value in method_rows[0].items():
                        if param_key.startswith('param_'):
                            summary_row[param_key] = param_value
                
                summary_data.append(summary_row)
            
            df_summary = pd.DataFrame(summary_data)
            csv_summary_path = output_dir / 'coverage_results_summary.csv'
            
            # Append if file exists, otherwise create new
            if csv_summary_path.exists():
                df_existing = pd.read_csv(csv_summary_path)
                df_summary = pd.concat([df_existing, df_summary], ignore_index=True)
                df_summary.to_csv(csv_summary_path, index=False, mode='w')
            else:
                df_summary.to_csv(csv_summary_path, index=False)
            
            # Print saved files
            print(f"Results saved:")
            print(f"  - Summary: {csv_summary_path}")
            print(f"  - All methods (detailed): {csv_path}")
            for method_file in saved_method_files:
                print(f"  - {method_file.replace('coverage_', '').replace('_detailed.csv', '')}: {method_file}")
        else:
            print("Warning: No results to save (all methods may have been skipped)")
    
    return coverage_results


# Removed experiment_nullspace_variance_mri and experiment_psnr_vs_uncertainty functions
# (not needed for coverage experiment)


# ============================================================================
# 4. Visualization Module
# ============================================================================

def plot_coverage_bar(coverage_results: Dict, save_path: Optional[str] = None):
    """Plot per-dimension coverage for each method."""
    n_methods = len(coverage_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, results) in enumerate(coverage_results.items()):
        ax = axes[idx]
        per_dim = results['per_dim_coverage']
        global_cov = results['global_coverage']
        
        ax.bar(range(16), per_dim, alpha=0.7)
        ax.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        ax.axhline(y=global_cov, color='g', linestyle='--', label=f'Global ({global_cov:.3f})')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Coverage')
        ax.set_title(f'{method_name}\nGlobal: {global_cov:.3f}')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Removed plot_nullspace_variance and plot_psnr_vs_uncertainty functions
# (not needed for coverage experiment)


# ============================================================================
# 5. Main Execution and Summary
# ============================================================================

def main():
    """Run coverage experiment and generate summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UQ Simulation Analysis')
    parser.add_argument('--experiment', type=str, choices=['coverage'],
                       default='coverage', help='Which experiment to run (only coverage is supported)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='Methods to test (default: all sampling methods)')
    parser.add_argument('--N', type=int, default=200, help='Number of test samples')
    parser.add_argument('--K', type=int, default=100, help='Number of posterior samples')
    parser.add_argument('--noise_std', type=float, default=DEFAULT_NOISE_STD, help='Noise std')
    parser.add_argument('--output_dir', type=str, default='exps/uq_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine methods to test
    if args.methods is None:
        sampling_methods = [m for m in SUPPORTED_METHODS.keys() 
                           if SUPPORTED_METHODS[m]['supports_sampling']]
    else:
        sampling_methods = args.methods
    
    # ========================================================================
    # Coverage Analysis (A=I)
    # ========================================================================
        coverage_dir = output_dir / 'coverage'
        coverage_dir.mkdir(parents=True, exist_ok=True)
        
        coverage_results = experiment_coverage_identity(
            methods=sampling_methods,
            N=args.N,
            K=args.K,
            noise_std=args.noise_std,
            device=device,
            output_dir=coverage_dir,
        )
        
        plot_coverage_bar(coverage_results, save_path=coverage_dir / 'coverage_identity.png')
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
