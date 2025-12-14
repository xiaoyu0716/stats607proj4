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
import itertools
import pandas as pd
import json

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem, make_mri_like_A_16


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


def experiment_nullspace_variance_mri(
    methods: List[str],
    N: int = 200,
    K: int = 100,
    noise_std: float = DEFAULT_NOISE_STD,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Experiment 2: Nullspace variance analysis for A=MRI-like.
    
    Analyze posterior variance in SVD space.
    
    Returns:
        Dictionary with variance results for each method
    """
    # Generate dataset with MRI-like A matrix
    dataset = generate_dataset('mri_like', N, noise_std)
    
    # Verify A is MRI-like (not identity)
    A = dataset['A']
    is_identity = np.allclose(A, np.eye(16))
    if is_identity:
        raise ValueError("A matrix is identity! Expected MRI-like A matrix for nullspace analysis.")
    
    S = dataset['S']  # (16,) singular values
    V = dataset['V']  # (16, 16) V matrix from SVD
    
    # Verify SVD structure: should have some zeros (nullspace dimensions)
    num_null_dims = np.sum(S < 0.5)  # SVD singular values should be 0 or 1 for MRI-like
    num_observed_dims = np.sum(S > 0.5)
    
    print(f"Using MRI-like A matrix: {num_observed_dims} observed dims, {num_null_dims} nullspace dims")
    
    results = {}
    all_results_data = []  # Store all results for CSV
    
    for method_name in methods:
        if not SUPPORTED_METHODS[method_name]['supports_sampling']:
            continue
        
        result = run_method_on_dataset(method_name, dataset, K, device=device, verbose=False)
        
        if result['samples'] is None:
            continue
        
        samples = result['samples']  # (N, K, 16) or potentially different shape
        config_params = result['meta']['config_params']
        
        # Ensure samples have correct shape (N, K, 16)
        if samples.shape != (N, K, 16):
            print(f"Warning: samples shape is {samples.shape}, expected ({N}, {K}, 16). Reshaping...")
            # If samples is 2D, reshape to (N, K, 16)
            if samples.ndim == 2:
                total_samples = samples.shape[0]
                expected_total = N * K
                if total_samples == expected_total:
                    samples = samples.reshape(N, K, 16)
                else:
                    # Handle case where we have more samples (e.g., all particles)
                    # Take first N*K samples
                    samples = samples[:expected_total].reshape(N, K, 16)
            elif samples.ndim == 3:
                # If shape is (N, num_particles, 16) where num_particles != K
                if samples.shape[1] != K:
                    # Take first K samples from each observation
                    samples = samples[:, :K, :]
        
        # Project samples to V space (SVD space)
        # Note: SVD is A = U @ diag(S) @ V^T, so V^T is the right singular vectors
        # To project to SVD space, we use V (not V^T)
        X = samples.reshape(-1, 16)  # (N*K, 16)
        Z = X @ V  # (N*K, 16) - project to SVD space using V
        Z = Z.reshape(N, K, 16)  # (N, K, 16)
        
        # Compute variance per singular dimension
        # For each dimension j, compute variance across K samples for each of N observations, then average
        var_per_singular_dim = []
        for j in range(16):
            # Z[:, :, j] is (N, K) - N observations, K samples per observation
            # Compute variance across K samples for each observation, then average over N
            var_j = np.mean(Z[:, :, j].var(axis=1, ddof=1))  # average over N
            var_per_singular_dim.append(var_j)
        
        var_per_singular_dim = np.array(var_per_singular_dim)
        
        # Separate observed and null dimensions
        # For MRI-like A, S should be exactly 0 or 1, but numerical errors may produce small values
        observed_mask = (S > 0.5)  # S should be 0 or 1
        null_mask = (S <= 0.5)
        
        var_observed_mean = var_per_singular_dim[observed_mask].mean() if observed_mask.any() else 0.0
        var_null_mean = var_per_singular_dim[null_mask].mean() if null_mask.any() else 0.0
        
        # Debug: print variance distribution
        if method_name == 'MCG_diff' or method_name == 'DPS':
            print(f"  {method_name} variance per dim: {var_per_singular_dim}")
            print(f"  Observed dims: {np.where(observed_mask)[0]}, variance: {var_per_singular_dim[observed_mask]}")
            print(f"  Null dims: {np.where(null_mask)[0]}, variance: {var_per_singular_dim[null_mask]}")
        
        results[method_name] = {
            'var_per_singular_dim': var_per_singular_dim,
            'var_observed_mean': var_observed_mean,
            'var_null_mean': var_null_mean,
        }
        
        ratio = var_null_mean / (var_observed_mean + 1e-8)
        print(f"{method_name} finished. Var (observed): {var_observed_mean:.6f}, Var (null): {var_null_mean:.6f}, Ratio: {ratio:.4f}")
        
        # Prepare data for CSV: save per-sample variance in SVD space
        for n in range(N):
            row_data = {
                'method': method_name,
                'sample_idx': n,
                'var_observed_mean': var_observed_mean,
                'var_null_mean': var_null_mean,
                'ratio': ratio,
            }
            
            # Add config parameters
            for param_key, param_value in config_params.items():
                row_data[f'param_{param_key}'] = param_value
            
            # Add variance per singular dimension
            for dim in range(16):
                row_data[f'var_singular_dim{dim}'] = var_per_singular_dim[dim]
                row_data[f'S_dim{dim}'] = S[dim]  # Singular value (0 or 1)
            
            # Add per-sample variance in SVD space
            for dim in range(16):
                var_n_dim = Z[n, :, dim].var(ddof=1).item()  # Variance across K samples
                row_data[f'var_sample_{n}_dim{dim}'] = var_n_dim
            
            all_results_data.append(row_data)
    
    # Save to CSV (always save, even if only one method)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if all_results_data:
            # Save detailed results CSV (all methods combined)
            df_detailed = pd.DataFrame(all_results_data)
            csv_path = output_dir / 'nullspace_results_detailed.csv'
            df_detailed.to_csv(csv_path, index=False)
            
            # Save individual CSV file for each method
            saved_method_files = []
            for method_name in results.keys():
                method_data = [r for r in all_results_data if r['method'] == method_name]
                if method_data:
                    df_method = pd.DataFrame(method_data)
                    method_csv_path = output_dir / f'nullspace_{method_name}_detailed.csv'
                    df_method.to_csv(method_csv_path, index=False)
                    saved_method_files.append(method_csv_path.name)
            
            # Save summary CSV (one row per method)
            summary_data = []
            for method_name, result in results.items():
                summary_row = {
                    'method': method_name,
                    'var_observed_mean': result['var_observed_mean'],
                    'var_null_mean': result['var_null_mean'],
                    'ratio': result['var_null_mean'] / (result['var_observed_mean'] + 1e-8),
                }
                # Add per-dimension variance
                for dim in range(16):
                    summary_row[f'var_singular_dim{dim}'] = result['var_per_singular_dim'][dim]
                
                # Get config params from first row of this method
                method_rows = [r for r in all_results_data if r['method'] == method_name]
                if method_rows:
                    for param_key, param_value in method_rows[0].items():
                        if param_key.startswith('param_'):
                            summary_row[param_key] = param_value
                
                summary_data.append(summary_row)
            
            df_summary = pd.DataFrame(summary_data)
            csv_summary_path = output_dir / 'nullspace_results_summary.csv'
            
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
                print(f"  - {method_file.replace('nullspace_', '').replace('_detailed.csv', '')}: {method_file}")
        else:
            print("Warning: No results to save (all methods may have been skipped)")
    
    return results, S


# Removed experiment_psnr_vs_uncertainty function (not needed for coverage experiment)


def plot_coverage_bar(coverage_results: Dict, save_path: Optional[str] = None):
    methods: List[str],
    hyperparam_grids: Optional[Dict[str, Dict]] = None,
    A_type: str = 'identity',
    N: int = 200,
    K: int = 100,
    noise_std: float = DEFAULT_NOISE_STD,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Experiment 3: PSNR vs uncertainty trade-off.
    
    Sweep hyperparameters and compute (PSNR, uncertainty) pairs for multiple methods.
    
    Args:
        methods: List of method names to test
        hyperparam_grids: Optional dict mapping method_name -> hyperparam_grid
                         If None, uses default hyperparameter grids for each method
        A_type: 'identity' or 'mri_like'
        N: Number of test samples
        K: Number of samples (for sampling methods)
        noise_std: Noise standard deviation
        device: Device to run on
        output_dir: Directory to save CSV results
    
    Returns:
        Dictionary mapping method_name -> {grid, psnr, avg_uncertainty}
    """
    # Default hyperparameter grids for each method
    # Note: Use correct parameter names from each method's config
    default_hyperparam_grids = {
        'DPS': {'guidance_scale': [1, 2, 4, 8, 10]},
        'MCG_diff': {'num_particles': [10, 50, 100, 200]},
        'PnPDM': {'lgvd_config.lr': [1e-4, 5e-4, 1e-3, 5e-3]},
        'DAPS': {'lgvd_config.lr': [1e-7, 1e-6, 5e-6, 1e-5]},  # DAPS uses lgvd_config.lr, not guidance_scale
        'DDRM': {'eta': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'DDNM': {'eta': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'DiffPIR': {'lamb': [0.05, 0.1, 0.5, 1.0, 2.0]},  # DiffPIR uses 'lamb', not 'lambda_'
        'ReDiff': {'observation_weight': [1.0, 5.0, 10.0, 20.0, 50.0]},  # ReDiff uses observation_weight, not guidance_scale
    }
    
    # Use provided grids or defaults
    if hyperparam_grids is None:
        hyperparam_grids = {}
    for method in methods:
        if method not in hyperparam_grids:
            if method in default_hyperparam_grids:
                hyperparam_grids[method] = default_hyperparam_grids[method]
            else:
                # No hyperparameters to sweep - use empty grid (single run)
                hyperparam_grids[method] = {}
    
    # Generate dataset
    dataset = generate_dataset(A_type, N, noise_std)
    x0 = dataset['x0']  # (N, 16)
    
    all_results = {}  # method_name -> {grid, psnr, avg_uncertainty}
    all_results_data = []  # Store all results for CSV (all methods combined)
    
    # Run tradeoff experiment for each method
    for method_name in methods:
        if method_name not in SUPPORTED_METHODS:
            print(f"Warning: Method {method_name} not supported, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Running tradeoff experiment for {method_name}")
        print(f"{'='*80}")
        
        hyperparam_grid = hyperparam_grids.get(method_name, {})
        
        # If no hyperparameters to sweep, run once with default config
        if not hyperparam_grid:
            print(f"  No hyperparameter grid specified, running with default config...")
            result = run_method_on_dataset(
                method_name, dataset, K, config_overrides=None, device=device, verbose=False
            )
            
            mean = result['mean']  # (N, 16)
            config_params = result['meta']['config_params']
            
            # Compute PSNR
            data_max = np.max(np.abs(x0))
            if data_max < 1e-8:
                data_max = 1.0
            mse = np.mean((mean - x0) ** 2)
            psnr = 20 * np.log10(data_max) - 10 * np.log10(mse + 1e-8)
            
            # Compute uncertainty
            if result['samples'] is not None:
                samples = result['samples']  # (N, K, 16)
                std = samples.std(axis=1, ddof=1)  # (N, 16)
                avg_uncertainty = std.mean()
            else:
                avg_uncertainty = 1e-6
            
            all_results[method_name] = {
                'grid': [{}],  # Empty config
                'psnr': np.array([psnr]),
                'avg_uncertainty': np.array([avg_uncertainty]),
            }
            
            # Prepare data for CSV
            row_data = {
                'method': method_name,
                'A_type': A_type,
                'psnr': psnr,
                'avg_uncertainty': avg_uncertainty,
            }
            
            # Add config parameters
            for param_key, param_value in config_params.items():
                row_data[f'param_{param_key}'] = param_value
            
            # Add per-sample PSNR and uncertainty
            for n in range(N):
                mse_n = np.mean((mean[n] - x0[n]) ** 2)
                psnr_n = 20 * np.log10(data_max) - 10 * np.log10(mse_n + 1e-8)
                row_data[f'psnr_sample_{n}'] = psnr_n
                
                if result['samples'] is not None:
                    std_n = std[n].mean().item()
                    row_data[f'uncertainty_sample_{n}'] = std_n
                else:
                    row_data[f'uncertainty_sample_{n}'] = 1e-6
            
            all_results_data.append(row_data)
            print(f"  {method_name} finished. PSNR: {psnr:.4f}, Uncertainty: {avg_uncertainty:.6f}")
        
        else:
            # Create grid of hyperparameter combinations
            param_names = list(hyperparam_grid.keys())
            param_values = list(hyperparam_grid.values())
            grid = list(itertools.product(*param_values))
            
            psnr_list = []
            uncertainty_list = []
            grid_configs = []
            
            for param_combo in tqdm(grid, desc=f"Sweeping {method_name} hyperparameters", disable=False):
                config_overrides = {param_names[i]: param_combo[i] for i in range(len(param_names))}
                grid_configs.append(config_overrides)
                
                # Run method
                result = run_method_on_dataset(
                    method_name, dataset, K, config_overrides, device=device, verbose=False
                )
                
                mean = result['mean']  # (N, 16)
                config_params = result['meta']['config_params']
                
                # Compute PSNR
                data_max = np.max(np.abs(x0))
                if data_max < 1e-8:
                    data_max = 1.0
                mse = np.mean((mean - x0) ** 2)
                psnr = 20 * np.log10(data_max) - 10 * np.log10(mse + 1e-8)
                psnr_list.append(psnr)
                
                # Compute uncertainty
                if result['samples'] is not None:
                    samples = result['samples']  # (N, K, 16)
                    std = samples.std(axis=1, ddof=1)  # (N, 16)
                    avg_uncertainty = std.mean()
                else:
                    avg_uncertainty = 1e-6
                
                uncertainty_list.append(avg_uncertainty)
                
                # Prepare data for CSV
                row_data = {
                    'method': method_name,
                    'A_type': A_type,
                    'psnr': psnr,
                    'avg_uncertainty': avg_uncertainty,
                }
                
                # Add hyperparameter values
                for param_name, param_value in config_overrides.items():
                    row_data[f'hyperparam_{param_name}'] = param_value
                
                # Add config parameters
                for param_key, param_value in config_params.items():
                    row_data[f'param_{param_key}'] = param_value
                
                # Add per-sample PSNR and uncertainty
                for n in range(N):
                    mse_n = np.mean((mean[n] - x0[n]) ** 2)
                    psnr_n = 20 * np.log10(data_max) - 10 * np.log10(mse_n + 1e-8)
                    row_data[f'psnr_sample_{n}'] = psnr_n
                    
                    if result['samples'] is not None:
                        std_n = std[n].mean().item()
                        row_data[f'uncertainty_sample_{n}'] = std_n
                    else:
                        row_data[f'uncertainty_sample_{n}'] = 1e-6
                
                all_results_data.append(row_data)
                
                print(f"  {method_name} finished. Hyperparams: {config_overrides}, PSNR: {psnr:.4f}, Uncertainty: {avg_uncertainty:.6f}")
            
            all_results[method_name] = {
                'grid': grid_configs,
                'psnr': np.array(psnr_list),
                'avg_uncertainty': np.array(uncertainty_list),
            }
    
    # Save to CSV (always save)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if all_results_data:
            # Save detailed results CSV (all methods combined)
            df_detailed = pd.DataFrame(all_results_data)
            csv_path = output_dir / 'psnr_vs_uncertainty_all_methods_detailed.csv'
            df_detailed.to_csv(csv_path, index=False)
            
            # Save individual CSV file for each method
            saved_method_files = []
            for method_name in all_results.keys():
                method_data = [r for r in all_results_data if r['method'] == method_name]
                if method_data:
                    df_method = pd.DataFrame(method_data)
                    method_csv_path = output_dir / f'psnr_vs_uncertainty_{method_name}_detailed.csv'
                    df_method.to_csv(method_csv_path, index=False)
                    saved_method_files.append(method_csv_path.name)
                    
                    # Save summary CSV for each method
                    summary_data = []
                    method_result = all_results[method_name]
                    for i, (config, psnr_val, unc_val) in enumerate(zip(
                        method_result['grid'], 
                        method_result['psnr'], 
                        method_result['avg_uncertainty']
                    )):
                        summary_row = {
                            'method': method_name,
                            'A_type': A_type,
                            'psnr': psnr_val,
                            'avg_uncertainty': unc_val,
                        }
                        # Add hyperparameter values
                        for param_name, param_value in config.items():
                            summary_row[f'hyperparam_{param_name}'] = param_value
                        
                        summary_data.append(summary_row)
                    
                    df_summary = pd.DataFrame(summary_data)
                    csv_summary_path = output_dir / f'psnr_vs_uncertainty_{method_name}_summary.csv'
                    
                    # Append if file exists, otherwise create new
                    if csv_summary_path.exists():
                        df_existing = pd.read_csv(csv_summary_path)
                        df_summary = pd.concat([df_existing, df_summary], ignore_index=True)
                        df_summary.to_csv(csv_summary_path, index=False, mode='w')
                    else:
                        df_summary.to_csv(csv_summary_path, index=False)
            
            # Print saved files
            print(f"\nResults saved:")
            print(f"  - All methods (detailed): {csv_path}")
            for method_file in saved_method_files:
                method_name = method_file.replace('psnr_vs_uncertainty_', '').replace('_detailed.csv', '')
                print(f"  - {method_name}: {method_file}")
                print(f"    Summary: psnr_vs_uncertainty_{method_name}_summary.csv")
        else:
            print("Warning: No results to save")
    
    return all_results


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


def plot_nullspace_variance(results: Dict, S: np.ndarray, save_path: Optional[str] = None):
    """Plot variance in SVD space, showing observed vs null dimensions."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 8))
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (method_name, result) in enumerate(results.items()):
        var_per_dim = result['var_per_singular_dim']
        var_obs = result['var_observed_mean']
        var_null = result['var_null_mean']
        
        # Top: per-dimension variance
        ax1 = axes[0, idx]
        observed_mask = (S > 0.5)
        null_mask = (S <= 0.5)
        
        ax1.bar(np.where(observed_mask)[0], var_per_dim[observed_mask], 
                alpha=0.7, color='blue', label='Observed (S=1)')
        ax1.bar(np.where(null_mask)[0], var_per_dim[null_mask], 
                alpha=0.7, color='red', label='Null (S=0)')
        ax1.set_xlabel('Singular Dimension')
        ax1.set_ylabel('Variance')
        ax1.set_title(f'{method_name} - Per-Dimension Variance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: mean comparison
        ax2 = axes[1, idx]
        ax2.bar(['Observed', 'Null'], [var_obs, var_null], 
                alpha=0.7, color=['blue', 'red'])
        ax2.set_ylabel('Mean Variance')
        ax2.set_title(f'Mean Variance Comparison\nRatio: {var_null/(var_obs+1e-8):.2f}')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_psnr_vs_uncertainty(tradeoff_result: Dict, method_name: str, save_path: Optional[str] = None):
    """Plot PSNR vs uncertainty scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    psnr = tradeoff_result['psnr']
    uncertainty = tradeoff_result['avg_uncertainty']
    grid = tradeoff_result['grid']
    
    # Create labels for each point
    labels = []
    for config in grid:
        label_parts = [f"{k}={v}" for k, v in config.items()]
        labels.append(", ".join(label_parts))
    
    scatter = ax.scatter(uncertainty, psnr, s=100, alpha=0.6, c=range(len(grid)), cmap='viridis')
    
    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (uncertainty[i], psnr[i]), 
                   fontsize=8, alpha=0.7, rotation=45)
    
    ax.set_xlabel('Average Uncertainty (std)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'{method_name} - PSNR vs Uncertainty Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Config Index')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5. Main Execution and Summary
# ============================================================================

def main():
    """Run all experiments and generate summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UQ Simulation Analysis')
    parser.add_argument('--experiment', type=str, choices=['coverage', 'nullspace', 'tradeoff', 'all'],
                       default='all', help='Which experiment to run')
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
    
    coverage_results = {}
    nullspace_results = {}
    S = None
    tradeoff_results = {}
    
    # ========================================================================
    # Experiment 1: Coverage Analysis (A=I)
    # ========================================================================
    if args.experiment in ['coverage', 'all']:
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
    
    # ========================================================================
    # Experiment 2: Nullspace Variance (A=MRI-like)
    # ========================================================================
    if args.experiment in ['nullspace', 'all']:
        nullspace_dir = output_dir / 'nullspace'
        nullspace_dir.mkdir(parents=True, exist_ok=True)
        
        nullspace_results, S = experiment_nullspace_variance_mri(
            methods=sampling_methods,
            N=args.N,
            K=args.K,
            noise_std=args.noise_std,
            device=device,
            output_dir=nullspace_dir,
        )
        
        plot_nullspace_variance(nullspace_results, S, 
                               save_path=nullspace_dir / 'nullspace_variance.png')
    
    # ========================================================================
    # Experiment 3: PSNR vs Uncertainty Trade-off
    # ========================================================================
    if args.experiment in ['tradeoff', 'all']:
        tradeoff_dir = output_dir / 'tradeoff'
        tradeoff_dir.mkdir(parents=True, exist_ok=True)
        
        # Run tradeoff experiment for all specified methods
        tradeoff_results = experiment_psnr_vs_uncertainty(
            methods=sampling_methods,
            hyperparam_grids=None,  # Use default grids
            A_type='identity',
            N=args.N,
            K=args.K,
            noise_std=args.noise_std,
            device=device,
            output_dir=tradeoff_dir,
        )
        
        # Plot results for each method
        for method_name, tradeoff_result in tradeoff_results.items():
            plot_psnr_vs_uncertainty(tradeoff_result, method_name,
                                     save_path=tradeoff_dir / f'psnr_vs_uncertainty_{method_name.lower()}.png')
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
