#!/usr/bin/env python3
"""
Quick test script to verify UQ simulation modules work correctly.
Run this before running the full analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.uq_simulation_analysis import (
    generate_dataset,
    run_method_on_dataset,
    experiment_coverage_identity,
    experiment_nullspace_variance_mri,
    experiment_psnr_vs_uncertainty,
)

def test_data_generation():
    """Test data generation module."""
    print("Testing data generation...")
    
    # Test identity
    dataset_id = generate_dataset('identity', N=10, noise_std=0.5)
    assert dataset_id['x0'].shape == (10, 16), f"Expected (10, 16), got {dataset_id['x0'].shape}"
    assert dataset_id['y'].shape == (10, 16), f"Expected (10, 16), got {dataset_id['y'].shape}"
    assert dataset_id['A'].shape == (16, 16), f"Expected (16, 16), got {dataset_id['A'].shape}"
    print("  ✓ Identity dataset generation works")
    
    # Test MRI-like
    dataset_mri = generate_dataset('mri_like', N=10, noise_std=0.5)
    assert dataset_mri['x0'].shape == (10, 16)
    assert dataset_mri['S'] is not None
    assert dataset_mri['V'] is not None
    print("  ✓ MRI-like dataset generation works")
    
    print("Data generation test passed!\n")


def test_method_execution():
    """Test method execution on small dataset."""
    print("Testing method execution...")
    
    dataset = generate_dataset('identity', N=5, noise_std=0.5)
    
    # Test DPS (should work)
    try:
        result = run_method_on_dataset('DPS', dataset, K=5, device='cpu', verbose=False)
        assert result['mean'].shape == (5, 16), f"Expected (5, 16), got {result['mean'].shape}"
        if result['samples'] is not None:
            assert result['samples'].shape == (5, 5, 16), f"Expected (5, 5, 16), got {result['samples'].shape}"
        print("  ✓ DPS execution works")
    except Exception as e:
        print(f"  ✗ DPS execution failed: {e}")
        raise
    
    print("Method execution test passed!\n")


def test_experiments():
    """Test experiment modules with small N and K."""
    print("Testing experiment modules...")
    
    # Test coverage experiment
    try:
        coverage_results = experiment_coverage_identity(
            methods=['DPS'],
            N=10,
            K=5,
            noise_std=0.5,
            device='cpu',
        )
        assert 'DPS' in coverage_results
        assert 'global_coverage' in coverage_results['DPS']
        print("  ✓ Coverage experiment works")
    except Exception as e:
        print(f"  ✗ Coverage experiment failed: {e}")
        raise
    
    # Test nullspace experiment
    try:
        nullspace_results, S = experiment_nullspace_variance_mri(
            methods=['DPS'],
            N=10,
            K=5,
            noise_std=0.5,
            device='cpu',
        )
        assert 'DPS' in nullspace_results
        assert 'var_observed_mean' in nullspace_results['DPS']
        print("  ✓ Nullspace variance experiment works")
    except Exception as e:
        print(f"  ✗ Nullspace variance experiment failed: {e}")
        raise
    
    # Test tradeoff experiment
    try:
        tradeoff_result = experiment_psnr_vs_uncertainty(
            method_name='DPS',
            hyperparam_grid={'guidance_scale': [1, 2]},
            A_type='identity',
            N=10,
            K=5,
            noise_std=0.5,
            device='cpu',
        )
        assert len(tradeoff_result['psnr']) == 2
        assert len(tradeoff_result['avg_uncertainty']) == 2
        print("  ✓ PSNR vs uncertainty experiment works")
    except Exception as e:
        print(f"  ✗ PSNR vs uncertainty experiment failed: {e}")
        raise
    
    print("All experiment tests passed!\n")


if __name__ == '__main__':
    print("="*80)
    print("UQ Simulation Modules Test")
    print("="*80)
    print()
    
    try:
        test_data_generation()
        test_method_execution()
        test_experiments()
        
        print("="*80)
        print("All tests passed! ✓")
        print("="*80)
        print("\nYou can now run the full analysis:")
        print("  python scripts/uq_simulation_analysis.py --experiment all --N 200 --K 100")
        
    except Exception as e:
        print("="*80)
        print(f"Test failed: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

