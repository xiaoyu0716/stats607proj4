#!/usr/bin/env python3
"""
Test script to verify all algorithms work with fixed-full-rank-16x16 A matrix.
"""
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem
from utils.scheduler import Scheduler
from models.ddpm import UNetModel

# Algorithms to test
ALGORITHMS = {
    'DPS': 'algo.dps.DPS',
    'DAPS': 'algo.daps.DAPS',
    'DDNM': 'algo.ddnm.DDNM',
    'DDRM': 'algo.ddrm.DDRM',
    'REDdiff': 'algo.reddiff.REDdiff',
    'MCGdiff': 'algo.mcgdiff.MCG_diff',
    'PnPDM': 'algo.pnpdm.PnPDM',
    'DiffPIR': 'algo.diffpir.DiffPIR',
}

def create_mock_net():
    """Create a mock network for testing."""
    # Create a simple UNet model
    net = UNetModel(
        img_resolution=4,
        img_channels=1,
        label_dim=0,
        use_fp16=False,
        model_type='DhariwalUNet',
        channel_mult=[1, 2],
        attention_resolutions=[],
        num_res_blocks=2,
    )
    return net

def create_problem():
    """Create the toy problem with fixed-full-rank-16x16 matrix."""
    problem = ToyGausscMoG8Problem(
        dim=16,
        A_type='fixed-full-rank-16x16',
        A_seed=1234,
        A_scale=1.0,
        noise_std=0.2236,
        gauss_rho=0.8,
        mog8_mu=2.0,
        mog8_wm_full=0.5,
        mog8_wp_full=0.5,
        A_obs_dim=16,
        device='cpu'
    )
    return problem

def test_algorithm(alg_name, alg_class, problem, net):
    """Test a single algorithm."""
    logger.info(f"Testing {alg_name}...")
    
    try:
        # Create scheduler config
        scheduler_config = {
            'num_steps': 10,  # Reduced for quick testing
            'sigma_min': 0.01,
            'sigma_max': 1.0,
            'rho': 7.0,
            'schedule_type': 'polynomial'
        }
        
        # Initialize algorithm based on type
        if alg_name == 'DPS':
            algo = alg_class(
                net=net,
                forward_op=problem,
                diffusion_scheduler_config=scheduler_config,
                guidance_scale=1.0,
                sde=True
            )
        elif alg_name == 'DAPS':
            from utils.diffusion import DiffusionSampler
            algo = alg_class(
                net=net,
                forward_op=problem,
                annealing_scheduler_config={'num_steps': 5, 'sigma_max': 1.0, 'sigma_min': 0.01},
                diffusion_scheduler_config=scheduler_config,
                lgvd_config={'num_steps': 3, 'lr': 0.1, 'tau': 0.01}
            )
        elif alg_name == 'DDNM':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                eta=0.85,
                L=0
            )
        elif alg_name == 'DDRM':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                eta=0.85,
                eta_b=0.1
            )
        elif alg_name == 'REDdiff':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                lambda_=1.0
            )
        elif alg_name == 'MCGdiff':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                guidance_scale=1.0
            )
        elif alg_name == 'PnPDM':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                lambda_=1.0,
                num_inner_steps=1
            )
        elif alg_name == 'DiffPIR':
            algo = alg_class(
                net=net,
                forward_op=problem,
                scheduler_config=scheduler_config,
                lambda_=1.0
            )
        else:
            logger.warning(f"Unknown algorithm: {alg_name}, skipping...")
            return False
        
        # Generate test observation
        x0, y = problem.generate_sample()
        observation = y.unsqueeze(0)
        
        # Run inference
        logger.info(f"  Running inference for {alg_name}...")
        recon = algo.inference(observation, num_samples=1)
        
        # Check output shape
        assert recon.shape == observation.shape, f"Output shape mismatch: {recon.shape} vs {observation.shape}"
        
        # Check for NaN/Inf
        if torch.isnan(recon).any() or torch.isinf(recon).any():
            logger.error(f"  {alg_name} produced NaN/Inf values!")
            return False
        
        logger.info(f"  ✓ {alg_name} passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ {alg_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Testing all algorithms with fixed-full-rank-16x16 A matrix")
    logger.info("=" * 60)
    
    # Create problem and network
    problem = create_problem()
    net = create_mock_net()
    
    # Test results
    results = {}
    
    # Test each algorithm
    for alg_name, alg_module_path in ALGORITHMS.items():
        try:
            # Import algorithm class
            module_path, class_name = alg_module_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            alg_class = getattr(module, class_name)
            
            # Test algorithm
            success = test_algorithm(alg_name, alg_class, problem, net)
            results[alg_name] = success
            
        except ImportError as e:
            logger.error(f"Failed to import {alg_name}: {e}")
            results[alg_name] = False
        except Exception as e:
            logger.error(f"Error testing {alg_name}: {e}")
            results[alg_name] = False
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    for alg_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {alg_name}: {status}")
    
    # Count successes
    num_passed = sum(results.values())
    num_total = len(results)
    logger.info(f"\nTotal: {num_passed}/{num_total} algorithms passed")
    
    if num_passed == num_total:
        logger.info("All algorithms passed! ✓")
        return 0
    else:
        logger.warning(f"{num_total - num_passed} algorithm(s) failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())






