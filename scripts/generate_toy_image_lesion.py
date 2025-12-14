"""
Script to generate images using trained diffusion model for toy_image_lesion.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import hydra
from omegaconf import DictConfig

from models.toy_mlp_diffusion import ToyDiffusionMLP
from algo.unconditional import UnconditionalDiffusionSampler

@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="toy_image_lesion")
def generate(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = cfg.path if hasattr(cfg, 'path') else "toy_image_lesion_diffusion.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        print("Run: python train_toy_image_lesion.py")
        return
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Build model
    model = hydra.utils.instantiate(cfg.model).to(device)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    elif 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dummy forward_op (not used for unconditional generation)
    class DummyForwardOp:
        def __init__(self):
            self.device = device
    
    forward_op = DummyForwardOp()
    
    # Create unconditional sampler
    diffusion_scheduler_config = {
        'num_steps': 1000,
        'schedule': 'vp',
        'timestep': 'vp',
        'scaling': 'vp'
    }
    
    sampler = UnconditionalDiffusionSampler(
        net=model,
        forward_op=forward_op,
        diffusion_scheduler_config=diffusion_scheduler_config,
        sde=True
    )
    
    # Generate images
    num_samples = 20
    print(f"Generating {num_samples} samples...")
    
    # Create dummy observation (not used, but required by interface)
    dummy_obs = torch.zeros(1, 1, 8, 8, device=device)
    
    with torch.no_grad():
        generated = sampler.inference(dummy_obs, num_samples=num_samples, verbose=True)
    
    print(f"Generated {generated.shape[0]} samples")
    print(f"Sample range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
    
    # Save visualization
    output_dir = "exps/generated_samples/toy_image_lesion"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create grid visualization
    n_cols = 5
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Generated Samples from Trained Diffusion Model (n={num_samples})', 
                 fontsize=14, fontweight='bold')
    
    # Compute global scale
    all_samples_np = generated[:, 0].cpu().numpy()
    global_vmin = all_samples_np.min()
    global_vmax = all_samples_np.max()
    margin = (global_vmax - global_vmin) * 0.05
    global_vmin = global_vmin - margin
    global_vmax = global_vmax + margin
    
    for i in range(num_samples):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        sample = generated[i, 0].cpu().numpy()
        sample_upsampled = zoom(sample, 4, order=1)
        
        im = ax.imshow(sample_upsampled, cmap='gray', vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'generated_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid visualization to {grid_path}")
    plt.close()
    
    # Save individual images
    individual_dir = os.path.join(output_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i in range(num_samples):
        sample = generated[i, 0].cpu().numpy()
        sample_upsampled = zoom(sample, 4, order=1)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(sample_upsampled, cmap='gray', vmin=global_vmin, vmax=global_vmax, interpolation='nearest')
        ax.set_title(f'Generated Sample {i+1}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        img_path = os.path.join(individual_dir, f'generated_{i+1:03d}.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_samples} individual images to {individual_dir}/")
    print("\nDone!")

if __name__ == "__main__":
    generate()







