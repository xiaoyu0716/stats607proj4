"""
Training script for Colab - no Hydra dependency
Usage in Colab:
  1. Upload this file and train_toy_image_lesion.py to Colab
  2. Upload configs/pretrain/toy_image_lesion.yaml (or set parameters below)
  3. Run this script
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import os

from models.toy_mlp_diffusion import ToyDiffusionMLP
from inverse_problems.toy_image_lesion import ToyImageLesionProblem

def train_colab():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model parameters (from config)
    dim = 256  # 16×16 = 256
    hidden = 128
    
    # Training parameters
    total_steps = 50000
    batch_size = 64
    lr = 1e-3
    num_samples = 10000
    
    # Prior parameters (current settings)
    blur_sigma = 1.0
    noise_std = 0.03
    tau = 0.06
    lesion_prior_weight = 0.1
    lesion_amplitude = 0.60
    lesion_radius = 2
    
    # Noise schedule
    sigma_min = 0.002
    sigma_max = 80.0
    
    # Build model
    model = ToyDiffusionMLP(dim=dim, hidden=hidden).to(device)
    opt = Adam(model.parameters(), lr=lr)
    
    print("Training diffusion prior for 16×16 image toy problem...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate or load dataset
    prior_data_path = "toy_image_lesion_prior.pt"
    if os.path.exists(prior_data_path):
        print(f"Loading prior data from {prior_data_path}")
        x0 = torch.load(prior_data_path)
    else:
        print("Generating prior data on the fly...")
        problem = ToyImageLesionProblem(
            blur_sigma=blur_sigma,
            noise_std=noise_std,
            tau=tau,
            lesion_prior_weight=lesion_prior_weight,
            lesion_amplitude=lesion_amplitude,
            lesion_radius=lesion_radius,
            device=device
        )
        result = problem.sample_prior(num_samples, exact_proportion=False)
        if isinstance(result, tuple):
            x0, labels = result
        else:
            x0 = result
        # Save for future use
        torch.save(x0, prior_data_path)
        print(f"Saved prior data to {prior_data_path}")
    
    x0 = x0.float()
    N = x0.shape[0]
    print(f"Training on {N} samples")
    
    # Training loop with progress bar
    from tqdm import tqdm
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        idx = torch.randint(0, N, (batch_size,))
        x = x0[idx].to(device)
        
        # Sample sigma (VP schedule)
        t = torch.rand(batch_size, device=device)
        sigma = sigma_min * (sigma_max / sigma_min) ** t
        
        noise = torch.randn_like(x)
        x_t = x + sigma.view(-1, 1, 1, 1) * noise
        
        eps_pred = model(x_t, sigma)
        loss = ((eps_pred - noise)**2).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Update progress bar
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if step % 1000 == 0:
            print(f"step {step} | loss={loss.item():.6f}")
    
    # Save model
    output_path = "toy_image_lesion_diffusion.pt"
    saved_dict = {'ema': model.state_dict(), 'net': model.state_dict()}
    torch.save(saved_dict, output_path)
    print(f"Saved model to {output_path}")
    
    # Download file in Colab
    try:
        from google.colab import files
        print("\nModel saved! You can download it using:")
        print("  files.download('toy_image_lesion_diffusion.pt')")
    except ImportError:
        print("\nModel saved locally.")

if __name__ == "__main__":
    train_colab()
