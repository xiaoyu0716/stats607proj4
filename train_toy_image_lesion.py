import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig
import os

from models.toy_mlp_diffusion import ToyDiffusionMLP
from inverse_problems.toy_image_lesion import ToyImageLesionProblem

@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="toy_image_lesion")
def train(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build model from config
    model = hydra.utils.instantiate(cfg.model).to(device)
    opt = Adam(model.parameters(), lr=cfg.training.lr)

    print("Training diffusion prior for 16×16 image toy problem...")

    # Generate or load dataset
    prior_data_path = "toy_image_lesion_prior.pt"
    if os.path.exists(prior_data_path):
        print(f"Loading prior data from {prior_data_path}")
        x0 = torch.load(prior_data_path)
    else:
        print("Generating prior data on the fly...")
        # Generate samples from prior using config
        problem = ToyImageLesionProblem(
            blur_sigma=cfg.prior.blur_sigma,
            noise_std=cfg.prior.noise_std,
            tau=cfg.prior.tau,
            lesion_prior_weight=cfg.prior.lesion_prior_weight,
            lesion_amplitude=cfg.prior.lesion_amplitude,
            lesion_radius=cfg.prior.lesion_radius,
            device=device
        )
        x0 = problem.sample_prior(cfg.training.num_samples)
        # Save for future use
        torch.save(x0, prior_data_path)
        print(f"Saved prior data to {prior_data_path}")
    
    x0 = x0.float()
    N = x0.shape[0]
    print(f"Training on {N} samples")

    for step in range(cfg.training.total_steps):
        idx = torch.randint(0, N, (cfg.training.batch_size,))
        x = x0[idx].to(device)

        # Sample sigma (VP schedule)
        t = torch.rand(cfg.training.batch_size, device=device)
        sigma = cfg.noise_schedule.sigma_min * (cfg.noise_schedule.sigma_max / cfg.noise_schedule.sigma_min) ** t

        noise = torch.randn_like(x)
        x_t = x + sigma.view(-1, 1, 1, 1) * noise

        eps_pred = model(x_t, sigma)
        loss = ((eps_pred - noise)**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"step {step} | loss={loss.item():.6f}")

    # Save in a format compatible with the framework
    saved_dict = {'ema': model.state_dict(), 'net': model.state_dict()}
    output_path = cfg.path if hasattr(cfg, 'path') else "toy_image_lesion_diffusion.pt"
    torch.save(saved_dict, output_path)
    print(f"Saved model to {output_path}")

if __name__ == "__main__":
    train()

