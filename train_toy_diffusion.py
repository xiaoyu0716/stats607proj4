import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig

from models.toy_mlp_diffusion import ToyDiffusionMLP
from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="toy_gausscmog8")
def train(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Build model from config
    model = hydra.utils.instantiate(cfg.model).to(device)
    opt = Adam(model.parameters(), lr=cfg.training.lr)


    print("Training diffusion prior for toy problem...")

    # Load dataset
    x0 = torch.load("toy_gausscmog8_prior.pt")
    x0 = x0.float()
    N = x0.shape[0]

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
    torch.save(saved_dict, "toy_gausscmog8_diffusion.pt")
    print("Saved model to toy_gausscmog8_diffusion.pt")

if __name__ == "__main__":
    train()
