import torch
import torch.nn as nn
from torch.optim import Adam
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.toy_diffusion import ToyDiffusionModel

# Load dataset
x0 = torch.load("toy_gausscmog8_prior.pt")  # [N,8]
x0 = x0.float()
N = x0.shape[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = ToyDiffusionModel(dim=8).to(device)
opt = Adam(model.parameters(), lr=1e-3)

num_steps = 10000
print("Training diffusion prior for toy problem...")

for step in range(num_steps):
    idx = torch.randint(0,N,(256,))
    x = x0[idx].to(device) # Shape: [B, 1, 2, 4]

    # sample sigma (VP schedule)
    sigma = torch.exp(torch.randn(256).to(device) * 1.0 - 2.0)

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
