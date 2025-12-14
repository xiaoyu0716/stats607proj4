import torch
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_problems.toy_gausscmog8 import ToyGausscMoG8Problem

# Create toy problem instance for sampling
problem = ToyGausscMoG8Problem(
    dim=16,
    A_type="random-gaussian",
    A_seed=1234,
    A_scale=1.0,
    noise_std=0.2236,
    gauss_rho=0.8,
    mog8_mu=2.0,
    mog8_wm_full=0.5,
    mog8_wp_full=0.5,
    device='cpu'
)

N = 50000
print(f"Generating {N} samples...")
x_img = problem.sample_prior(N)  # shape = [N, 1, 4, 4]

torch.save(x_img.cpu(), "toy_gausscmog8_prior.pt")
print("Saved to toy_gausscmog8_prior.pt")
