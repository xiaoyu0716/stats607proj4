import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_svd
    
# ----------------------------------------------------------------------------------
# Paper: Denoising Diffusion Restoration Models
# Official implementation: https://github.com/bahjat-kawar/ddrm
# ----------------------------------------------------------------------------------


class DDRM(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 eta,
                 eta_b):
        super(DDRM, self).__init__(net, forward_op)
        assert has_svd(forward_op), "DDRM only works with linear forward operators, which can be decomposed via SVD"
        self.scheduler = Scheduler(**scheduler_config)
        self.eta = eta
        self.eta_b = eta_b

    def score(self, model, x, sigma):
        """
            Computes the score function for the given model.

            Parameters:
                model (DiffusionModel): Diffusion model.
                x (torch.Tensor): Input tensor.
                sigma (float): Sigma value.

            Returns:
                torch.Tensor: The computed score.
        """
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        # print(d.min(), d.max())
        return (d - x) / sigma**2
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        
        observation = observation / self.forward_op.unnorm_scale - self.forward_op.forward(self.forward_op.unnorm_shift * torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device),unnormalize=False)
        sigma_y = self.forward_op.sigma_noise / self.forward_op.unnorm_scale
        observation_t = self.forward_op.Ut(observation)*(self.forward_op.M/self.forward_op.S)
        z = torch.randn(num_samples, *self.forward_op.M.shape, device=device)
        x_t = self.forward_op.M * (observation_t + z * torch.sqrt(self.scheduler.sigma_max**2 - self.forward_op.M*sigma_y**2/self.forward_op.S**2)) + (1 - self.forward_op.M) * z * self.scheduler.sigma_max / self.scheduler.scaling_steps[0]
        pbar = tqdm.trange(self.scheduler.num_steps)
        for step in pbar:
            sigma = self.scheduler.sigma_steps[step]
            x = self.forward_op.V(x_t)  # Returns 4D [B, 1, 4, 4]
            x_next_t_4d = self.forward_op.Vt(self.net(x, torch.as_tensor(sigma).to(x.device)))  # Returns 4D [B, 1, 4, 4]
            
            # Expand x_next_t to match x_t's shape if needed
            if x_t.dim() == 5 and x_next_t_4d.dim() == 4:
                # x_t is [B, 1, 1, 4, 4], expand x_next_t to match
                x_next_t = x_next_t_4d.unsqueeze(1)  # [B, 1, 1, 4, 4]
            else:
                x_next_t = x_next_t_4d

            sigma_next = self.scheduler.sigma_steps[step + 1]
            # Clip x_next_t to prevent explosion (use wider bounds for better reconstruction)
            x_next_t = torch.clamp(x_next_t, min=-4.0, max=4.0)
            
            # Compute step coefficient with numerical stability
            step_coeff = np.sqrt(1 - self.eta**2) * sigma_next / max(sigma, 1e-8)
            # Clip step coefficient to prevent large steps
            step_coeff = min(step_coeff, 1.5)
            x_masked = x_next_t + step_coeff * (x_t - x_next_t) + self.eta * sigma_next * torch.randn_like(x_t)
            x_masked = torch.clamp(x_masked, min=-4.0, max=4.0)

            mask = (self.forward_op.S >= sigma_y/sigma_next) # For numerical stability
            # Compute sqrt term with numerical stability
            sqrt_arg = sigma_next**2 - mask * sigma_y**2/self.forward_op.S**2
            sqrt_arg = torch.clamp(sqrt_arg, min=0.0)  # Ensure non-negative
            sqrt_term = torch.sqrt(sqrt_arg)
            x_obs_1 = x_next_t * (1 - self.eta_b) + self.eta_b * observation_t + sqrt_term * torch.randn_like(x_t)
            x_obs_1 = torch.clamp(x_obs_1, min=-4.0, max=4.0)
            
            # else:
            if sigma_y <= 1e-5: # For numerical stability
                x_obs_2 = torch.zeros_like(x_next_t)
            else:
                # Ensure S/sigma_y doesn't cause issues
                sigma_y_safe = max(sigma_y, 1e-8)  # Ensure sigma_y is not too small
                S_over_sigma_y = self.forward_op.S / sigma_y_safe
                # Clip S_over_sigma_y to prevent explosion
                S_over_sigma_y = torch.clamp(S_over_sigma_y, max=50.0)
                step_coeff_obs = np.sqrt(1 - self.eta**2) * sigma_next * S_over_sigma_y
                step_coeff_obs = torch.clamp(step_coeff_obs, max=2.0)
                x_obs_2 = x_next_t + step_coeff_obs * (observation_t - x_next_t) + self.eta * sigma_next * torch.randn_like(x_t)
                x_obs_2 = torch.clamp(x_obs_2, min=-4.0, max=4.0)
            
            x_t = self.forward_op.M * x_obs_1 * (self.forward_op.S >= sigma_y/sigma_next) + self.forward_op.M * x_obs_2 * (self.forward_op.S < sigma_y/sigma_next) + (1 - self.forward_op.M) * x_masked
            x_t = torch.clamp(x_t, min=-4.0, max=4.0)
            
            # Check for NaN/Inf and handle
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                x_t = torch.where(torch.isnan(x_t) | torch.isinf(x_t), torch.zeros_like(x_t), x_t)
        result = self.forward_op.V(x_t)
        # Final check for NaN/Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)
        return result
