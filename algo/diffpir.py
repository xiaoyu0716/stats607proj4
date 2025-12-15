import torch
import tqdm
from .base import Algo
import numpy as np
import wandb
from utils.scheduler import Scheduler

# -----------------------------------------------------------------------------------------------
# Paper: Denoising Diffusion Models for Plug-and-Play Image Restoration
# Official implementation: https://github.com/yuanzhi-zhu/DiffPIR
# -----------------------------------------------------------------------------------------------


class DiffPIR(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config, sigma_n, lamb, xi, linear=False):
        super(DiffPIR, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sigma_n = sigma_n
        self.lamb = lamb
        self.xi = xi
        self.linear = linear
        
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        pbar = tqdm.trange(self.scheduler.num_steps)
        xt= torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        for step in pbar:
            sigma, sigma_next = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step+1]
            scaling_step = self.scheduler.scaling_steps[step]
            
            # FIX: Model outputs epsilon, need to convert to x0
            x_scaled = xt / scaling_step
            eps_pred = self.net(x_scaled, torch.as_tensor(sigma).to(xt.device))
            x0 = (x_scaled - sigma * eps_pred).clone().requires_grad_(True)
            
            rho =  (2*self.lamb*self.sigma_n**2)/(sigma*scaling_step)**2
            if self.linear:
                # Linear: use closed-form proximal solution
                # x* = (A^T A + alpha*I)^{-1} (A^T y + alpha*x0)
                # where alpha = lambda * sigma_n^2 / sigma^2
                # Note: rho in code = 2*lambda*sigma_n^2/(sigma*scaling)^2
                # so alpha = rho * scaling^2 / 2, or use theoretical formula directly
                alpha = (self.lamb * (self.sigma_n ** 2)) / max(sigma ** 2, 1e-8)
                
                # Convert to vector form
                x0_vec = x0.view(num_samples, -1)  # [B, 16]
                y_vec = self.forward_op._img_to_vec(observation)  # [B, 16]
                y_obs = y_vec[:, :self.forward_op.A_obs_dim]  # [B, 16] for A=I
                
                # Get A matrix
                A = self.forward_op.A.to(device)  # [16, 16]
                AT = A.T
                ATA = AT @ A  # [16, 16]
                
                # Compute (A^T y + alpha*x0)
                ATy = (AT @ y_obs.T).T  # [B, 16]
                rhs = ATy + alpha * x0_vec  # [B, 16]
                
                # Solve (ATA + alpha*I) x = rhs
                H = ATA + alpha * torch.eye(ATA.shape[0], device=device)
                x0hat_vec = torch.linalg.solve(H, rhs.T).T  # [B, 16]
                x0hat = self.forward_op._vec_to_img(x0hat_vec)  # [B, 1, 4, 4]
                
                # Compute data fitting loss
                Ax0hat = (A @ x0hat_vec.T).T  # [B, 16]
                data_fitting_loss = ((Ax0hat - y_obs) ** 2).sum(dim=1).mean().item()
                loss_scale = torch.tensor(data_fitting_loss, device=device)
            else:
                # Nonlinear: use gradient descent
                # Note: forward_op.gradient returns -A^T(Ax-y), not divided by sigma_n^2
                # so need to manually compute correct gradient
                with torch.enable_grad():
                    grad_neg, loss_scale = self.forward_op.gradient(x0, observation, return_loss=True)
                    # grad_neg = -A^T(Ax-y), so correct data gradient is -grad_neg / sigma_n^2 = A^T(Ax-y) / sigma_n^2
                    # For proximal step, we need: x0hat = x0 - step_size * grad_total
                    # where grad_total = grad_data + grad_prior
                    # grad_data = A^T(Ax-y) / sigma_n^2 = -grad_neg / sigma_n^2
                    # grad_prior = lambda * (x - x0) / sigma^2 = alpha * (x - x0)
                    
                    # Manual computation (safer)
                    Ax0 = self.forward_op.forward(x0)
                    x0_vec = self.forward_op._img_to_vec(x0)
                    Ax0_vec = self.forward_op._img_to_vec(Ax0)
                    y_vec = self.forward_op._img_to_vec(observation)
                    y_obs = y_vec[:, :self.forward_op.A_obs_dim]
                    Ax0_obs = Ax0_vec[:, :self.forward_op.A_obs_dim]
                    
                    A_full = self.forward_op.A[:self.forward_op.A_obs_dim, :].to(device)
                    residual = Ax0_obs - y_obs
                    grad_data_vec = (residual @ A_full) / (self.sigma_n ** 2)
                    grad_data = self.forward_op._vec_to_img(grad_data_vec)
                    
                    # Prior term gradient
                    alpha = (self.lamb * (self.sigma_n ** 2)) / max(sigma ** 2, 1e-8)
                    grad_prior = alpha * (x0 - x0.detach())  # x0 is detached here, so grad_prior=0
                    # Actually prior term is ||x - x0||^2, so grad_prior = alpha * (x - x0)
                    # But x0 is detached, so should use detached version of x0
                    x0_detached = x0.detach()
                    grad_prior = alpha * (x0 - x0_detached)
                    
                    grad_total = grad_data + grad_prior
                
                # Proximal step: x0hat = x0 - step_size * grad_total
                # Use rho as inverse of step size
                step_size = 1.0 / max(rho, 1e-8)
                x0hat = x0 - step_size * grad_total
            
            # Update xt with effect term
            sigma_safe = max(sigma, 1e-8)
            effect = (x_scaled - x0hat) / sigma_safe
            # Clip effect to prevent explosion
            effect = torch.clamp(effect, min=-10.0, max=10.0)
            
            if self.xi > 0:
                noise = np.sqrt(self.xi) * torch.randn_like(xt) + np.sqrt(1 - self.xi) * effect
            else:
                noise = torch.randn_like(xt)
            
            xt = x0hat + noise * sigma_next

            if step < self.scheduler.num_steps-1:
                xt = xt * self.scheduler.scaling_steps[step+1]
            
            # Mild clipping to prevent extreme values
            xt = torch.clamp(xt, min=-20.0, max=20.0) 
            pbar.set_description(f'Iteration {step + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})
        
        # Return final result (remove scaling)
        return xt / self.scheduler.scaling_steps[-1] if len(self.scheduler.scaling_steps) > 0 else xt