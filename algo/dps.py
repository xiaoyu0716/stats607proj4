import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

# -----------------------------------------------------------------------------------------------
# Paper: Diffusion Posterior Sampling for General Noisy Inverse Problems
# Official implementation: https://github.com/DPS2022/diffusion-posterior-sampling
# -----------------------------------------------------------------------------------------------


class DPS(Algo):
    
    '''
    DPS algorithm implemented in EDM framework.
    Official DPS style: VP-SDE + autograd backprop from x0 to x_t
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 sde=True):
        super(DPS, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        
        # For VP-SDE: compute beta, dt, sigma, t from scheduler
        if self.scheduler.schedule == 'vp' and self.scheduler.timestep == 'vp':
            self._setup_vp_sde_params()
        else:
            # Fallback to EDM style (original implementation)
            self.use_vp_sde = False
    
    def _setup_vp_sde_params(self):
        """Setup VP-SDE parameters: beta, dt, sigma, t from scheduler"""
        self.use_vp_sde = True
        beta_d = 19.9
        beta_min = 0.1
        
        # Extract time steps from scheduler
        time_steps = self.scheduler.time_steps  # [num_steps + 1]
        sigma_steps = self.scheduler.sigma_steps  # [num_steps + 1]
        
        # Compute beta(t) = beta_d * t + beta_min
        self.beta = np.array([beta_d * t + beta_min for t in time_steps[:-1]])  # [num_steps]
        
        # Compute dt = time_steps[i] - time_steps[i+1]
        self.dt = np.array([time_steps[i] - time_steps[i+1] for i in range(self.scheduler.num_steps)])  # [num_steps]
        
        # sigma for each step (model uses sigma, not t)
        self.sigma = sigma_steps[:-1]  # [num_steps]
        self.t = time_steps[:-1]  # [num_steps] (for reference, but model uses sigma)
        
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        
        # Start from noise (VP-SDE starts from standard Gaussian, not scaled)
        x = torch.randn(num_samples, self.net.img_channels, 
                       self.net.img_resolution, self.net.img_resolution, 
                       device=device)
        
        pbar = tqdm(range(self.scheduler.num_steps))
        
        for i in pbar:
            x = x.detach().requires_grad_(True)
            
            if self.use_vp_sde:
                # VP-SDE style (official DPS)
                beta = torch.tensor(self.beta[i], device=device, dtype=x.dtype)
                dt = torch.tensor(self.dt[i], device=device, dtype=x.dtype)
                sigma = torch.tensor(self.sigma[i], device=device, dtype=x.dtype)
                
                # Ensure gradients flow through the network
                with torch.enable_grad():
                    # 1. Score model gives ε(x_t, sigma_t)
                    # Model interface: net(x, sigma)
                    eps = self.net(x, sigma)
                    
                    # 2. Predict x0: x0 = x_t - sigma_t * eps
                    x0 = (x - sigma * eps)
                
                # 3. Likelihood gradient on x0
                grad_x0, loss_scale = self.forward_op.gradient(x0, observation, return_loss=True)
                
                # Backprop ∂x0/∂x to get ∂LL/∂x
                grad_ll = torch.autograd.grad(x0, x, grad_x0, retain_graph=False)[0]
                
                # Normalize (InverseBench style)
                grad_ll = grad_ll * 0.5 / torch.sqrt(loss_scale + 1e-8)
                
                # 4. Posterior ODE drift term: -0.5*beta*x - beta*eps + scale*beta*grad_ll
                drift = -0.5 * beta * x - beta * eps + self.scale * beta * grad_ll
                
                # 5. Update: ODE (deterministic) or SDE (stochastic)
                if self.sde:
                    # SDE: add random noise term for posterior sampling
                    # dx = drift * dt + sqrt(beta) * dW
                    noise = torch.randn_like(x)
                    x = x + drift * dt + torch.sqrt(beta * dt) * noise
                else:
                    # ODE: deterministic update
                    x = x + drift * dt
                
                pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale).item():.4f}')
                
            else:
                # EDM style (fallback to original)
                x_cur = x
                sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
                
                # Ensure gradients flow through the network
                with torch.enable_grad():
                    denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
                
                gradient, loss_scale = self.forward_op.gradient(denoised, observation, return_loss=True)

                ll_grad = torch.autograd.grad(denoised, x_cur, gradient, retain_graph=False)[0]
                ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale + 1e-8)

                score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale).item():.4f}')
                
                if self.sde:
                    epsilon = torch.randn_like(x_cur)
                    x = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
                else:
                    x = x_cur * scaling_factor + factor * score * 0.5 
                x -= ll_grad * self.scale
        
        # Detach before returning to avoid gradient issues in outer context
        return x.detach()
