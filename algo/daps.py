import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler

# ------------------------------------------------------------------------------------
# Paper: Improving diffusion inverse problem solving with decoupled noise annealing
# Official implementation: https://github.com/zhangbingliang2019/DAPS
# ------------------------------------------------------------------------------------


class LangevinDynamics:
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        """
            Initializes the Langevin dynamics sampler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the sampling process.
                lr (float): Learning rate.
                tau (float): Noise parameter.
                lr_min_ratio (float): Minimum learning rate ratio.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False):
        """
            Samples using Langevin dynamics.

            Parameters:
                x0hat (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                sigma (float): Current sigma value.
                ratio (float): Current step ratio.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x0hat = x0hat.detach()
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()

            gradient = operator.gradient(x, measurement) / (2 * self.tau ** 2)
            gradient += (x - x0hat) / sigma ** 2
            
            # Clip gradient to prevent explosion
            grad_norm = gradient.norm()
            if grad_norm > 100.0:
                gradient = gradient / grad_norm * 100.0
            
            x.grad = gradient

            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon
                
                # Clip x to prevent explosion
                x.data = torch.clamp(x.data, min=-5.0, max=5.0)
                
                # For toy 8D problem: enforce zero-padding constraint (last 8 dimensions should be 0)
                # But skip this if using full 16x16 matrix (A_type == 'fixed-full-rank-16x16')
                if hasattr(operator, 'dim_true') and hasattr(operator, 'dim_padded'):
                    if operator.dim_true == 8 and operator.dim_padded == 16:
                        # Check if using full 16x16 matrix
                        if not (hasattr(operator, 'A_type') and operator.A_type == 'fixed-full-rank-16x16'):
                            # Reshape to vector, zero out last 8 dims, reshape back
                            x_vec = x.data.view(x.data.shape[0], -1)
                            x_vec[:, 8:] = 0  # Force last 8 dimensions to be 0
                            x.data = x_vec.view(x.data.shape)

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

        return x.detach()
    
    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
    
    
class DAPS(Algo):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, net, forward_op, annealing_scheduler_config={}, diffusion_scheduler_config={}, lgvd_config={}):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super(DAPS, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        # annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
        #                                                                      diffusion_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    
    def inference(self, observation, num_samples=1, verbose=True):
        """
            Samples using the DAPS method.

            Parameters:
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        device = self.forward_op.device
        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.annealing_scheduler.sigma_max
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            x0hat = sampler.sample(self.net, xt, SDE=False, verbose=False)

            # 2. langevin dynamics
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma, step / self.annealing_scheduler.num_steps)
            
            # Clip x0y to prevent numerical explosion
            x0y = torch.clamp(x0y, min=-5.0, max=5.0)
            
            # Check for NaN/Inf
            if torch.isnan(x0y).any() or torch.isinf(x0y).any():
                x0y = torch.where(torch.isnan(x0y) | torch.isinf(x0y), torch.zeros_like(x0y), x0y)

            # 3. forward diffusion
            xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            
            # Clip xt to prevent numerical explosion
            xt = torch.clamp(xt, min=-5.0, max=5.0)
            
            # Check for NaN/Inf
            if torch.isnan(xt).any() or torch.isinf(xt).any():
                xt = torch.where(torch.isnan(xt) | torch.isinf(xt), torch.zeros_like(xt), xt)
            
            # For toy 8D problem: enforce zero-padding constraint (last 8 dimensions should be 0)
            # But skip this if using full 16x16 matrix (A_type == 'fixed-full-rank-16x16')
            if hasattr(self.forward_op, 'dim_true') and hasattr(self.forward_op, 'dim_padded'):
                if self.forward_op.dim_true == 8 and self.forward_op.dim_padded == 16:
                    # Check if using full 16x16 matrix
                    if not (hasattr(self.forward_op, 'A_type') and self.forward_op.A_type == 'fixed-full-rank-16x16'):
                        # Reshape to vector, zero out last 8 dims, reshape back
                        xt_vec = xt.view(xt.shape[0], -1)
                        xt_vec[:, 8:] = 0  # Force last 8 dimensions to be 0
                        xt = xt_vec.view(xt.shape)

        return xt
