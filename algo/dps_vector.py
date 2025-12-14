import torch
import torch.nn as nn
from tqdm import tqdm
from algo.base import Algo

class DPSVector(Algo):
    """
    Vector version of DPS for toy problems (no images).
    x is [B, D]
    A is [D, D]
    """

    def __init__(self, net, forward_op, scheduler_config, guidance_scale=1.0):
        super().__init__(net, forward_op)
        from utils.scheduler import Scheduler
        self.scheduler = Scheduler(**scheduler_config)
        self.A = forward_op.A.to(forward_op.device)                  # [D,D]
        self.A_T = self.A.t()
        self.noise_std = forward_op.noise_std
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        D = self.A.shape[0]

        # initialize x_T ~ N(0, sigma_max^2 I)
        x = torch.randn(num_samples, D, device=device) * self.scheduler.sigma_max

        y = observation.to(device) # observation is already [B, D]
        if y.shape[0] != num_samples:
            y = y.repeat(num_samples, 1)

        pbar = tqdm(range(self.scheduler.num_steps))
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            sigma_next = self.scheduler.sigma_steps[i+1]
            step_size = sigma**2 - sigma_next**2

            # ---- prior score: -(1/sigma)*eps_pred
            eps_pred = self.net(x, torch.tensor(sigma, device=device))
            score_prior = -(1.0 / sigma) * eps_pred

            # ---- likelihood gradient: A^T (y - A x) / noise_std^2
            Ax = x @ self.A_T
            resid = y - Ax
            grad_lik = resid @ self.A / (self.noise_std**2)

            # ---- Predict x0 from prior
            x0_pred_prior = x + (sigma**2) * score_prior

            # ---- Likelihood gradient correction
            Ax0 = x0_pred_prior @ self.A_T
            resid = y - Ax0
            grad_lik_x0 = resid @ self.A / (self.noise_std**2)

            # ---- Corrected x0 prediction
            x0_pred_posterior = x0_pred_prior + self.guidance_scale * (sigma**2) * grad_lik_x0

            # ---- DDIM-like update to the next step
            x = (sigma_next / sigma) * x + (sigma**2 - sigma_next**2).sqrt() * ( (x - x0_pred_posterior) / sigma )


        return x
