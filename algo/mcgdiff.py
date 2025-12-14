import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_svd

# -------------------------------------------------------------------------------------------
# Paper: MCG-Diff: Monte Carlo guided diffusion for Bayesian linear inverse problems
# Official implementation: https://github.com/gabrielvc/mcg_diff
# -------------------------------------------------------------------------------------------

class MCG_diff(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 scheduler_config,
                 num_particles):
        super(MCG_diff, self).__init__(net, forward_op)
        assert has_svd(forward_op), "MCG_diff only works with linear forward operators, which can be decomposed via SVD"
        self.scheduler = Scheduler(**scheduler_config)
        self.num_particles = num_particles

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
        return (d - x) / sigma**2
    
    def K(self, t):
        if t == self.scheduler.num_steps:
            return 1
        return self.scheduler.factor_steps[t] / (self.scheduler.factor_steps[t]+ self.scheduler.sigma_steps[t]**2)
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        # -------------------------------------------------------
        # 0. preprocess y (原始代码，不动)
        # -------------------------------------------------------
        observation = (
            observation / self.forward_op.unnorm_scale
            - self.forward_op.forward(
                self.forward_op.unnorm_shift *
                torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device),
                unnormalize=False
            )
        )

        # -------------------------------------------------------
        # 1. 真正正确的 SVD mask（flat 16 维）
        # -------------------------------------------------------
        # SVD space is 16-dim: do NOT reshape to 4x4!
        S_flat = self.forward_op.S.view(1, 1, 1, 1, 16).to(device)   # [1,1,1,1,16]
        svd_mask = (S_flat > 0.1).float()                            # observed dims
        null_mask = 1.0 - svd_mask                                   # null dims

        # Safe S (avoid division by tiny S)
        S_safe = torch.where(svd_mask > 0, S_flat, torch.ones_like(S_flat))

        # -------------------------------------------------------
        # 2. 将 y 投影到 16 维 SVD basis
        # -------------------------------------------------------
        obs_ut_flat = self.forward_op.Ut(observation).view(1, 1, 1, 1, 16)
        observation_t = (obs_ut_flat / S_safe) * svd_mask            # only observed dims

        # Broadcast to particles: [1,P,1,1,16]
        observation_t = observation_t.expand(-1, self.num_particles, -1, -1, -1)

        # -------------------------------------------------------
        # 3. 初始化粒子 x_t in SVD space  (flat)
        # -------------------------------------------------------
        z = torch.randn(1, self.num_particles, 1, 1, 16, device=device)
        x_t = self.scheduler.sigma_max * z                           # pure noise init

        # -------------------------------------------------------
        # 4. Diffusion Loop
        # -------------------------------------------------------
        pbar = tqdm.trange(self.scheduler.num_steps)

        for step in pbar:
            sigma      = self.scheduler.sigma_steps[step]
            sigma_next = self.scheduler.sigma_steps[step+1]
            factor     = self.scheduler.factor_steps[step]
            scaling_factor = self.scheduler.scaling_factor[step]
            scaling_step   = self.scheduler.scaling_steps[step]

            # ---------------------------------------------------
            # 4.1 x_t → image → score → back to SVD space
            # ---------------------------------------------------
            x_img = self.forward_op.V(x_t.view(1*self.num_particles, 1, 4, 4))

            denoised = []
            MAX_BS = 128
            for i in range(0, x_img.shape[0], MAX_BS):
                bs = x_img[i:i+MAX_BS] / scaling_step
                out = self.net(bs, torch.as_tensor(sigma, device=bs.device))
                vt = self.forward_op.Vt(out).view(-1, 1, 1, 1, 16)
                denoised.append(vt)
            denoised = torch.cat(denoised, dim=0).view(1, self.num_particles, 1, 1, 16)

            score = (denoised - x_t / scaling_step) / (sigma**2) / scaling_step
            x_next_t = x_t * scaling_factor + factor * score

            # ---------------------------------------------------
            # 4.2 Resampling based on data likelihood
            # ---------------------------------------------------
            ll_new = -(((observation_t - x_next_t) * svd_mask).flatten(2).norm(dim=-1)**2) / (2*(sigma_next**2 + factor))
            ll_old =  (((observation_t - x_t     ) * svd_mask).flatten(2).norm(dim=-1)**2) / (2*sigma**2)
            log_prob = ll_new + ll_old

            # Normalize for numerical stability
            log_prob -= log_prob.max(dim=1, keepdim=True)[0]
            prob = torch.softmax(log_prob, dim=1)

            idx = torch.multinomial(prob, self.num_particles, replacement=True)

            # Gather particles
            gather_idx = idx[...,None,None,None].expand(-1, -1, 1, 1, 16)
            x_next_t = torch.gather(x_next_t, 1, gather_idx)

            # ---------------------------------------------------
            # 4.3 Update observed vs null separately (核心修复)
            # ---------------------------------------------------
            K = self.K(step+1)
            K_tensor = torch.tensor(K, device=device)

            # Observed dims use conditional update
            x_obs_update = (
                K_tensor * observation_t +
                (1 - K_tensor) * x_next_t +
                torch.sqrt(K_tensor) * sigma_next * torch.randn_like(x_t)
            )

            # Null dims use PURE prior reverse SDE
            x_null_update = x_t + torch.sqrt(torch.tensor(factor, device=device)) * torch.randn_like(x_t)

            # Combine correctly by mask (16-D aligned!)
            x_t = svd_mask * x_obs_update + null_mask * x_null_update

        # -------------------------------------------------------
        # 5. 返回 posterior sample — 选择最可能的粒子
        # -------------------------------------------------------
        # 计算所有 particles 的 likelihood（基于 observed dims）
        # 使用最终的 x_t 和 observation_t
        # Likelihood: -||(observation_t - x_t) * svd_mask||^2 / (2 * sigma_final^2)
        sigma_final = self.scheduler.sigma_steps[-1]  # 最后一步的 sigma
        if sigma_final < 1e-6:
            sigma_final = 1e-6  # 避免除以 0
        
        # 计算每个粒子的 likelihood（只考虑 observed dims）
        # x_t: [1, num_particles, 1, 1, 16]
        # observation_t: [1, num_particles, 1, 1, 16]
        log_likelihood = -(((observation_t - x_t) * svd_mask).flatten(2).norm(dim=-1)**2) / (2 * sigma_final**2)
        # log_likelihood: [1, num_particles]
        
        # 选择 likelihood 最高的粒子
        best_pid = log_likelihood.argmax(dim=1).item()  # [1] -> scalar
        
        # 返回最可能的粒子
        x_final_svd = x_t[0, best_pid:best_pid+1].view(1, 1, 4, 4)
        # Convert back to image
        x_img = self.forward_op.V(x_final_svd)
        return x_img
