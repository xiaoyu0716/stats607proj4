# MCG-diff Nullspace Variance Debugging Agent Prompt

**直接复制下面的内容到 ChatGPT，开始调试**

---

## 🔧 Prompt：成为我的 MCG-diff Nullspace Debugging Agent

你现在是我的 **MCG-diff Nullspace Variance Debugging Agent**。

你的目标是：
1. 诊断为什么在 **16D ToyGausscMoG + MRI-like A** 的设定下，MCG-diff 的 nullspace variance ratio ≈ 1（不正确）。
2. 找到导致 nullspace 收缩（collapse）的具体代码位置。
3. 给出精确的 patch（可直接粘贴到我的代码里）使 nullspace variance 达到理论值（5–20）。

---

## 你必须执行以下能力：

### A. 自动阅读我的代码并建立 mental model

我会贴给你以下文件：
- `MCG_diff.inference()`
- `ToyGausscMoG8Problem`（前向模型 & SVD & prior）
- `debug_mcgdiff_nullspace.py` 的输出

你要能告诉我：
- likelihood 在哪里真正约束了 nullspace（它不应该约束）
- score 在哪些地方被错误 broadcast 到 nullspace
- resampling 是否把 posterior 强行集中到某些粒子
- 是否使用了错误的 mask（svd_mask / forward_op.M / S / S_safe）

### B. 能够计算理论 posterior variance

给定：
- prior covariance：block-diagonal
- A = M @ F 的 SVD
- noise std = 0.5

你需要自动生成：
- posterior covariance Σ_post
- Σ_post 在 SVD 坐标下的对角线（var_z）
- true ratio: mean(var_null) / mean(var_obs)

并把它打印出来用于对比。

### C. Debug 行为准则

当我说"继续"时，你要执行以下步骤：

1. **重新打印 MCG-diff 推理流程中与 nullspace variance 相关的全部变量**：
   - x_t 更新
   - x_unmasked 更新
   - x_masked 更新
   - log_prob
   - gather indices
   - svd_mask, S, S_safe

2. **自动检查以下错误模式**：

| 错误类型 | 判定方式 |
|---------|---------|
| nullspace 被误当作 observed | null dims 出现在 likelihood 约束里 |
| score 强行作用在 nullspace | denoised_t 使用了错误 broadcast |
| resampling 意外收缩 nullspace | multinomial 选择只集中在少数粒子 |
| 初始化不符合 prior | sigma_max 初始化导致 nullspace variance 太小 |

3. **针对检测到的问题，提供精确的代码 patch**，例如：

```python
# WRONG
x_unmasked = x_next_t + sqrt(factor) * randn

# FIX
x_unmasked = x_t + sqrt(factor) * randn   # pure prior reverse diffusion
```

4. **运行"patch 后 MCG-diff"的预期行为对比**（不用实际运行，给出理论预期）

### D. 最终目标：

当我说："请确认 nullspace variance 达到理论值"

你要检查：
- 输出中 null dims 的 variance 是否 ≈ prior variance（2–5）
- 输出中 observed dims 的 variance 是否 ≈ 0.1–0.3
- ratio 是否 ≥ 5

并告诉我"MCG-diff 是否已恢复正确的欠定性不确定性建模"。

---

## 📋 代码上下文

### 1. MCG_diff.inference() 完整代码

```python
import torch
import tqdm
from .base import Algo
import numpy as np
from utils.scheduler import Scheduler
from utils.helper import has_svd

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
        observation = observation / self.forward_op.unnorm_scale - self.forward_op.forward(self.forward_op.unnorm_shift * torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device),unnormalize=False)

        # Mask for observed SVD dimensions
        S = self.forward_op.S.to(device)
        # For MRI-like A, S should be exactly 0 or 1, but numerical errors may produce
        # values close to 0 (e.g., 1e-8). Use a stricter threshold (0.1) to only keep
        # dimensions where S is close to 1, avoiding numerical explosion when dividing.
        # This ensures we only use well-conditioned dimensions.
        svd_mask = (S > 0.1).float()

        # Compute observation_t = Ut(y) / S  for observed dims only
        obs_ut = self.forward_op.Ut(observation)
        
        # Clip S to avoid numerical explosion when dividing by very small values
        # For MRI-like A, S should be 0 or 1, so clip small values to a safe minimum
        # Use 0.1 as minimum to avoid dividing by values that cause explosion
        # (For S < 0.1, the dimension should be treated as unobserved anyway)
        S_clipped = torch.clamp(S, min=0.1)  # Clip S to minimum 0.1
        S_safe = torch.where(svd_mask > 0, S_clipped, torch.ones_like(S))
        observation_t = (obs_ut / S_safe) * svd_mask
        
        # Initialize x_t in SVD space
        z = torch.randn(num_samples, self.num_particles, *self.forward_op.M.shape, device=device)
        K0 = self.K(0)  # K(0) = 1
        x_t = self.scheduler.sigma_max * z
        
        pbar = tqdm.trange(self.scheduler.num_steps)

        MAX_BATCH_SIZE = 128
        for step in pbar:
            sigma, sigma_next, factor, scaling_factor, scaling_step = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step], self.scheduler.scaling_steps[step]
            x = self.forward_op.V(x_t.flatten(0,1))

            denoised_t = []
            for i in range(0, x.shape[0], MAX_BATCH_SIZE):
                # Follow reference implementation: Vt(...).view(-1, num_particles, *M.shape)
                denoised_t.append(self.forward_op.Vt(self.net(x[i:i+MAX_BATCH_SIZE]/scaling_step, torch.as_tensor(sigma).to(x.device))).view(-1, self.num_particles, *self.forward_op.M.shape))
            denoised_t = torch.cat(denoised_t, dim=0)
            score = (denoised_t - x_t / scaling_step) / sigma ** 2 / scaling_step
            x_next_t = x_t * scaling_factor + factor * score
            
            # Compute log probability for resampling (follow reference implementation)
            # Use svd_mask consistently for all likelihood terms
            log_prob = -torch.linalg.norm(
                ((observation_t - x_next_t) * svd_mask).flatten(2),
                dim=-1
            )**2 / (2 * (sigma_next**2 + factor))
            # FIXED: Changed from self.forward_op.M to svd_mask for consistency
            log_prob += torch.linalg.norm(((observation_t - x_t) * svd_mask).flatten(2), dim=-1)**2 / (2 * sigma**2)
            
            log_prob -= log_prob.min(dim=1, keepdim=True)[0]
            log_prob = torch.clamp(log_prob, max=60)
            # Ensure numerical stability: clamp to avoid exp(very negative) = 0, and ensure no NaN/Inf
            log_prob = torch.clamp(log_prob, min=-700, max=60)  # exp(-700) is approximately 0, safe for float32
            # Check for NaN/Inf and replace with very negative value
            log_prob = torch.where(torch.isfinite(log_prob), log_prob, torch.tensor(-700.0, device=log_prob.device))
            # Compute probabilities and ensure they are valid
            prob = torch.exp(log_prob)
            # Normalize to ensure valid probability distribution
            prob_sum = prob.sum(dim=1, keepdim=True)
            # If sum is too small (all probabilities near zero), use uniform distribution as fallback
            uniform_prob = torch.ones_like(prob) / self.num_particles
            prob = torch.where(prob_sum > 1e-10, prob / prob_sum, uniform_prob)
            # Ensure all probabilities are non-negative and finite
            prob = torch.clamp(prob, min=0.0, max=1.0)
            prob = torch.where(torch.isfinite(prob), prob, uniform_prob)
            # Final normalization to ensure sum is exactly 1.0
            prob_sum = prob.sum(dim=1, keepdim=True)
            prob = torch.where(prob_sum > 1e-10, prob / prob_sum, uniform_prob)
            # Final check: ensure sum is valid before multinomial
            prob_sum = prob.sum(dim=1, keepdim=True)
            if (prob_sum <= 0).any():
                # Fallback to uniform if any batch has invalid sum
                prob = torch.where(prob_sum <= 0, uniform_prob, prob)
            indices = torch.multinomial(prob, self.num_particles, replacement=True)
            

            K = self.K(step+1)
            # Gather indices: x_next_t is [B, num_particles, *M.shape], indices is [B, num_particles]
            # Need to expand indices to match x_next_t's shape for gathering along dimension 1
            # x_next_t shape: [B, num_particles, *M.shape], e.g., [B, num_particles, 1, 1, 4, 4]
            # We need gather_indices with same shape as x_next_t
            gather_indices = indices.unsqueeze(-1)  # [B, num_particles, 1]
            # Add trailing dimensions: need len(M.shape) - 1 more dims (since we already have 1)
            # x_next_t is [B, num_particles] + M.shape, so we need indices: [B, num_particles] + M.shape
            # From [B, num_particles, 1], we need to add the remaining dims from M.shape
            for _ in range(len(self.forward_op.M.shape) - 1):
                gather_indices = gather_indices.unsqueeze(-1)
            # Now expand the last dimensions to match M.shape[1:] (skip the first dim of M which is batch-like)
            # gather_indices is [B, num_particles, 1, 1, 1, ...], need to expand to match x_next_t
            gather_indices = gather_indices.expand(list(gather_indices.shape[:2]) + list(x_next_t.shape[2:]))
            x_next_t = torch.gather(x_next_t, 1, gather_indices)
            
            # Update x_t using masked and unmasked updates (follow reference implementation)
            # Use svd_mask consistently (already defined at the beginning)
            x_masked = (
                K * observation_t * svd_mask +
                (1 - K) * x_next_t + 
                np.sqrt(K) * sigma_next * torch.randn_like(x_t)
            )
            x_unmasked = x_next_t + np.sqrt(factor) * torch.randn_like(x_t)
            # pure prior reverse diffusion, not guided by likelihood or score
            # x_unmasked = x_t + np.sqrt(factor) * torch.randn_like(x_t)

            x_t = svd_mask * x_masked + (1 - svd_mask) * x_unmasked
            
        # Return final result: convert from SVD space to image space
        # MCG_diff particles are internal Monte Carlo objects for guidance, NOT posterior samples
        # Each inference() call should return exactly ONE posterior sample
        # True posterior sampling is done by multiple independent inference() calls
        # x_t shape: [B=1, num_particles=P, *M.shape]
        
        if num_samples == 1:
            # Choose ONE particle as the posterior sample (not average, not all particles)
            # Use first particle for deterministic behavior, or random for stochastic
            # x_t[0] has shape [num_particles, *M.shape]
            # Select first particle: x_t[0, 0] -> [*M.shape]
            x_final_svd = x_t[0, 0:1]  # [1, *M.shape] - select first particle, keep batch dim
        else:
            # For multiple batches, select first particle from each batch
            # x_t shape: [B, num_particles, *M.shape]
            x_final_svd = x_t[:, 0:1]  # [B, 1, *M.shape] - select first particle from each batch
        
        # Convert to image space
        # x_final_svd: [1, *M.shape] or [B, 1, *M.shape]
        x_final_img = self.forward_op.V(x_final_svd)  # [1, 1, 4, 4] or [B, 1, 4, 4]
        return x_final_img
```

### 2. 问题设置

**Prior X (先验分布)**:
- 16维混合高斯分布 (Mixture of Gaussians, MoG)
- 前8维 (dim 0-7): MoG，2个分量，均值在第7维分别为-2.0和+2.0，协方差是Toeplitz结构 (rho=0.8)
- 后8维 (dim 8-15): 弱高斯先验，均值为0，方差为5.0
- 先验协方差矩阵：前8×8块是Toeplitz矩阵，后8×8块是对角矩阵（对角线元素为5.0）

**A (前向算子矩阵)**:
- 类型: MRI-like (A = M @ F)
- F: 16×16 类Fourier矩阵（正交）
- M: 对角mask矩阵，只有9个1，7个0
- SVD分解:
  - 观测维度 (S > 0.1): [0, 1, 2, 3, 4, 5, 6, 7, 8] (共9个)
  - Null空间维度 (S ≤ 0.1): [9, 10, 11, 12, 13, 14, 15] (共7个)

**观测模型**: y = A @ x + noise, noise_std = 0.5

### 3. 当前诊断结果

从 `debug_mcgdiff_nullspace.py` 的输出：
- **SVD一致性**: ✅ 正常
- **Nullspace存在**: ✅ 9个observed dims，7个null dims
- **MCG-diff返回**: ✅ 样本有方差
- **Nullspace泄漏**: ✅ 无泄漏
- **问题**: ❌ Nullspace variance ratio = 0.8072（期望 >> 1，理想5-20）

**关键发现**:
- 观测维度方差: 0.1588
- Null空间方差: 0.1282
- Ratio: 0.8072（应该 >> 1）

### 4. 理论预期

对于线性逆问题 y = Ax + noise，后验协方差为：
- Σ_post = (A^T A / σ_noise^2 + Σ_prior^-1)^-1

在SVD空间（z = V^T x）中：
- Observed dims: variance ≈ σ_noise^2 / S^2（当S=1时，≈ 0.25）
- Null dims: variance ≈ prior variance（≈ 5.0 for dims 8-15）

**预期 ratio**: mean(var_null) / mean(var_obs) ≈ 5.0 / 0.25 = 20

---

## ✨ 开始调试

请按照上述步骤开始分析 nullspace variance collapse 的原因。

**第一步**: 请分析代码，找出所有可能导致 nullspace variance 被抑制的地方。

**第二步**: 计算理论后验方差，并与当前输出对比。

**第三步**: 提供精确的代码 patch。

---

**提示**: 当你准备好时，说"继续"，我会提供更多调试信息。
