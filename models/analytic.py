import torch

class AnalyticPrior(torch.nn.Module):
    """
    An analytic prior that provides score and denoise functions for a given distribution.
    """
    def __init__(self, prior_type, gauss_rho, mog8_mu, mog8_wm_full, mog8_wp_full, **kwargs):
        super().__init__()
        self.prior_type = prior_type
        if self.prior_type == 'gausscmog8':
            self.dim = 8
            self.img_channels = 8  # Treat as an 8-channel, 1x1 'image'
            self.img_resolution = 1 # This will make DPS create a (N, 8, 1, 1) tensor
            mu = float(mog8_mu)
            self.means = torch.zeros(2, self.dim)
            self.means[0, 7] = -mu
            self.means[1, 7] = +mu

            self.weights = torch.tensor([mog8_wm_full, mog8_wp_full])
            self.weights = self.weights / self.weights.sum()

            rho = float(gauss_rho)
            idx = torch.arange(self.dim).float()
            absdiff = (idx[:, None] - idx[None, :]).abs()
            self.Sigma0 = rho ** absdiff
            self.Sigma0_inv = torch.linalg.inv(self.Sigma0)
        else:
            raise NotImplementedError(f"Prior type {self.prior_type} not implemented")

    def score(self, x, t=None):
        """Calculates the score grad_x log p(x)."""
        if self.prior_type == 'gausscmog8':
            # Squeeze the spatial dimensions if they exist
            if x.dim() == 4:
                x = x.squeeze(-1).squeeze(-1)
            x = x.to(torch.float32)
            
            x = x.to(self.means.device)
            self.Sigma0_inv = self.Sigma0_inv.to(x.device)
            
            # Numerator: w1 * N(x|mu1, Sigma) * (-Sigma_inv @ (x-mu1)) + w2 * ...
            # Denominator: w1 * N(x|mu1, Sigma) + w2 * ...
            # Simplified: sum_i (w_i * pdf_i * score_i) / sum_i (w_i * pdf_i)

            # logpdf: -0.5 * (x-mu).T @ Sigma_inv @ (x-mu)
            diffs = x.unsqueeze(1) - self.means.unsqueeze(0)  # [B, M, D]
            log_pdfs = -0.5 * torch.einsum('bmd,de,bme->bm', diffs, self.Sigma0_inv, diffs)
            
            # scores: -Sigma_inv @ (x-mu)
            scores_per_comp = -torch.einsum('de,bme->bmd', self.Sigma0_inv, diffs) # [B, M, D]

            # softmax on log_pdfs with weights to get component probabilities
            log_probs = torch.log(self.weights.view(1, -1)) + log_pdfs # [B, M]
            probs = torch.softmax(log_probs, dim=1) # [B, M]

            # weighted average of scores
            final_score = torch.einsum('bm,bmd->bd', probs, scores_per_comp) # [B, D]
            return final_score
        else:
            raise NotImplementedError

    def denoise(self, x, t, **kwargs):
        """Approximates the denoiser e(x, t) using the score function.
           e(x,t) ~ (x_t - x_0) / (-sigma_t)
           For DPS, it expects a denoiser that predicts x0.
           x0_pred = xt + sigma_t^2 * score(xt)
        """
        # This is a simplified denoiser for a prior on x0, not a time-dependent diffusion process.
        # We assume the input 'x' is a noisy version of x0, and 't' gives the noise level.
        # The algorithms in this repo often use VP or VE schedulers.
        # Let's assume sigma is passed via 't' somehow, or we can approximate.
        # For DPS, the effective sigma is often related to the data noise.
        # Here, we just use the score of the prior p(x0).
        
        # A simple interpretation for a static prior is that the denoiser is a projection.
        # x0_pred = x + noise_var * grad_log_p(x)
        # Let's assume `t` is sigma_t^2, a common convention.
        # The DPS implementation might pass sigma via `t`.
        sigma_sq = t.view(-1, 1, 1, 1).to(x.dtype)
        score_val = self.score(x)
        # Clip the score to prevent explosion
        score_val = torch.clamp(score_val, -1e5, 1e5)

        # Reshape score to match input shape if needed
        if x.dim() == 4 and score_val.dim() == 2:
            score_val = score_val.unsqueeze(-1).unsqueeze(-1)
        
        # Tweedie's formula for denoising
        denoised_x = x + sigma_sq * score_val
        return denoised_x.to(x.dtype)

    def forward(self, *args, **kwargs):
        # DPS calls model(x, t), which maps to denoise.
        return self.denoise(*args, **kwargs)
