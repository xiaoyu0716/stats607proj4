import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from .base import BaseOperator

class ToyImageLesionProblem(BaseOperator):
    """
    Toy 16×16 image problem with blur + downsampling forward operator.
    Prior: Two-component Gaussian mixture (normal background vs lesion).
    """
    C, H, W = 1, 16, 16  # Input image size
    H_obs, W_obs = 8, 8  # Observation size after downsampling
    
    def __init__(self, 
                 blur_sigma=1.0,  # Gaussian blur sigma
                 noise_std=0.03,  # Observation noise std (pixels in [0,1])
                 tau=0.2,  # Prior std: Σ = τ²I
                 lesion_prior_weight=0.1,  # π (lesion prior weight)
                 lesion_amplitude=0.25,  # Amplitude of lesion
                 lesion_radius=3,  # Radius of lesion in pixels
                 lesion_center=None,  # Center of lesion (if None, use center of image)
                 **kwargs):
        super().__init__(**kwargs)
        
        self.blur_sigma = float(blur_sigma)
        self.noise_std = float(noise_std)
        if 'sigma_noise' not in kwargs:
            self.sigma_noise = self.noise_std
        self.tau = float(tau)
        self.lesion_prior_weight = float(lesion_prior_weight)
        self.lesion_amplitude = float(lesion_amplitude)
        self.lesion_radius = int(lesion_radius)
        self.lesion_center = lesion_center  # Can be None or (y, x) tuple
        
        # Prior means: μ₀ (normal) and μ₁ (lesion)
        self.mu_0 = torch.zeros(self.H, self.W, device=self.device)  # Normal background (zero)
        self.mu_1 = self._create_lesion_pattern()  # Lesion pattern
        
        # Prior weights
        self.pi_0 = 1.0 - self.lesion_prior_weight
        self.pi_1 = self.lesion_prior_weight
        self.weights = torch.tensor([self.pi_0, self.pi_1], device=self.device)
        
        # Prior covariance: Σ = τ²I
        self.Sigma_prior = (self.tau ** 2) * torch.eye(self.H * self.W, device=self.device)
        
        # Create blur kernel
        self._create_blur_kernel()
        
        # Precompute forward operator matrix A for posterior computation
        self._compute_forward_matrix()
    
    def _create_lesion_pattern(self):
        """Create lesion pattern: μ₁ = μ₀ + δ, where δ is a bright circular spot."""
        mu_1 = torch.zeros(self.H, self.W, device=self.device)
        
        # Center of lesion
        if self.lesion_center is None:
            center_y, center_x = self.H // 2, self.W // 2
        else:
            center_y, center_x = self.lesion_center
        
        # Create circular lesion with Gaussian smoothing
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=torch.float32),
            torch.arange(self.W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        dist_sq = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
        # Gaussian-smoothed circle
        lesion_mask = torch.exp(-dist_sq / (2 * (self.lesion_radius ** 2)))
        mu_1 = self.lesion_amplitude * lesion_mask
        
        return mu_1
    
    def _create_blur_kernel(self):
        """Create Gaussian blur kernel."""
        kernel_size = int(6 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 2D Gaussian kernel
        kernel_1d = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (kernel_1d / self.blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Reshape for conv2d: [1, 1, H, W]
        self.blur_kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    def _compute_forward_matrix(self):
        """
        Compute forward operator matrix A = D ∘ S (downsample ∘ blur).
        A: [H_obs*W_obs, H*W] matrix
        """
        # Create identity image for each pixel to compute A
        A_list = []
        for i in range(self.H * self.W):
            x = torch.zeros(1, 1, self.H, self.W, device=self.device)
            x.view(-1)[i] = 1.0
            
            # Apply forward operator
            y = self._apply_forward(x)
            A_list.append(y.reshape(-1))
        
        self.A = torch.stack(A_list, dim=1)  # [H_obs*W_obs, H*W]
        self.A_T = self.A.T  # [H*W, H_obs*W_obs]
    
    def _apply_blur(self, x):
        """Apply Gaussian blur to image."""
        # x: [B, 1, H, W]
        padding = self.blur_kernel.shape[2] // 2
        x_blurred = F.conv2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            self.blur_kernel,
            padding=0
        )
        return x_blurred
    
    def _apply_downsample(self, x):
        """Apply 2× downsampling (stride=2, take even indices)."""
        # x: [B, 1, H, W]
        # Downsample by taking every 2nd pixel (even indices: 0, 2, 4, ...)
        return x[:, :, ::2, ::2]
    
    def _apply_forward(self, x):
        """Apply forward operator: blur + downsample."""
        x_blurred = self._apply_blur(x)
        x_downsampled = self._apply_downsample(x_blurred)
        return x_downsampled
    
    def forward(self, x_img):
        """
        Forward operator: A(x) = D(S(x)) where S is blur and D is downsampling.
        Input: x_img [B, 1, 16, 16]
        Output: y_img [B, 1, 8, 8]
        """
        y = self._apply_forward(x_img)
        # Add noise (only to observation, not to padding if any)
        noise = self.noise_std * torch.randn_like(y)
        return y + noise
    
    def sample_prior(self, n, exact_proportion=True):
        """
        Sample from prior: mixture of two Gaussians.
        
        Args:
            n: Number of samples
            exact_proportion: If True, ensure exactly 10% have lesion, 90% normal.
                           If False, use random multinomial sampling.
        """
        if exact_proportion:
            # Ensure exactly 10% have lesion, 90% normal
            n_lesion = int(n * self.lesion_prior_weight)  # 10% with lesion
            n_normal = n - n_lesion  # 90% without lesion
            
            # Create component assignments
            comp = torch.zeros(n, dtype=torch.long, device=self.device)
            comp[:n_lesion] = 1  # First n_lesion samples have lesion
            
            # Shuffle to randomize order
            perm = torch.randperm(n, device=self.device)
            comp = comp[perm]
        else:
            # Random multinomial sampling (original behavior)
            comp = torch.multinomial(self.weights, n, replacement=True)  # [n]
        
        # Sample from each component
        samples = []
        for i in range(n):
            if comp[i] == 0:
                # Normal component: N(μ₀, τ²I)
                z = torch.randn(self.H * self.W, device=self.device) * self.tau
                x = (self.mu_0.view(-1) + z).view(1, 1, self.H, self.W)
            else:
                # Lesion component: N(μ₁, τ²I)
                z = torch.randn(self.H * self.W, device=self.device) * self.tau
                x = (self.mu_1.view(-1) + z).view(1, 1, self.H, self.W)
            samples.append(x)
        
        return torch.cat(samples, dim=0)
    
    def generate_sample(self):
        """Generate a sample from prior and its observation."""
        x0 = self.sample_prior(1)
        y = self.forward(x0)
        return x0[0], y[0]
    
    def compute_posterior_variance(self, observation=None):
        """
        Compute closed-form posterior for the linear Gaussian inverse problem.
        
        For y = A @ x + noise, where noise ~ N(0, σ_y² I)
        Prior: p(x) = π₀ N(x | μ₀, τ²I) + π₁ N(x | μ₁, τ²I)
        
        Posterior: p(x|y) = w₀(y) N(x | μ₀^post, Σ_post) + w₁(y) N(x | μ₁^post, Σ_post)
        
        where:
        - Σ_post = (1/τ² I + 1/σ_y² A^T A)^(-1)  (common to both components)
        - μ_k^post = μ_k + τ² A^T (σ_y² I + τ² A A^T)^(-1) (y - A μ_k)
        - w_k(y) ∝ π_k N(y | A μ_k, σ_y² I + τ² A A^T)
        
        Args:
            observation: Optional observation y [B, 1, 8, 8]. If None, returns only Σ_post.
        
        Returns:
            Always includes:
                - posterior_covariance: Common posterior covariance matrix [H*W, H*W]
                - posterior_variance_diag: Diagonal of posterior covariance [H*W]
            
            If observation is provided, also includes:
                - posterior_means: dict with posterior means for each component
                - updated_component_weights: Updated mixture weights w_k(y) [B, 2]
                - total_posterior_variance: Total variance of mixture posterior [B, H*W]
                - total_posterior_mean: Total mean of mixture posterior [B, H*W]
        """
        sigma_y_sq = self.noise_std ** 2
        tau_sq = self.tau ** 2
        
        # Compute posterior covariance (common to both components)
        # Σ_post^(-1) = 1/τ² I + 1/σ_y² A^T A
        A_T_A = self.A_T @ self.A  # [H*W, H*W]
        Sigma_post_inv = (1.0 / tau_sq) * torch.eye(self.H * self.W, device=self.device) + (1.0 / sigma_y_sq) * A_T_A
        Sigma_post = torch.linalg.inv(Sigma_post_inv)
        
        result = {
            'posterior_covariance': Sigma_post,  # [H*W, H*W]
            'posterior_variance_diag': torch.diag(Sigma_post),  # [H*W]
        }
        
        if observation is not None:
            import math
            # observation: [B, 1, 8, 8]
            y_vec = observation.view(observation.shape[0], -1)  # [B, 64]
            B = y_vec.shape[0]
            
            # Compute A @ μ_k for both components
            mu_0_vec = self.mu_0.view(-1)  # [256]
            mu_1_vec = self.mu_1.view(-1)  # [256]
            
            A_mu_0 = (self.A @ mu_0_vec).unsqueeze(0)  # [1, 64]
            A_mu_1 = (self.A @ mu_1_vec).unsqueeze(0)  # [1, 64]
            
            # Compute covariance of y: Σ_y = σ_y² I + τ² A A^T
            A_A_T = self.A @ self.A_T  # [64, 64]
            Sigma_y = sigma_y_sq * torch.eye(self.H_obs * self.W_obs, device=self.device) + tau_sq * A_A_T
            Sigma_y_inv = torch.linalg.inv(Sigma_y)
            log_det_Sigma_y = torch.logdet(Sigma_y)
            
            # Compute posterior means for each component
            # μ_k^post = μ_k + τ² A^T (σ_y² I + τ² A A^T)^(-1) (y - A μ_k)
            # = μ_k + τ² A^T Σ_y^(-1) (y - A μ_k)
            posterior_means = {}
            for k, (mu_k_vec, A_mu_k, pi_k) in enumerate([
                (mu_0_vec, A_mu_0, self.pi_0),
                (mu_1_vec, A_mu_1, self.pi_1)
            ]):
                # y - A μ_k: [B, 64]
                diff_y = y_vec - A_mu_k  # [B, 64]
                
                # τ² A^T Σ_y^(-1) (y - A μ_k): [B, 256]
                correction = tau_sq * (self.A_T @ (Sigma_y_inv @ diff_y.T)).T  # [B, 256]
                
                mu_post = mu_k_vec.unsqueeze(0) + correction  # [B, 256]
                posterior_means[f'component_{k}'] = mu_post
            
            # Compute updated mixture weights w_k(y)
            # w_k(y) ∝ π_k N(y | A μ_k, Σ_y)
            logliks = []
            for k, (A_mu_k, pi_k) in enumerate([
                (A_mu_0, self.pi_0),
                (A_mu_1, self.pi_1)
            ]):
                diff_y = y_vec - A_mu_k  # [B, 64]
                # Quadratic form: diff_y^T @ Sigma_y_inv @ diff_y
                quad = torch.einsum('bi,ij,bj->b', diff_y, Sigma_y_inv, diff_y)  # [B]
                loglik = -0.5 * (quad + log_det_Sigma_y + self.H_obs * self.W_obs * math.log(2 * math.pi))
                logliks.append(loglik)
            
            logliks = torch.stack(logliks, dim=1)  # [B, 2]
            log_weights = torch.log(self.weights).unsqueeze(0) + logliks  # [B, 2]
            log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)  # Normalize
            w_tilde = torch.exp(log_weights)  # [B, 2]
            
            # Compute total mixture posterior mean and variance
            mu_post_0 = posterior_means['component_0']  # [B, 256]
            mu_post_1 = posterior_means['component_1']  # [B, 256]
            mu_post_stack = torch.stack([mu_post_0, mu_post_1], dim=1)  # [B, 2, 256]
            
            # Total mean: m = sum_k w_k * μ_k^post
            m = (w_tilde.unsqueeze(-1) * mu_post_stack).sum(dim=1)  # [B, 256]
            
            # Total variance: Var[x|y] = E[Var[x|y,k]] + Var[E[x|y,k]]
            # = Σ_post + sum_k w_k * (μ_k^post @ μ_k^post^T) - m @ m^T
            mu_post_sq = mu_post_stack ** 2  # [B, 2, 256]
            E_mu_sq = (w_tilde.unsqueeze(-1) * mu_post_sq).sum(dim=1)  # [B, 256]
            m_sq = m ** 2  # [B, 256]
            total_var_diag = result['posterior_variance_diag'].unsqueeze(0) + E_mu_sq - m_sq  # [B, 256]
            
            result['posterior_means'] = posterior_means
            result['original_component_weights'] = self.weights.unsqueeze(0).repeat(B, 1)  # [B, 2]
            result['updated_component_weights'] = w_tilde  # [B, 2]
            result['total_posterior_mean'] = m  # [B, 256]
            result['total_posterior_variance'] = total_var_diag  # [B, 256]
        
        return result
    
    def gradient(self, x_img, y_img, return_loss=False):
        """Compute gradient of data consistency loss."""
        x_vec = x_img.view(x_img.shape[0], -1)  # [B, 256]
        y_vec = y_img.view(y_img.shape[0], -1)  # [B, 64]
        
        # Forward: y_pred = A @ x
        y_pred_vec = (self.A @ x_vec.T).T  # [B, 64]
        
        # Gradient: A^T @ (y_pred - y)
        grad_vec = (self.A_T @ (y_pred_vec - y_vec).T).T  # [B, 256]
        grad_img = grad_vec.view(x_img.shape)
        
        if return_loss:
            loss = ((y_pred_vec - y_vec) ** 2).sum(dim=1)  # [B]
            return grad_img, loss
        else:
            return grad_img
    
    def __call__(self, data, **kwargs):
        return self.forward(data['target'])
    
    def unnormalize(self, x):
        return x
    
    def normalize(self, x):
        return x

import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from .base import BaseOperator

class ToyImageLesionProblem(BaseOperator):
    """
    Toy 16×16 image problem with blur + downsampling forward operator.
    Prior: Two-component Gaussian mixture (normal background vs lesion).
    """
    C, H, W = 1, 16, 16  # Input image size
    H_obs, W_obs = 8, 8  # Observation size after downsampling
    
    def __init__(self, 
                 blur_sigma=1.0,  # Gaussian blur sigma
                 noise_std=0.03,  # Observation noise std (pixels in [0,1])
                 tau=0.2,  # Prior std: Σ = τ²I
                 lesion_prior_weight=0.1,  # π (lesion prior weight)
                 lesion_amplitude=0.25,  # Amplitude of lesion
                 lesion_radius=3,  # Radius of lesion in pixels
                 lesion_center=None,  # Center of lesion (if None, use center of image)
                 **kwargs):
        super().__init__(**kwargs)
        
        self.blur_sigma = float(blur_sigma)
        self.noise_std = float(noise_std)
        if 'sigma_noise' not in kwargs:
            self.sigma_noise = self.noise_std
        self.tau = float(tau)
        self.lesion_prior_weight = float(lesion_prior_weight)
        self.lesion_amplitude = float(lesion_amplitude)
        self.lesion_radius = int(lesion_radius)
        self.lesion_center = lesion_center  # Can be None or (y, x) tuple
        
        # Prior means: μ₀ (normal) and μ₁ (lesion)
        self.mu_0 = torch.zeros(self.H, self.W, device=self.device)  # Normal background (zero)
        self.mu_1 = self._create_lesion_pattern()  # Lesion pattern
        
        # Prior weights
        self.pi_0 = 1.0 - self.lesion_prior_weight
        self.pi_1 = self.lesion_prior_weight
        self.weights = torch.tensor([self.pi_0, self.pi_1], device=self.device)
        
        # Prior covariance: Σ = τ²I
        self.Sigma_prior = (self.tau ** 2) * torch.eye(self.H * self.W, device=self.device)
        
        # Create blur kernel
        self._create_blur_kernel()
        
        # Precompute forward operator matrix A for posterior computation
        self._compute_forward_matrix()
    
    def _create_lesion_pattern(self):
        """Create lesion pattern: μ₁ = μ₀ + δ, where δ is a bright circular spot."""
        mu_1 = torch.zeros(self.H, self.W, device=self.device)
        
        # Center of lesion
        if self.lesion_center is None:
            center_y, center_x = self.H // 2, self.W // 2
        else:
            center_y, center_x = self.lesion_center
        
        # Create circular lesion with Gaussian smoothing
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=torch.float32),
            torch.arange(self.W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        dist_sq = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
        # Gaussian-smoothed circle
        lesion_mask = torch.exp(-dist_sq / (2 * (self.lesion_radius ** 2)))
        mu_1 = self.lesion_amplitude * lesion_mask
        
        return mu_1
    
    def _create_blur_kernel(self):
        """Create Gaussian blur kernel."""
        kernel_size = int(6 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 2D Gaussian kernel
        kernel_1d = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (kernel_1d / self.blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Reshape for conv2d: [1, 1, H, W]
        self.blur_kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    def _compute_forward_matrix(self):
        """
        Compute forward operator matrix A = D ∘ S (downsample ∘ blur).
        A: [H_obs*W_obs, H*W] matrix
        """
        # Create identity image for each pixel to compute A
        A_list = []
        for i in range(self.H * self.W):
            x = torch.zeros(1, 1, self.H, self.W, device=self.device)
            x.view(-1)[i] = 1.0
            
            # Apply forward operator
            y = self._apply_forward(x)
            A_list.append(y.reshape(-1))
        
        self.A = torch.stack(A_list, dim=1)  # [H_obs*W_obs, H*W]
        self.A_T = self.A.T  # [H*W, H_obs*W_obs]
    
    def _apply_blur(self, x):
        """Apply Gaussian blur to image."""
        # x: [B, 1, H, W]
        padding = self.blur_kernel.shape[2] // 2
        x_blurred = F.conv2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            self.blur_kernel,
            padding=0
        )
        return x_blurred
    
    def _apply_downsample(self, x):
        """Apply 2× downsampling (stride=2, take even indices)."""
        # x: [B, 1, H, W]
        # Downsample by taking every 2nd pixel (even indices: 0, 2, 4, ...)
        return x[:, :, ::2, ::2]
    
    def _apply_forward(self, x):
        """Apply forward operator: blur + downsample."""
        x_blurred = self._apply_blur(x)
        x_downsampled = self._apply_downsample(x_blurred)
        return x_downsampled
    
    def forward(self, x_img):
        """
        Forward operator: A(x) = D(S(x)) where S is blur and D is downsampling.
        Input: x_img [B, 1, 16, 16]
        Output: y_img [B, 1, 8, 8]
        """
        y = self._apply_forward(x_img)
        # Add noise (only to observation, not to padding if any)
        noise = self.noise_std * torch.randn_like(y)
        return y + noise
    
    def sample_prior(self, n, exact_proportion=True):
        """
        Sample from prior: mixture of two Gaussians.
        
        Args:
            n: Number of samples
            exact_proportion: If True, ensure exactly 10% have lesion, 90% normal.
                           If False, use random multinomial sampling.
        """
        if exact_proportion:
            # Ensure exactly 10% have lesion, 90% normal
            n_lesion = int(n * self.lesion_prior_weight)  # 10% with lesion
            n_normal = n - n_lesion  # 90% without lesion
            
            # Create component assignments
            comp = torch.zeros(n, dtype=torch.long, device=self.device)
            comp[:n_lesion] = 1  # First n_lesion samples have lesion
            
            # Shuffle to randomize order
            perm = torch.randperm(n, device=self.device)
            comp = comp[perm]
        else:
            # Random multinomial sampling (original behavior)
            comp = torch.multinomial(self.weights, n, replacement=True)  # [n]
        
        # Sample from each component
        samples = []
        for i in range(n):
            if comp[i] == 0:
                # Normal component: N(μ₀, τ²I)
                z = torch.randn(self.H * self.W, device=self.device) * self.tau
                x = (self.mu_0.view(-1) + z).view(1, 1, self.H, self.W)
            else:
                # Lesion component: N(μ₁, τ²I)
                z = torch.randn(self.H * self.W, device=self.device) * self.tau
                x = (self.mu_1.view(-1) + z).view(1, 1, self.H, self.W)
            samples.append(x)
        
        return torch.cat(samples, dim=0)
    
    def generate_sample(self):
        """Generate a sample from prior and its observation."""
        x0 = self.sample_prior(1)
        y = self.forward(x0)
        return x0[0], y[0]
    
    def compute_posterior_variance(self, observation=None):
        """
        Compute closed-form posterior for the linear Gaussian inverse problem.
        
        For y = A @ x + noise, where noise ~ N(0, σ_y² I)
        Prior: p(x) = π₀ N(x | μ₀, τ²I) + π₁ N(x | μ₁, τ²I)
        
        Posterior: p(x|y) = w₀(y) N(x | μ₀^post, Σ_post) + w₁(y) N(x | μ₁^post, Σ_post)
        
        where:
        - Σ_post = (1/τ² I + 1/σ_y² A^T A)^(-1)  (common to both components)
        - μ_k^post = μ_k + τ² A^T (σ_y² I + τ² A A^T)^(-1) (y - A μ_k)
        - w_k(y) ∝ π_k N(y | A μ_k, σ_y² I + τ² A A^T)
        
        Args:
            observation: Optional observation y [B, 1, 8, 8]. If None, returns only Σ_post.
        
        Returns:
            Always includes:
                - posterior_covariance: Common posterior covariance matrix [H*W, H*W]
                - posterior_variance_diag: Diagonal of posterior covariance [H*W]
            
            If observation is provided, also includes:
                - posterior_means: dict with posterior means for each component
                - updated_component_weights: Updated mixture weights w_k(y) [B, 2]
                - total_posterior_variance: Total variance of mixture posterior [B, H*W]
                - total_posterior_mean: Total mean of mixture posterior [B, H*W]
        """
        sigma_y_sq = self.noise_std ** 2
        tau_sq = self.tau ** 2
        
        # Compute posterior covariance (common to both components)
        # Σ_post^(-1) = 1/τ² I + 1/σ_y² A^T A
        A_T_A = self.A_T @ self.A  # [H*W, H*W]
        Sigma_post_inv = (1.0 / tau_sq) * torch.eye(self.H * self.W, device=self.device) + (1.0 / sigma_y_sq) * A_T_A
        Sigma_post = torch.linalg.inv(Sigma_post_inv)
        
        result = {
            'posterior_covariance': Sigma_post,  # [H*W, H*W]
            'posterior_variance_diag': torch.diag(Sigma_post),  # [H*W]
        }
        
        if observation is not None:
            import math
            # observation: [B, 1, 8, 8]
            y_vec = observation.view(observation.shape[0], -1)  # [B, 64]
            B = y_vec.shape[0]
            
            # Compute A @ μ_k for both components
            mu_0_vec = self.mu_0.view(-1)  # [256]
            mu_1_vec = self.mu_1.view(-1)  # [256]
            
            A_mu_0 = (self.A @ mu_0_vec).unsqueeze(0)  # [1, 64]
            A_mu_1 = (self.A @ mu_1_vec).unsqueeze(0)  # [1, 64]
            
            # Compute covariance of y: Σ_y = σ_y² I + τ² A A^T
            A_A_T = self.A @ self.A_T  # [64, 64]
            Sigma_y = sigma_y_sq * torch.eye(self.H_obs * self.W_obs, device=self.device) + tau_sq * A_A_T
            Sigma_y_inv = torch.linalg.inv(Sigma_y)
            log_det_Sigma_y = torch.logdet(Sigma_y)
            
            # Compute posterior means for each component
            # μ_k^post = μ_k + τ² A^T (σ_y² I + τ² A A^T)^(-1) (y - A μ_k)
            # = μ_k + τ² A^T Σ_y^(-1) (y - A μ_k)
            posterior_means = {}
            for k, (mu_k_vec, A_mu_k, pi_k) in enumerate([
                (mu_0_vec, A_mu_0, self.pi_0),
                (mu_1_vec, A_mu_1, self.pi_1)
            ]):
                # y - A μ_k: [B, 64]
                diff_y = y_vec - A_mu_k  # [B, 64]
                
                # τ² A^T Σ_y^(-1) (y - A μ_k): [B, 256]
                correction = tau_sq * (self.A_T @ (Sigma_y_inv @ diff_y.T)).T  # [B, 256]
                
                mu_post = mu_k_vec.unsqueeze(0) + correction  # [B, 256]
                posterior_means[f'component_{k}'] = mu_post
            
            # Compute updated mixture weights w_k(y)
            # w_k(y) ∝ π_k N(y | A μ_k, Σ_y)
            logliks = []
            for k, (A_mu_k, pi_k) in enumerate([
                (A_mu_0, self.pi_0),
                (A_mu_1, self.pi_1)
            ]):
                diff_y = y_vec - A_mu_k  # [B, 64]
                # Quadratic form: diff_y^T @ Sigma_y_inv @ diff_y
                quad = torch.einsum('bi,ij,bj->b', diff_y, Sigma_y_inv, diff_y)  # [B]
                loglik = -0.5 * (quad + log_det_Sigma_y + self.H_obs * self.W_obs * math.log(2 * math.pi))
                logliks.append(loglik)
            
            logliks = torch.stack(logliks, dim=1)  # [B, 2]
            log_weights = torch.log(self.weights).unsqueeze(0) + logliks  # [B, 2]
            log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)  # Normalize
            w_tilde = torch.exp(log_weights)  # [B, 2]
            
            # Compute total mixture posterior mean and variance
            mu_post_0 = posterior_means['component_0']  # [B, 256]
            mu_post_1 = posterior_means['component_1']  # [B, 256]
            mu_post_stack = torch.stack([mu_post_0, mu_post_1], dim=1)  # [B, 2, 256]
            
            # Total mean: m = sum_k w_k * μ_k^post
            m = (w_tilde.unsqueeze(-1) * mu_post_stack).sum(dim=1)  # [B, 256]
            
            # Total variance: Var[x|y] = E[Var[x|y,k]] + Var[E[x|y,k]]
            # = Σ_post + sum_k w_k * (μ_k^post @ μ_k^post^T) - m @ m^T
            mu_post_sq = mu_post_stack ** 2  # [B, 2, 256]
            E_mu_sq = (w_tilde.unsqueeze(-1) * mu_post_sq).sum(dim=1)  # [B, 256]
            m_sq = m ** 2  # [B, 256]
            total_var_diag = result['posterior_variance_diag'].unsqueeze(0) + E_mu_sq - m_sq  # [B, 256]
            
            result['posterior_means'] = posterior_means
            result['original_component_weights'] = self.weights.unsqueeze(0).repeat(B, 1)  # [B, 2]
            result['updated_component_weights'] = w_tilde  # [B, 2]
            result['total_posterior_mean'] = m  # [B, 256]
            result['total_posterior_variance'] = total_var_diag  # [B, 256]
        
        return result
    
    def gradient(self, x_img, y_img, return_loss=False):
        """Compute gradient of data consistency loss."""
        x_vec = x_img.view(x_img.shape[0], -1)  # [B, 256]
        y_vec = y_img.view(y_img.shape[0], -1)  # [B, 64]
        
        # Forward: y_pred = A @ x
        y_pred_vec = (self.A @ x_vec.T).T  # [B, 64]
        
        # Gradient: A^T @ (y_pred - y)
        grad_vec = (self.A_T @ (y_pred_vec - y_vec).T).T  # [B, 256]
        grad_img = grad_vec.view(x_img.shape)
        
        if return_loss:
            loss = ((y_pred_vec - y_vec) ** 2).sum(dim=1)  # [B]
            return grad_img, loss
        else:
            return grad_img
    
    def __call__(self, data, **kwargs):
        return self.forward(data['target'])
    
    def unnormalize(self, x):
        return x
    
    def normalize(self, x):
        return x







