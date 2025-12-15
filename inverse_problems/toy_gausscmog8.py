import torch
import torch.nn.functional as F
from .base import BaseOperator
import math
import torch
import torch

def make_mri_like_A_16(num_samples: int = 8, seed: int = 0):
    """
    Construct an MRI-like 16×16 linear operator A = M @ F
    - F: 16×16 orthogonal matrix (Fourier-like)
    - M: 0/1 diagonal mask with num_samples ones
    Returns:
        A: [16, 16] torch.float32
        mask: [16] 0/1 vector (which "frequencies" are sampled)
    """
    torch.manual_seed(seed)

    N = 16

    F = make_dft_matrix(N)                     # F: [16,16]

    # 2) Construct a 0/1 mask (select num_samples rows)
    mask = torch.zeros(N)
    idx = torch.randperm(N)[:num_samples]
    mask[idx] = 1.0                    # These "frequencies" are observed

    M = torch.diag(mask)               # [16,16]

    # 3) MRI-like A = M @ F
    A = M @ F                          # [16,16]

    return A.to(torch.float32), mask

def make_dft_matrix(N):
    F = torch.zeros(N, N, dtype=torch.float32)
    for k in range(N):
        for n in range(N):
            angle = 2.0 * math.pi * k * n / N
            # Use only cos part to make a real "Fourier-like" matrix
            F[k, n] = math.cos(angle) / math.sqrt(N)
    # F is not strictly orthogonal, but condition number is mild
    return F


class ToyGausscMoG8Problem(BaseOperator):
    """
    Toy 16D problem wrapped as a 1x4x4 pseudo-image.
    True 16D Gaussian mixture prior: first 8D uses MoG, last 8D uses weak Gaussian prior.
    """
    C, H, W = 1, 4, 4
    dim_true = 16
    dim_padded = 16

    def __init__(self, dim, A_type, A_seed, A_scale, noise_std,
                 gauss_rho, mog8_mu, mog8_wm_full, mog8_wp_full,
                 A_obs_dim=None,  # Observation dimension (for underdetermined case, should be < 8)
                 **kwargs):
        super().__init__(**kwargs)
        assert dim == self.dim_padded, f"Dimension must be {self.dim_padded} for 1x4x4 image"

        self.noise_std = float(noise_std)
        # Set sigma_noise for DDNM compatibility (if not already set via kwargs)
        if 'sigma_noise' not in kwargs:
            self.sigma_noise = self.noise_std
        torch.manual_seed(A_seed)

        # Store A_type for use in forward/gradient methods
        self.A_type = A_type

        # A operates only on the first 8 dimensions
        if A_type == 'random-gaussian':
            # If A_obs_dim is specified and < 8, create underdetermined matrix
            if A_obs_dim is not None and A_obs_dim < self.dim_true:
                # Underdetermined: A is A_obs_dim x 8 (fewer observations than unknowns)
                A_obs_x8 = torch.randn(A_obs_dim, self.dim_true, dtype=torch.float32) * A_scale
                self.A_obs_dim = A_obs_dim
                # Pad A to 16x16 with zeros for image compatibility
                # A_16x16: first A_obs_dim rows, first 8 columns contain the actual A
                self.A = torch.zeros(16, 16, dtype=torch.float32)
                self.A[:A_obs_dim, :8] = A_obs_x8
            else:
                # Determined or overdetermined: A is 8x8 (square)
                A_8x8 = torch.randn(self.dim_true, self.dim_true, dtype=torch.float32) * A_scale
                self.A_obs_dim = self.dim_true
                # Pad A to 16x16 with zeros
                self.A = F.pad(A_8x8, (0, 8, 0, 8), "constant", 0)
        elif A_type == 'fixed-full-rank-16x16':
            # Use a fixed 16x16 full-rank integer matrix
            # This is a 16x16 full-rank matrix with integer entries
            A_16x16 = torch.tensor([
                [2, -1,  0,  3,  1, -2,  1,  0, -1,  2,  1,  0,  3, -1,  2,  1],
                [1,  1,  2, -1,  0,  1, -2,  3,  1,  0, -1,  2,  1,  1, -1,  0],
                [0,  2, -1,  1,  3,  0,  1, -2,  2,  1,  0, -1,  1,  2,  1,  3],
                [-1,  0,  1,  2, -1,  3,  0,  1, -2,  1,  2,  1,  0, -1,  3,  2],
                [3,  1, -2,  0,  1,  2, -1,  0,  3, -1,  2,  1, -2,  3,  1,  0],
                [1, -2,  3,  1,  0, -1,  2,  1,  0,  3, -1,  2,  1,  0, -2,  3],
                [0,  1,  0, -2,  3,  1,  1, -1,  2,  1,  0,  3, -1,  2,  1,  0],
                [2,  1, -1,  3,  1,  0, -2,  1,  3,  0,  1, -1,  2,  1,  0,  3],
                [-1,  3,  1,  0, -1,  2,  1,  0,  1, -2,  3,  1,  0, -1,  2,  1],
                [1,  0,  3, -1,  2,  1,  0,  3, -1,  2,  1,  0,  3, -1,  2,  1],
                [0, -1,  2,  1,  0,  3, -1,  2,  1,  0,  3, -1,  2,  1,  0,  3],
                [3,  1,  0, -1,  2,  1,  3,  0, -1,  2,  1,  3,  0, -1,  2,  1],
                [1,  2,  1,  0,  3, -1,  2,  1,  0,  3, -1,  2,  1,  0,  3, -1],
                [-2,  1,  3,  0, -1,  2,  1,  3,  0, -1,  2,  1,  3,  0, -1,  2],
                [1,  0, -1,  2,  1,  3,  0, -1,  2,  1,  3,  0, -1,  2,  1,  3],
                [2,  1,  0, -1,  3,  1,  2,  1,  0, -1,  3,  1,  2,  1,  0, -1]
            ], dtype=torch.float32) * A_scale
            self.A = A_16x16
            # For 16x16 full-rank matrix, A_obs_dim should be 16
            if A_obs_dim is not None:
                self.A_obs_dim = A_obs_dim
            else:
                self.A_obs_dim = 16
        elif A_type == 'mri-like':
            A_16x16, mask = make_mri_like_A_16(num_samples=A_obs_dim, seed=A_seed)
            self.A = A_16x16
            self.A_obs_dim = int(mask.sum().item())  
        elif A_type == 'Identity':
            self.A = torch.eye(16, dtype=torch.float32)
            self.A_obs_dim = 16
        else:
            raise NotImplementedError(f"A_type '{A_type}' is not implemented. Supported types: 'random-gaussian', 'fixed-full-rank-16x16', 'Identity'")

        mu = float(mog8_mu)
        # True 16D MoG prior: first 8D uses MoG, last 8D uses weak Gaussian prior
        self.means = torch.zeros(2, self.dim_true)  # [2, 16]
        # First 8D: MoG components (only last dimension differs)
        self.means[0, 7] = -mu
        self.means[1, 7] = +mu
        # Last 8D: both components have zero mean (weak prior)

        w = torch.tensor([mog8_wm_full, mog8_wp_full])
        self.weights = w / w.sum()

        rho = float(gauss_rho)
        # Build 16D prior covariance: block-diagonal structure
        # First 8D: Toeplitz covariance (original MoG structure)
        idx_8 = torch.arange(8)
        absdiff_8 = (idx_8[:, None] - idx_8[None, :]).abs()
        Sigma0_8x8 = (rho ** absdiff_8)  # [8, 8]
        
        # Last 8D: weak Gaussian prior with large variance (e.g., 5.0)
        weak_variance = 5.0
        Sigma0_16x16 = torch.zeros(16, 16, dtype=torch.float32)
        Sigma0_16x16[:8, :8] = Sigma0_8x8  # First 8x8 block: original MoG covariance
        Sigma0_16x16[8:, 8:] = torch.eye(8) * weak_variance  # Last 8x8 block: weak prior
        
        self.L = torch.linalg.cholesky(Sigma0_16x16)
        self.Sigma_prior = Sigma0_16x16  # Store 16x16 prior covariance for posterior computation
        
        # Compute SVD for DDNM compatibility
        self._compute_svd()

    def _compute_svd(self):
        """Compute SVD decomposition of A for DDNM compatibility."""
        # A is 16x16, but forward() only uses A[:A_obs_dim, :8]
        # For DDNM, we need SVD that matches the actual forward operation
        # Since forward pads the result to 16D, we compute SVD of the full 16x16 A
        # but ensure it's consistent with forward()
        A_full = self.A.to(self.device)  # [16, 16]
        U, S, Vt = torch.linalg.svd(A_full, full_matrices=True)
        # U: [16, 16]
        # S: [16] (singular values, many will be zero or near-zero)
        # Vt: [16, 16]
        
        # Store SVD components
        self.U = U.to(self.device)  # [16, 16]
        self.S_vec = S.to(self.device)  # [16]
        self._Vt_matrix = Vt.to(self.device)  # [16, 16] - stored as _Vt_matrix to avoid conflict with Vt() method
        
        # M is a mask indicating which singular values are non-zero (observed dimensions)
        # For DDNM, M should be 1 for non-zero singular values, 0 for zero singular values
        # M needs to be in image format [1, 1, 4, 4]
        M_vec = (S.abs() > 1e-10).float()  # [16], 1 for non-zero, 0 for zero
        self._M = self._vec_to_img(M_vec.unsqueeze(0))  # [1, 1, 4, 4]
        
        # S also needs to be in image format [1, 1, 4, 4]
        # Use small value for zero singular values to avoid division by zero
        S_vec_safe = S.clone()
        # For zero or very small singular values, set to a safe minimum
        # This will be masked by M anyway, so it's safe
        S_vec_safe = torch.clamp(S_vec_safe, min=1e-8)  # Use a larger minimum for better stability
        self._S = self._vec_to_img(S_vec_safe.unsqueeze(0))  # [1, 1, 4, 4]

    def _img_to_vec(self, x_img):
        # [B, 1, 4, 4] -> [B, 16]
        return x_img.reshape(x_img.shape[0], -1)

    def _vec_to_img(self, v):
        # [B, 16] -> [B, 1, 4, 4]
        return v.reshape(v.shape[0], self.C, self.H, self.W)

    def sample_prior(self, n):
        # True 16D prior: sample directly from 16D MoG
        comp = torch.multinomial(self.weights, n, True)
        means = self.means[comp].to(self.device)  # [n, 16]
        z = torch.randn(n, self.dim_true, device=self.device) @ self.L.to(self.device).T  # [n, 16]
        v_16d = means + z  # [n, 16]
        return self._vec_to_img(v_16d)  # [n, 1, 4, 4]

    def generate_sample(self):
        x0_img = self.sample_prior(1)
        y_img = self.forward(x0_img)
        return x0_img[0], y_img[0]

    def forward(self, x_img, **kwargs):
        # Accept **kwargs for compatibility with algorithms that pass unnormalize=False
        x_vec = self._img_to_vec(x_img)  # [B, 16]
        
        # True 16D forward: y = A @ x + noise, where A is 16x16 (or A_obs_dim x 16)
        A_full = self.A[:self.A_obs_dim, :].to(x_vec.device)  # [A_obs_dim, 16]
        y_vec_obs = (x_vec @ A_full.T)  # [B, A_obs_dim]
        
        # Add noise to observations
        noise = self.noise_std * torch.randn_like(y_vec_obs)  # [B, A_obs_dim]
        y_vec_obs = y_vec_obs + noise
        
        # Pad y_vec_obs to 16D for image compatibility (only for image format, not used in computation)
        y_vec = F.pad(y_vec_obs, (0, 16 - self.A_obs_dim), "constant", 0)  # [B, 16]
        y_img = self._vec_to_img(y_vec)
        return y_img

    def gradient(self, x_img, y_img, return_loss=False):
        # True 16D gradient: ∇_x log p(y|x) = -A^T @ (Ax - y) / sigma^2
        x_vec = self._img_to_vec(x_img)  # [B, 16]
        y_vec = self._img_to_vec(y_img)  # [B, 16]
        y_obs = y_vec[:, :self.A_obs_dim]  # [B, A_obs_dim]
        
        # Use full 16x16 matrix A with full 16D x
        A_full = self.A[:self.A_obs_dim, :].to(x_vec.device)  # [A_obs_dim, 16]
        pred_vec_obs = x_vec @ A_full.T  # [B, A_obs_dim]
        diff_obs = pred_vec_obs - y_obs  # [B, A_obs_dim]
        
        # Return negative gradient: -A^T @ (Ax - y) = A^T @ (y - Ax)
        # This points in the direction of decreasing loss (for DPS algorithm)
        grad_vec = -diff_obs @ A_full  # [B, 16]
        
        grad_img = self._vec_to_img(grad_vec)

        if return_loss:
            loss = (diff_obs ** 2).sum(dim=1)  # Only compute loss on observation dimensions
            return grad_img, loss
        else:
            return grad_img

    def __call__(self, data, **kwargs):
        return self.forward(data['target'])

    def unnormalize(self, x):
        return x
    
    # SVD methods for DDNM compatibility
    def V(self, x):
        """
        Apply right singular vectors V to x.
        Input: x in SVD space [B, 1, 4, 4] or [B, particles, 1, 4, 4] or [B, 1, 1, 4, 4] or [B*particles, 1, 1, 4, 4] (image format)
        Output: x in image space [B, 1, 4, 4] (always 4D for network compatibility)
        For DDNM/DDRM: V transforms from SVD space back to image space
        V = Vt^T, so V @ x = Vt^T @ x
        """
        original_shape = x.shape
        # Handle 5D input
        if x.dim() == 5:
            # Case 1: [B, 1, 1, 4, 4] from DDRM (M shape is [1, 1, 4, 4])
            # Case 2: [B, particles, 1, 4, 4] from MCGdiff (before flatten)
            # Case 3: [B*particles, 1, 1, 4, 4] from MCGdiff (after flatten)
            # Check total elements to determine correct reshape
            total_elements = x.numel()
            B = x.shape[0]
            expected_elements = B * 16  # Each sample should have 16 elements (1*4*4)
            
            if total_elements == expected_elements:
                # Reshape directly to [B, 1, 4, 4]
                x = x.reshape(B, 1, 4, 4)
            elif x.shape[1] == 1 and x.shape[2] == 1:
                # DDRM or MCGdiff after flatten: compress [B, 1, 1, 4, 4] -> [B, 1, 4, 4]
                x = x.reshape(B, 1, x.shape[3], x.shape[4])  # [B, 1, 4, 4]
            elif len(x.shape) == 5 and x.shape[2] == 1:
                # MCGdiff before flatten: [B, particles, 1, 4, 4] -> [B*particles, 1, 4, 4]
                B, particles = x.shape[0], x.shape[1]
                # Preserve channel dimension: [B, particles, 1, 4, 4] -> [B*particles, 1, 4, 4]
                x = x.reshape(B * particles, x.shape[2], x.shape[3], x.shape[4])  # [B*particles, 1, 4, 4]
            else:
                # Generic case: try to flatten first two dimensions and preserve last 3
                B = x.shape[0]
                x = x.reshape(B, *x.shape[2:])  # Remove dimension 1, keep rest
                # If still 5D, try another approach
                if x.dim() == 5:
                    # Try to reshape assuming [B, C, H, W, D] -> [B, C*H, W, D] or similar
                    # But we expect [B, 1, 4, 4], so try to get there
                    x = x.reshape(B, -1, x.shape[-2], x.shape[-1])  # [B, ?, 4, 4]
                    # If middle dimension is not 1, we have a problem
                    if x.shape[1] != 1:
                        # Try to fix by checking if we can reshape to [B, 1, 4, 4]
                        if x.numel() == B * 16:
                            x = x.reshape(B, 1, 4, 4)
                        else:
                            raise ValueError(f"Cannot reshape 5D input from {original_shape} to [B, 1, 4, 4]. Total elements: {x.numel()}, expected: {B*16}")
        
        if x.dim() == 4:
            # x is in SVD space [B, 1, 4, 4], apply V transformation
            # Ensure x has correct shape [B, 1, 4, 4]
            # If shape is wrong, try to fix it by checking total elements
            total_elements = x.numel()
            B = x.shape[0]
            if x.shape[1] != 1 or x.shape[2] != 4 or x.shape[3] != 4:
                if total_elements == B * 16:
                    # Reshape to [B, 1, 4, 4]
                    x = x.reshape(B, 1, 4, 4)
                elif total_elements % (B * 16) == 0:
                    # If total elements is a multiple of B*16, we might have extra dimensions
                    # Try to reshape by flattening middle dimensions
                    # e.g., [B, 64, 4, 4] -> [B, 1, 4, 4] if 64*4*4 = 16 (but that's not right)
                    # Actually, if we have [B, C, H, W] and total_elements = B*16, we need C*H*W = 16
                    # So if we have [B, 64, 4, 4], that's B*64*4*4 = B*1024, not B*16
                    # Let's try a different approach: if we have [B, C, H, W] and C*H*W > 16,
                    # we might need to take only the first 16 elements per sample
                    # But that doesn't make sense either...
                    # Actually, let's just try to reshape assuming the last two dimensions are 4x4
                    # and the middle dimension should be 1
                    if x.shape[2] == 4 and x.shape[3] == 4:
                        # We have [B, C, 4, 4] where C != 1
                        # If C*4*4 = 16, then C = 1, which contradicts C != 1
                        # So this case shouldn't happen if the input is correct
                        # But if it does, let's try to take only the first channel
                        if x.shape[1] > 1:
                            x = x[:, 0:1, :, :]  # Take first channel: [B, 1, 4, 4]
                        else:
                            raise ValueError(f"Cannot reshape x from {x.shape} to [B, 1, 4, 4]. Total elements: {total_elements}, expected: {B*16}")
                    else:
                        raise ValueError(f"Cannot reshape x from {x.shape} to [B, 1, 4, 4]. Total elements: {total_elements}, expected: {B*16}")
                else:
                    raise ValueError(f"Cannot reshape x from {x.shape} to [B, 1, 4, 4]. Total elements: {total_elements}, expected: {B*16}")
            x_vec = self._img_to_vec(x)  # [B, 16]
            # V transforms from SVD space to image space: x_img = V @ z_svd
            # Since _Vt_matrix is V^T (from SVD: A = U @ diag(S) @ V^T)
            # We have: V = _Vt_matrix.T
            # For row vectors [B, 16], V @ z = z @ V^T = z @ _Vt_matrix
            # So we compute: x_vec @ self._Vt_matrix
            x_img = x_vec @ self._Vt_matrix  # [B, 16] @ [16, 16] = [B, 16]
            result = self._vec_to_img(x_img)  # [B, 1, 4, 4]
            
            # Always return 4D for network compatibility
            # Algorithms that need 5D should handle reshaping themselves
            return result
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected [B, 1, 4, 4] or [B, particles, 1, 4, 4] or [B, 1, 1, 4, 4]")
    
    def Ut(self, y):
        """
        Apply U^T to observation y.
        Input: y in image space [B, 1, 4, 4] or [B, particles, 1, 4, 4]
        Output: y in SVD space [B, 1, 4, 4] or [B, particles, 1, 4, 4] (image format)
        For DDNM: Ut transforms observation to SVD space
        Ut(y) = U^T @ y
        """
        original_shape = y.shape
        # Handle 5D input (e.g., from MCGdiff with particles: [B, particles, 1, 4, 4])
        if y.dim() == 5:
            # Reshape to [B*particles, 1, 4, 4] for processing
            B, particles = y.shape[0], y.shape[1]
            y = y.view(B * particles, *y.shape[2:])  # [B*particles, 1, 4, 4]
        
        if y.dim() == 4:
            y_vec = self._img_to_vec(y)  # [B, 16]
            # U^T @ y_vec = (y_vec^T @ U)^T, but we compute: y_vec @ U^T
            # Actually, U is stored, so U^T = U.T
            y_svd = y_vec @ self.U.T  # [B, 16] @ [16, 16] = [B, 16]
            result = self._vec_to_img(y_svd)  # [B, 1, 4, 4]
            
            # Reshape back to original shape if it was 5D
            if len(original_shape) == 5:
                B, particles = original_shape[0], original_shape[1]
                result = result.view(B, particles, *result.shape[1:])  # [B, particles, 1, 4, 4]
            
            return result
        else:
            raise ValueError(f"Unexpected input shape: {y.shape}, expected [B, 1, 4, 4] or [B, particles, 1, 4, 4]")
    
    def Vt(self, x):
        """
        Apply V^T to x (transpose of V).
        Input: x in image space [B, 1, 4, 4] (from network, always 4D)
        Output: x in SVD space [B, 1, 4, 4] (image format)
        For DDRM: Vt transforms from image space to SVD space
        Vt = V^T, so Vt @ x = V^T @ x = (Vt stored as matrix) @ x
        Note: Always returns 4D. Algorithms needing 5D should handle reshaping.
        """
        if x.dim() == 4:
            # x is in image space [B, 1, 4, 4], apply Vt transformation
            x_vec = self._img_to_vec(x)  # [B, 16]
            # Vt transforms from image space to SVD space: z_svd = V^T @ x_img
            # Since _Vt_matrix is V^T, and x_vec is [B, 16] (row vectors)
            # We have: V^T @ x = x @ V (for row vectors)
            # Since V = _Vt_matrix.T, we compute: x_vec @ _Vt_matrix.T
            x_svd = x_vec @ self._Vt_matrix.T  # [B, 16] @ [16, 16] = [B, 16]
            result = self._vec_to_img(x_svd)  # [B, 1, 4, 4]
            return result
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected [B, 1, 4, 4]")
    
    @property
    def M(self):
        """
        Mask matrix indicating observed dimensions.
        Returns: [1, 1, 4, 4] tensor (image format) for broadcasting with images
        """
        return self._M
    
    @property
    def S(self):
        """
        Singular values in image format.
        Returns: [1, 1, 4, 4] tensor (image format)
        """
        return self._S
    
    def compute_posterior_variance(self, observation=None):
        """
        Compute the true posterior variance for the linear Gaussian inverse problem.
        
        For a linear model: y = A @ x + noise, where noise ~ N(0, sigma_noise^2 * I)
        Prior is Mixture of Gaussians: p(x) = sum_k w_k N(x | mu_k, Sigma_prior)
        
        Posterior is also a mixture: p(x|y) = sum_k w_tilde_k(y) N(x | mu_post,k, Sigma_post)
        where:
        - Sigma_post is COMMON to all components (since prior covariance is the same)
        - w_tilde_k(y) are updated weights based on likelihood
        - mu_post,k are component-specific posterior means
        
        This function returns:
        1. Common posterior covariance Sigma_post (same for all components)
        2. Component-specific posterior means (if observation provided)
        3. Updated mixture weights w_tilde_k(y) (if observation provided)
        4. Total mixture posterior variance Var[x|y] (if observation provided)
        
        Args:
            observation: Optional observation y. If None, returns only the common posterior covariance.
                         If provided, also computes posterior means, updated weights, and total variance.
        
        Returns:
            Always includes:
                - posterior_covariance: Common posterior covariance matrix (8x8) for all components
                - posterior_variance_diag: Diagonal of posterior covariance (8D vector)
            
            If observation is provided, also includes:
                - posterior_means: dict with posterior means for each component
                - updated_component_weights: Updated mixture weights w_tilde_k(y) [B, K]
                - total_posterior_variance: Total variance of the mixture posterior Var[x|y] [B, 8]
                - total_posterior_mean: Total mean of the mixture posterior E[x|y] [B, 8]
        """
        # Get effective A matrix
        if self.A_type == 'fixed-full-rank-16x16' or self.A_type == 'mri-like' or self.A_type == 'Identity':
            # Use full 16x16 matrix A
            A_effective = self.A[:self.A_obs_dim, :].to(self.device)  # [A_obs_dim, 16]
            # For 16D case, we need to extend the prior covariance to 16D
            # Since the prior is only defined on first 8 dimensions, we pad it
            # self.Sigma_prior is [16, 16], extract the first 8x8 block
            Sigma_prior_8x8 = self.Sigma_prior[:8, :8].to(self.device)  # [8, 8]
            # Pad to 16x16: first 8x8 block is the actual prior, rest is identity (or large variance)
            Sigma_prior = torch.zeros(16, 16, device=self.device)
            Sigma_prior[:8, :8] = Sigma_prior_8x8
            # For dimensions 8-15, use a large variance (or identity) since they have no prior
            # Using identity with large scale for unconstrained dimensions
            Sigma_prior[8:, 8:] = torch.eye(8, device=self.device) * 1e6  # Large variance for unconstrained dims
        else:
            # For random-gaussian, only use the first 8 dimensions
            A_effective = self.A[:self.A_obs_dim, :8].to(self.device)  # [A_obs_dim, 8]
            Sigma_prior = self.Sigma_prior.to(self.device)  # [8, 8]
        
        sigma_noise_sq = self.noise_std ** 2
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        
        # Compute posterior covariance for each component
        # Sigma_posterior^{-1} = Sigma_prior^{-1} + (1/sigma_noise^2) * A^T @ A
        A_T = A_effective.T  # [16, A_obs_dim] or [8, A_obs_dim]
        Sigma_posterior_inv = Sigma_prior_inv + (1.0 / sigma_noise_sq) * (A_T @ A_effective)
        Sigma_posterior = torch.linalg.inv(Sigma_posterior_inv)
        
        # Extract relevant dimensions for result
        if self.A_type == 'fixed-full-rank-16x16' or self.A_type == 'mri-like' or self.A_type == 'Identity':
            result = {
                'posterior_covariance': Sigma_posterior,  # 16x16
                'posterior_variance_diag': torch.diag(Sigma_posterior),  # 16D vector of variances
            }
        else:
            result = {
                'posterior_covariance': Sigma_posterior,  # 8x8
                'posterior_variance_diag': torch.diag(Sigma_posterior),  # 8D vector of variances
            }
        
        # If observation is provided, also compute posterior means, updated weights, and total variance
        if observation is not None:
            import math
            y_vec = self._img_to_vec(observation)  # [B, 16]
            y_obs = y_vec[:, :self.A_obs_dim]  # [B, A_obs_dim]
            B = y_obs.shape[0]
            K = len(self.means)
            
            # 1. Compute posterior means for each component
            posterior_means = {}
            for i, mu_prior_full in enumerate(self.means):
                mu_prior_full = mu_prior_full.to(self.device)  # [16] - self.means is [2, 16]
                
                if self.A_type == 'fixed-full-rank-16x16' or self.A_type == 'mri-like' or self.A_type == 'Identity':
                    # mu_prior_full is already [16], use it directly
                    mu_prior = mu_prior_full  # [16]
                else:
                    # For random-gaussian, Sigma_prior is [8, 8], so use only first 8 dimensions
                    mu_prior = mu_prior_full[:8]  # [8]
                
                # Posterior mean: mu_posterior = Sigma_posterior @ (Sigma_prior^{-1} @ mu_prior + A^T @ (1/sigma_noise^2) @ y)
                term1 = Sigma_prior_inv @ mu_prior  # [16] or [8]
                term2 = (1.0 / sigma_noise_sq) * (A_T @ y_obs.T).T  # [B, 16] or [B, 8]
                term_sum = term1.unsqueeze(0) + term2  # [B, 16] or [B, 8]
                mu_posterior = (Sigma_posterior @ term_sum.T).T  # [B, 16] or [B, 8]
                
                posterior_means[f'component_{i}'] = mu_posterior
            
            # 2. Compute updated mixture weights w_tilde_k(y)
            # w_tilde_k(y) ∝ w_k * N(y | A*mu_k, A*Sigma_prior*A^T + sigma^2*I)
            Sigma_y = A_effective @ Sigma_prior @ A_effective.T + sigma_noise_sq * torch.eye(self.A_obs_dim, device=self.device)  # [m, m]
            Sigma_y_inv = torch.linalg.inv(Sigma_y)
            log_det_Sigma_y = torch.logdet(Sigma_y)
            
            logliks = []
            for i, mu_prior_full in enumerate(self.means):
                mu_prior_full = mu_prior_full.to(self.device)  # [16] - self.means is [2, 16]
                
                if self.A_type == 'fixed-full-rank-16x16' or self.A_type == 'mri-like' or self.A_type == 'Identity':
                    # mu_prior_full is already [16], use it directly
                    mu_prior = mu_prior_full  # [16]
                else:
                    # For random-gaussian, A_effective is [A_obs_dim, 8], so use only first 8 dimensions
                    mu_prior = mu_prior_full[:8]  # [8]
                
                mu_y = A_effective @ mu_prior  # [m]
                diff_y = y_obs - mu_y.unsqueeze(0)  # [B, m]
                # Quadratic form: diff_y^T @ Sigma_y_inv @ diff_y
                quad = torch.einsum('bm,mn,bn->b', diff_y, Sigma_y_inv, diff_y)  # [B]
                loglik = -0.5 * (quad + log_det_Sigma_y + self.A_obs_dim * math.log(2 * math.pi))
                logliks.append(loglik)
            
            logliks = torch.stack(logliks, dim=1)  # [B, K]
            log_weights = torch.log(self.weights.to(self.device)).unsqueeze(0) + logliks  # [B, K]
            log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)  # Normalize
            w_tilde = torch.exp(log_weights)  # [B, K]
            
            # 3. Compute total mixture posterior mean: m = sum_k w_tilde_k * mu_post,k
            mu_post_stack = torch.stack([posterior_means[f'component_{k}'] for k in range(K)], dim=1)  # [B, K, 16] or [B, K, 8]
            m = (w_tilde.unsqueeze(-1) * mu_post_stack).sum(dim=1)  # [B, 16] or [B, 8]
            
            # 4. Compute total mixture posterior variance: Var[x|y] = E[Var[x|y,k]] + Var[E[x|y,k]]
            # = sum_k w_tilde_k * (Sigma_post + mu_post,k @ mu_post,k^T) - m @ m^T
            # For diagonal variance only:
            # Var[x_i|y] = sum_k w_tilde_k * (Sigma_post[i,i] + mu_post,k[i]^2) - m[i]^2
            # = Sigma_post[i,i] + sum_k w_tilde_k * mu_post,k[i]^2 - m[i]^2
            mu_post_sq = mu_post_stack ** 2  # [B, K, 16] or [B, K, 8]
            E_mu_sq = (w_tilde.unsqueeze(-1) * mu_post_sq).sum(dim=1)  # [B, 16] or [B, 8]
            m_sq = m ** 2  # [B, 16] or [B, 8]
            total_var_diag = result['posterior_variance_diag'].unsqueeze(0) + E_mu_sq - m_sq  # [B, 16] or [B, 8]
            
            result['posterior_means'] = posterior_means
            result['original_component_weights'] = self.weights.to(self.device)
            result['updated_component_weights'] = w_tilde  # [B, K]
            result['total_posterior_mean'] = m  # [B, 8]
            result['total_posterior_variance'] = total_var_diag  # [B, 8]
        
        return result
