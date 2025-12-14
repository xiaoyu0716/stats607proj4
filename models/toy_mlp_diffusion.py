import torch
import torch.nn as nn

class ToyDiffusionMLP(nn.Module):
    """
    Tiny diffusion model for 8D toy vector.
    x: [batch,8]
    sigma: scalar noise level
    Returns: eps_hat same shape
    """
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.dim = dim
        # Add attributes expected by the framework for compatibility
        # Calculate img_resolution from dim (assuming square images)
        # For dim=16: 4×4 image, for dim=256: 16×16 image
        self.img_channels = 1
        import math
        self.img_resolution = int(math.sqrt(dim))
        
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, sigma):
        """
        sigma is scalar or [batch] tensor.
        We concatenate x with sigma as a feature.
        """
        # Ensure x is a 2D tensor
        if x.dim() == 4:
            x_flat = x.view(x.shape[0], -1)  # Use x.shape for robust reshaping
        else:
            x_flat = x

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.view(1).repeat(x.shape[0])
        
        sigma = sigma.view(x_flat.shape[0], 1).to(x_flat.dtype)
        inp = torch.cat([x_flat, sigma], dim=1)
        out_flat = self.net(inp)
        # Reshape output back to the original image shape
        out = out_flat.view(x.shape)
        
        # The output of DPS should be a vector, no need to reshape back to 4D
        return out
