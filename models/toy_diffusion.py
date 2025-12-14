import torch
import torch.nn as nn

class ToyDiffusionModel(nn.Module):
    """
    Tiny diffusion model for 8D toy vector.
    x: [batch,8]
    sigma: scalar noise level
    Returns: eps_pred same shape
    """
    def __init__(self, dim=8, hidden=64):
        super().__init__()
        # Add attributes expected by the framework
        self.img_channels = 1 # C
        self.img_resolution = (2, 4) # H, W
        
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
            
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], device=x.device, dtype=x.dtype)
        if sigma.ndim == 0:
            sigma = sigma.view(1).repeat(x.shape[0])
        
        # Flatten the image input
        x_flat = x.view(x.shape[0], -1) # [B, C*H*W]

        sigma = sigma.view(x.shape[0], 1).to(x.dtype)
        inp = torch.cat([x_flat, sigma], dim=1)
        out_flat = self.net(inp)
        
        # Reshape output back to image shape
        out = out_flat.view_as(x)
        return out
