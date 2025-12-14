"""
New implementation of ToyImageLesionProblem with organ template and smooth texture.
This is a temporary file to test the new implementation before replacing the old one.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

def _gaussian_kernel1d(sigma, ksize=None, device='cpu', dtype=torch.float32):
    if ksize is None:
        ksize = int(2*math.ceil(3*sigma)+1)
    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize-1)/2
    g = torch.exp(-0.5*(x/sigma)**2)
    g = g / g.sum()
    return g.view(1,1,-1), ksize

def _gaussian_blur(img, sigma):
    # img: (B,1,H,W)
    if sigma <= 0:
        return img
    g1d, k = _gaussian_kernel1d(sigma, device=img.device, dtype=img.dtype)
    pad = k//2
    # separable conv
    img = F.pad(img, (0,0,pad,pad), mode='reflect')
    img = F.conv2d(img, g1d.unsqueeze(3), groups=1)
    img = F.pad(img, (pad,pad,0,0), mode='reflect')
    img = F.conv2d(img, g1d.unsqueeze(2), groups=1)
    return img







