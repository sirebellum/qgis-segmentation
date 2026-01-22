# SPDX-License-Identifier: BSD-3-Clause
"""Training-only reconstruction losses (Option C + D).

This module provides:
- TinyReconDecoder: lightweight decoder for stride-4 RGB reconstruction
- Blur + gradient loss targets computed from augmented input RGB
- EMA normalization for auxiliary loss weighting

All components are training-only and excluded from deployment artifacts.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Fixed Sobel kernels (no learnable params, deterministic)
# ---------------------------------------------------------------------------

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3) / 4.0

_LUMA_WEIGHTS = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)


def _to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [B,3,H,W] to luminance [B,1,H,W]."""
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB tensor [B,3,H,W]")
    weights = _LUMA_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def sobel_gradients(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute gradient magnitude from luminance using fixed Sobel kernels.
    
    Args:
        img: input tensor [B,C,H,W] or [B,1,H,W] (will convert to luma if C>1)
        eps: small value for numerical stability
        
    Returns:
        Gradient magnitude [B,1,H,W]
    """
    if img.dim() != 4:
        raise ValueError("Expected 4D tensor [B,C,H,W]")
    
    if img.shape[1] > 1:
        luma = _to_luminance(img)
    else:
        luma = img
    
    sobel_x = _SOBEL_X.to(device=luma.device, dtype=luma.dtype)
    sobel_y = _SOBEL_Y.to(device=luma.device, dtype=luma.dtype)
    
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    
    grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
    return grad_mag


def gaussian_blur_2d(img: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    """Apply Gaussian blur to each channel independently.
    
    Args:
        img: input tensor [B,C,H,W]
        sigma: Gaussian standard deviation
        kernel_size: kernel size (must be odd)
        
    Returns:
        Blurred tensor [B,C,H,W]
    """
    if img.dim() != 4:
        raise ValueError("Expected 4D tensor [B,C,H,W]")
    if kernel_size % 2 == 0:
        kernel_size += 1
    if sigma <= 0:
        return img
    
    # Build 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (coords / sigma).pow(2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Separable 2D Gaussian: [1, 1, 1, K] and [1, 1, K, 1]
    gauss_h = gauss_1d.view(1, 1, 1, kernel_size)
    gauss_v = gauss_1d.view(1, 1, kernel_size, 1)
    
    c = img.shape[1]
    pad_h = kernel_size // 2
    
    # Apply horizontal then vertical
    out = F.pad(img, (pad_h, pad_h, 0, 0), mode="reflect")
    out = out.view(-1, 1, out.shape[-2], out.shape[-1])
    out = F.conv2d(out, gauss_h.to(img.dtype))
    out = out.view(-1, c, out.shape[-2], out.shape[-1])
    
    out = F.pad(out, (0, 0, pad_h, pad_h), mode="reflect")
    out = out.view(-1, 1, out.shape[-2], out.shape[-1])
    out = F.conv2d(out, gauss_v.to(img.dtype))
    out = out.view(-1, c, out.shape[-2], out.shape[-1])
    
    return out


def downsample_area(img: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """Downsample using area averaging (stride factor).
    
    Args:
        img: input tensor [B,C,H,W]
        factor: downsample factor
        
    Returns:
        Downsampled tensor [B,C,H//factor,W//factor]
    """
    if factor <= 1:
        return img
    h, w = img.shape[-2:]
    target_h = max(1, h // factor)
    target_w = max(1, w // factor)
    return F.interpolate(img, size=(target_h, target_w), mode="area")


# ---------------------------------------------------------------------------
# TinyReconDecoder (training-only)
# ---------------------------------------------------------------------------

class TinyReconDecoder(nn.Module):
    """Lightweight decoder producing stride-4 RGB from stride-4 features.
    
    This module is training-only and should NOT be included in deployment
    artifacts. It takes pre-projection feature maps from the student and
    produces RGB predictions for reconstruction loss computation.
    
    Architecture:
        - 2 blocks of: 3x3 conv (stride=1) + GroupNorm + ReLU
        - 1x1 conv to 3 channels
        - Sigmoid to bound output in [0,1]
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 2,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._training_only = True  # marker for artifact separation
        
        layers = []
        current_ch = in_channels
        for _ in range(max(1, num_blocks)):
            layers.append(nn.Conv2d(current_ch, hidden_channels, kernel_size=3, stride=1, padding=1))
            groups_eff = max(1, min(groups, hidden_channels))
            while hidden_channels % groups_eff != 0 and groups_eff > 1:
                groups_eff -= 1
            layers.append(nn.GroupNorm(groups_eff, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            current_ch = hidden_channels
        
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to RGB.
        
        Args:
            features: [B,C,H,W] feature map at stride-4
            
        Returns:
            RGB prediction [B,3,H,W] in [0,1] range
        """
        if features.dim() != 4:
            raise ValueError("Expected 4D feature tensor")
        x = self.blocks(features)
        x = self.head(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Reconstruction targets + loss computation
# ---------------------------------------------------------------------------

def build_recon_targets(
    rgb_aug: torch.Tensor,
    stride: int = 4,
    blur_sigma: float = 1.0,
    blur_kernel: int = 5,
) -> Dict[str, torch.Tensor]:
    """Build reconstruction targets from augmented RGB.
    
    Args:
        rgb_aug: augmented RGB input [B,3,H,W] normalized to [0,1]
        stride: downsample factor (matches encoder stride)
        blur_sigma: Gaussian blur sigma for low-pass target
        blur_kernel: Gaussian kernel size
        
    Returns:
        dict with keys:
            - rgb_ds: downsampled RGB [B,3,H//stride,W//stride]
            - rgb_blur: blurred downsampled RGB [B,3,H//stride,W//stride]
            - grad_target: gradient magnitude of blurred RGB [B,1,H//stride,W//stride]
    """
    rgb_ds = downsample_area(rgb_aug, factor=stride)
    rgb_blur = gaussian_blur_2d(rgb_ds, sigma=blur_sigma, kernel_size=blur_kernel)
    grad_target = sobel_gradients(rgb_blur)
    
    return {
        "rgb_ds": rgb_ds,
        "rgb_blur": rgb_blur,
        "grad_target": grad_target,
    }


def reconstruction_loss(
    pred_rgb: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    grad_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses (Option C + D).
    
    Args:
        pred_rgb: predicted RGB [B,3,H,W] from decoder, expected in [0,1]
        targets: dict from build_recon_targets()
        grad_weight: relative weight for gradient loss vs blur loss
        
    Returns:
        dict with keys:
            - recon_blur: L1 loss on blurred RGB
            - recon_grad: L1 loss on gradient magnitude
            - recon_raw: combined raw loss (blur + grad_weight * grad)
    """
    rgb_blur = targets["rgb_blur"]
    grad_target = targets["grad_target"]
    
    # Ensure shapes match
    if pred_rgb.shape[-2:] != rgb_blur.shape[-2:]:
        pred_rgb = F.interpolate(pred_rgb, size=rgb_blur.shape[-2:], mode="bilinear", align_corners=False)
    
    # Option C: low-pass reconstruction
    loss_blur = F.l1_loss(pred_rgb, rgb_blur)
    
    # Option D: gradient/edge consistency
    pred_grad = sobel_gradients(pred_rgb)
    loss_grad = F.l1_loss(pred_grad, grad_target)
    
    # Combined
    loss_raw = loss_blur + grad_weight * loss_grad
    
    return {
        "recon_blur": loss_blur,
        "recon_grad": loss_grad,
        "recon_raw": loss_raw,
    }


# ---------------------------------------------------------------------------
# EMA-normalized auxiliary weighting
# ---------------------------------------------------------------------------

def update_recon_ema(
    ema_state: Dict[str, torch.Tensor],
    loss_raw: torch.Tensor,
    key: str = "recon_ema",
    decay: float = 0.99,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Update EMA and compute normalized loss.
    
    Args:
        ema_state: dict storing EMA values (modified in place)
        loss_raw: raw reconstruction loss (scalar tensor)
        key: key name for this EMA entry
        decay: EMA decay factor
        eps: small value for numerical stability
        
    Returns:
        (normalized_loss, updated_ema_state)
    """
    detached = loss_raw.detach()
    
    if key not in ema_state:
        ema_state[key] = detached.clone()
    else:
        ema_state[key] = decay * ema_state[key] + (1.0 - decay) * detached
    
    ema_val = ema_state[key].clamp(min=eps)
    normalized = loss_raw / ema_val
    
    return normalized, ema_state


def apply_recon_loss(
    normalized_loss: torch.Tensor,
    lambda_recon: float = 0.01,
) -> torch.Tensor:
    """Apply small coefficient to normalized reconstruction loss.
    
    Args:
        normalized_loss: EMA-normalized reconstruction loss
        lambda_recon: coefficient (kept small to avoid dominating other losses)
        
    Returns:
        Weighted contribution to add to total loss
    """
    return lambda_recon * normalized_loss
