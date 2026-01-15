# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Simple gradient utilities used for edge-aware losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def image_gradients(x: torch.Tensor) -> torch.Tensor:
    """Compute per-channel gradient magnitude using Sobel kernels.

    Args:
        x: Tensor of shape [B, C, H, W].
    Returns:
        Tensor [B, C, H, W] with gradient magnitude per channel.
    """
    if x.dim() != 4:
        raise ValueError("image_gradients expects 4D tensor")
    device = x.device
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    padding = 1
    grad_x = F.conv2d(x, kx.repeat(x.shape[1], 1, 1, 1), padding=padding, groups=x.shape[1])
    grad_y = F.conv2d(x, ky.repeat(x.shape[1], 1, 1, 1), padding=padding, groups=x.shape[1])
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
