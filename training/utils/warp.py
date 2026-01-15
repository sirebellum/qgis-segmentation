# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Warping helpers using torch grid_sample."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_warp(tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Warp tensor using a precomputed sampling grid.

    Args:
        tensor: [B, C, H, W]
        grid: [B, H, W, 2] in normalized coords.
    Returns:
        Warped tensor with same shape as input.
    """
    if grid is None:
        return tensor
    if tensor.dim() != 4:
        raise ValueError("apply_warp expects 4D tensor")
    padding_mode = "border"
    if tensor.device.type == "mps":
        # MPS backend does not support border padding; fall back to zeros for compatibility.
        padding_mode = "zeros"
    return F.grid_sample(tensor, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)


def identity_grid(height: int, width: int, device: torch.device, batch: int) -> torch.Tensor:
    """Create an identity sampling grid."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)
    return base.unsqueeze(0).repeat(batch, 1, 1, 1)
