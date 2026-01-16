# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Resampling helpers for feature alignment."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def resample_to_match(elev: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Resample elevation to match target spatial dims.

    Args:
        elev: [B, 1, H, W]
        target_shape: torch.Size with 4 dims (B, C, Ht, Wt) or 3 dims (Ht, Wt).
    """
    if elev is None:
        return None
    if elev.dim() != 4 or elev.shape[1] != 1:
        raise ValueError("elev must be [B,1,H,W]")
    if len(target_shape) == 4:
        _, _, h, w = target_shape
    elif len(target_shape) == 2:
        h, w = target_shape
    elif len(target_shape) == 3:
        _, h, w = target_shape
    else:
        raise ValueError("target_shape must describe spatial dims")
    return F.interpolate(elev, size=(h, w), mode="bilinear", align_corners=False)


def downsample_factor(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return x
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)
