# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Paired two-view augmentations with shared geometry and photometric jitter."""
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import AugConfig
from ..utils.warp import identity_grid


def _random_affine_matrix(cfg: AugConfig, device: torch.device) -> torch.Tensor:
    angle = math.radians(random.choice(cfg.rotate_choices))
    tx = 0.0
    ty = 0.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    matrix = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=torch.float32, device=device
    )
    return matrix


def _apply_affine(img: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    b, c, h, w = img.shape
    grid = F.affine_grid(matrix.unsqueeze(0).expand(b, -1, -1), (b, c, h, w), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _color_jitter(x: torch.Tensor, cfg: AugConfig) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("color jitter expects [B,C,H,W]")
    b = x.shape[0]
    jittered = x
    for _ in range(b):
        brightness, contrast, saturation, hue = cfg.color_jitter
        alpha = 1.0 + random.uniform(-brightness, brightness)
        jittered = jittered * alpha
        if contrast > 0:
            c_scale = 1.0 + random.uniform(-contrast, contrast)
            jittered = (jittered - jittered.mean()) * c_scale + jittered.mean()
        if saturation > 0 and jittered.shape[1] >= 3:
            gray = jittered.mean(dim=1, keepdim=True)
            s_scale = random.uniform(-saturation, saturation)
            jittered = jittered + s_scale * (jittered - gray)
        if hue > 0:
            jittered = torch.clamp(jittered + random.uniform(-hue, hue), 0.0, 1.0)
    if cfg.gaussian_noise_std > 0:
        jittered = jittered + torch.randn_like(jittered) * cfg.gaussian_noise_std
    return torch.clamp(jittered, 0.0, 1.0)


def paired_views(
    rgb: torch.Tensor,
    elev: Optional[torch.Tensor],
    cfg: AugConfig,
    elev_present: bool,
) -> Tuple[dict, dict, torch.Tensor]:
    """Create two augmented views sharing geometry with independent color jitter.

    Returns view1, view2, and an identity warp grid (views share geometry).
    """
    if rgb.dim() != 4:
        raise ValueError("rgb must be [B,3,H,W]")
    device = rgb.device
    matrix = _random_affine_matrix(cfg, device)
    rgb_geo = _apply_affine(rgb, matrix)
    elev_geo = _apply_affine(elev, matrix) if elev is not None else None

    def _build_view(base_rgb, base_elev):
        jittered = _color_jitter(base_rgb.clone(), cfg)
        view = {"rgb": jittered}
        if base_elev is not None and elev_present:
            view["elev"] = base_elev
        return view

    view1 = _build_view(rgb_geo, elev_geo)
    view2 = _build_view(rgb_geo, elev_geo)
    grid = identity_grid(rgb.shape[-2], rgb.shape[-1], device=device, batch=rgb.shape[0])
    return view1, view2, grid
