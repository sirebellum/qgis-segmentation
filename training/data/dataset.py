# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Augmented two-view wrapper for RGB-only samples."""
from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..config import AugConfig, DataConfig
from ..utils.warp import identity_grid


class UnsupervisedRasterDataset(Dataset):
    """Build two stochastic views from preloaded RGB tensors.

    Samples are dictionaries with an ``"rgb"`` key containing a tensor or
    numpy array shaped [3, H, W]. Elevation is intentionally unsupported.
    """

    def __init__(self, samples: List[Dict], data_cfg: DataConfig, aug_cfg: AugConfig):
        self.samples = samples
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _to_tensor(self, rgb) -> torch.Tensor:
        if isinstance(rgb, torch.Tensor):
            tensor = rgb.float()
        else:
            tensor = torch.as_tensor(rgb, dtype=torch.float32)
        if tensor.dim() != 3 or tensor.shape[0] != 3:
            raise ValueError("rgb sample must be [3,H,W]")
        return tensor.unsqueeze(0) / 255.0 if tensor.max() > 1.0 else tensor.unsqueeze(0)

    def _augment(self, rgb: torch.Tensor) -> torch.Tensor:
        out = rgb
        if random.random() < self.aug_cfg.flip_prob:
            out = torch.flip(out, dims=[3])
        if self.aug_cfg.rotate_choices:
            k = random.choice(self.aug_cfg.rotate_choices)
            if k % 90 != 0:
                raise ValueError("rotate_choices must be multiples of 90 degrees")
            turns = (k // 90) % 4
            if turns:
                out = torch.rot90(out, turns, dims=(2, 3))
        if self.aug_cfg.gaussian_noise_std > 0:
            noise = torch.randn_like(out) * float(self.aug_cfg.gaussian_noise_std)
            out = (out + noise).clamp(0.0, 1.0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Dict]:
        sample = self.samples[idx % len(self.samples)]
        rgb = self._to_tensor(sample["rgb"])
        view1_rgb = self._augment(rgb.clone())
        view2_rgb = self._augment(rgb.clone())
        grid = identity_grid(view1_rgb.shape[-2], view1_rgb.shape[-1], device=view1_rgb.device, batch=1)
        return {
            "view1": {"rgb": view1_rgb},
            "view2": {"rgb": view2_rgb},
            "warp_grid": grid,
        }
