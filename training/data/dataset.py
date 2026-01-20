# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Augmented two-view wrapper for RGB-only samples."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..config import AugConfig, DataConfig
from ..augmentations import apply_augmentations, make_rng
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

    def _augment(self, rgb: torch.Tensor, *, view_id: int, sample_idx: int) -> torch.Tensor:
        rng = make_rng(self.aug_cfg.seed, sample_index=sample_idx, view_id=view_id)
        augmented = apply_augmentations(rgb, aug_cfg=self.aug_cfg, rng=rng)["rgb"]
        return augmented

    def __getitem__(self, idx: int) -> Dict[str, Dict]:
        sample = self.samples[idx % len(self.samples)]
        rgb = self._to_tensor(sample["rgb"])
        view1_rgb = self._augment(rgb.clone(), view_id=0, sample_idx=idx)
        view2_rgb = self._augment(rgb.clone(), view_id=1, sample_idx=idx)
        grid = identity_grid(view1_rgb.shape[-2], view1_rgb.shape[-1], device=view1_rgb.device, batch=1)
        return {
            "view1": {"rgb": view1_rgb},
            "view2": {"rgb": view2_rgb},
            "warp_grid": grid,
        }
