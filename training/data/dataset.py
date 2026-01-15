# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Datasets for unlabeled RGB (optional elevation) training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..config import AugConfig, DataConfig
from .augmentations import paired_views

try:  # optional raster backend
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None


class UnsupervisedRasterDataset(Dataset):
    """Patch sampler for unlabeled RGB with optional elevation.

    Each sample can be provided as in-memory tensors (recommended for tests)
    or file paths (rgb_path, elev_path). When file paths are used, rasterio is
    attempted first; otherwise, a clear error is raised.
    """

    def __init__(self, samples: List[Dict[str, Any]], data_cfg: DataConfig, aug_cfg: AugConfig):
        self.samples = samples
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rgb = sample.get("rgb")
        elev = sample.get("elev")
        if rgb is not None:
            return {"rgb": rgb, "elev": elev}

        rgb_path = sample.get("rgb_path")
        elev_path = sample.get("elev_path")
        if rgb_path is None:
            raise ValueError("Sample must provide 'rgb' tensor or 'rgb_path'")
        if rasterio is None:
            raise ImportError("rasterio not installed; provide in-memory tensors for tests")
        with rasterio.open(rgb_path) as src:
            rgb_arr = src.read(out_dtype="uint8")
        rgb_tensor = torch.from_numpy(rgb_arr).float() / 255.0
        elev_tensor = None
        if elev_path:
            with rasterio.open(elev_path) as src:
                elev_arr = src.read(out_dtype="float32")
            elev_tensor = torch.from_numpy(elev_arr)
            if elev_tensor.ndim == 3:
                elev_tensor = elev_tensor[:1]
        return {"rgb": rgb_tensor, "elev": elev_tensor}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tensors = self._load_tensor(sample)
        rgb = tensors["rgb"].float()
        elev = tensors.get("elev")
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError("RGB tensor must be [3,H,W]")
        rgb = rgb.unsqueeze(0)
        elev_present = elev is not None
        if elev is not None:
            elev = elev.float().unsqueeze(0)
            if random.random() < self.data_cfg.elevation_dropout:
                elev_present = False
                elev = None
        view1, view2, grid = paired_views(rgb, elev, self.aug_cfg, elev_present)
        return {
            "view1": view1,
            "view2": view2,
            "warp_grid": grid,
            "elev_present": elev_present,
            "meta": {"id": sample.get("id", idx)},
        }
