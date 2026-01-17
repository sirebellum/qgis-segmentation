# SPDX-License-Identifier: BSD-3-Clause
"""GeoTIFF patch loader for RGB-only training.

This dataset samples fixed-size windows (default 512x512) from a list of
3-band GeoTIFF paths. It optionally loads aligned target rasters for
auxiliary supervised losses/metrics. Elevation or multi-band inputs are
rejected at load time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ..config import DataConfig


@dataclass
class GeoPatchSample:
    rgb: torch.Tensor  # [3, H, W]
    target: Optional[torch.Tensor]  # [H, W] int64 or None
    meta: dict


class GeoTiffPatchDataset(Dataset):
    """Sample random RGB patches from a set of GeoTIFFs.

    Args:
        rasters: iterable of GeoTIFF paths (3-band RGB required).
        targets: optional iterable of aligned target rasters (same shape as rasters).
        data_cfg: DataConfig providing patch_size/stride and optional max_samples.
        with_targets: load targets when available; otherwise ignore.
    """

    def __init__(
        self,
        rasters: Iterable[str | Path],
        *,
        targets: Optional[Iterable[str | Path]] = None,
        data_cfg: Optional[DataConfig] = None,
        with_targets: bool = False,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg or DataConfig()
        self.patch = int(max(1, self.data_cfg.patch_size))
        self.rasters: List[Path] = [Path(p).expanduser().resolve() for p in rasters]
        if not self.rasters:
            raise ValueError("At least one raster path is required")
        self.targets: Optional[List[Path]] = None
        if targets is not None:
            tgt_list = [Path(p).expanduser().resolve() for p in targets]
            if len(tgt_list) != len(self.rasters):
                raise ValueError("Targets must align 1:1 with rasters when provided")
            self.targets = tgt_list
        self.with_targets = with_targets and self.targets is not None
        self.max_samples = self.data_cfg.max_samples

    def __len__(self) -> int:  # pragma: no cover - length unused for random sampling
        # Use a large virtual length when max_samples is None to enable endless sampling.
        return self.max_samples if self.max_samples is not None else 10_000_000

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            if src.count != 3:
                raise ValueError(f"Expected 3 bands, found {src.count} in {path}")
            window = GeoTiffPatchDataset._random_window(src, 512)
            array = src.read(window=window)
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _random_window(src: rasterio.io.DatasetReader, patch: int):
        h = src.height
        w = src.width
        if h < patch or w < patch:
            raise ValueError(f"Raster smaller than patch size {patch}: {(h, w)}")
        y0 = random.randint(0, h - patch)
        x0 = random.randint(0, w - patch)
        return rasterio.windows.Window(x0, y0, patch, patch)

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        if rgb.shape[0] != 3:
            raise ValueError("RGB array must be channel-first [3, H, W]")
        tensor = torch.from_numpy(rgb) / 255.0
        return tensor

    @staticmethod
    def _load_target(path: Path, window) -> torch.Tensor:
        with rasterio.open(path) as src:
            array = src.read(1, window=window)
        tensor = torch.from_numpy(array.astype(np.int64, copy=False))
        return tensor

    def __getitem__(self, index: int) -> GeoPatchSample:
        idx = index % len(self.rasters)
        raster_path = self.rasters[idx]
        window = None
        with rasterio.open(raster_path) as src:
            if src.count != 3:
                raise ValueError(f"Expected 3 bands, found {src.count} in {raster_path}")
            patch = int(max(1, self.patch))
            if src.height < patch or src.width < patch:
                raise ValueError(f"Raster smaller than patch size {patch}: {raster_path}")
            window = self._random_window(src, patch)
            rgb = src.read(window=window)
        rgb_tensor = self._to_tensor(rgb)
        target_tensor: Optional[torch.Tensor] = None
        if self.with_targets and self.targets is not None:
            target_path = self.targets[idx]
            target_tensor = self._load_target(target_path, window)
        meta = {"raster": str(raster_path), "window": (window.col_off, window.row_off, window.width, window.height)}
        return GeoPatchSample(rgb=rgb_tensor, target=target_tensor, meta=meta)
