# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Manifest-based loader for NAIP RGB + USGS 3DEP DEM tiles.

This loader expects a jsonl manifest produced by scripts/data/prepare_naip_3dep_dataset.py.
Each row must include rgb_path, dem_path (relative to an anchor), and metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..config import AugConfig, DataConfig
from .dataset import UnsupervisedRasterDataset

try:  # optional backend
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None


@dataclass
class ManifestEntry:
    aoi_name: str
    tile_id: str
    rgb_path: Path
    dem_path: Path
    epsg: int
    pixel_size: tuple
    width: int
    height: int
    bounds: tuple
    geotransform: tuple
    source_naip: Optional[str]
    source_dem: Optional[str]
    dem_tier: Optional[str] = None


def load_manifest(manifest_path: Path, anchor: Optional[Path] = None, limit: Optional[int] = None) -> List[ManifestEntry]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if anchor is None:
        try:
            anchor = manifest_path.parents[3]
        except IndexError:
            anchor = manifest_path.parent
    entries: List[ManifestEntry] = []
    with manifest_path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            rec = json.loads(line)
            rgb_path = (anchor / rec["rgb_path"]).expanduser().resolve()
            dem_path = (anchor / rec["dem_path"]).expanduser().resolve()
            entries.append(
                ManifestEntry(
                    aoi_name=rec.get("aoi_name", "unknown"),
                    tile_id=rec["tile_id"],
                    rgb_path=rgb_path,
                    dem_path=dem_path,
                    epsg=int(rec["epsg"]),
                    pixel_size=tuple(rec.get("pixel_size", ())),
                    width=int(rec.get("width", 0)),
                    height=int(rec.get("height", 0)),
                    bounds=tuple(rec.get("bounds", ())),
                    geotransform=tuple(rec.get("geotransform", ())),
                    source_naip=rec.get("source_naip"),
                    source_dem=rec.get("source_dem"),
                    dem_tier=rec.get("dem_tier"),
                )
            )
    return entries


class Naip3DepDataset(UnsupervisedRasterDataset):
    """Dataset that loads manifest entries and standardizes elevation."""

    def __init__(self, manifest_path: Path, data_cfg: DataConfig, aug_cfg: AugConfig, limit: Optional[int] = None, validate: bool = False):
        if rasterio is None:
            raise ImportError("rasterio is required to load NAIP/3DEP tiles")
        entries = load_manifest(manifest_path, limit=limit)
        samples: List[Dict[str, Any]] = []
        for entry in entries:
            if not entry.rgb_path.exists():
                raise FileNotFoundError(f"RGB tile missing: {entry.rgb_path}")
            if not entry.dem_path.exists():
                raise FileNotFoundError(f"DEM tile missing: {entry.dem_path}")
            samples.append({
                "rgb_path": str(entry.rgb_path),
                "elev_path": str(entry.dem_path),
                "id": entry.tile_id,
                "aoi": entry.aoi_name,
            })
        super().__init__(samples, data_cfg, aug_cfg)
        self._validate = validate

    def _load_tensor(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tensors = super()._load_tensor(sample)
        rgb = tensors["rgb"].float()
        elev = tensors.get("elev")
        if elev is not None and self.data_cfg.standardize_elevation:
            mean = float(elev.mean())
            std = float(elev.std())
            eps = max(self.data_cfg.elevation_eps, 1e-6)
            elev = (elev - mean) / max(std, eps)
        tensors["rgb"] = rgb
        if elev is not None:
            tensors["elev"] = elev
        if self._validate:
            if rgb.shape[-2:] != (self.data_cfg.patch_size, self.data_cfg.patch_size):
                raise ValueError("RGB patch size does not match config")
            if elev is not None and elev.shape[-2:] != rgb.shape[-2:]:
                raise ValueError("Elevation shape mismatch")
        return tensors


def build_dataset_from_manifest(cfg: DataConfig, aug_cfg: AugConfig) -> Naip3DepDataset:
    if not cfg.manifest_path:
        raise ValueError("DataConfig.manifest_path is required for NAIP/3DEP dataset")
    manifest_path = Path(cfg.manifest_path).expanduser().resolve()
    return Naip3DepDataset(manifest_path, cfg, aug_cfg, limit=cfg.max_samples, validate=False)
