# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Iterable GeoTIFF shard loader with optional per-worker caching."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import AugConfig, DataConfig
from ..augmentations import apply_augmentations, make_rng
from ..utils.warp import identity_grid


class _LruCache:
    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        value = self._store.pop(key, None)
        if value is not None:
            self._store[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)


class ShardedTifDataset(IterableDataset):
    """Stream records from shard directories for a given split.

    The dataset partitions shards across DataLoader workers for throughput.
    It supports optional per-worker LRU caching to avoid repeated disk reads.
    """

    def __init__(
        self,
        *,
        processed_root: str | Path,
        dataset_id: str,
        split: str,
        data_cfg: DataConfig,
        aug_cfg: Optional[AugConfig] = None,
        with_augmentations: bool = True,
        include_targets: bool = False,
        cache_mode: str = "none",
        cache_max_items: int = 0,
        require_slic: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.root = Path(processed_root).expanduser().resolve()
        self.dataset_id = dataset_id
        self.split = split
        self.data_cfg = data_cfg
        self.aug_cfg = aug_cfg or AugConfig()
        self.with_augmentations = with_augmentations
        self.include_targets = include_targets
        self.cache_mode = cache_mode
        self.cache_max_items = int(cache_max_items)
        self.max_samples = data_cfg.max_samples
        self.require_slic = bool(require_slic if require_slic is not None else getattr(data_cfg, "require_slic", True))
        self._cache: Optional[_LruCache] = None

        base = self.root / split / dataset_id
        self._shard_dirs = sorted(base.glob("shard-*"))
        if not self._shard_dirs:
            raise FileNotFoundError(f"No shards found under {base}")

    def _make_cache(self) -> Optional[_LruCache]:
        if self.cache_mode.lower() != "lru" or self.cache_max_items <= 0:
            return None
        if self._cache is None:
            self._cache = _LruCache(self.cache_max_items)
        return self._cache

    def _load_array(self, path: Path, cache: Optional[_LruCache], expected_channels: Optional[int] = 3) -> np.ndarray:
        key = str(path)
        if cache:
            cached = cache.get(key)
            if cached is not None:
                return cached
        with rasterio.open(path) as src:
            array = src.read()
            if expected_channels is not None and array.shape[0] != expected_channels:
                raise ValueError(f"Expected {expected_channels} channels, found {array.shape[0]} in {path}")
        if cache:
            cache.put(key, array)
        return array

    def _load_slic(self, path: Path, cache: Optional[_LruCache]) -> np.ndarray:
        key = f"slic::{path}"
        if cache:
            cached = cache.get(key)
            if cached is not None:
                return cached
        with np.load(path) as npz:
            labels = npz["labels"]
        if cache:
            cache.put(key, labels)
        return labels

    @staticmethod
    def _to_tensor(array: np.ndarray, *, add_batch: bool) -> torch.Tensor:
        tensor = torch.from_numpy(array).float()
        if tensor.dim() != 3:
            raise ValueError("Expected array shaped [C,H,W]")
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0) if add_batch else tensor

    def _augment_view(
        self,
        rgb: torch.Tensor,
        slic: torch.Tensor,
        *,
        rng: Optional[torch.Generator],
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        augmented = apply_augmentations(rgb, slic=slic, target=target, aug_cfg=self.aug_cfg, rng=rng)
        return augmented["rgb"], augmented["slic"], augmented.get("target")

    @staticmethod
    def _normalize_target(array: np.ndarray) -> torch.Tensor:
        if array.ndim != 3:
            raise ValueError("Target array must be [C,H,W]")
        plane = array[0].astype(np.int64, copy=False)
        positives = sorted(v for v in np.unique(plane) if v > 0)
        if not positives:
            return torch.from_numpy((plane * 0).astype(np.int64))
        remapped = np.zeros_like(plane, dtype=np.int64)
        for idx, value in enumerate(positives, start=1):
            remapped[plane == value] = idx
        return torch.from_numpy(remapped)

    def _iter_entries(self, shard_dir: Path) -> Iterator[Dict[str, object]]:
        index_path = shard_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index file: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        worker = get_worker_info()
        cache = self._make_cache()
        shard_dirs = self._shard_dirs
        if worker is not None:
            shard_dirs = shard_dirs[worker.id :: worker.num_workers]
        produced = 0
        for shard_dir in shard_dirs:
            for entry in self._iter_entries(shard_dir):
                if self.include_targets and not entry.get("has_target"):
                    continue
                input_rel = entry.get("input")
                slic_rel = entry.get("slic")
                if not input_rel:
                    raise ValueError("Shard index entry must include an input field")
                if self.require_slic and not slic_rel:
                    raise ValueError("Shard index entry missing precomputed SLIC; set require_slic=False to bypass")

                rgb_path = shard_dir / str(input_rel)
                rgb_array = self._load_array(rgb_path, cache)

                slic_array = None
                if slic_rel:
                    slic_path = shard_dir / str(slic_rel)
                    slic_array = self._load_slic(slic_path, cache)
                    if slic_array.shape[-2:] != rgb_array.shape[-2:]:
                        raise ValueError(f"SLIC shape {slic_array.shape[-2:]} does not match RGB {rgb_array.shape[-2:]} for {slic_path}")
                else:
                    # Backward compatibility for shards created before SLIC precomputation.
                    slic_array = np.zeros(rgb_array.shape[-2:], dtype=np.int64)

                target_tensor: Optional[torch.Tensor] = None
                if self.include_targets and entry.get("target"):
                    target_path = shard_dir / str(entry["target"])
                    target_array = self._load_array(target_path, cache, expected_channels=None)
                    target_tensor = self._normalize_target(target_array)

                slic_tensor = torch.from_numpy(slic_array.astype(np.int64, copy=False))

                if self.with_augmentations:
                    rgb_tensor = self._to_tensor(rgb_array, add_batch=True)
                    slic_tensor = slic_tensor.unsqueeze(0).unsqueeze(0)
                    rng_base = self.aug_cfg.seed
                    worker_id = worker.id if worker is not None else 0
                    item_id = entry.get("item_id")
                    view1_rng = make_rng(rng_base, worker_id=worker_id, sample_index=produced, item_id=item_id, view_id=0)
                    view2_rng = make_rng(rng_base, worker_id=worker_id, sample_index=produced, item_id=item_id, view_id=1)
                    view1_rgb, view1_slic, view1_target = self._augment_view(
                        rgb_tensor.clone(), slic_tensor.clone(), rng=view1_rng, target=target_tensor
                    )
                    view2_rgb, view2_slic, view2_target = self._augment_view(
                        rgb_tensor.clone(), slic_tensor.clone(), rng=view2_rng, target=target_tensor
                    )
                    view1_slic = view1_slic.long().squeeze(0) if view1_slic is not None else None
                    view2_slic = view2_slic.long().squeeze(0) if view2_slic is not None else None
                    target_out = view1_target if view1_target is not None else target_tensor
                    grid = identity_grid(view1_rgb.shape[-2], view1_rgb.shape[-1], device=view1_rgb.device, batch=1)
                    yield {
                        "view1": {"rgb": view1_rgb, "slic": view1_slic},
                        "view2": {"rgb": view2_rgb, "slic": view2_slic},
                        "warp_grid": grid,
                        "target": target_out,
                        "meta": {
                            "item_id": entry.get("item_id"),
                            "split": entry.get("split"),
                            "slic_meta": entry.get("slic_meta"),
                        },
                    }
                else:
                    rgb_tensor = self._to_tensor(rgb_array, add_batch=False)
                    yield {
                        "rgb": rgb_tensor,
                        "slic": slic_tensor,
                        "target": target_tensor,
                        "meta": {
                            "item_id": entry.get("item_id"),
                            "split": entry.get("split"),
                            "slic_meta": entry.get("slic_meta"),
                        },
                    }

                produced += 1
                if self.max_samples is not None and produced >= self.max_samples:
                    return