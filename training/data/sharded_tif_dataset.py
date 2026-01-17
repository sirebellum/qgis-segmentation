# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Iterable GeoTIFF shard loader with optional per-worker caching."""
from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import AugConfig, DataConfig
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

    def _load_array(self, path: Path, cache: Optional[_LruCache]) -> np.ndarray:
        key = str(path)
        if cache:
            cached = cache.get(key)
            if cached is not None:
                return cached
        with rasterio.open(path) as src:
            array = src.read()
        if cache:
            cache.put(key, array)
        return array

    @staticmethod
    def _to_tensor(array: np.ndarray, *, add_batch: bool) -> torch.Tensor:
        tensor = torch.from_numpy(array).float()
        if tensor.dim() != 3:
            raise ValueError("Expected array shaped [C,H,W]")
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0) if add_batch else tensor

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
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
                if not input_rel:
                    continue
                rgb_path = shard_dir / str(input_rel)
                rgb_array = self._load_array(rgb_path, cache)
                target_tensor: Optional[torch.Tensor] = None
                if self.include_targets and entry.get("target"):
                    target_path = shard_dir / str(entry["target"])
                    target_array = self._load_array(target_path, cache)
                    target_tensor = self._normalize_target(target_array)

                if self.with_augmentations:
                    rgb_tensor = self._to_tensor(rgb_array, add_batch=True)
                    view1 = self._augment(rgb_tensor.clone())
                    view2 = self._augment(rgb_tensor.clone())
                    grid = identity_grid(view1.shape[-2], view1.shape[-1], device=view1.device, batch=1)
                    yield {
                        "view1": {"rgb": view1},
                        "view2": {"rgb": view2},
                        "warp_grid": grid,
                        "target": target_tensor,
                        "meta": {"item_id": entry.get("item_id"), "split": entry.get("split")},
                    }
                else:
                    rgb_tensor = self._to_tensor(rgb_array, add_batch=False)
                    yield {
                        "rgb": rgb_tensor,
                        "target": target_tensor,
                        "meta": {"item_id": entry.get("item_id"), "split": entry.get("split")},
                    }

                produced += 1
                if self.max_samples is not None and produced >= self.max_samples:
                    return