# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Synthetic RGB-only sample generator for offline tests and smoke runs."""
from __future__ import annotations

from typing import Dict, List

import torch

from ..config import DataConfig


class SyntheticDataset:
    def __init__(self, num_samples: int, cfg: DataConfig):
        self.num_samples = max(1, int(num_samples))
        self.cfg = cfg
        self.samples: List[Dict] = []
        for _ in range(self.num_samples):
            rgb = torch.rand(3, self.cfg.patch_size, self.cfg.patch_size)
            self.samples.append({"rgb": rgb})

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx % self.num_samples]
