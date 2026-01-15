# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Synthetic data generator for smoke tests."""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch.utils.data import Dataset


def _make_pattern(seed: int, with_elev: bool) -> dict:
    torch.manual_seed(seed)
    size = 512
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    centers = torch.tensor([[0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3]])
    rgb = torch.zeros(3, size, size)
    elev = torch.zeros(1, size, size) if with_elev else None
    for idx, (cy, cx) in enumerate(centers):
        radius = 0.18 + 0.05 * math.sin(idx + seed)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
        color = torch.rand(3, 1, 1) * (0.4 + 0.1 * idx)
        rgb = torch.where(mask, color.expand_as(rgb), rgb)
        if elev is not None:
            elev = torch.where(mask, elev + (idx + 1) * 0.5, elev)
    rgb = rgb + 0.05 * torch.randn_like(rgb)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return {"rgb": rgb, "elev": elev, "id": f"synthetic_{seed}"}


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 4, with_elevation: bool = True, seed: int = 0):
        self.samples: List[dict] = []
        for i in range(num_samples):
            self.samples.append(_make_pattern(seed + i, with_elevation))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
