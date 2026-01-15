# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Refinement placeholders (learned + fast smoothing)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RefineHead(nn.Module):
    def __init__(self, num_classes: int = 16):
        super().__init__()
        self.learned = nn.Sequential(
            nn.Conv2d(num_classes + 3, num_classes, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, 3, padding=1),
        )

    def forward(self, probs: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        rgb_resized = F.interpolate(rgb, size=probs.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([probs, rgb_resized], dim=1)
        refined = self.learned(x)
        return torch.softmax(refined, dim=1)


def fast_smooth(probs: torch.Tensor, iters: int = 1) -> torch.Tensor:
    if iters <= 0:
        return probs
    kernel = torch.ones(1, 1, 3, 3, device=probs.device) / 9.0
    x = probs
    for _ in range(iters):
        x = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
    return x
