# SPDX-License-Identifier: BSD-3-Clause
"""Teacher interface for training-only distillation.

Runtime code must never import these modules. Teachers run on GPU0 by default
and emit stride-aligned feature maps for distillation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TeacherOutput:
    features: torch.Tensor  # [B, C, H, W]
    stride: int


class TeacherBase(torch.nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or self._default_device()

    @staticmethod
    def _default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def forward(self, rgb: torch.Tensor) -> TeacherOutput:  # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def extract(self, rgb: torch.Tensor) -> TeacherOutput:
        return self.forward(rgb)


class FakeTeacher(TeacherBase):
    """Small conv teacher used for offline tests or when downloads are unavailable."""

    def __init__(self, embed_dim: int = 256, stride: int = 8, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.stride = int(max(1, stride))
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(embed_dim // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=self.stride // 4, padding=1),
            torch.nn.BatchNorm2d(embed_dim),
            torch.nn.ReLU(inplace=True),
        ).to(self.device)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, rgb: torch.Tensor) -> TeacherOutput:
        x = rgb.to(self.device, non_blocking=True)
        feat = self.net(x)
        return TeacherOutput(features=feat, stride=self.stride)
