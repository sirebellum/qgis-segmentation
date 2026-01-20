# SPDX-License-Identifier: BSD-3-Clause
"""DINOv2 teacher adapter (training-only).

Attempts to load a pretrained ViT (e.g., DINOv2 ViT-L/14). If the model or
weights are unavailable, falls back to a lightweight FakeTeacher to keep
offline tests functional. Features are projected to a configurable dimension
and resampled to stride 4 for student alignment.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .teacher_base import FakeTeacher, TeacherBase, TeacherOutput


def _maybe_load_dinov2(model_name: str):
    try:
        from torch.hub import load
    except Exception:
        return None
    try:
        return load("facebookresearch/dinov2", model_name)
    except Exception:
        return None


class Dinov2Teacher(TeacherBase):
    def __init__(
        self,
        *,
        model_name: str = "dinov2_vitl14_reg",
        proj_dim: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self.proj_dim = int(proj_dim)
        backbone = _maybe_load_dinov2(model_name)
        if backbone is None:
            # Offline fallback
            self.fallback = FakeTeacher(embed_dim=proj_dim, stride=8, device=self.device)
            self.backbone = None
            self.proj = None
            self.stride = self.fallback.stride
            return
        self.backbone = backbone.to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.fallback = None
        self.stride = 14  # ViT-L/14 patch size
        self.proj = nn.Conv2d(self.backbone.embed_dim, proj_dim, kernel_size=1, bias=False).to(self.device)
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> TeacherOutput:
        if self.backbone is None and self.fallback is not None:
            return self.fallback(rgb)
        assert self.backbone is not None and self.proj is not None
        # Expect rgb in [0,1], BCHW
        x = rgb.to(self.device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=self.device.type if self.device.type != "cpu" else "cpu", enabled=self.device.type != "cpu"):
            tokens = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
            feat = self.proj(tokens)
        # Resample to stride 4 alignment
        b, c, h, w = feat.shape
        # Use float ratio to preserve the intended stride-4 alignment (avoid floor from integer division).
        target_h = int(round(h * (self.stride / 4.0)))
        target_w = int(round(w * (self.stride / 4.0)))
        feat_resized = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return TeacherOutput(features=feat_resized, stride=4)
