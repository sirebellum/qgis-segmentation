# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Torch-backed inference runtime for the next-gen variable-K segmenter.

This backend mirrors the numpy runtime but executes the forward pass in PyTorch
and prefers GPU devices when available (CUDA → MPS → CPU). It consumes the same
``model.npz`` / ``meta.json`` artifacts exported by the training pipeline.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .runtime_numpy import RuntimeMeta, load_runtime_artifacts

Array = np.ndarray


def _select_device(preference: str = "auto") -> torch.device:
    pref = (preference or "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchSegmenter:
    def __init__(self, weights: Dict[str, Array], meta: RuntimeMeta, *, device_preference: str = "auto"):
        self.meta = meta
        self.device = _select_device(device_preference)
        self.device_label = str(self.device)
        self.backend = "torch"

        self.w = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in weights.items()}
        self.max_k = max(2, min(16, int(meta.max_k)))
        self.temperature = float(meta.temperature)
        self.cluster_iters_default = max(1, int(meta.cluster_iters_default))
        self.smooth_iters_default = max(0, int(meta.smooth_iters_default))
        self.input_mean = torch.tensor(meta.input_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.input_std = torch.tensor(meta.input_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.input_scale = float(meta.input_scale)
        self.stride = max(1, int(meta.stride))

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h, w = x.shape
        pad_h = (self.stride - (h % self.stride)) % self.stride
        pad_w = (self.stride - (w % self.stride)) % self.stride
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (pad_h, pad_w)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.w["stem.conv1.weight"], self.w.get("stem.conv1.bias"), stride=2, padding=1)
        x = F.batch_norm(x, self.w["stem.bn1.running_mean"], self.w["stem.bn1.running_var"], self.w["stem.bn1.weight"], self.w["stem.bn1.bias"], training=False)
        x = F.relu(x)
        x = F.conv2d(x, self.w["stem.conv2.weight"], self.w.get("stem.conv2.bias"), stride=2, padding=1)
        x = F.batch_norm(x, self.w["stem.bn2.running_mean"], self.w["stem.bn2.running_var"], self.w["stem.bn2.weight"], self.w["stem.bn2.bias"], training=False)
        x = F.relu(x)
        return x

    def _resblock(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        out = F.conv2d(x, self.w[f"{prefix}.conv1.weight"], self.w.get(f"{prefix}.conv1.bias"), padding=1)
        out = F.batch_norm(out, self.w[f"{prefix}.bn1.running_mean"], self.w[f"{prefix}.bn1.running_var"], self.w[f"{prefix}.bn1.weight"], self.w[f"{prefix}.bn1.bias"], training=False)
        out = F.relu(out)
        out = F.conv2d(out, self.w[f"{prefix}.conv2.weight"], self.w.get(f"{prefix}.conv2.bias"), padding=1)
        out = F.batch_norm(out, self.w[f"{prefix}.bn2.running_mean"], self.w[f"{prefix}.bn2.running_var"], self.w[f"{prefix}.bn2.weight"], self.w[f"{prefix}.bn2.bias"], training=False)
        out = F.relu(out + x)
        return out

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._pad_to_multiple(x)
        x = self._stem(x)
        x = self._resblock(x, "block1")
        x = self._resblock(x, "block2")
        return x

    def _soft_kmeans(self, emb: torch.Tensor, k: int, cluster_iters: Optional[int]) -> torch.Tensor:
        b, c, h, w = emb.shape
        k_eff = max(2, min(self.max_k, int(k)))
        seeds_w = self.w["seed_proj.weight"][:k_eff]
        seeds_b = self.w.get("seed_proj.bias")
        if seeds_b is not None:
            seeds_b = seeds_b[:k_eff]
        seeds = F.conv2d(emb, seeds_w, seeds_b, stride=1, padding=0)
        logits = seeds.view(b, k_eff, -1)
        assign = torch.softmax(logits, dim=1)
        feats = emb.view(b, c, -1)
        denom = assign.sum(dim=2, keepdim=True) + 1e-6
        prototypes = torch.einsum("bkn,bcn->bkc", assign, feats) / denom

        iters = max(1, cluster_iters if cluster_iters is not None else self.cluster_iters_default)
        for _ in range(iters):
            proto = prototypes
            diff = feats.unsqueeze(1) - proto.unsqueeze(-1)
            dist = (diff * diff).sum(dim=2)
            logits = -dist / max(self.temperature, 1e-6)
            assign = torch.softmax(logits, dim=1)
            denom = assign.sum(dim=2, keepdim=True) + 1e-6
            prototypes = torch.einsum("bkn,bcn->bkc", assign, feats) / denom

        logits_map = logits.view(b, k_eff, h, w)
        probs = torch.softmax(logits_map, dim=1)
        return probs

    def _depthwise_box_filter(self, x: torch.Tensor, iters: int) -> torch.Tensor:
        if iters <= 0:
            return x
        k = x.shape[1]
        kernel = torch.ones((k, 1, 3, 3), device=x.device, dtype=x.dtype) / 9.0
        out = x
        for _ in range(iters):
            out = F.conv2d(out, kernel, bias=None, stride=1, padding=1, groups=k)
        return out

    @torch.no_grad()
    def forward(self, rgb: Array, k: int, smooth_iters: Optional[int] = None, cluster_iters: Optional[int] = None) -> Array:
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        x = torch.as_tensor(rgb, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_h, target_w = x.shape[-2], x.shape[-1]
        x = x * self.input_scale
        x = (x - self.input_mean) / torch.clamp(self.input_std, min=1e-6)
        emb = self._encode(x)
        probs_latent = self._soft_kmeans(emb, k=k, cluster_iters=cluster_iters)
        probs = F.interpolate(probs_latent, size=(target_h, target_w), mode="bilinear", align_corners=False)
        smooth = smooth_iters if smooth_iters is not None else self.smooth_iters_default
        if smooth > 0:
            probs = self._depthwise_box_filter(probs, iters=smooth)
        return probs.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_labels(self, rgb: Array, k: int, smooth_iters: Optional[int] = None, cluster_iters: Optional[int] = None) -> Array:
        probs = self.forward(rgb, k=k, smooth_iters=smooth_iters, cluster_iters=cluster_iters)
        labels = np.argmax(probs, axis=0).astype(np.uint8)
        return labels


def load_runtime_model_torch(
    model_dir: str,
    *,
    device_preference: str = "auto",
    status_callback: Optional[callable] = None,
) -> TorchSegmenter:
    weights, meta = load_runtime_artifacts(model_dir, status_callback=status_callback)
    if status_callback:
        try:
            status_callback(f"Loading torch runtime on {device_preference or 'auto'} device preference...")
        except Exception:
            pass
    runtime = TorchSegmenter(weights, meta, device_preference=device_preference)
    return runtime


__all__ = ["TorchSegmenter", "load_runtime_model_torch"]
