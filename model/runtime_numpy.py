# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Numpy-only inference runtime for the next-gen variable-K segmenter.

This runtime avoids heavyweight ML backends and consumes exported numpy weights from
``model/best``. It implements a minimal subset of the training architecture:
- stride-4 encoder with two residual blocks
- 1x1 seed projection + soft k-means refinement
- optional fast smoothing (box filter)
- bilinear upsampling back to full resolution
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

Array = np.ndarray

RUNTIME_META_VERSION = "numpy-nextgen-v1"
REQUIRED_META_KEYS = {
    "version",
    "max_k",
    "embed_dim",
    "temperature",
    "cluster_iters_default",
    "smooth_iters_default",
    "input_mean",
    "input_std",
    "input_scale",
    "stride",
    "supports_learned_refine",
}


def _parse_runtime_meta(meta_raw: Dict[str, object]) -> RuntimeMeta:
    missing = sorted(REQUIRED_META_KEYS.difference(meta_raw.keys()))
    if missing:
        raise ValueError(f"Runtime metadata missing required keys: {missing}")

    version = str(meta_raw.get("version"))
    if version != RUNTIME_META_VERSION:
        raise ValueError(
            f"Runtime artifact version mismatch: expected {RUNTIME_META_VERSION}, found {version}"
        )

    def _tuple3(name: str) -> Tuple[float, float, float]:
        value = meta_raw.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"meta.json[{name}] must be a 3-item list/tuple")
        try:
            parsed = tuple(float(v) for v in value)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"meta.json[{name}] must contain numeric values") from exc
        return parsed

    max_k = int(meta_raw.get("max_k", 0))
    embed_dim = int(meta_raw.get("embed_dim", 0))
    stride = int(meta_raw.get("stride", 0))
    input_scale = float(meta_raw.get("input_scale", 0.0))

    if max_k < 2:
        raise ValueError("meta.json[max_k] must be >= 2")
    if embed_dim <= 0:
        raise ValueError("meta.json[embed_dim] must be > 0")
    if stride <= 0:
        raise ValueError("meta.json[stride] must be > 0")
    if input_scale <= 0:
        raise ValueError("meta.json[input_scale] must be > 0")

    return RuntimeMeta(
        version=version,
        max_k=max_k,
        embed_dim=embed_dim,
        temperature=float(meta_raw.get("temperature", 0.8)),
        cluster_iters_default=int(meta_raw.get("cluster_iters_default", 1)),
        smooth_iters_default=int(meta_raw.get("smooth_iters_default", 0)),
        input_mean=_tuple3("input_mean"),
        input_std=_tuple3("input_std"),
        input_scale=input_scale,
        stride=stride,
        supports_learned_refine=bool(meta_raw.get("supports_learned_refine", False)),
    )


def _validate_required_weights(weights: Dict[str, Array]) -> None:
    required = {
        "stem.conv1.weight",
        "stem.conv1.bias",
        "stem.bn1.weight",
        "stem.bn1.bias",
        "stem.bn1.running_mean",
        "stem.bn1.running_var",
        "stem.conv2.weight",
        "stem.conv2.bias",
        "stem.bn2.weight",
        "stem.bn2.bias",
        "stem.bn2.running_mean",
        "stem.bn2.running_var",
        "block1.conv1.weight",
        "block1.conv1.bias",
        "block1.bn1.weight",
        "block1.bn1.bias",
        "block1.bn1.running_mean",
        "block1.bn1.running_var",
        "block1.conv2.weight",
        "block1.conv2.bias",
        "block1.bn2.weight",
        "block1.bn2.bias",
        "block1.bn2.running_mean",
        "block1.bn2.running_var",
        "block2.conv1.weight",
        "block2.conv1.bias",
        "block2.bn1.weight",
        "block2.bn1.bias",
        "block2.bn1.running_mean",
        "block2.bn1.running_var",
        "block2.conv2.weight",
        "block2.conv2.bias",
        "block2.bn2.weight",
        "block2.bn2.bias",
        "block2.bn2.running_mean",
        "block2.bn2.running_var",
        "seed_proj.weight",
        "seed_proj.bias",
    }
    missing = sorted(name for name in required if name not in weights)
    if missing:
        raise ValueError(f"Runtime weights missing required tensors: {missing}")


@dataclass
class RuntimeMeta:
    version: str
    max_k: int
    embed_dim: int
    temperature: float
    cluster_iters_default: int
    smooth_iters_default: int
    input_mean: Tuple[float, float, float]
    input_std: Tuple[float, float, float]
    input_scale: float
    stride: int
    supports_learned_refine: bool


def _softmax(x: Array, axis: int = 0) -> Array:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-8)


def _relu(x: Array) -> Array:
    return np.maximum(x, 0.0)


def _pad_to_multiple(x: Array, factor: int) -> Tuple[Array, Tuple[int, int]]:
    _, h, w = x.shape
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    padded = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (pad_h, pad_w)


def _conv2d(x: Array, weight: Array, bias: Optional[Array], stride: int = 1, padding: int = 0) -> Array:
    c_out, c_in, kh, kw = weight.shape
    _, h, w = x.shape
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="reflect")
        h += 2 * padding
        w += 2 * padding
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((c_out, out_h, out_w), dtype=np.float32)
    for oy in range(out_h):
        sy = oy * stride
        for ox in range(out_w):
            sx = ox * stride
            region = x[:, sy : sy + kh, sx : sx + kw]
            out[:, oy, ox] = np.tensordot(weight, region, axes=((1, 2, 3), (0, 1, 2)))
    if bias is not None:
        out += bias.reshape(-1, 1, 1)
    return out


def _batch_norm(x: Array, weight: Array, bias: Array, running_mean: Array, running_var: Array, eps: float = 1e-5) -> Array:
    return (x - running_mean.reshape(-1, 1, 1)) / np.sqrt(running_var.reshape(-1, 1, 1) + eps) * weight.reshape(-1, 1, 1) + bias.reshape(-1, 1, 1)


def _bilinear_resize(x: Array, size: Tuple[int, int]) -> Array:
    # x: [C,H,W]
    c, h, w = x.shape
    out_h, out_w = size
    if h == out_h and w == out_w:
        return x
    y_grid = np.linspace(0, h - 1, out_h)
    x_grid = np.linspace(0, w - 1, out_w)
    y0 = np.floor(y_grid).astype(int)
    x0 = np.floor(x_grid).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y_alpha = (y_grid - y0).reshape(-1, 1)
    x_alpha = (x_grid - x0).reshape(1, -1)
    top = (1 - x_alpha) * x[:, y0][:, :, x0] + x_alpha * x[:, y0][:, :, x1]
    bottom = (1 - x_alpha) * x[:, y1][:, :, x0] + x_alpha * x[:, y1][:, :, x1]
    out = (1 - y_alpha) * top + y_alpha * bottom
    return out.astype(np.float32)


def _depthwise_box_filter(x: Array, iters: int = 1) -> Array:
    if iters <= 0:
        return x
    c, h, w = x.shape
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    out = x
    for _ in range(iters):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
        next_out = np.zeros_like(out)
        for oy in range(h):
            for ox in range(w):
                patch = padded[:, oy : oy + 3, ox : ox + 3]
                next_out[:, oy, ox] = np.sum(patch * kernel, axis=(1, 2))
        out = next_out
    return out


class NumpySegmenter:
    def __init__(self, weights: Dict[str, Array], meta: RuntimeMeta):
        self.w = weights
        self.meta = meta
        self.max_k = max(2, min(16, int(meta.max_k)))
        self.temperature = float(meta.temperature)
        self.cluster_iters_default = max(1, int(meta.cluster_iters_default))
        self.smooth_iters_default = max(0, int(meta.smooth_iters_default))
        self.input_mean = np.array(meta.input_mean, dtype=np.float32).reshape(3, 1, 1)
        self.input_std = np.array(meta.input_std, dtype=np.float32).reshape(3, 1, 1)
        self.input_scale = float(meta.input_scale)
        self.stride = max(1, int(meta.stride))

    def _stem(self, x: Array) -> Array:
        x = _conv2d(x, self.w["stem.conv1.weight"], self.w["stem.conv1.bias"], stride=2, padding=1)
        x = _batch_norm(x, self.w["stem.bn1.weight"], self.w["stem.bn1.bias"], self.w["stem.bn1.running_mean"], self.w["stem.bn1.running_var"])
        x = _relu(x)
        x = _conv2d(x, self.w["stem.conv2.weight"], self.w["stem.conv2.bias"], stride=2, padding=1)
        x = _batch_norm(x, self.w["stem.bn2.weight"], self.w["stem.bn2.bias"], self.w["stem.bn2.running_mean"], self.w["stem.bn2.running_var"])
        x = _relu(x)
        return x

    def _resblock(self, x: Array, prefix: str) -> Array:
        out = _conv2d(x, self.w[f"{prefix}.conv1.weight"], self.w[f"{prefix}.conv1.bias"], padding=1)
        out = _batch_norm(out, self.w[f"{prefix}.bn1.weight"], self.w[f"{prefix}.bn1.bias"], self.w[f"{prefix}.bn1.running_mean"], self.w[f"{prefix}.bn1.running_var"])
        out = _relu(out)
        out = _conv2d(out, self.w[f"{prefix}.conv2.weight"], self.w[f"{prefix}.conv2.bias"], padding=1)
        out = _batch_norm(out, self.w[f"{prefix}.bn2.weight"], self.w[f"{prefix}.bn2.bias"], self.w[f"{prefix}.bn2.running_mean"], self.w[f"{prefix}.bn2.running_var"])
        out = _relu(out + x)
        return out

    def _encode(self, x: Array) -> Array:
        x, pads = _pad_to_multiple(x, factor=self.stride)
        x = self._stem(x)
        x = self._resblock(x, "block1")
        x = self._resblock(x, "block2")
        return x

    def _soft_kmeans(self, emb: Array, k: int, cluster_iters: Optional[int]) -> Tuple[Array, Array]:
        c, h, w = emb.shape
        k_eff = max(2, min(self.max_k, int(k)))
        seeds_w = self.w["seed_proj.weight"][:k_eff]
        seeds_b = self.w.get("seed_proj.bias")
        if seeds_b is not None:
            seeds_b = seeds_b[:k_eff]
        seeds = _conv2d(emb, seeds_w, seeds_b, stride=1, padding=0)
        logits = seeds.reshape(k_eff, -1)
        assign = _softmax(logits, axis=0)
        feats = emb.reshape(c, -1)
        denom = np.sum(assign, axis=1, keepdims=True) + 1e-6
        prototypes = (assign @ feats.T) / denom
        iters = max(1, cluster_iters if cluster_iters is not None else self.cluster_iters_default)
        for _ in range(iters):
            proto = prototypes
            diff = feats[None, :, :] - proto[:, :, None]
            dist = np.sum(diff * diff, axis=1)  # [K,N]
            logits = -dist / max(self.temperature, 1e-6)
            assign = _softmax(logits, axis=0)
            denom = np.sum(assign, axis=1, keepdims=True) + 1e-6
            prototypes = (assign @ feats.T) / denom
        logits_map = logits.reshape(k_eff, h, w)
        probs = _softmax(logits_map, axis=0).astype(np.float32)
        return probs, prototypes

    def _upsample(self, probs: Array, target_shape: Tuple[int, int]) -> Array:
        return _bilinear_resize(probs, target_shape)

    def forward(self, rgb: Array, k: int, smooth_iters: Optional[int] = None, cluster_iters: Optional[int] = None) -> Array:
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        rgb = rgb * self.input_scale
        rgb = (rgb - self.input_mean) / np.maximum(self.input_std, 1e-6)
        emb = self._encode(rgb)
        probs_latent, _ = self._soft_kmeans(emb, k=k, cluster_iters=cluster_iters)
        probs = self._upsample(probs_latent, target_shape=rgb.shape[-2:])
        smooth = smooth_iters if smooth_iters is not None else self.smooth_iters_default
        if smooth > 0:
            probs = _depthwise_box_filter(probs, iters=smooth)
        return probs

    def predict_labels(self, rgb: Array, k: int, smooth_iters: Optional[int] = None, cluster_iters: Optional[int] = None) -> Array:
        probs = self.forward(rgb, k=k, smooth_iters=smooth_iters, cluster_iters=cluster_iters)
        labels = np.argmax(probs, axis=0).astype(np.uint8)
        return labels


def load_runtime_model(model_dir: str, status_callback: Optional[Callable[[str], None]] = None) -> NumpySegmenter:
    meta_path = os.path.join(model_dir, "meta.json")
    weights_path = os.path.join(model_dir, "model.npz")
    if not os.path.exists(meta_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing runtime artifacts in {model_dir}; expected meta.json and model.npz")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    meta = _parse_runtime_meta(meta_raw)
    if status_callback:
        try:
            status_callback(f"Loading next-gen numpy runtime (version {meta.version})...")
        except Exception:
            pass
    with np.load(weights_path) as weights:
        named = {k: weights[k] for k in weights.files}
    _validate_required_weights(named)
    return NumpySegmenter(named, meta)


__all__ = ["RuntimeMeta", "NumpySegmenter", "load_runtime_model", "RUNTIME_META_VERSION"]
