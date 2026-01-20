import itertools
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, cast, Callable, Any

try:
    from .dependency_manager import ensure_dependencies
except ImportError:  # pragma: no cover
    from dependency_manager import ensure_dependencies

try:
    from .autoencoder_utils import TextureAutoencoderManager
except ImportError:  # pragma: no cover
    from autoencoder_utils import TextureAutoencoderManager

ensure_dependencies()

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class SegmentationCanceled(Exception):
    """Raised when a segmentation task is canceled mid-flight."""


def _maybe_raise_cancel(cancel_token):
    if cancel_token is None:
        return
    checker = getattr(cancel_token, "raise_if_cancelled", None)
    if callable(checker):
        checker()
        return
    probe = getattr(cancel_token, "is_cancelled", None)
    if callable(probe) and probe():
        raise SegmentationCanceled()


def _coerce_torch_device(device_like):
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, str):
        try:
            return torch.device(device_like)
        except (TypeError, ValueError):
            return None
    return None


def _quantization_device(device_hint=None):
    candidate = _coerce_torch_device(device_hint)
    if candidate and candidate.type in {"cuda", "mps"}:
        if candidate.type == "cuda" and torch.cuda.is_available():
            return candidate
        if candidate.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return candidate
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _runtime_float_dtype(device_hint=None):
    """Prefer float16 tensors on GPU backends while retaining float32 on CPU."""
    device = _coerce_torch_device(device_hint)
    if device is not None and device.type in {"cuda", "mps"}:
        return torch.float16
    if device is None:
        try:
            if torch.cuda.is_available():
                return torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.float16
        except Exception:
            pass
    return torch.float32


def _quant_chunk_size(device, num_clusters):
    base = 262144 if device.type == "cuda" else 131072
    scale = max(1, num_clusters // 8)
    return max(2048, base // scale)


def _gpu_cluster_assignments(data, centers, device, status_callback=None):
    if data.size == 0 or centers.size == 0:
        return None
    dtype = _runtime_float_dtype(device)
    try:
        centers_tensor = torch.as_tensor(centers, dtype=dtype, device=device)
        centers_norm = centers_tensor.pow(2).sum(dim=1)
        data_tensor = torch.as_tensor(data, dtype=dtype)
    except Exception:
        return None

    chunk = _quant_chunk_size(device, centers_tensor.shape[0])
    total = data_tensor.shape[0]
    outputs = np.empty(total, dtype=np.int64)

    try:
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            batch = data_tensor[start:end].to(device, non_blocking=True)
            batch_norm = batch.pow(2).sum(dim=1, keepdim=True)
            distances = batch_norm - 2.0 * batch @ centers_tensor.T + centers_norm.unsqueeze(0)
            idx = torch.argmin(distances, dim=1).cpu().numpy().astype(np.int64)
            outputs[start:end] = idx
            if status_callback:
                percent = int((end / max(total, 1)) * 100)
                status_callback(f"GPU cluster assignment {percent}% complete.")
    except RuntimeError:
        return None

    return outputs


def _materialize_raster(raster_input):
    if isinstance(raster_input, np.ndarray):
        return raster_input
    if callable(raster_input):
        materialized = raster_input()
        if not isinstance(materialized, np.ndarray):
            raise TypeError("Raster loader must return a numpy.ndarray")
        return materialized
    if isinstance(raster_input, str):
        source = raster_input.split("|")[0]
        try:
            from osgeo import gdal  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires QGIS runtime
            raise RuntimeError("GDAL is required to read raster sources.") from exc
        dataset = gdal.Open(source)
        if dataset is None:
            raise RuntimeError(f"Unable to open raster source: {source}")
        array = dataset.ReadAsArray()
        dataset = None
        if array is None:
            raise RuntimeError("Raster source returned no data.")
        return np.ascontiguousarray(array)
    raise TypeError(f"Unsupported raster input type: {type(raster_input)!r}")


def _materialize_model(model_or_loader, device):
    model: Any = model_or_loader() if callable(model_or_loader) else model_or_loader
    if model is None:
        raise RuntimeError("CNN model provider returned no model instance.")
    to_fn = getattr(model, "to", None)
    if callable(to_fn) and device is not None:
        try:
            model = to_fn(device)
        except Exception:  # pragma: no cover - best effort device placement
            pass
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        try:
            model = eval_fn()
        except Exception:
            pass
    return model


def _apply_texture_autoencoder(
    raster: np.ndarray,
    labels: np.ndarray,
    texture_manager: Optional[TextureAutoencoderManager],
    status_callback,
    cancel_token,
):
    if texture_manager is None:
        raise RuntimeError("Texture autoencoder manager is required for segmentation output.")
    _maybe_raise_cancel(cancel_token)
    remap = texture_manager.refresh_and_remap(
        raster,
        labels,
        status_callback=status_callback,
    )
    _maybe_raise_cancel(cancel_token)
    if remap is None:
        raise RuntimeError("Texture autoencoder failed to produce a remapped result.")
    _emit_status(status_callback, "Texture autoencoder remap applied.")
    return remap


def blur_segmentation_map(
    labels: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    status_callback=None,
    cancel_token=None,
):
    """Lightweight label blur that smooths jagged masks without altering class IDs."""
    if labels is None or labels.size == 0 or labels.ndim != 2:
        return labels
    kernel = int(max(1, kernel_size))
    iterations = int(max(1, iterations))
    if kernel <= 1 or iterations <= 0:
        return labels
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    labels_src = np.ascontiguousarray(labels)
    num_segments = int(labels_src.max()) + 1
    if num_segments <= 0:
        return labels
    _maybe_raise_cancel(cancel_token)
    dtype = _runtime_float_dtype(None)
    tensor = torch.from_numpy(labels_src.astype(np.int64, copy=False))
    one_hot = F.one_hot(tensor, num_classes=num_segments).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    weight = torch.ones(num_segments, 1, kernel, kernel, dtype=one_hot.dtype) / float(kernel * kernel)
    result = one_hot
    for idx in range(iterations):
        _maybe_raise_cancel(cancel_token)
        padded = F.pad(result, (pad, pad, pad, pad), mode="replicate")
        result = F.conv2d(padded, weight, groups=result.shape[1])
        _emit_status(
            status_callback,
            f"Legacy blur smoothing {idx + 1}/{iterations} ({int(((idx + 1) / max(iterations, 1)) * 100)}% complete, kernel={kernel}px).",
        )
    blurred = torch.argmax(result, dim=1).squeeze(0).to(tensor.dtype)
    return blurred.cpu().numpy()


def _apply_legacy_blur(labels, blur_config, status_callback, cancel_token=None):
    if blur_config is None:
        return labels
    kernel = int(max(1, blur_config.get("kernel_size", 1) or 1))
    iterations = int(max(1, blur_config.get("iterations", 1) or 1))
    if kernel <= 1 or iterations <= 0:
        return labels
    return blur_segmentation_map(
        labels,
        kernel,
        iterations,
        status_callback=status_callback,
        cancel_token=cancel_token,
    )


def legacy_kmeans_segmentation(
    array,
    num_segments,
    resolution,
    status_callback=None,
    blur_config: Optional[Dict[str, int]] = None,
    sample_scale: float = 1.0,
    device_hint=None,
    cancel_token=None,
):
    """Legacy main-branch flow: run basic K-Means and blur the output for stability."""
    _emit_status(status_callback, "Running legacy K-Means segmentation flow...")
    materialized = _materialize_raster(array)
    labels = predict_kmeans(
        materialized,
        num_segments=num_segments,
        resolution=resolution,
        status_callback=status_callback,
        sample_scale=sample_scale,
        cancel_token=cancel_token,
        device_hint=device_hint,
    )
    return _apply_legacy_blur(labels, blur_config, status_callback, cancel_token=cancel_token)


def legacy_cnn_segmentation(
    cnn_model,
    array,
    num_segments,
    tile_size,
    device,
    status_callback=None,
    blur_config: Optional[Dict[str, int]] = None,
    heuristic_overrides: Optional[Dict[str, object]] = None,
    profile_tier: Optional[str] = None,
    cancel_token=None,
):
    """Legacy main-branch flow for CNN inference with blur post-processing."""
    _emit_status(status_callback, "Running legacy CNN segmentation flow...")
    raster = _materialize_raster(array)
    model = _materialize_model(cnn_model, device)
    overrides = heuristic_overrides or {}
    tile_override = overrides.get("tile_size")
    if isinstance(tile_override, (int, float)):
        tile_size = int(np.clip(tile_override, MIN_TILE_SIZE, MAX_TILE_SIZE))
    latent_knn_config = overrides.get("latent_knn")
    labels = predict_cnn(
        model,
        raster,
        num_segments,
        tile_size=tile_size,
        device=device,
        status_callback=status_callback,
        profile_tier=profile_tier,
        latent_knn_config=latent_knn_config if isinstance(latent_knn_config, dict) else None,
        cancel_token=cancel_token,
    )
    return _apply_legacy_blur(labels, blur_config, status_callback, cancel_token=cancel_token)

DEFAULT_MEMORY_BUDGET = 128 * 1024 * 1024

# VRAM budget ratios based on backend characteristics.
# These conservative fractions (<=1%) leave headroom for runtime allocations and UI rendering.
VRAM_RATIO_CUDA = 0.009   # 0.9% of free CUDA memory
VRAM_RATIO_MPS = 0.0075   # 0.75% of free MPS memory
VRAM_RATIO_CPU = 0.01     # 1.0% of available system RAM

# Tile size bounds for chunk processing.
# MIN_TILE_SIZE: Minimum tile size in pixels to ensure meaningful feature extraction.
# Below 128px, CNN receptive fields may not capture enough context.
MIN_TILE_SIZE = 128
# MAX_TILE_SIZE: Maximum tile size in pixels to prevent excessive memory usage per chunk.
# Larger tiles increase memory pressure; 512px balances quality and efficiency.
MAX_TILE_SIZE = 512
SMOOTH_CHUNK_BYTES = 256 * 1024 * 1024
LATENT_KNN_DEFAULTS = {
    "enabled": True,
    "neighbors": 12,
    "temperature": 2.0,
    "mix": 0.65,
    "iterations": 2,
    "spatial_weight": 0.08,
    "chunk_size": 65536,
    "index_points": 120000,
    "query_batch": 32768,
    "hierarchy_factor": 1,
    "hierarchy_passes": 1,
    "mixed_precision_smoothing": True,
}


@dataclass
class ChunkPlan:
    chunk_size: int
    overlap: int
    budget_bytes: int
    ratio: float
    prefetch_depth: int

    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")
    @property
    def stride(self) -> int:
        return max(1, self.chunk_size - self.overlap)

    def should_chunk(self, height: int, width: int) -> bool:
        return height > self.chunk_size or width > self.chunk_size


try:  # Optional dependency for better memory stats
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class AdaptiveSettings:
    safety_factor: int = 8
    prefetch_depth: int = 2


_ADAPTIVE_SETTINGS_MAP: Dict[str, AdaptiveSettings] = {"default": AdaptiveSettings()}
_ADAPTIVE_OPTIONS_MAP: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {"default": []}
_ADAPTIVE_DEFAULT_TIER = "default"


def _copy_setting(settings: AdaptiveSettings) -> AdaptiveSettings:
    return AdaptiveSettings(safety_factor=settings.safety_factor, prefetch_depth=settings.prefetch_depth)


def get_adaptive_settings(memory_bytes: Optional[int] = None, tier: Optional[str] = None) -> AdaptiveSettings:
    tier_key = tier or _ADAPTIVE_DEFAULT_TIER
    settings = _ADAPTIVE_SETTINGS_MAP.get(tier_key) or _ADAPTIVE_SETTINGS_MAP.get(_ADAPTIVE_DEFAULT_TIER)
    if settings is None:
        settings = AdaptiveSettings()
    options = _ADAPTIVE_OPTIONS_MAP.get(tier_key, [])
    if memory_bytes is not None and options:
        for threshold, opt_settings in reversed(options):
            if memory_bytes >= threshold:
                return opt_settings
        return options[0][1]
    return settings


def set_adaptive_settings(
    settings: Optional[Dict[str, AdaptiveSettings]] = None,
    options: Optional[Dict[str, List[Tuple[int, AdaptiveSettings]]]] = None,
    default_tier: Optional[str] = None,
) -> None:
    global _ADAPTIVE_SETTINGS_MAP, _ADAPTIVE_OPTIONS_MAP, _ADAPTIVE_DEFAULT_TIER

    if settings is None:
        settings = {"default": AdaptiveSettings()}
    elif not isinstance(settings, dict):  # backward compatibility
        settings = {default_tier or "default": settings}

    _ADAPTIVE_SETTINGS_MAP = {tier: _copy_setting(cfg) for tier, cfg in settings.items()}

    if options is None:
        _ADAPTIVE_OPTIONS_MAP = {tier: [] for tier in _ADAPTIVE_SETTINGS_MAP}
    else:
        normalized: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {}
        for tier, entries in options.items():
            sorted_entries = sorted(entries, key=lambda item: item[0])
            normalized[tier] = [
                (int(max(0, threshold)), _copy_setting(cfg))
                for threshold, cfg in sorted_entries
            ]
        for tier in _ADAPTIVE_SETTINGS_MAP:
            normalized.setdefault(tier, [])
        _ADAPTIVE_OPTIONS_MAP = normalized

    if default_tier and default_tier in _ADAPTIVE_SETTINGS_MAP:
        _ADAPTIVE_DEFAULT_TIER = default_tier
    elif _ADAPTIVE_DEFAULT_TIER not in _ADAPTIVE_SETTINGS_MAP:
        _ADAPTIVE_DEFAULT_TIER = next(iter(_ADAPTIVE_SETTINGS_MAP))


def get_adaptive_options(tier: Optional[str] = None) -> List[Tuple[int, AdaptiveSettings]]:
    tier_key = tier or _ADAPTIVE_DEFAULT_TIER
    return list(_ADAPTIVE_OPTIONS_MAP.get(tier_key, []))


def _emit_status(callback, message):
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass  # Silently ignore callback errors to avoid interrupting processing


def _process_in_chunks(
    array,
    plan,
    num_segments,
    infer_fn,
    status_callback,
    smoothing_scale=1.0,
    cancel_token=None,
):
    height, width = array.shape[1], array.shape[2]
    if not plan.should_chunk(height, width):
        return infer_fn(array)

    stride = plan.stride
    y_starts = _compute_chunk_starts(height, plan.chunk_size, stride)
    x_starts = _compute_chunk_starts(width, plan.chunk_size, stride)
    total = len(y_starts) * len(x_starts)
    aggregator = _ChunkAggregator(
        height,
        width,
        num_segments,
        plan.chunk_size,
        status_callback=status_callback,
        smoothing_scale=smoothing_scale,
    )

    for idx, (y, x) in enumerate(itertools.product(y_starts, x_starts), start=1):
        _maybe_raise_cancel(cancel_token)
        y_end = min(y + plan.chunk_size, height)
        x_end = min(x + plan.chunk_size, width)
        chunk = array[:, y:y_end, x:x_end]
        if status_callback:
            status_callback(f"Chunk {idx}/{total}: rows {y}-{y_end}, cols {x}-{x_end}")
        inference_result = infer_fn(chunk)
        labels, scores = _normalize_inference_output(inference_result)
        aggregator.add(labels, (y, x, y_end, x_end), chunk_data=chunk, scores=scores)

    if status_callback:
        megapixels = (height * width) / 1_000_000
        status_callback(
            f"Smoothing out {total} chunks (~{megapixels:.2f} MP of coverage)..."
        )
    _maybe_raise_cancel(cancel_token)
    return aggregator.finalize()


def _compute_chunk_starts(length, chunk_size, stride):
    if length <= chunk_size:
        return [0]
    starts = list(range(0, max(1, length - chunk_size), stride))
    last_start = length - chunk_size
    if last_start > 0 and (not starts or starts[-1] != last_start):
        starts.append(last_start)
    return sorted(set(starts))


def _normalize_inference_output(result):
    scores = None
    labels = result
    if isinstance(result, dict):
        labels = result.get("labels")
        scores = result.get("scores")
    elif isinstance(result, (list, tuple)) and len(result) == 2:
        labels, scores = result
    if labels is None:
        raise ValueError("Inference result must include labels.")
    return labels, scores


def _label_to_one_hot(label_map, num_segments):
    labels = np.clip(label_map.astype(np.int64, copy=False), 0, num_segments - 1)
    one_hot = np.eye(num_segments, dtype=np.float32)[labels]
    return one_hot.transpose(2, 0, 1)


def _derive_chunk_size(array_shape, device, profile_tier: Optional[str] = None):
    channels = array_shape[0]
    free_bytes = _free_vram_bytes(device)
    if device.type == "cuda":
        ratio = VRAM_RATIO_CUDA
    elif device.type == "mps":
        ratio = VRAM_RATIO_MPS
    else:
        ratio = VRAM_RATIO_CPU
    budget = max(int(free_bytes * ratio), 64 * 1024 * 1024)
    bytes_per_pixel = channels * 4
    settings = get_adaptive_settings(free_bytes, tier=profile_tier)
    safety = settings.safety_factor
    max_pixels = max(budget // (bytes_per_pixel * safety), 1)
    tile_side = int(math.sqrt(max_pixels))
    tile_side = max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, tile_side))
    return tile_side, budget, ratio, settings


def _free_vram_bytes(device):
    if device.type == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(device.index or torch.cuda.current_device())
        return free
    return _system_available_memory()


def _system_available_memory():
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except (ValueError, AttributeError, OSError):  # pragma: no cover
        return 1_000_000_000  # Fallback 1 GB


class _ChunkAggregator:
    def __init__(
        self,
        height,
        width,
        num_segments,
        chunk_size,
        status_callback=None,
        smoothing_scale=1.0,
    ):
        self.height = height
        self.width = width
        self.num_segments = num_segments
        self.scores = np.zeros((num_segments, height, width), dtype=np.float32)
        self.weight = np.zeros((height, width), dtype=np.float32)
        self.weight_template = _build_weight_mask(chunk_size)
        self.chunk_size = chunk_size
        self._status_callback = status_callback
        self._smoothing_scale = max(0.1, float(smoothing_scale))
        # Track running color prototypes to align labels across tiles.
        self._palette_threshold = 24.0
        self._feature_dim = None
        self._prototype_vectors = None
        self._prototype_counts = np.zeros(num_segments, dtype=np.int64)

    def add(self, labels, region, chunk_data=None, scores=None):
        y0, x0, y1, x1 = region
        h = y1 - y0
        w = x1 - x0
        harmonized = self._harmonize_labels(labels[:h, :w], chunk_data)
        chunk = harmonized.astype(np.int64, copy=False)
        mask = self.weight_template[:h, :w]
        if scores is not None:
            chunk_scores = scores[:, :h, :w].astype(np.float32, copy=False)
            weighted_scores = chunk_scores * mask[np.newaxis, ...]
        else:
            one_hot = np.eye(self.num_segments, dtype=np.float32)[chunk]
            chunk_scores = one_hot.transpose(2, 0, 1)
            weighted_scores = chunk_scores * mask
        self.scores[:, y0:y1, x0:x1] += weighted_scores
        self.weight[y0:y1, x0:x1] += mask

    def finalize(self):
        sigma = max(1.0, min((self.chunk_size / 10.0) * self._smoothing_scale, 32.0))
        _emit_status(self._status_callback, "Smoothing CNN logits with GPU gradients...")
        smoothed_scores = _gaussian_blur_channels(
            self.scores,
            sigma,
            status_callback=lambda msg: _emit_status(self._status_callback, msg),
            stage_label="scores",
        )
        smoothed_weight = _gaussian_blur_channels(
            self.weight[np.newaxis, ...],
            sigma,
            status_callback=lambda msg: _emit_status(self._status_callback, msg),
            stage_label="weights",
        )[0]
        weight = np.maximum(smoothed_weight, 1e-6)
        probs = smoothed_scores / weight
        return np.argmax(probs, axis=0).astype(np.uint8)

    def _harmonize_labels(self, labels, chunk_data):
        if chunk_data is None:
            return labels
        label_vectors = self._extract_label_vectors(chunk_data, labels)
        if not label_vectors:
            return labels
        if self._feature_dim is None:
            sample = next(iter(label_vectors.values()))
            self._feature_dim = sample.shape[0]
            self._prototype_vectors = np.zeros((self.num_segments, self._feature_dim), dtype=np.float32)
        remapped = labels.copy()
        used_targets = set()
        for label_id, vector in label_vectors.items():
            target = self._select_target_index(vector, used_targets)
            if target is None:
                continue
            used_targets.add(target)
            self._update_prototype(target, vector)
            if target != label_id:
                remapped[labels == label_id] = target
        return remapped

    def _extract_label_vectors(self, chunk_data, labels):
        if chunk_data.ndim != 3:
            return {}
        channels = chunk_data.shape[0]
        flat_pixels = chunk_data.reshape(channels, -1).astype(np.float32, copy=False)
        flat_labels = labels.reshape(-1)
        vectors = {}
        for label_id in np.unique(flat_labels):
            mask = flat_labels == label_id
            if not np.any(mask):
                continue
            pixels = flat_pixels[:, mask]
            vectors[int(label_id)] = pixels.mean(axis=1)
        return vectors

    def _select_target_index(self, vector, used_targets):
        if self._prototype_vectors is None:
            return self._next_unused_index(used_targets)
        active_indices = [idx for idx, count in enumerate(self._prototype_counts) if count > 0 and idx not in used_targets]
        candidate = None
        min_distance = None
        for idx in active_indices:
            distance = float(np.linalg.norm(self._prototype_vectors[idx] - vector))
            if min_distance is None or distance < min_distance:
                min_distance = distance
                candidate = idx
        if candidate is not None and min_distance is not None and min_distance <= self._palette_threshold:
            return candidate
        fallback = self._next_unused_index(used_targets)
        if fallback is not None:
            return fallback
        return candidate

    def _update_prototype(self, idx, vector):
        if self._prototype_vectors is None:
            return
        current = self._prototype_counts[idx]
        vector = vector.astype(np.float32, copy=False)
        if current == 0:
            self._prototype_vectors[idx] = vector
        else:
            total = current + 1
            self._prototype_vectors[idx] = (self._prototype_vectors[idx] * current + vector) / total
        self._prototype_counts[idx] = current + 1

    def _next_unused_index(self, used_targets):
        for idx in range(self.num_segments):
            if self._prototype_counts[idx] == 0 and idx not in used_targets:
                return idx
        for idx in range(self.num_segments):
            if idx not in used_targets:
                return idx
        return None


def _resize_latent_map(latent_map, size):
    dtype = _runtime_float_dtype(None)
    tensor = torch.from_numpy(latent_map).unsqueeze(0).to(dtype=dtype)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()


def _resize_label_map(label_map, size):
    dtype = _runtime_float_dtype(None)
    tensor = torch.from_numpy(label_map.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return resized.squeeze().cpu().numpy().astype(np.int32, copy=False)


def _stratified_sample_indices(labels, max_points):
    total = labels.size
    if total <= max_points:
        return None
    flat = labels.reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    proportions = counts / total
    allocations = np.maximum(1, (proportions * max_points).astype(int))
    diff = allocations.sum() - max_points
    # Adjust allocations so the total matches max_points
    while diff > 0:
        for idx in range(len(allocations)):
            if allocations[idx] > 1 and diff > 0:
                allocations[idx] -= 1
                diff -= 1
    samples = []
    rng = np.random.default_rng()
    for value, take in zip(unique, allocations):
        candidates = np.where(flat == value)[0]
        take = min(take, candidates.size)
        if take <= 0:
            continue
        samples.append(rng.choice(candidates, size=take, replace=False))
    if not samples:
        return None
    combined = np.concatenate(samples)
    if combined.size > max_points:
        combined = combined[:max_points]
    return np.sort(combined)


def _latent_knn_soft_refine(
    latent_map,
    lowres_labels,
    centers,
    num_segments,
    status_callback=None,
    config=None,
    cancel_token=None,
    return_posteriors=False,
):
    cfg = dict(LATENT_KNN_DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", False):
        result = lowres_labels.astype(np.uint8, copy=False)
        if return_posteriors:
            return result, None
        return result

    h, w = lowres_labels.shape
    factor = max(int(cfg.get("hierarchy_factor", 1)), 1)
    passes = max(int(cfg.get("hierarchy_passes", 1)), 1)
    seed_labels = lowres_labels
    if factor > 1 and passes > 1:
        for level in range(passes - 1, 0, -1):
            _maybe_raise_cancel(cancel_token)
            scale = factor * level
            if min(h, w) < scale:
                continue
            coarse_h = max(1, h // scale)
            coarse_w = max(1, w // scale)
            coarse_latent = _resize_latent_map(latent_map, (coarse_h, coarse_w))
            coarse_labels = _resize_label_map(seed_labels, (coarse_h, coarse_w))
            coarse_refined = _latent_knn_core(
                coarse_latent,
                coarse_labels,
                centers,
                num_segments,
                cfg,
                status_callback,
                stage_label=f"coarse@{scale}",
                cancel_token=cancel_token,
            )
            seed_labels = _resize_label_map(coarse_refined, (h, w))

    refined = _latent_knn_core(
        latent_map,
        seed_labels,
        centers,
        num_segments,
        cfg,
        status_callback,
        stage_label="fine",
        cancel_token=cancel_token,
        return_posteriors=return_posteriors,
    )
    return refined


def _latent_knn_core(
    latent_map,
    seed_labels,
    centers,
    num_segments,
    cfg,
    status_callback,
    stage_label="latent",
    cancel_token=None,
    return_posteriors=False,
):
    h, w = seed_labels.shape
    channels = latent_map.shape[0]
    _maybe_raise_cancel(cancel_token)

    vectors = latent_map.reshape(channels, h * w).transpose(1, 0).astype(np.float32, copy=False)
    labels_stack = np.clip(seed_labels.reshape(-1), 0, num_segments - 1)
    coords_y, coords_x = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([coords_y.reshape(-1), coords_x.reshape(-1)], axis=1)
    base_size = vectors.shape[0]
    spatial_weight = float(cfg.get("spatial_weight", 0.0))
    augmented = np.concatenate([vectors, coords * spatial_weight], axis=1)

    temperature = max(float(cfg.get("temperature", 1.0)), 1e-6)
    diffs = vectors[:, None, :] - centers[None, :, :]
    logits = -np.sum(diffs * diffs, axis=2) / temperature
    posteriors = _softmax(logits)

    base = np.eye(num_segments, dtype=np.float32)[labels_stack]
    posteriors = 0.5 * posteriors + 0.5 * base

    neighbors = int(cfg.get("neighbors", 8))
    index_cap = int(cfg.get("index_points", augmented.shape[0]))
    sample_idx = _stratified_sample_indices(labels_stack.reshape(-1), index_cap)
    index_data = augmented if sample_idx is None else augmented[sample_idx]
    nn = NearestNeighbors(n_neighbors=min(neighbors, index_data.shape[0]), algorithm="auto")
    nn.fit(index_data)

    query_batch = max(int(cfg.get("query_batch", 32768)), neighbors)
    total = augmented.shape[0]
    neighbor_idx = np.empty((total, neighbors), dtype=np.int64)
    for start in range(0, total, query_batch):
        end = min(start + query_batch, total)
        idx_chunk = nn.kneighbors(augmented[start:end], return_distance=False)
        if sample_idx is not None:
            idx_chunk = sample_idx[idx_chunk]
        neighbor_idx[start:end] = idx_chunk

    mix = float(cfg.get("mix", 0.5))
    iterations = max(int(cfg.get("iterations", 1)), 1)
    chunk_size = max(int(cfg.get("chunk_size", 32768)), 1024)

    for iteration in range(iterations):
        neighbor_probs = np.empty_like(posteriors)
        for start in range(0, neighbor_idx.shape[0], chunk_size):
            end = min(start + chunk_size, neighbor_idx.shape[0])
            chunk = neighbor_idx[start:end]
            neighbor_probs[start:end] = posteriors[chunk].mean(axis=1)
        posteriors = (1.0 - mix) * posteriors + mix * neighbor_probs
        percent = int(((iteration + 1) / max(iterations, 1)) * 100)
        _emit_status(
            status_callback,
            f"Latent KNN refinement ({stage_label}) {iteration + 1}/{iterations} ({percent}% complete).",
        )

    refined_base = np.argmax(posteriors[:base_size], axis=1)
    refined_map = refined_base.reshape(h, w).astype(np.uint8)
    if return_posteriors:
        base_scores = posteriors[:base_size]
        scores = base_scores.reshape(h, w, num_segments).transpose(2, 0, 1).astype(np.float32, copy=False)
        return refined_map, scores
    return refined_map


def _softmax(x):
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-8)


def _gaussian_blur_channels(array, sigma, status_callback=None, stage_label="scores"):
    if sigma <= 0:
        return array
    return _chunked_gaussian_blur(array, sigma, status_callback=status_callback, stage_label=stage_label)


def _chunked_gaussian_blur(
    array,
    sigma,
    max_chunk_bytes=SMOOTH_CHUNK_BYTES,
    status_callback=None,
    stage_label="scores",
):
    sigma = max(float(sigma), 1e-6)
    channels, height, width = array.shape
    radius = int(max(1, round(3 * sigma)))
    device = _smoothing_device()
    preferred_dtype = _runtime_float_dtype(device)
    use_fp16 = preferred_dtype == torch.float16 and LATENT_KNN_DEFAULTS.get("mixed_precision_smoothing", True)
    dtype = torch.float16 if use_fp16 else torch.float32
    kernel = _build_gaussian_kernel(radius, sigma, channels, device, dtype=dtype)
    result = np.zeros_like(array, dtype=np.float32)
    bytes_per_row = max(channels * width * 4, 1)
    chunk_rows = max(radius * 2 + 1, int(max_chunk_bytes // bytes_per_row))
    chunk_rows = max(chunk_rows, radius * 2 + 1)

    start = 0
    total_chunks = max(1, math.ceil(height / chunk_rows))
    chunk_index = 0
    while start < height:
        end = min(height, start + chunk_rows)
        pad_top = min(radius, start)
        pad_bottom = min(radius, height - end)
        slice_start = start - pad_top
        slice_end = min(height, end + pad_bottom)
        chunk = array[:, slice_start:slice_end, :]
        tensor = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=dtype)
        padded = F.pad(tensor, (radius, radius, radius, radius), mode="reflect")
        smoothed = F.conv2d(padded, kernel, groups=channels).squeeze(0)
        usable = smoothed[:, pad_top : pad_top + (end - start), :].to(dtype)
        result[:, start:end, :] = usable.detach().cpu().numpy()
        start = end
        chunk_index += 1
        if status_callback:
            percent = int((chunk_index / total_chunks) * 100)
            status_callback(
                f"GPU smoothing [{stage_label}] chunk {chunk_index}/{total_chunks} ({percent}% complete)."
            )

    return result.astype(np.float32)


def _build_gaussian_kernel(radius, sigma, channels, device, dtype=torch.float32):
    if radius <= 0:
        kernel = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
        return kernel.repeat(channels, 1, 1, 1)
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
    return kernel.repeat(channels, 1, 1, 1)


def _smoothing_device():
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _build_weight_mask(size):
    if size <= 1:
        return np.ones((1, 1), dtype=np.float32)
    window = np.hanning(size)
    assert np.max(window) != 0, "np.hanning(size) returned all zeros, which should never happen for size > 1"
    mask = np.outer(window, window)
    mask = mask / np.max(mask)
    return mask.astype(np.float32)


def recommended_chunk_plan(array_shape, device, profile_tier: Optional[str] = None):
    """Advanced helper retained for future optimizations (unused in legacy flow)."""
    chunk_size, budget, ratio, settings = _derive_chunk_size(array_shape, device, profile_tier=profile_tier)
    overlap = 0
    return ChunkPlan(
        chunk_size=chunk_size,
        overlap=overlap,
        budget_bytes=budget,
        ratio=ratio,
        prefetch_depth=settings.prefetch_depth,
    )


def execute_kmeans_segmentation(
    array,
    num_segments,
    resolution,
    chunk_plan,
    status_callback=None,
    texture_manager: Optional[TextureAutoencoderManager] = None,
    device_hint=None,
    cancel_token=None,
):
    """Advanced chunked K-Means path retained for future work (unused in legacy flow)."""
    _maybe_raise_cancel(cancel_token)
    height, width = array.shape[1], array.shape[2]
    if chunk_plan and chunk_plan.should_chunk(height, width):
        labels = _process_in_chunks(
            array,
            chunk_plan,
            num_segments,
            lambda data: predict_kmeans(
                data,
                num_segments,
                resolution,
                status_callback=status_callback,
                cancel_token=cancel_token,
                device_hint=device_hint,
            ),
            status_callback,
            smoothing_scale=1.0,
            cancel_token=cancel_token,
        )
    else:
        labels = predict_kmeans(
            array,
            num_segments,
            resolution,
            status_callback=status_callback,
                cancel_token=cancel_token,
                device_hint=device_hint,
        )
    return _apply_texture_autoencoder(
        array,
        labels,
        texture_manager,
        status_callback,
        cancel_token,
    )


def execute_cnn_segmentation(
    cnn_model,
    array,
    num_segments,
    chunk_plan,
    tile_size,
    device,
    status_callback=None,
    profile_tier: Optional[str] = None,
    heuristic_overrides: Optional[Dict[str, object]] = None,
    texture_manager: Optional[TextureAutoencoderManager] = None,
    cancel_token=None,
):
    """Advanced chunked CNN path retained for future work (unused in legacy flow)."""
    _maybe_raise_cancel(cancel_token)
    heuristic_overrides = heuristic_overrides or {}
    raw_tile_size = heuristic_overrides.get("tile_size")
    tile_size_override = int(raw_tile_size) if isinstance(raw_tile_size, (int, float)) else 0
    raw_smoothing = heuristic_overrides.get("smoothing_scale")
    smoothing_scale = float(raw_smoothing) if isinstance(raw_smoothing, (int, float)) else 1.0
    raw_latent = heuristic_overrides.get("latent_knn")
    latent_knn_config = raw_latent if isinstance(raw_latent, dict) else None

    if tile_size_override:
        tile_size = max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, tile_size_override))

    height, width = array.shape[1], array.shape[2]
    effective_tile = min(tile_size, chunk_plan.chunk_size) if chunk_plan else tile_size

    def _tile_for_data(data):
        size = min(effective_tile, data.shape[1], data.shape[2])
        return max(64, size)

    if chunk_plan and chunk_plan.should_chunk(height, width):
        result = _process_in_chunks(
            array,
            chunk_plan,
            num_segments,
            lambda data: predict_cnn(
                cnn_model,
                data,
                num_segments,
                tile_size=_tile_for_data(data),
                device=device,
                status_callback=status_callback,
                memory_budget=chunk_plan.budget_bytes,
                prefetch_depth=chunk_plan.prefetch_depth,
                profile_tier=profile_tier,
                latent_knn_config=latent_knn_config,
                cancel_token=cancel_token,
                return_scores=True,
            ),
            status_callback,
            smoothing_scale=smoothing_scale,
            cancel_token=cancel_token,
        )
    else:
        result = predict_cnn(
            cnn_model,
            array,
            num_segments,
            tile_size=_tile_for_data(array),
            device=device,
            status_callback=status_callback,
            memory_budget=chunk_plan.budget_bytes if chunk_plan else None,
            prefetch_depth=chunk_plan.prefetch_depth if chunk_plan else None,
            profile_tier=profile_tier,
            latent_knn_config=latent_knn_config,
            cancel_token=cancel_token,
        )

    return _apply_texture_autoencoder(
        array,
        result,
        texture_manager,
        status_callback,
        cancel_token,
    )

# Predict coverage map using kmeans
def predict_kmeans(
    array,
    num_segments=8,
    resolution=16,
    status_callback=None,
    sample_scale: float = 1.0,
    return_lowres=False,
    return_centers=False,
    cancel_token=None,
    device_hint=None,
):
    _maybe_raise_cancel(cancel_token)
    device = _quantization_device(device_hint)
    # Instantiate kmeans model
    kmeans = KMeans(n_clusters=num_segments)
    _emit_status(
        status_callback,
        f"K-Means initialized ({num_segments} clusters, resolution={resolution}).",
    )

    # Pad to resolution
    channel_pad = (0, 0)
    height_pad = (0, resolution - array.shape[1] % resolution)
    width_pad = (0, resolution - array.shape[2] % resolution)
    _maybe_raise_cancel(cancel_token)
    array_padded = np.pad(
        array, (channel_pad, height_pad, width_pad), mode="constant"
    )
    _emit_status(status_callback, "Raster padded for block processing.")

    # Reshape into 2d
    array_2d = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // resolution,
        resolution,
        array_padded.shape[2] // resolution,
        resolution,
    )
    array_2d = array_2d.transpose(1, 3, 0, 2, 4)
    array_2d = array_2d.reshape(
        array_2d.shape[0] * array_2d.shape[1],
        array_2d.shape[2] * resolution * resolution,
    )
    _emit_status(status_callback, f"Prepared {array_2d.shape[0]} blocks for clustering.")

    # Fit kmeans model to random subset
    sample_scale = float(np.clip(sample_scale, 0.3, 2.0))
    max_sample = int(round(10000 * sample_scale))
    size = min(array_2d.shape[0], max_sample if max_sample > 0 else array_2d.shape[0])
    size = max(200, size)
    idx = np.random.randint(0, array_2d.shape[0], size=size)
    _emit_status(status_callback, f"Fitting K-Means on {size} sampled blocks.")
    _maybe_raise_cancel(cancel_token)
    kmeans = kmeans.fit(array_2d[idx])

    # Get clusters
    _maybe_raise_cancel(cancel_token)
    clusters = None
    if device is not None:
        backend = device.type.upper()
        _emit_status(status_callback, f"Assigning clusters on {backend} backend...")
        clusters = _gpu_cluster_assignments(
            array_2d,
            kmeans.cluster_centers_,
            device,
            status_callback=status_callback,
        )
        if clusters is None:
            _emit_status(status_callback, "GPU assignment unavailable; falling back to CPU.")
    if clusters is None:
        clusters = kmeans.predict(array_2d)
    clusters = _normalize_cluster_labels(clusters, kmeans.cluster_centers_)
    _emit_status(status_callback, "Cluster assignment complete.")

    # Reshape clusters to match map
    clusters = clusters.reshape(
        1,
        1,
        array_padded.shape[1] // resolution,
        array_padded.shape[2] // resolution,
    )

    # Get rid of padding
    clusters = clusters[
        :, :, : array.shape[1] // resolution, : array.shape[2] // resolution
    ]

    lowres = clusters.copy()

    # Upsample to original size
    _maybe_raise_cancel(cancel_token)
    clusters = torch.tensor(clusters)
    clusters = torch.nn.Upsample(
        size=(array.shape[-2], array.shape[-1]), mode="nearest"
    )(clusters.byte())
    clusters = clusters.detach().squeeze()

    _emit_status(status_callback, "K-Means output upsampled to raster resolution.")
    result = clusters.cpu().numpy()

    if not (return_lowres or return_centers):
        return result

    extras = []
    if return_lowres:
        extras.append(lowres.squeeze().astype(np.uint8))
    if return_centers:
        extras.append(kmeans.cluster_centers_.astype(np.float32))

    return (result, *extras)

def predict_cnn(
    cnn_model,
    array,
    num_segments,
    tile_size=256,
    device="cpu",
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
    profile_tier: Optional[str] = None,
    latent_knn_config: Optional[Dict[str, object]] = None,
    cancel_token=None,
    return_scores=False,
):
    _maybe_raise_cancel(cancel_token)
    device_obj = cast(torch.device, _coerce_torch_device(device) or torch.device("cpu"))
    compute_dtype = _runtime_float_dtype(device_obj)
    effective_budget = memory_budget or DEFAULT_MEMORY_BUDGET
    settings = get_adaptive_settings(effective_budget, tier=profile_tier)
    prefetch_depth = prefetch_depth or settings.prefetch_depth
    memory_budget = effective_budget

    tiles, (height_pad, width_pad), grid_shape = tile_raster(array, tile_size)
    _emit_status(
        status_callback,
        f"Raster tiled into {tiles.shape[0]} patches of size {tile_size}x{tile_size}.",
    )

    total_tiles = tiles.shape[0]
    batch_size = _recommended_batch_size(
        tiles.shape[1], tiles.shape[2], tiles.shape[3], memory_budget, settings=settings
    )
    batch_size = max(1, min(batch_size, total_tiles))
    _emit_status(
        status_callback,
        f"Adaptive batch size: {batch_size} tile(s) with prefetch depth {prefetch_depth}.",
    )
    coverage_map = []
    last_report = -1

    def _autocast_ctx():
        if compute_dtype != torch.float16:
            return nullcontext()
        if device_obj.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.amp.autocast(dtype=compute_dtype)
        if device_obj.type == "mps":
            try:
                return torch.autocast(device_type="mps", dtype=compute_dtype)
            except Exception:
                return nullcontext()
        return nullcontext()

    for tensor_batch, start, end, total in _prefetch_batches(
        tiles,
        batch_size,
        device_obj,
        depth=prefetch_depth,
        cancel_token=cancel_token,
        dtype=compute_dtype,
    ):
        _maybe_raise_cancel(cancel_token)
        with torch.no_grad():
            with _autocast_ctx():
                result = cnn_model.forward(tensor_batch)
        vectors = result[1].detach().cpu().numpy()
        coverage_map.append(vectors)
        percent = int((end / total) * 100)
        if percent // 5 > last_report:
            last_report = percent // 5
            _emit_status(status_callback, f"CNN inference {min(percent, 100)}% complete.")
    coverage_map = np.concatenate(coverage_map, axis=0)

    channels = coverage_map.shape[1]
    feat_h = coverage_map.shape[2]
    feat_w = coverage_map.shape[3]
    rows, cols = grid_shape
    coverage_map = coverage_map.transpose(1, 0, 2, 3)
    coverage_map = coverage_map.reshape(
        channels,
        rows,
        cols,
        feat_h,
        feat_w,
    )
    coverage_map = coverage_map.transpose(0, 1, 3, 2, 4)
    coverage_map = coverage_map.reshape(
        channels,
        rows * feat_h,
        cols * feat_w,
    )

    latent_grid = coverage_map.copy()
    kmeans_outputs = predict_kmeans(
        coverage_map,
        num_segments=num_segments,
        resolution=1,
        status_callback=status_callback,
        return_lowres=True,
        return_centers=True,
        cancel_token=cancel_token,
        device_hint=device_obj,
    )

    if isinstance(kmeans_outputs, tuple):
        _, lowres_labels, centers = kmeans_outputs
    else:
        lowres_labels = None
        centers = None

    lowres_scores = None
    if lowres_labels is not None and centers is not None:
        refined_output = _latent_knn_soft_refine(
            latent_grid,
            lowres_labels,
            centers,
            num_segments,
            status_callback=status_callback,
            config=latent_knn_config,
            cancel_token=cancel_token,
            return_posteriors=return_scores,
        )
        if return_scores:
            refined_lowres, lowres_scores = refined_output
        else:
            refined_lowres = refined_output
    elif lowres_labels is not None:
        refined_lowres = lowres_labels.astype(np.uint8, copy=False)
        if return_scores:
            lowres_scores = _label_to_one_hot(refined_lowres, num_segments)
    else:
        refined_lowres = kmeans_outputs[0] if isinstance(kmeans_outputs, tuple) else kmeans_outputs
        if return_scores:
            lowres_scores = _label_to_one_hot(refined_lowres, num_segments)

    coverage_map = torch.tensor(refined_lowres).unsqueeze(0).unsqueeze(0)
    coverage_map = torch.nn.Upsample(
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad), mode="nearest"
    )(coverage_map.byte())

    coverage_map = coverage_map[0, 0, : array.shape[1], : array.shape[2]]

    _emit_status(status_callback, "CNN segmentation map reconstructed.")
    _maybe_raise_cancel(cancel_token)
    labels_full = coverage_map.cpu().numpy()
    labels_full, rotation_plan = _auto_orient_tile_grid(labels_full, tile_size)
    if not return_scores:
        return labels_full

    if lowres_scores is None:
        lowres_scores = _label_to_one_hot(refined_lowres, num_segments)
    score_tensor = torch.from_numpy(lowres_scores).unsqueeze(0)
    score_tensor = F.interpolate(
        score_tensor,
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad),
        mode="bilinear",
        align_corners=False,
    )
    score_tensor = score_tensor[0, :, : array.shape[1], : array.shape[2]]
    scores_np = score_tensor.cpu().numpy()
    if rotation_plan is not None:
        scores_np = _apply_rotation_plan_to_volume(scores_np, tile_size, rotation_plan)
    return labels_full, scores_np

def tile_raster(array, tile_size):
    padding = lambda shape: 0 if shape % tile_size == 0 else tile_size - shape % tile_size
    channel_pad = (0, 0)
    height_pad = (0, padding(array.shape[1]))
    width_pad = (0, padding(array.shape[2]))
    array_padded = np.pad(
        array,
        (channel_pad, height_pad, width_pad),
        mode="constant",
    )

    tiles = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // tile_size,
        tile_size,
        array_padded.shape[2] // tile_size,
        tile_size,
    )
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    tiles = tiles.reshape(
        tiles.shape[0] * tiles.shape[1],
        array_padded.shape[0],
        tile_size,
        tile_size,
    )

    grid_shape = (
        array_padded.shape[1] // tile_size,
        array_padded.shape[2] // tile_size,
    )

    return tiles, (height_pad[1], width_pad[1]), grid_shape


ROTATION_ALIGNMENT_THRESHOLD = 0.6
ROTATION_IMPROVEMENT_MARGIN = 0.05


def _auto_orient_tile_grid(label_map: np.ndarray, tile_size: int):
    if tile_size <= 1:
        return label_map, None
    height, width = label_map.shape
    rows = max(1, math.ceil(height / tile_size))
    cols = max(1, math.ceil(width / tile_size))
    oriented = np.zeros_like(label_map)
    rotation_plan = np.zeros((rows, cols), dtype=np.uint8)
    changed = False

    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = label_map[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[0] != tile.shape[1]:
                oriented[y0:y1, x0:x1] = tile
                rotation_plan[r, c] = 0
                continue
            top_edge = oriented[y0 - 1, x0:x1] if y0 > 0 else None
            left_edge = oriented[y0:y1, x0 - 1] if x0 > 0 else None
            best_k, best_tile = _select_tile_rotation(tile, top_edge, left_edge)
            rotation_plan[r, c] = best_k
            if best_k != 0:
                changed = True
            oriented[y0:y1, x0:x1] = best_tile

    if not changed:
        return label_map, None
    return oriented, rotation_plan


def _select_tile_rotation(tile: np.ndarray, top_edge: Optional[np.ndarray], left_edge: Optional[np.ndarray]):
    best_k = 0
    best_score = float("-inf")
    best_tile = tile
    base_score = None
    for k in range(4):
        candidate = np.rot90(tile, k=k)
        score = 0.0
        if top_edge is not None and top_edge.size > 0:
            overlap = min(candidate.shape[1], top_edge.shape[-1])
            if overlap > 0:
                score += float(np.mean(candidate[0, :overlap] == top_edge[:overlap]))
        if left_edge is not None and left_edge.size > 0:
            overlap = min(candidate.shape[0], left_edge.shape[0])
            if overlap > 0:
                score += float(np.mean(candidate[:overlap, 0] == left_edge[:overlap]))
        if k == 0:
            base_score = score
        if score > best_score + 1e-6:
            best_score = score
            best_k = k
            best_tile = candidate
    if best_k != 0:
        baseline = 0.0 if base_score is None else base_score
        if best_score < ROTATION_ALIGNMENT_THRESHOLD:
            return 0, tile
        if (best_score - baseline) < ROTATION_IMPROVEMENT_MARGIN:
            return 0, tile
    return best_k, best_tile


def _apply_rotation_plan_to_volume(volume: np.ndarray, tile_size: int, plan: np.ndarray):
    rotated = np.zeros_like(volume)
    channels, height, width = volume.shape
    rows, cols = plan.shape
    for r in range(rows):
        y0 = r * tile_size
        if y0 >= height:
            break
        y1 = min(y0 + tile_size, height)
        for c in range(cols):
            x0 = c * tile_size
            if x0 >= width:
                break
            x1 = min(x0 + tile_size, width)
            tile = volume[:, y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.shape[1] != tile.shape[2]:
                rotated[:, y0:y1, x0:x1] = tile
                continue
            k = int(plan[r, c]) if plan is not None else 0
            if k == 0:
                rotated[:, y0:y1, x0:x1] = tile
            else:
                rotated[:, y0:y1, x0:x1] = np.rot90(tile, k=k, axes=(-2, -1))
    return rotated


def _normalize_cluster_labels(labels, centers):
    # Validate that all label indices are within bounds
    n_centers = centers.shape[0]
    flat = labels.reshape(-1)
    if not np.all((flat >= 0) & (flat < n_centers)):
        raise ValueError(
            f"All label indices must be in [0, {n_centers-1}]. Found out-of-bounds values: {flat[(flat < 0) | (flat >= n_centers)]}"
        )
    ordering = np.argsort(centers.mean(axis=1))
    mapping = np.zeros_like(ordering)
    mapping[ordering] = np.arange(ordering.size)
    flat = mapping[flat]
    return flat.reshape(labels.shape)


def _recommended_batch_size(channels, height, width, memory_budget, settings):
    # Ensure all dimensions are at least 1 to avoid division by zero
    channels = max(1, channels)
    height = max(1, height)
    width = max(1, width)
    bytes_per_tile = channels * height * width * 4
    # bytes_per_tile is guaranteed > 0 since all dimensions are clamped to >= 1
    budget = max(memory_budget or DEFAULT_MEMORY_BUDGET, bytes_per_tile)
    safety = max(1, settings.safety_factor)
    depth = max(1, settings.prefetch_depth)
    effective_budget = max(1, budget // safety)
    denom = max(bytes_per_tile * (1 + depth), 1)
    return max(1, effective_budget // denom)


def _prefetch_batches(tiles, batch_size, device, depth=2, cancel_token=None, dtype=None):
    total = tiles.shape[0]
    if total == 0:
        return
    target_dtype = dtype or _runtime_float_dtype(device)
    if device.type == "cuda" and torch.cuda.is_available():
        yield from _prefetch_batches_cuda(tiles, batch_size, device, cancel_token, dtype=target_dtype)
        return
    yield from _prefetch_batches_threaded(tiles, batch_size, device, depth, cancel_token, dtype=target_dtype)


def _prefetch_batches_threaded(tiles, batch_size, device, depth=2, cancel_token=None, dtype=None):
    total = tiles.shape[0]
    depth = max(1, depth or 1)
    executor = ThreadPoolExecutor(max_workers=depth)
    futures = deque()
    index = 0

    try:
        while index < total or futures:
            while index < total and len(futures) < depth:
                _maybe_raise_cancel(cancel_token)
                start = index
                end = min(start + batch_size, total)
                batch = tiles[start:end]
                future = executor.submit(_batch_to_tensor, batch, device, dtype)
                futures.append((future, start, end))
                index = end
            future, start, end = futures.popleft()
            try:
                _maybe_raise_cancel(cancel_token)
                yield future.result(), start, end, total
            except Exception:
                while futures:
                    leftover_future, _, _ = futures.popleft()
                    try:
                        leftover_future.result()
                    except Exception:
                        pass
                raise
    finally:
        executor.shutdown(wait=True)


def _prefetch_batches_cuda(tiles, batch_size, device, cancel_token=None, dtype=None):
    total = tiles.shape[0]
    if total == 0:
        return
    stream = torch.cuda.Stream(device=device)
    next_start = 0
    next_tensor = None
    next_bounds = None

    def _stage_copy(start, end):
        _maybe_raise_cancel(cancel_token)
        batch = torch.from_numpy(tiles[start:end])
        if not batch.is_floating_point():
            batch = batch.float().div_(255.0)
        else:
            batch = batch / 255.0
        if dtype is not None and batch.dtype != dtype:
            batch = batch.to(dtype)
        pinned = batch.pin_memory()
        with torch.cuda.stream(stream):
            gpu_tensor = pinned.to(device, non_blocking=True)
        del pinned
        return gpu_tensor

    if next_start < total:
        staged_end = min(next_start + batch_size, total)
        next_tensor = _stage_copy(next_start, staged_end)
        next_bounds = (next_start, staged_end)
        next_start = staged_end

    current_stream = torch.cuda.current_stream(device)
    while next_tensor is not None:
        current_stream.wait_stream(stream)
        tensor = next_tensor
        if next_bounds is None:
            break
        start, end = next_bounds
        if next_start < total:
            staged_end = min(next_start + batch_size, total)
            next_tensor = _stage_copy(next_start, staged_end)
            next_bounds = (next_start, staged_end)
            next_start = staged_end
        else:
            next_tensor = None
            next_bounds = None
        yield tensor, start, end, total


def _batch_to_tensor(batch, device, dtype):
    tensor = torch.from_numpy(batch)
    if not tensor.is_floating_point():
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor / 255.0
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.to(device)
