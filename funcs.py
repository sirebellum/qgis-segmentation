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


def _quant_chunk_size(device, num_clusters):
    base = 262144 if device.type == "cuda" else 131072
    scale = max(1, num_clusters // 8)
    return max(2048, base // scale)


def _gpu_cluster_assignments(data, centers, device, status_callback=None):
    if data.size == 0 or centers.size == 0:
        return None
    try:
        centers_tensor = torch.as_tensor(centers, dtype=torch.float32, device=device)
        centers_norm = centers_tensor.pow(2).sum(dim=1)
        data_tensor = torch.as_tensor(data, dtype=torch.float32)
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
    tensor = torch.from_numpy(labels_src.astype(np.int64, copy=False))
    one_hot = F.one_hot(tensor, num_classes=num_segments).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
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


def _resize_latent_map(latent_map, size):
    tensor = torch.from_numpy(latent_map).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).cpu().numpy()


def _resize_label_map(label_map, size):
    tensor = torch.from_numpy(label_map.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
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
    amp_enabled = bool(device_obj.type == "cuda" and torch.cuda.is_available())
    amp_context = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for tensor_batch, start, end, total in _prefetch_batches(
        tiles, batch_size, device_obj, depth=prefetch_depth, cancel_token=cancel_token
    ):
        _maybe_raise_cancel(cancel_token)
        with torch.no_grad():
            with amp_context():
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


def _prefetch_batches(tiles, batch_size, device, depth=2, cancel_token=None):
    total = tiles.shape[0]
    if total == 0:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        yield from _prefetch_batches_cuda(tiles, batch_size, device, cancel_token)
        return
    yield from _prefetch_batches_threaded(tiles, batch_size, device, depth, cancel_token)


def _prefetch_batches_threaded(tiles, batch_size, device, depth=2, cancel_token=None):
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
                future = executor.submit(_batch_to_tensor, batch, device)
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


def _prefetch_batches_cuda(tiles, batch_size, device, cancel_token=None):
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
            batch = batch.float()
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


def _batch_to_tensor(batch, device):
    tensor = torch.from_numpy(batch)
    if not tensor.is_floating_point():
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor / 255.0
    return tensor.to(device)
