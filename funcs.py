import itertools
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
        labels = infer_fn(chunk)
        aggregator.add(labels, (y, x, y_end, x_end))

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

    def add(self, labels, region):
        y0, x0, y1, x1 = region
        h = y1 - y0
        w = x1 - x0
        chunk = labels[:h, :w].astype(np.int64, copy=False)
        mask = self.weight_template[:h, :w]
        one_hot = np.eye(self.num_segments, dtype=np.float32)[chunk]
        one_hot = one_hot.transpose(2, 0, 1)
        self.scores[:, y0:y1, x0:x1] += one_hot * mask
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
):
    cfg = dict(LATENT_KNN_DEFAULTS)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", False):
        return lowres_labels.astype(np.uint8, copy=False)

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
):
    h, w = seed_labels.shape
    channels = latent_map.shape[0]
    _maybe_raise_cancel(cancel_token)
    vectors = latent_map.reshape(channels, h * w).transpose(1, 0).astype(np.float32, copy=False)

    coords_y, coords_x = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([coords_y.reshape(-1), coords_x.reshape(-1)], axis=1)
    spatial_weight = float(cfg.get("spatial_weight", 0.0))
    augmented = np.concatenate([vectors, coords * spatial_weight], axis=1)

    temperature = max(float(cfg.get("temperature", 1.0)), 1e-6)
    diffs = vectors[:, None, :] - centers[None, :, :]
    logits = -np.sum(diffs * diffs, axis=2) / temperature
    posteriors = _softmax(logits)

    base = np.eye(num_segments, dtype=np.float32)[np.clip(seed_labels.reshape(-1), 0, num_segments - 1)]
    posteriors = 0.5 * posteriors + 0.5 * base

    neighbors = int(cfg.get("neighbors", 8))
    index_cap = int(cfg.get("index_points", augmented.shape[0]))
    sample_idx = _stratified_sample_indices(seed_labels, index_cap)
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
        _emit_status(
            status_callback,
            f"Latent KNN refinement ({stage_label}) {iteration + 1}/{iterations} complete.",
        )

    refined = np.argmax(posteriors, axis=1)
    return refined.reshape(h, w).astype(np.uint8)


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
    use_fp16 = device.type in {"cuda", "mps"} and LATENT_KNN_DEFAULTS.get("mixed_precision_smoothing", True)
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
        usable = smoothed[:, pad_top : pad_top + (end - start), :].to(torch.float32)
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
    cancel_token=None,
):
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
    _maybe_raise_cancel(cancel_token)
    heuristic_overrides = heuristic_overrides or {}
    tile_size_override = int(heuristic_overrides.get("tile_size", 0) or 0)
    smoothing_scale = float(heuristic_overrides.get("smoothing_scale", 1.0) or 1.0)
    latent_knn_config = heuristic_overrides.get("latent_knn")

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
    num_segments=16,
    resolution=16,
    status_callback=None,
    return_lowres=False,
    return_centers=False,
    cancel_token=None,
):
    _maybe_raise_cancel(cancel_token)
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
    size = 10000 if array_2d.shape[0] > 10000 else array_2d.shape[0]
    idx = np.random.randint(0, array_2d.shape[0], size=size)
    _emit_status(status_callback, f"Fitting K-Means on {size} sampled blocks.")
    _maybe_raise_cancel(cancel_token)
    kmeans = kmeans.fit(array_2d[idx])

    # Get clusters
    _maybe_raise_cancel(cancel_token)
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
):
    _maybe_raise_cancel(cancel_token)
    effective_budget = memory_budget or DEFAULT_MEMORY_BUDGET
    settings = get_adaptive_settings(effective_budget, tier=profile_tier)
    prefetch_depth = prefetch_depth or settings.prefetch_depth
    memory_budget = effective_budget

    tiles, (height_pad, width_pad), grid_shape = tile_raster(array, tile_size)
    _emit_status(
        status_callback,
        f"Raster tiled into {tiles.shape[0]} patches of size {tile_size}x{tile_size}.",
    )

    tiles = tiles.astype("float32") / 255

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

    for tensor_batch, start, end, total in _prefetch_batches(
        tiles, batch_size, device, depth=prefetch_depth, cancel_token=cancel_token
    ):
        _maybe_raise_cancel(cancel_token)
        with torch.no_grad():
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
    )

    if isinstance(kmeans_outputs, tuple):
        _, lowres_labels, centers = kmeans_outputs
    else:
        lowres_labels = None
        centers = None

    if lowres_labels is not None and centers is not None:
        refined_lowres = _latent_knn_soft_refine(
            latent_grid,
            lowres_labels,
            centers,
            num_segments,
            status_callback=status_callback,
            config=latent_knn_config,
            cancel_token=cancel_token,
        )
    elif lowres_labels is not None:
        refined_lowres = lowres_labels.astype(np.uint8, copy=False)
    else:
        refined_lowres = kmeans_outputs[0] if isinstance(kmeans_outputs, tuple) else kmeans_outputs

    coverage_map = torch.tensor(refined_lowres).unsqueeze(0).unsqueeze(0)
    coverage_map = torch.nn.Upsample(
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad), mode="nearest"
    )(coverage_map.byte())

    coverage_map = coverage_map[0, 0, : array.shape[1], : array.shape[2]]

    _emit_status(status_callback, "CNN segmentation map reconstructed.")
    _maybe_raise_cancel(cancel_token)
    return coverage_map.cpu().numpy()

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
            except Exception as exc:
                # Ensure all remaining futures are awaited so exceptions are not lost
                while futures:
                    leftover_future, _, _ = futures.popleft()
                    try:
                        leftover_future.result()
                    except Exception:
                        pass  # Optionally log or collect these exceptions
                raise
    finally:
        executor.shutdown(wait=True)


def _batch_to_tensor(batch, device):
    tensor = torch.from_numpy(batch)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.to(device)
