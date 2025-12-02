import itertools
import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    from .dependency_manager import ensure_dependencies
except ImportError:  # pragma: no cover
    from dependency_manager import ensure_dependencies

ensure_dependencies()

import numpy as np
import torch
from sklearn.cluster import KMeans

DEFAULT_MEMORY_BUDGET = 128 * 1024 * 1024


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


_ADAPTIVE_SETTINGS = AdaptiveSettings()


def get_adaptive_settings() -> AdaptiveSettings:
    return _ADAPTIVE_SETTINGS


def set_adaptive_settings(settings: AdaptiveSettings) -> None:
    global _ADAPTIVE_SETTINGS
    _ADAPTIVE_SETTINGS = settings


def _emit_status(callback, message):
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


def _process_in_chunks(array, plan, num_segments, infer_fn, status_callback):
    height, width = array.shape[1], array.shape[2]
    if not plan.should_chunk(height, width):
        return infer_fn(array)

    stride = plan.stride
    y_starts = _compute_chunk_starts(height, plan.chunk_size, stride)
    x_starts = _compute_chunk_starts(width, plan.chunk_size, stride)
    total = len(y_starts) * len(x_starts)
    aggregator = _ChunkAggregator(height, width, num_segments, plan.chunk_size)

    for idx, (y, x) in enumerate(itertools.product(y_starts, x_starts), start=1):
        y_end = min(y + plan.chunk_size, height)
        x_end = min(x + plan.chunk_size, width)
        chunk = array[:, y:y_end, x:x_end]
        if status_callback:
            status_callback(f"Chunk {idx}/{total}: rows {y}-{y_end}, cols {x}-{x_end}")
        labels = infer_fn(chunk)
        aggregator.add(labels, (y, x, y_end, x_end))

    return aggregator.finalize()


def _compute_chunk_starts(length, chunk_size, stride):
    if length <= chunk_size:
        return [0]
    starts = list(range(0, max(1, length - chunk_size), stride))
    last_start = length - chunk_size
    if last_start > 0 and (not starts or starts[-1] != last_start):
        starts.append(last_start)
    return sorted(set(starts))


def _derive_chunk_size(array_shape, device):
    channels = array_shape[0]
    free_bytes = _free_vram_bytes(device)
    ratio = 0.009 if device.type == "cuda" else 0.0075 if device.type == "mps" else 0.01
    budget = max(int(free_bytes * ratio), 64 * 1024 * 1024)
    bytes_per_pixel = channels * 4
    safety = 8
    max_pixels = max(budget // (bytes_per_pixel * safety), 1)
    tile_side = int(math.sqrt(max_pixels))
    tile_side = max(128, min(512, tile_side))
    return tile_side, budget, ratio


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
    def __init__(self, height, width, num_segments, chunk_size):
        self.height = height
        self.width = width
        self.num_segments = num_segments
        self.scores = np.zeros((num_segments, height, width), dtype=np.float32)
        self.weight = np.zeros((height, width), dtype=np.float32)
        self.weight_template = _build_weight_mask(chunk_size)

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
        weight = np.maximum(self.weight, 1e-6)
        probs = self.scores / weight
        return np.argmax(probs, axis=0).astype(np.uint8)


def _build_weight_mask(size):
    if size <= 1:
        return np.ones((1, 1), dtype=np.float32)
    window = np.hanning(size)
    if np.max(window) == 0:
        window = np.ones(size)
    mask = np.outer(window, window)
    mask = mask / np.max(mask)
    return mask.astype(np.float32)


def recommended_chunk_plan(array_shape, device):
    chunk_size, budget, ratio = _derive_chunk_size(array_shape, device)
    overlap = max(32, int(chunk_size * 0.25))
    settings = get_adaptive_settings()
    return ChunkPlan(
        chunk_size=chunk_size,
        overlap=overlap,
        budget_bytes=budget,
        ratio=ratio,
        prefetch_depth=settings.prefetch_depth,
    )


def execute_kmeans_segmentation(array, num_segments, resolution, chunk_plan, status_callback=None):
    height, width = array.shape[1], array.shape[2]
    if chunk_plan and chunk_plan.should_chunk(height, width):
        return _process_in_chunks(
            array,
            chunk_plan,
            num_segments,
            lambda data: predict_kmeans(data, num_segments, resolution, status_callback=None),
            status_callback,
        )
    return predict_kmeans(array, num_segments, resolution, status_callback=status_callback)


def execute_cnn_segmentation(
    cnn_model,
    array,
    num_segments,
    chunk_plan,
    tile_size,
    device,
    status_callback=None,
):
    height, width = array.shape[1], array.shape[2]
    effective_tile = min(tile_size, chunk_plan.chunk_size) if chunk_plan else tile_size

    def _tile_for_data(data):
        size = min(effective_tile, data.shape[1], data.shape[2])
        return max(64, size)

    if chunk_plan and chunk_plan.should_chunk(height, width):
        return _process_in_chunks(
            array,
            chunk_plan,
            num_segments,
            lambda data: predict_cnn(
                cnn_model,
                data,
                num_segments,
                tile_size=_tile_for_data(data),
                device=device,
                status_callback=None,
                memory_budget=chunk_plan.budget_bytes,
                prefetch_depth=chunk_plan.prefetch_depth,
            ),
            status_callback,
        )
    return predict_cnn(
        cnn_model,
        array,
        num_segments,
        tile_size=_tile_for_data(array),
        device=device,
        status_callback=status_callback,
        memory_budget=chunk_plan.budget_bytes if chunk_plan else None,
        prefetch_depth=chunk_plan.prefetch_depth if chunk_plan else None,
    )

# Predict coverage map using kmeans
def predict_kmeans(array, num_segments=16, resolution=16, status_callback=None):
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
    kmeans = kmeans.fit(array_2d[idx])

    # Get clusters
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

    # Upsample to original size
    clusters = torch.tensor(clusters)
    clusters = torch.nn.Upsample(
        size=(array.shape[-2], array.shape[-1]), mode="nearest"
    )(clusters.byte())
    clusters = clusters.squeeze()

    _emit_status(status_callback, "K-Means output upsampled to raster resolution.")
    return clusters.cpu().numpy()

def predict_cnn(
    cnn_model,
    array,
    num_segments,
    tile_size=256,
    device="cpu",
    status_callback=None,
    memory_budget=None,
    prefetch_depth=None,
):
    settings = get_adaptive_settings()
    prefetch_depth = prefetch_depth or settings.prefetch_depth
    memory_budget = memory_budget or DEFAULT_MEMORY_BUDGET

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
        tiles, batch_size, device, depth=prefetch_depth
    ):
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

    coverage_map = predict_kmeans(
        coverage_map,
        num_segments=num_segments,
        resolution=1,
        status_callback=status_callback,
    )

    coverage_map = torch.tensor(coverage_map).unsqueeze(0).unsqueeze(0)
    coverage_map = torch.nn.Upsample(
        size=(array.shape[1] + height_pad, array.shape[2] + width_pad), mode="nearest"
    )(coverage_map.byte())

    coverage_map = coverage_map[0, 0, : array.shape[1], : array.shape[2]]
    coverage_map = coverage_map.squeeze()

    _emit_status(status_callback, "CNN segmentation map reconstructed.")
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
    # bytes_per_tile is guaranteed > 0 since all factors are >= 1
    budget = max(memory_budget or DEFAULT_MEMORY_BUDGET, bytes_per_tile)
    safety = max(1, settings.safety_factor)
    depth = max(1, settings.prefetch_depth)
    effective_budget = max(1, budget // safety)
    denom = max(bytes_per_tile * (1 + depth), 1)
    return max(1, effective_budget // denom)


def _prefetch_batches(tiles, batch_size, device, depth=2):
    total = tiles.shape[0]
    depth = max(1, depth or 1)
    executor = ThreadPoolExecutor(max_workers=depth)
    futures = deque()
    index = 0

    try:
        while index < total or futures:
            while index < total and len(futures) < depth:
                start = index
                end = min(start + batch_size, total)
                batch = tiles[start:end]
                future = executor.submit(_batch_to_tensor, batch, device)
                futures.append((future, start, end))
                index = end
            future, start, end = futures.popleft()
            try:
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
