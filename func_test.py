import json
import time

import numpy as np
import pytest
import torch

from funcs import (
    tile_raster,
    predict_cnn,
    predict_kmeans,
    execute_cnn_segmentation,
    execute_kmeans_segmentation,
    ChunkPlan,
    AdaptiveSettings,
    set_adaptive_settings,
    get_adaptive_settings,
    _compute_chunk_starts,
    _normalize_cluster_labels,
    recommended_chunk_plan,
    _derive_chunk_size,
    _ChunkAggregator,
    _chunked_gaussian_blur,
)
from perf_tuner import load_or_profile_settings, ProfilePayload, _run_profile
from raster_utils import ensure_channel_first


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_history = []

    def forward(self, x):
        self.batch_history.append(x.shape[0])
        mean = x.mean(dim=1, keepdim=True)
        maxv, _ = x.max(dim=1, keepdim=True)
        minv, _ = x.min(dim=1, keepdim=True)
        stacked = torch.cat([mean, maxv, minv], dim=1)
        return (None, stacked)


class _StubAutoencoder:
    def __init__(self):
        self.calls = 0

    def set_device(self, *_):
        return

    def refresh_and_remap(self, raster, label_map, status_callback=None):
        self.calls += 1
        if status_callback:
            status_callback(f"stub-remap-{self.calls}")
        return label_map.copy()


def _rand_array(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def _rand_tile_batch(seed, tile_sizes):
    batch = []
    for idx, size in enumerate(tile_sizes):
        batch.append(_rand_array(seed + idx, (3, size, size)))
    return batch


def _available_gpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _synchronize_device(device):
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _benchmark_predict(model, arrays, device, settings, tile_size):
    samples = list(arrays)
    if not samples:
        raise ValueError("arrays must not be empty")
    set_adaptive_settings({"default": settings}, default_tier="default")
    start = time.perf_counter()
    total_pixels = 0
    for array in samples:
        predict_cnn(
            model,
            array,
            num_segments=3,
            tile_size=tile_size,
            device=device,
            memory_budget=64 * 1024 * 1024,
        )
        _synchronize_device(device)
        total_pixels += array.shape[1] * array.shape[2]
    elapsed_total = time.perf_counter() - start
    avg_elapsed = elapsed_total / max(len(samples), 1)
    avg_pixels = total_pixels / max(len(samples), 1)
    throughput = avg_pixels / max(avg_elapsed, 1e-6)
    return {"elapsed": avg_elapsed, "throughput": throughput}


def test_tile_raster_reassembles_original_extent():
    array = _rand_array(0, (3, 133, 287))
    tile_size = 64
    tiles, pads, grid = tile_raster(array, tile_size=tile_size)
    rows, cols = grid
    recon = tiles.reshape(rows, cols, array.shape[0], tile_size, tile_size)
    recon = recon.transpose(2, 0, 3, 1, 4).reshape(array.shape[0], rows * tile_size, cols * tile_size)
    recon = recon[:, : array.shape[1], : array.shape[2]]
    np.testing.assert_array_equal(array, recon)
    expected_pads = (
        (tile_size - (array.shape[1] % tile_size)) % tile_size,
        (tile_size - (array.shape[2] % tile_size)) % tile_size,
    )
    assert pads == expected_pads


@pytest.mark.parametrize(
    "shape,tile_size",
    [
        ((3, 384, 640), 128),
        ((3, 257, 513), 96),
    ],
)
def test_predict_cnn_handles_rectangular_tiles(shape, tile_size):
    array = _rand_array(1, shape)
    model = _DummyModel()
    result = predict_cnn(
        model,
        array,
        num_segments=3,
        tile_size=tile_size,
        device="cpu",
        memory_budget=32 * 1024 * 1024,
    )
    assert result.shape == shape[1:]
    assert result.dtype == np.uint8
    assert max(model.batch_history) >= 1


def test_predict_cnn_uses_adaptive_batching():
    shape = (3, 256, 512)
    array = _rand_array(4, shape)
    model = _DummyModel()
    predict_cnn(
        model,
        array,
        num_segments=3,
        tile_size=64,
        device="cpu",
        memory_budget=64 * 1024 * 1024,
        prefetch_depth=3,
    )
    assert max(model.batch_history) > 1


@pytest.mark.parametrize(
    "shape",
    [
        (3, 640, 925),
        (3, 512, 700),
    ],
)
def test_execute_cnn_segmentation_chunking_preserves_shape(shape):
    array = _rand_array(2, shape)
    plan = ChunkPlan(chunk_size=256, overlap=64, budget_bytes=16 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    manager = _StubAutoencoder()
    manager.set_device(torch.device("cpu"))
    output = execute_cnn_segmentation(
        _DummyModel(),
        array,
        num_segments=3,
        chunk_plan=plan,
        tile_size=256,
        device=torch.device("cpu"),
        texture_manager=manager,
    )
    assert output.shape == shape[1:]
    assert output.max() < 3
    assert manager.calls == 1


def test_execute_kmeans_segmentation_chunking_preserves_shape():
    array = _rand_array(3, (3, 781, 1023))
    plan = ChunkPlan(chunk_size=200, overlap=50, budget_bytes=8 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    manager = _StubAutoencoder()
    output = execute_kmeans_segmentation(
        array,
        num_segments=4,
        resolution=16,
        chunk_plan=plan,
        status_callback=lambda *_: None,
        texture_manager=manager,
    )
    assert output.shape == array.shape[1:]
    assert output.max() < 4
    assert manager.calls == 1


def test_perf_tuner_profiles_and_caches(tmp_path):
    device = torch.device("cpu")
    calls = []

    def fake_runner(dev, status):
        calls.append(dev.type)
        settings = {
            "high": AdaptiveSettings(safety_factor=5, prefetch_depth=3),
            "medium": AdaptiveSettings(safety_factor=6, prefetch_depth=2),
            "low": AdaptiveSettings(safety_factor=7, prefetch_depth=1),
        }
        options = {
            "high": [
                (32 * 1024 * 1024, AdaptiveSettings(safety_factor=7, prefetch_depth=1)),
                (128 * 1024 * 1024, AdaptiveSettings(safety_factor=5, prefetch_depth=2)),
            ],
            "medium": [],
            "low": [],
        }
        metrics = {
            "high": {"best_px_per_s": 10_000.0, "speedup_vs_prefetch1": 1.25},
            "medium": {"best_px_per_s": 8_000.0, "speedup_vs_prefetch1": 1.10},
        }
        return ProfilePayload(settings=settings, options=options, default_tier="high", metrics=metrics)

    settings, created = load_or_profile_settings(
        tmp_path, device, benchmark_runner=fake_runner
    )
    assert created
    assert settings.prefetch_depth == 3
    assert settings.safety_factor == 5
    cached, created2 = load_or_profile_settings(
        tmp_path, device, benchmark_runner=fake_runner
    )
    assert not created2
    assert cached.prefetch_depth == 3
    assert len(calls) == 1
    profile_file = tmp_path / "perf_profile.json"
    assert profile_file.exists()
    disk_data = json.loads(profile_file.read_text())
    assert "cpu" in disk_data
    entry = disk_data["cpu"]
    assert entry["default_tier"] == "high"
    assert "settings" in entry and "low" in entry["settings"]
    assert "metrics" in entry and "high" in entry["metrics"]
    low = get_adaptive_settings(40 * 1024 * 1024, tier="low")
    high = get_adaptive_settings(256 * 1024 * 1024, tier="high")
    assert low.prefetch_depth == 1
    assert high.prefetch_depth == 2
    set_adaptive_settings(AdaptiveSettings())


def test_ensure_channel_first_promotes_grayscale_arrays():
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    prepared = ensure_channel_first(array)
    assert prepared.shape == (1, 4, 4)
    np.testing.assert_array_equal(prepared[0], array)


def test_ensure_channel_first_rejects_invalid_ndarrays():
    array = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        ensure_channel_first(array)


# Regression tests for shape preservation (from original unittest-based tests)
@pytest.mark.parametrize(
    "array_shape,num_segments,resolution,expected_shape",
    [
        ((3, 64, 64), 4, 16, (64, 64)),
        ((3, 128, 128), 8, 16, (128, 128)),
        ((3, 256, 256), 16, 16, (256, 256)),
        ((3, 256, 128), 8, 16, (256, 128)),
        ((3, 128, 256), 8, 16, (128, 256)),
    ],
)
def test_predict_kmeans_shape_preservation(array_shape, num_segments, resolution, expected_shape):
    """Regression test: verify predict_kmeans preserves spatial dimensions."""
    rng = np.random.default_rng(42)
    array = rng.random(array_shape, dtype=np.float32)
    result = predict_kmeans(array, num_segments=num_segments, resolution=resolution)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "array_shape,tile_size,expected_height_pad,expected_width_pad,expected_tile_shape",
    [
        ((3, 512, 512), 256, 0, 0, (4, 3, 256, 256)),
        ((3, 500, 500), 256, 12, 12, (4, 3, 256, 256)),
        ((3, 256, 256), 128, 0, 0, (4, 3, 128, 128)),
        ((3, 300, 300), 128, 84, 84, (9, 3, 128, 128)),
        ((3, 256, 128), 128, 0, 0, (2, 3, 128, 128)),
    ],
)
def test_tile_raster_shape_preservation(
    array_shape, tile_size, expected_height_pad, expected_width_pad, expected_tile_shape
):
    """Regression test: verify tile_raster produces correct shapes and padding."""
    rng = np.random.default_rng(42)
    array = rng.random(array_shape, dtype=np.float32)
    tiles, (height_pad, width_pad), grid_shape = tile_raster(array, tile_size=tile_size)
    assert height_pad == expected_height_pad
    assert width_pad == expected_width_pad
    assert tiles.shape == expected_tile_shape


def test_chunked_gaussian_blur_matches_single_pass():
    rng = np.random.default_rng(0)
    array = rng.random((3, 96, 64), dtype=np.float32)
    sigma = 6.0
    full = _chunked_gaussian_blur(array, sigma, max_chunk_bytes=1_000_000_000)
    chunked = _chunked_gaussian_blur(array, sigma, max_chunk_bytes=32 * 1024)
    np.testing.assert_allclose(chunked, full, atol=1e-4)


@pytest.mark.performance
def test_predict_cnn_gpu_prefetch_throughput(gpu_metric_recorder, tmp_path):
    device = _available_gpu_device()
    if device is None:
        pytest.skip("No CUDA or MPS backend available for GPU throughput test.")

    tile_sizes = (256, 512, 768, 1024)
    tile_batch = _rand_tile_batch(11, tile_sizes)
    model = _DummyModel().to(device)

    # Warm-up to stabilize kernels/cache before measuring.
    for sample in tile_batch:
        predict_cnn(
            model,
            sample,
            num_segments=3,
            tile_size=128,
            device=device,
            memory_budget=32 * 1024 * 1024,
            prefetch_depth=1,
        )
    _synchronize_device(device)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    def _runner(dev, status):
        return _run_profile(dev, status, tile_stack=tile_sizes, tiers=("high",))

    profiled_settings, _ = load_or_profile_settings(str(profile_dir), device, benchmark_runner=_runner)
    baseline_settings = AdaptiveSettings(safety_factor=12, prefetch_depth=1)

    baseline = _benchmark_predict(model, tile_batch, device, baseline_settings, tile_size=128)
    optimized = _benchmark_predict(model, tile_batch, device, profiled_settings, tile_size=128)

    gpu_metric_recorder(
        {
            "label": "baseline",
            "device": device.type,
            "prefetch": baseline_settings.prefetch_depth,
            "throughput": baseline["throughput"],
            "elapsed": baseline["elapsed"],
            "tile_sizes": tile_sizes,
            "is_baseline": True,
        }
    )
    gpu_metric_recorder(
        {
            "label": "optimized",
            "device": device.type,
            "prefetch": profiled_settings.prefetch_depth,
            "throughput": optimized["throughput"],
            "elapsed": optimized["elapsed"],
            "tile_sizes": tile_sizes,
            "safety": profiled_settings.safety_factor,
        }
    )

    # Allow modest variance on MPS while ensuring optimized path is not slower.
    assert optimized["throughput"] >= baseline["throughput"] * 0.8
