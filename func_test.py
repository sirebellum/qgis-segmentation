import numpy as np
import pytest
import torch

from funcs import (
    tile_raster,
    predict_cnn,
    execute_cnn_segmentation,
    execute_kmeans_segmentation,
    ChunkPlan,
    AdaptiveSettings,
    set_adaptive_settings,
)
from perf_tuner import load_or_profile_settings


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


def _rand_array(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


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
        memory_budget=512 * 1024 * 1024,
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
        memory_budget=1024 * 1024 * 1024,
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
    plan = ChunkPlan(chunk_size=256, overlap=64, budget_bytes=128 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    output = execute_cnn_segmentation(
        _DummyModel(),
        array,
        num_segments=3,
        chunk_plan=plan,
        tile_size=256,
        device=torch.device("cpu"),
    )
    assert output.shape == shape[1:]
    assert output.max() < 3


def test_execute_kmeans_segmentation_chunking_preserves_shape():
    array = _rand_array(3, (3, 781, 1023))
    plan = ChunkPlan(chunk_size=200, overlap=50, budget_bytes=64 * 1024 * 1024, ratio=0.0075, prefetch_depth=2)
    output = execute_kmeans_segmentation(
        array,
        num_segments=4,
        resolution=16,
        chunk_plan=plan,
        status_callback=lambda *_: None,
    )
    assert output.shape == array.shape[1:]
    assert output.max() < 4


def test_perf_tuner_profiles_and_caches(tmp_path):
    device = torch.device("cpu")
    calls = []

    def fake_runner(dev, status):
        calls.append(dev.type)
        return AdaptiveSettings(safety_factor=5, prefetch_depth=3)

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
    set_adaptive_settings(AdaptiveSettings())
