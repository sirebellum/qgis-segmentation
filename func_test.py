"""
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
"""
import json

import numpy as np
import pytest

from funcs import predict_nextgen_numpy, tile_raster
from model import load_runtime_model
from raster_utils import ensure_channel_first


def test_tile_raster_reassembles_original_extent():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(3, 133, 287), dtype=np.uint8)
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
    rng = np.random.default_rng(42)
    array = rng.random(array_shape, dtype=np.float32)
    tiles, (height_pad, width_pad), grid_shape = tile_raster(array, tile_size=tile_size)
    assert height_pad == expected_height_pad
    assert width_pad == expected_width_pad
    assert tiles.shape == expected_tile_shape
    rows, cols = grid_shape
    assert rows * tile_size >= array.shape[1]
    assert cols * tile_size >= array.shape[2]


def test_funcs_runtime_is_numpy_only():
    import funcs  # noqa: F401

    assert "torch" not in funcs.__dict__


def test_ensure_channel_first_promotes_grayscale_arrays():
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)
    prepared = ensure_channel_first(array)
    assert prepared.shape == (1, 4, 4)
    np.testing.assert_array_equal(prepared[0], array)


def test_ensure_channel_first_rejects_invalid_ndarrays():
    array = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        ensure_channel_first(array)


def test_runtime_numpy_smoke(tmp_path):
    meta = {
        "version": "test",
        "max_k": 4,
        "embed_dim": 4,
        "temperature": 1.0,
        "cluster_iters_default": 2,
        "smooth_iters_default": 0,
        "input_mean": [0.0, 0.0, 0.0],
        "input_std": [1.0, 1.0, 1.0],
        "input_scale": 1.0,
        "stride": 4,
        "supports_elevation": False,
        "supports_learned_refine": False,
    }
    weights = {
        "stem.conv1.weight": np.zeros((2, 3, 3, 3), dtype=np.float32),
        "stem.conv1.bias": np.zeros((2,), dtype=np.float32),
        "stem.bn1.weight": np.ones((2,), dtype=np.float32),
        "stem.bn1.bias": np.zeros((2,), dtype=np.float32),
        "stem.bn1.running_mean": np.zeros((2,), dtype=np.float32),
        "stem.bn1.running_var": np.ones((2,), dtype=np.float32),
        "stem.conv2.weight": np.zeros((4, 2, 3, 3), dtype=np.float32),
        "stem.conv2.bias": np.zeros((4,), dtype=np.float32),
        "stem.bn2.weight": np.ones((4,), dtype=np.float32),
        "stem.bn2.bias": np.zeros((4,), dtype=np.float32),
        "stem.bn2.running_mean": np.zeros((4,), dtype=np.float32),
        "stem.bn2.running_var": np.ones((4,), dtype=np.float32),
        "block1.conv1.weight": np.zeros((4, 4, 3, 3), dtype=np.float32),
        "block1.conv1.bias": np.zeros((4,), dtype=np.float32),
        "block1.bn1.weight": np.ones((4,), dtype=np.float32),
        "block1.bn1.bias": np.zeros((4,), dtype=np.float32),
        "block1.bn1.running_mean": np.zeros((4,), dtype=np.float32),
        "block1.bn1.running_var": np.ones((4,), dtype=np.float32),
        "block1.conv2.weight": np.zeros((4, 4, 3, 3), dtype=np.float32),
        "block1.conv2.bias": np.zeros((4,), dtype=np.float32),
        "block1.bn2.weight": np.ones((4,), dtype=np.float32),
        "block1.bn2.bias": np.zeros((4,), dtype=np.float32),
        "block1.bn2.running_mean": np.zeros((4,), dtype=np.float32),
        "block1.bn2.running_var": np.ones((4,), dtype=np.float32),
        "block2.conv1.weight": np.zeros((4, 4, 3, 3), dtype=np.float32),
        "block2.conv1.bias": np.zeros((4,), dtype=np.float32),
        "block2.bn1.weight": np.ones((4,), dtype=np.float32),
        "block2.bn1.bias": np.zeros((4,), dtype=np.float32),
        "block2.bn1.running_mean": np.zeros((4,), dtype=np.float32),
        "block2.bn1.running_var": np.ones((4,), dtype=np.float32),
        "block2.conv2.weight": np.zeros((4, 4, 3, 3), dtype=np.float32),
        "block2.conv2.bias": np.zeros((4,), dtype=np.float32),
        "block2.bn2.weight": np.ones((4,), dtype=np.float32),
        "block2.bn2.bias": np.zeros((4,), dtype=np.float32),
        "block2.bn2.running_mean": np.zeros((4,), dtype=np.float32),
        "block2.bn2.running_var": np.ones((4,), dtype=np.float32),
        "seed_proj.weight": np.zeros((4, 4, 1, 1), dtype=np.float32),
        "seed_proj.bias": np.zeros((4,), dtype=np.float32),
    }
    model_dir = tmp_path / "best"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "model.npz", **weights)
    (model_dir / "meta.json").write_text(json.dumps(meta))
    runtime = load_runtime_model(str(model_dir))
    rgb = np.ones((3, 32, 32), dtype=np.float32)
    labels = runtime.predict_labels(rgb, k=3)
    assert labels.shape == (32, 32)
    assert labels.max() == 0


def test_predict_nextgen_numpy_tiles_and_stitches(monkeypatch):
    calls = {"predict": 0}

    class _StubRuntime:
        def predict_labels(self, tile, k):
            calls["predict"] += 1
            return np.zeros((tile.shape[1], tile.shape[2]), dtype=np.uint8)

    def loader():
        return _StubRuntime()

    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(3, 130, 260), dtype=np.uint8)
    tile_size = 64
    labels = predict_nextgen_numpy(loader, array, num_segments=3, tile_size=tile_size)
    assert labels.shape == (130, 260)
    assert labels.dtype == np.uint8
    expected_tiles = ((array.shape[1] + tile_size - 1) // tile_size) * ((array.shape[2] + tile_size - 1) // tile_size)
    assert calls["predict"] == expected_tiles
