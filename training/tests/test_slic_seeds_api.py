import numpy as np
import pytest

from training.datasets.build_shards import _lab_image, _run_slic_labels, _run_seeds_labels


@pytest.fixture()
def _cv2_ximgproc():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "ximgproc"):
        pytest.skip("cv2.ximgproc not available")
    return cv2


def _demo_rgb(shape=(3, 32, 32)):
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def test_run_slic_labels_basic(_cv2_ximgproc):
    rgb = _demo_rgb()
    lab = _lab_image(rgb)
    labels = _run_slic_labels(lab, region_size=8, ruler=10.0, iterations=2, algorithm="slico")
    assert labels.shape == rgb.shape[1:]
    assert labels.dtype == np.int32


def test_run_seeds_labels_basic(_cv2_ximgproc):
    rgb = _demo_rgb()
    bgr = np.moveaxis(rgb, 0, 2)
    labels = _run_seeds_labels(bgr, approx_region=8, iterations=2)
    assert labels is not None
    assert labels.shape == rgb.shape[1:]
    assert labels.dtype == np.int32
