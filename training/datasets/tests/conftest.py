# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _stub_slic(monkeypatch):
    def _fake_slic(input_path, slic_dir, item_id, slic_spec, *, want_preview):
        slic_dir.mkdir(parents=True, exist_ok=True)
        out = slic_dir / f"{item_id}.npz"
        np.savez(out, labels=np.zeros((1, 1), dtype=np.uint16))
        return {
            "meta": {"file": str(out.relative_to(input_path.parent.parent)), "num_superpixels": 1},
            "preview": None,
        }

    monkeypatch.setattr("training.datasets.build_shards._compute_and_write_slic", _fake_slic)
    monkeypatch.setattr("training.datasets.build_shards._render_preview", lambda **_: None)
    # Keep raster copy + index writing intact; only bypass heavy OpenCV paths.
    yield
