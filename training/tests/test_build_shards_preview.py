# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from pathlib import Path

from training.datasets import build_shards as bs
from training.datasets.build_shards import _render_preview
from training.datasets.header_schema import DatasetHeader, PairingSpec, ShardingSpec, SplitSpec, ValidationSpec
from training.datasets.build_shards import DatasetItem


@pytest.fixture()
def _cv2_required():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "ximgproc"):
        pytest.skip("cv2.ximgproc not available")
    return cv2


def test_render_preview_pads_rows_to_same_width(tmp_path, _cv2_required):
    h, w = 8, 8
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    labels = np.zeros((h, w), dtype=np.int32)
    labels_fine = labels + 1
    labels_coarse = labels + 2
    edges = np.zeros((h, w), dtype=bool)
    edges_fine = np.zeros((h, w), dtype=bool)
    edges_coarse = np.zeros((h, w), dtype=bool)
    labels_seeds = labels + 3
    edges_seeds = np.zeros((h, w), dtype=bool)

    out = _render_preview(
        rgb=rgb,
        labels_main=labels,
        edges_main=edges,
        labels_fine=labels_fine,
        labels_coarse=labels_coarse,
        edges_fine=edges_fine,
        edges_coarse=edges_coarse,
        labels_seeds=labels_seeds,
        edges_seeds=edges_seeds,
        preview_dir=tmp_path,
        item_id="sample",
        seed=42,
    )
    assert out is not None
    assert out.exists()

    cv2 = _cv2_required
    img = cv2.imread(str(out))
    assert img is not None
    # Panels: rgb, mean fill, overlay, overlay_seeds => 4 tiles wide, 1 tile high.
    assert img.shape[1] == w * 4
    assert img.shape[0] == h


def test_shard_items_generates_preview_on_first_item_only(monkeypatch, tmp_path):
    calls = []
    preview_file = tmp_path / "preview.png"

    def fake_process_item(item, header, shard_dir, target_split, preview_root, shard_name, *, generate_preview):
        entry = {
            "dataset_id": item.dataset_id,
            "item_id": item.item_id,
            "raw_split": item.raw_split,
            "split": target_split,
            "input": f"inputs/{item.item_id}.tif",
            "has_target": False,
            "slic": f"slic/{item.item_id}.npz",
            "slic_meta": {},
        }
        path = None
        if generate_preview:
            preview_file.touch()
            path = preview_file
        calls.append((item.item_id, generate_preview))
        return entry, path

    monotonic_values = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(bs, "_process_item", fake_process_item)
    monkeypatch.setattr(bs.time, "monotonic", lambda: next(monotonic_values))

    header = DatasetHeader(
        dataset_id="ds",
        version="0.0",
        description="",
        source_root=".",
        modalities=[],
        pairing=PairingSpec(strategy="by_stem", input_modality=None, target_modality=None),
        splits=SplitSpec(raw_splits=["train"]),
        sharding=ShardingSpec(shard_size=4),
        validation=ValidationSpec(),
        slic=None,
    )

    items = [
        DatasetItem(dataset_id="ds", item_id=f"item-{idx}", raw_split="train", input_path=Path(f"in-{idx}.tif"))
        for idx in range(3)
    ]
    assignments = {"train": items}

    output_root = tmp_path / "out"
    summary = bs._shard_items(
        assignments,
        header,
        output_root,
        shard_size=4,
        max_items=None,
        overwrite=True,
        viz_interval=5.0,
        threads=3,
    )

    assert summary["train"]["shards"] == 1
    preview_flags = [flag for _, flag in calls]
    assert preview_flags.count(True) == 1
    assert calls[0] == ("item-0", True)
