# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from training.datasets.metrics import masked_iou


def test_masked_iou_ignores_non_positive_labels() -> None:
    pred = np.array([[1, 1], [0, 2]], dtype=np.uint8)
    gt = np.array([[0, 1], [0, 2]], dtype=np.uint8)

    result = masked_iou(pred, gt, ignore_label_leq=0)

    assert result["has_valid_labels"] is True
    assert 0 not in result["per_class"]
    assert set(result["per_class"].keys()) == {1, 2}
    assert result["per_class"][1] == 1.0
    assert result["per_class"][2] == 1.0
