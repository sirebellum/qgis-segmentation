# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def masked_iou(pred: np.ndarray, gt: np.ndarray, *, ignore_label_leq: int = 0) -> Dict[str, object]:
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have matching shapes")
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)

    valid_mask = gt_arr > ignore_label_leq
    if not np.any(valid_mask):
        return {
            "has_valid_labels": False,
            "mean_iou": None,
            "per_class": {},
            "ignored_threshold": ignore_label_leq,
        }

    labels = np.unique(gt_arr[valid_mask])
    per_class: Dict[int, float] = {}
    for label in labels:
        gt_mask = gt_arr == label
        pred_mask = pred_arr == label
        intersection = float(np.logical_and(gt_mask, pred_mask).sum())
        union = float(np.logical_or(gt_mask, pred_mask).sum())
        iou = float(intersection / union) if union > 0 else 0.0
        per_class[int(label)] = iou

    mean_iou: Optional[float] = None
    if per_class:
        mean_iou = float(np.mean(list(per_class.values())))

    return {
        "has_valid_labels": True,
        "mean_iou": mean_iou,
        "per_class": per_class,
        "ignored_threshold": ignore_label_leq,
    }
