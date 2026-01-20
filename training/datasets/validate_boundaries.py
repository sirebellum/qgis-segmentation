# SPDX-License-Identifier: BSD-3-Clause
"""Validation utility for comparing superpixel boundaries against hand masks.

Usage:
    python -m training.datasets.validate_boundaries --pred path/to/slic.npz --gt path/to/boundary_mask.tif \
        [--pred-key edges] [--fallback-labels labels]

The ground-truth mask is expected to be a single-channel image where nonzero
pixels mark boundaries. Predicted boundaries are loaded either from a stored
edge mask (e.g., `edges`, `edges_seeds`) or derived from a label map when the
requested key is missing. Metrics reported: precision, recall, and F1 on the
boundary foreground.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must be 2D")
    h, w = labels.shape
    edges = np.zeros((h, w), dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edges


def _load_pred_mask(npz_path: Path, pred_key: str, fallback_labels_key: Optional[str]) -> np.ndarray:
    with np.load(npz_path) as npz:
        if pred_key in npz:
            arr = npz[pred_key]
            return arr.astype(bool)
        if fallback_labels_key and fallback_labels_key in npz:
            labels = npz[fallback_labels_key]
            return _label_boundaries(labels)
    raise KeyError(f"Neither '{pred_key}' nor '{fallback_labels_key}' found in {npz_path}")


def _load_gt_mask(gt_path: Path) -> np.ndarray:
    with rasterio.open(gt_path) as src:
        arr = src.read(1)
    return arr.astype(bool)


def _precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred_fg = pred.astype(bool)
    gt_fg = gt.astype(bool)
    tp = np.logical_and(pred_fg, gt_fg).sum()
    fp = np.logical_and(pred_fg, ~gt_fg).sum()
    fn = np.logical_and(~pred_fg, gt_fg).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = float(2 * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, f1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate superpixel boundaries against hand masks")
    parser.add_argument("--pred", required=True, type=Path, help="Path to slic npz file")
    parser.add_argument("--gt", required=True, type=Path, help="Path to ground-truth boundary mask (tif)")
    parser.add_argument("--pred-key", default="edges", help="Key in npz to use for predicted boundaries")
    parser.add_argument("--fallback-labels", default="labels", help="Key to derive boundaries if pred-key missing")
    args = parser.parse_args(argv)

    pred = _load_pred_mask(args.pred, args.pred_key, args.fallback_labels)
    gt = _load_gt_mask(args.gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    precision, recall, f1 = _precision_recall_f1(pred, gt)
    print(
        f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  (key={args.pred_key}, fallback={args.fallback_labels})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
