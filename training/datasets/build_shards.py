# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import rasterio
import shutil

from .header_schema import DatasetHeader, ModalitySpec, load_header, load_headers, SlicSpec


def _try_get_ximgproc():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        return None, f"{exc}"
    if not hasattr(cv2, "ximgproc"):
        return None, "cv2.ximgproc is missing"
    return cv2, None


def _ensure_ximgproc():
    cv2, reason = _try_get_ximgproc()
    if cv2 is None:
        raise ImportError(
            "OpenCV with ximgproc is required for deterministic superpixels. Install opencv-contrib-python (or -headless)."
        )
    return cv2


def _stable_int_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _fallback_edge_stats(rgb: np.ndarray) -> Dict[str, float]:
    # Lightweight gradient proxy when OpenCV is unavailable; deterministic on input.
    l_chan = rgb[0].astype(np.float32)
    gx = np.abs(np.diff(l_chan, axis=1))
    gy = np.abs(np.diff(l_chan, axis=0))
    mag = np.pad(np.sqrt(np.square(gx[:-1, :]) + np.square(gy[:, :-1])), ((0, 1), (0, 1)))
    p50 = float(np.median(mag))
    p90 = float(np.percentile(mag, 90.0))
    p99 = float(np.percentile(mag, 99.0))
    mean = float(np.mean(mag))
    density = max(1e-3, p90)
    return {"p50": p50, "p90": p90, "p99": p99, "mean": mean, "edge_density": density}


def _grid_labels(h: int, w: int, region: int) -> np.ndarray:
    region = max(1, int(region))
    y_ids = np.arange(h) // region
    x_ids = np.arange(w) // region
    labels = (y_ids[:, None] * ((w + region - 1) // region)) + x_ids[None, :]
    return labels.astype(np.int32, copy=False)


def _fallback_slic(rgb: np.ndarray, item_id: str, slic_spec: SlicSpec) -> Dict[str, object]:
    # Deterministic grid-based superpixels as a no-op fallback when cv2.ximgproc is absent.
    h, w = rgb.shape[1:]
    patch_max = float(max(h, w))
    base_px = int(np.clip(round(slic_spec.region_size * patch_max / 512.0), slic_spec.min_size, slic_spec.max_size))
    factors = slic_spec.scales if slic_spec.scales else [0.75, 1.0, 1.35]
    factors = factors[:3] if len(factors) >= 3 else (factors + [factors[-1]] * (3 - len(factors)))
    region_sizes = {
        "fine": int(np.clip(round(base_px * factors[0]), slic_spec.min_size, slic_spec.max_size)),
        "medium": int(np.clip(round(base_px * factors[1]), slic_spec.min_size, slic_spec.max_size)),
        "coarse": int(np.clip(round(base_px * factors[2]), slic_spec.min_size, slic_spec.max_size)),
    }

    labels_fine = _grid_labels(h, w, region_sizes["fine"])
    labels_main = _grid_labels(h, w, region_sizes["medium"])
    labels_coarse = _grid_labels(h, w, region_sizes["coarse"])

    edges_fine = _label_boundaries(labels_fine)
    edges_main = _label_boundaries(labels_main)
    edges_coarse = _label_boundaries(labels_coarse)

    meta = {
        "file": None,
        "algorithm": "grid_fallback",
        "ruler": slic_spec.ruler,
        "iterations": slic_spec.iterations,
        "edge_stats": _fallback_edge_stats(rgb),
        "policy": {
            "base_region_size_512": int(round(base_px * 512.0 / patch_max)) if patch_max > 0 else base_px,
            "base_region_size_px": base_px,
            "bounds_px": {"min": int(slic_spec.min_size), "max": int(slic_spec.max_size)},
            "factors": factors,
            "region_sizes": region_sizes,
        },
        "patch_shape": [h, w],
        "num_superpixels": int(np.max(labels_main) + 1),
        "dtype": str(labels_main.dtype),
        "seeds_available": False,
        "fallback_used": True,
        "fallback_reason": "cv2.ximgproc unavailable",
    }

    arrays = {
        "labels": labels_main,
        "labels_fine": labels_fine,
        "labels_coarse": labels_coarse,
        "edges": edges_main.astype(np.uint8),
        "edges_fine": edges_fine.astype(np.uint8),
        "edges_coarse": edges_coarse.astype(np.uint8),
    }

    return {"arrays": arrays, "meta": meta}


def _lab_image(rgb: np.ndarray) -> np.ndarray:
    image = np.moveaxis(rgb, 0, 2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    cv2 = _ensure_ximgproc()
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def _edge_stats(lab_image: np.ndarray) -> Dict[str, float]:
    cv2 = _ensure_ximgproc()
    l_chan = lab_image[:, :, 0].astype(np.float32)
    grad_x = cv2.Sobel(l_chan, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    p50 = float(np.median(mag))
    p90 = float(np.percentile(mag, 90.0))
    p99 = float(np.percentile(mag, 99.0))
    mean = float(np.mean(mag))
    density = max(1e-3, p90)
    return {"p50": p50, "p90": p90, "p99": p99, "mean": mean, "edge_density": density}


def _select_region_sizes(stats: Dict[str, float], shape: Tuple[int, int], slic_spec: SlicSpec) -> Dict[str, object]:
    height, width = shape
    patch_max = float(max(height, width))
    min_px = max(6, int(round(slic_spec.min_size * patch_max / 512.0)))
    max_px = max(min_px + 1, int(round(slic_spec.max_size * patch_max / 512.0)))

    base_512 = int(np.clip(round(48.0 / (stats["edge_density"] + 1e-3)), slic_spec.min_size, slic_spec.max_size))
    base_px = int(np.clip(round(base_512 * patch_max / 512.0), min_px, max_px))

    factors = slic_spec.scales if slic_spec.scales else [0.75, 1.0, 1.35]
    if len(factors) < 3:
        factors = factors + [factors[-1]] * (3 - len(factors))

    scale_names = ["fine", "medium", "coarse"]
    region_sizes = {}
    for name, factor in zip(scale_names, factors[:3]):
        region_sizes[name] = int(np.clip(round(base_px * float(factor)), min_px, max_px))

    return {
        "base_region_size_512": base_512,
        "base_region_size_px": base_px,
        "region_sizes": region_sizes,
        "bounds_px": {"min": min_px, "max": max_px},
        "factors": factors[:3],
    }


def _slic_algorithm_code(cv2, name: str) -> int:
    normalized = (name or "slico").strip().lower()
    if normalized == "slic":
        return cv2.ximgproc.SLIC
    if normalized == "slicm":
        return cv2.ximgproc.SLIC
    return cv2.ximgproc.SLICO


def _run_slic_labels(lab_image: np.ndarray, *, region_size: int, ruler: float, iterations: int, algorithm: str) -> np.ndarray:
    cv2 = _ensure_ximgproc()
    image = np.ascontiguousarray(lab_image)
    region = max(2, int(region_size))
    iter_count = max(1, int(iterations))
    algo = _slic_algorithm_code(cv2, algorithm)
    slic = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=algo, region_size=region, ruler=float(ruler))
    slic.iterate(iter_count)
    labels = slic.getLabels()
    return labels.astype(np.int32, copy=False)


def _run_seeds_labels(bgr_image: np.ndarray, *, approx_region: int, iterations: int) -> Optional[np.ndarray]:
    try:
        cv2 = _ensure_ximgproc()
    except Exception:
        return None
    image = np.ascontiguousarray(bgr_image)
    h, w, c = image.shape
    region = max(2, int(approx_region))
    # Estimate target count from area and region size
    est_superpixels = max(8, int(round((h * w) / float(region * region))))
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        int(w),
        int(h),
        int(c),
        int(est_superpixels),
        4,
    )
    seeds.iterate(image, max(1, int(iterations)))
    labels = seeds.getLabels()
    return labels.astype(np.int32, copy=False)


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    edges = np.zeros((h, w), dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edges


def _mean_color_fill(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    flat = labels.reshape(-1)
    num_labels = int(flat.max()) + 1 if flat.size else 0
    if num_labels <= 0:
        return np.moveaxis(rgb, 0, 2)
    rgb_hw = np.moveaxis(rgb, 0, 2).reshape(-1, 3).astype(np.float32)
    counts = np.bincount(flat, minlength=num_labels).astype(np.float32)
    counts[counts == 0] = 1.0
    colors = np.stack([
        np.bincount(flat, weights=rgb_hw[:, c], minlength=num_labels) / counts for c in range(3)
    ], axis=1).astype(np.uint8)
    filled = colors[flat].reshape(h, w, 3)
    return filled


def _random_color_fill(labels: np.ndarray, seed: int) -> np.ndarray:
    flat = labels.reshape(-1)
    num_labels = int(flat.max()) + 1 if flat.size else 0
    if num_labels <= 0:
        return np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
    return palette[flat].reshape(labels.shape[0], labels.shape[1], 3)


def _overlay_edges(rgb: np.ndarray, *, edges_main: np.ndarray, edges_fine: Optional[np.ndarray] = None, edges_coarse: Optional[np.ndarray] = None) -> np.ndarray:
    canvas = np.moveaxis(rgb, 0, 2).copy()
    if edges_fine is not None:
        canvas[edges_fine] = (0, 255, 0)
    canvas[edges_main] = (255, 0, 0)
    if edges_coarse is not None:
        canvas[edges_coarse] = (0, 0, 255)
    return canvas


def _render_preview(
    *,
    rgb: np.ndarray,
    labels_main: np.ndarray,
    edges_main: np.ndarray,
    labels_fine: Optional[np.ndarray],
    labels_coarse: Optional[np.ndarray],
    edges_fine: Optional[np.ndarray],
    edges_coarse: Optional[np.ndarray],
    labels_seeds: Optional[np.ndarray],
    edges_seeds: Optional[np.ndarray],
    preview_dir: Path,
    item_id: str,
    seed: int,
) -> Optional[Path]:
    try:
        cv2 = _ensure_ximgproc()
    except Exception:
        return None

    rgb_hwc = np.moveaxis(rgb, 0, 2)
    mean_fill = _mean_color_fill(rgb, labels_main)

    overlay_main = _overlay_edges(
        rgb,
        edges_main=edges_main,
        edges_fine=edges_fine,
        edges_coarse=edges_coarse,
    )
    overlay_seeds = None
    if edges_seeds is not None:
        overlay_seeds = np.moveaxis(rgb, 0, 2).copy()
        overlay_seeds[edges_seeds] = (255, 165, 0)

    panels = [rgb_hwc, mean_fill, overlay_main]
    if overlay_seeds is not None:
        panels.append(overlay_seeds)
    canvas = np.concatenate(panels, axis=1)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    preview_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / f"{item_id}.png"
    if not cv2.imwrite(str(path), bgr):
        return None
    return path


def _has_white_border(rgb: np.ndarray, *, threshold: int = 250, min_fraction: float = 0.2) -> bool:
    if rgb.ndim != 3 or rgb.shape[0] < 3:
        return False
    white = (rgb >= threshold).all(axis=0)
    edges = np.zeros_like(white, dtype=bool)
    edges[0, :] = True
    edges[-1, :] = True
    edges[:, 0] = True
    edges[:, -1] = True
    edge_white = white & edges
    if not edge_white.any():
        return False
    fraction = edge_white.sum() / edge_white.size
    return fraction >= min_fraction


def _preview_orders_for_threads(total_items: int, workers: int) -> set[int]:
    # Keep previews lightweight: emit exactly one preview per shard (first item only).
    if total_items <= 0 or workers <= 0:
        return set()
    return {0}


def _load_rgb_array(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read()
    if array.shape[0] != 3:
        raise ValueError(f"SLIC requires exactly 3 channels; found {array.shape[0]} in {path}")
    return array[:3]


def _write_slic(shard_dir: Path, slic_dir: Path, item_id: str, arrays: Dict[str, np.ndarray]) -> Path:
    slic_dir.mkdir(parents=True, exist_ok=True)
    path = slic_dir / f"{item_id}.npz"
    payload = {}
    for key, arr in arrays.items():
        if arr is None:
            continue
        dtype = np.uint16 if arr.max() < 65535 else np.int32
        payload[key] = arr.astype(dtype, copy=False)
    np.savez_compressed(path, **payload)
    return path


def _compute_and_write_slic(
    input_path: Path,
    slic_dir: Path,
    item_id: str,
    slic_spec: SlicSpec,
    *,
    want_preview: bool,
) -> Dict[str, object]:
    rgb = _load_rgb_array(input_path)
    slic_payload: Optional[Dict[str, object]] = None
    fallback_reason: Optional[str] = None

    cv2, reason = _try_get_ximgproc()
    if cv2 is not None:
        try:
            lab = _lab_image(rgb)
            stats = _edge_stats(lab)
            size_info = _select_region_sizes(stats, (rgb.shape[1], rgb.shape[2]), slic_spec)

            region_sizes = size_info["region_sizes"]
            ruler = slic_spec.ruler if slic_spec.algorithm != "slico" else max(1.0, slic_spec.ruler)

            labels_fine = _run_slic_labels(lab, region_size=region_sizes.get("fine", size_info["base_region_size_px"]), ruler=ruler, iterations=slic_spec.iterations, algorithm=slic_spec.algorithm)
            labels_main = _run_slic_labels(lab, region_size=region_sizes.get("medium", size_info["base_region_size_px"]), ruler=ruler, iterations=slic_spec.iterations, algorithm=slic_spec.algorithm)
            labels_coarse = _run_slic_labels(lab, region_size=region_sizes.get("coarse", size_info["base_region_size_px"]), ruler=ruler, iterations=slic_spec.iterations, algorithm=slic_spec.algorithm)

            edges_fine = _label_boundaries(labels_fine)
            edges_main = _label_boundaries(labels_main)
            edges_coarse = _label_boundaries(labels_coarse)

            seeds_labels = None
            seeds_edges = None
            if slic_spec.enable_seeds:
                bgr = np.moveaxis(rgb, 0, 2)
                if bgr.dtype != np.uint8:
                    bgr = np.clip(bgr, 0.0, 1.0)
                    bgr = (bgr * 255.0).astype(np.uint8)
                seeds_labels = _run_seeds_labels(np.ascontiguousarray(bgr), approx_region=size_info["base_region_size_px"], iterations=slic_spec.iterations)
                if seeds_labels is not None:
                    seeds_edges = _label_boundaries(seeds_labels)

            arrays = {
                "labels": labels_main,
                "labels_fine": labels_fine,
                "labels_coarse": labels_coarse,
                "edges": edges_main.astype(np.uint8),
                "edges_fine": edges_fine.astype(np.uint8),
                "edges_coarse": edges_coarse.astype(np.uint8),
            }
            if seeds_labels is not None:
                arrays["labels_seeds"] = seeds_labels
                if seeds_edges is not None:
                    arrays["edges_seeds"] = seeds_edges.astype(np.uint8)

            meta = {
                "algorithm": slic_spec.algorithm,
                "ruler": ruler,
                "iterations": slic_spec.iterations,
                "edge_stats": stats,
                "policy": {
                    "base_region_size_512": size_info["base_region_size_512"],
                    "base_region_size_px": size_info["base_region_size_px"],
                    "bounds_px": size_info["bounds_px"],
                    "factors": size_info["factors"],
                    "region_sizes": region_sizes,
                },
                "patch_shape": [int(rgb.shape[1]), int(rgb.shape[2])],
                "num_superpixels": int(np.max(labels_main) + 1),
                "dtype": str(labels_main.dtype),
                "seeds_available": bool(seeds_labels is not None),
                "fallback_used": False,
                "fallback_reason": None,
            }

            preview = None
            if want_preview:
                seed = _stable_int_hash(item_id)
                preview = {
                    "rgb": rgb,
                    "labels_main": labels_main,
                    "labels_fine": labels_fine,
                    "labels_coarse": labels_coarse,
                    "edges_main": edges_main,
                    "edges_fine": edges_fine,
                    "edges_coarse": edges_coarse,
                    "labels_seeds": seeds_labels,
                    "edges_seeds": seeds_edges,
                    "seed": seed,
                }

            slic_payload = {"arrays": arrays, "meta": meta, "preview": preview}
        except Exception as exc:  # pragma: no cover - fallback path exercised elsewhere
            fallback_reason = str(exc)
            slic_payload = None
    else:
        fallback_reason = reason

    if slic_payload is None:
        slic_payload = _fallback_slic(rgb, item_id, slic_spec)
        if fallback_reason:
            slic_payload["meta"]["fallback_reason"] = fallback_reason

    arrays = slic_payload["arrays"]
    meta = slic_payload["meta"]
    path = _write_slic(shard_dir=input_path.parent.parent, slic_dir=slic_dir, item_id=item_id, arrays=arrays)
    meta["file"] = str(path.relative_to(input_path.parent.parent))

    preview = None
    if want_preview and slic_payload.get("preview") and not meta.get("fallback_used"):
        preview = slic_payload["preview"]

    return {"meta": meta, "preview": preview}


@dataclass
class DatasetItem:
    dataset_id: str
    item_id: str
    raw_split: str
    input_path: Path
    target_path: Optional[Path] = None
    assigned_split: Optional[str] = None


def _normalize_split_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    if normalized in {"metrics_train", "metrics", "train"}:
        return "train"
    if normalized in {"val", "validation"}:
        return "val"
    if normalized == "test":
        return "test"
    return "train"


def _glob_for_modality(modality: ModalitySpec, source_root: Path, raw_split: str) -> Dict[str, Path]:
    pattern = modality.pattern.format(split=raw_split)
    matches: Dict[str, Path] = {}
    for path in sorted(source_root.glob(pattern)):
        if not path.is_file():
            continue
        matches[path.stem] = path
    return matches


def _glob_all_for_modality(modality: ModalitySpec, source_root: Path) -> Dict[str, Path]:
    wildcard = "**" if "**" in modality.pattern else "*"
    pattern = modality.pattern
    if "{split}" in pattern:
        pattern = pattern.format(split=wildcard)
    matches: Dict[str, Path] = {}
    for path in sorted(source_root.glob(pattern)):
        if path.is_file():
            matches[path.stem] = path
    return matches


def _load_manifest_stems(manifest_path: Path) -> List[str]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    stems = []
    for line in lines:
        name = line.strip()
        if not name:
            continue
        stems.append(Path(name).stem)
    return list(dict.fromkeys(stems))


def _collect_items(header: DatasetHeader, source_root: Path) -> List[DatasetItem]:
    if header.pairing.strategy != "by_stem":
        raise ValueError(f"Unsupported pairing strategy: {header.pairing.strategy}")

    modalities = {m.name: m for m in header.modalities}
    input_mod = modalities.get(header.pairing.input_modality or "")
    target_mod = modalities.get(header.pairing.target_modality or "") if header.pairing.target_modality else None
    if input_mod is None:
        raise ValueError("Input modality required for pairing")

    items: List[DatasetItem] = []
    if header.splits.manifest_files:
        input_matches = _glob_all_for_modality(input_mod, source_root)
        target_matches = _glob_all_for_modality(target_mod, source_root) if target_mod else {}
        for split_name, manifest_rel in header.splits.manifest_files.items():
            manifest_path = Path(manifest_rel)
            if not manifest_path.is_absolute():
                manifest_path = source_root / manifest_path
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found for split '{split_name}': {manifest_path}")
            stems = _load_manifest_stems(manifest_path)
            for stem in stems:
                input_path = input_matches.get(stem)
                if input_path is None:
                    print(f"[build_shards] Skipping missing input for stem '{stem}' from manifest {manifest_path}")
                    continue
                items.append(
                    DatasetItem(
                        dataset_id=header.dataset_id,
                        item_id=stem,
                        raw_split=split_name,
                        input_path=input_path,
                        target_path=target_matches.get(stem),
                    )
                )
        return items

    for raw_split in header.splits.raw_splits:
        input_matches = _glob_for_modality(input_mod, source_root, raw_split)
        target_matches = _glob_for_modality(target_mod, source_root, raw_split) if target_mod else {}
        all_stems = sorted(set(input_matches) | set(target_matches))
        for stem in all_stems:
            input_path = input_matches.get(stem)
            target_path = target_matches.get(stem)
            if input_path is None:
                raise ValueError(f"Missing input for stem '{stem}' in split '{raw_split}'")
            items.append(
                DatasetItem(
                    dataset_id=header.dataset_id,
                    item_id=stem,
                    raw_split=raw_split,
                    input_path=input_path,
                    target_path=target_path,
                )
            )
    return items


def _assign_splits(items: List[DatasetItem], header: DatasetHeader, seed_override: Optional[int]) -> Dict[str, List[DatasetItem]]:
    if header.splits.manifest_files:
        assignments: Dict[str, List[DatasetItem]] = {"train": [], "val": [], "test": []}
        for item in items:
            split = _normalize_split_name(item.raw_split)
            item.assigned_split = split
            assignments.setdefault(split, []).append(item)
        for name in ("train", "val", "test"):
            assignments.setdefault(name, [])
        return assignments

    if getattr(header.splits, "preserve_raw_split", False):
        assignments: Dict[str, List[DatasetItem]] = {"train": [], "val": [], "test": []}
        for item in items:
            split = _normalize_split_name(item.raw_split)
            item.assigned_split = split
            assignments.setdefault(split, []).append(item)
        for name in ("train", "val", "test"):
            assignments.setdefault(name, [])
        return assignments

    rng = random.Random(seed_override if seed_override is not None else header.splits.seed)
    labeled_fraction = header.splits.train_metric_fraction_of_labeled
    raw_assignments: Dict[str, List[DatasetItem]] = {}

    ordered_items = sorted(items, key=lambda itm: (itm.raw_split, itm.item_id))
    single_train_split = False
    if len(header.splits.raw_splits) == 1:
        raw_name = (header.splits.raw_splits[0] or "").strip().lower()
        single_train_split = raw_name in {"train", "training"}

    if header.splits.ratios:
        rng.shuffle(ordered_items)
        ratios = header.splits.ratios
        assert ratios is not None  # for type checkers
        names_order = ["train", "val", "test"] + sorted({name for name in ratios if name not in {"train", "val", "test"}})
        names_order = [name for name in names_order if name in ratios]
        total_ratio = sum(ratios.values())
        raw_counts = {name: (len(ordered_items) * (ratios[name] / total_ratio)) for name in names_order}
        base_counts = {name: int(count) for name, count in raw_counts.items()}
        remainder = len(ordered_items) - sum(base_counts.values())
        fractional = sorted(
            ((raw_counts[name] - base_counts[name], name) for name in names_order),
            key=lambda pair: (-pair[0], names_order.index(pair[1])),
        )
        idx = 0
        while remainder > 0 and fractional:
            _, name = fractional[idx % len(fractional)]
            base_counts[name] += 1
            remainder -= 1
            idx += 1

        base_lookup: List[str] = []
        for name in names_order:
            base_lookup.extend([name] * base_counts.get(name, 0))
        base_lookup = base_lookup[: len(ordered_items)]
    else:
        if single_train_split:
            base_lookup = ["train"] * len(ordered_items)
        else:
            base_lookup = ["train" if itm.target_path is None else "val" for itm in ordered_items]

    # Precompute base splits for determinism.
    base_assignments: List[str] = []
    for idx, item in enumerate(ordered_items):
        base_split = base_lookup[idx] if idx < len(base_lookup) else ("train" if item.target_path is None else "val")
        base_assignments.append(base_split)

    # Optionally move a subset of labeled items into metrics_train deterministically.
    use_metrics = header.splits.labeled_policy == "validation_only_with_metrics_subset" and labeled_fraction > 0
    candidate_indices = []
    if use_metrics:
        for idx, (item, base_split) in enumerate(zip(ordered_items, base_assignments)):
            if item.target_path is None:
                continue
            if header.splits.ratios is not None and base_split != "train":
                continue
            candidate_indices.append(idx)
        if candidate_indices:
            metrics_count = int(round(labeled_fraction * len(candidate_indices)))
            metrics_count = max(0, min(metrics_count, len(candidate_indices)))
            rng.shuffle(candidate_indices)
            metrics_selection = set(candidate_indices[:metrics_count])
        else:
            metrics_selection = set()
    else:
        metrics_selection = set()

    for idx, item in enumerate(ordered_items):
        base_split = base_assignments[idx]
        split = base_split
        if idx in metrics_selection:
            split = "metrics_train"
        raw_assignments.setdefault(split, []).append(item)

    normalized: Dict[str, List[DatasetItem]] = {"train": [], "val": [], "test": []}
    for split_name, split_items in raw_assignments.items():
        target_split = _normalize_split_name(split_name)
        for item in split_items:
            item.assigned_split = target_split
        normalized.setdefault(target_split, []).extend(split_items)

    for name in ("train", "val", "test"):
        normalized.setdefault(name, [])

    return normalized


def _copy_uncompressed(src: Path, dst: Path, *, force_uncompressed: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as src_ds:
        profile = src_ds.profile.copy()
        profile.pop("compress", None)
        if force_uncompressed:
            profile["compress"] = None
        else:
            compress = src_ds.profile.get("compress")
            if compress:
                profile["compress"] = compress
        with rasterio.open(dst, "w", **profile) as dst_ds:
            for band in range(1, src_ds.count + 1):
                dst_ds.write(src_ds.read(band), band)


def _write_index(shard_dir: Path, entries: List[Dict[str, object]]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path = shard_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")


def _process_item(
    item: DatasetItem,
    header: DatasetHeader,
    shard_dir: Path,
    target_split: str,
    preview_root: Path,
    shard_name: str,
    *,
    generate_preview: bool,
) -> Optional[Tuple[Dict[str, object], Optional[Path]]]:
    inputs_dir = shard_dir / "inputs"
    targets_dir = shard_dir / "targets"
    slic_dir = shard_dir / "slic"

    rgb = _load_rgb_array(item.input_path)
    if _has_white_border(rgb):
        return None

    input_dst = inputs_dir / f"{item.item_id}.tif"
    _copy_uncompressed(item.input_path, input_dst, force_uncompressed=header.sharding.force_uncompressed_tiff)

    slic_result = _compute_and_write_slic(
        input_dst,
        slic_dir,
        item.item_id,
        slic_spec=header.slic or SlicSpec(),
        want_preview=generate_preview,
    )

    target_dst: Optional[Path] = None
    if item.target_path is not None:
        target_dst = targets_dir / f"{item.item_id}.tif"
        _copy_uncompressed(item.target_path, target_dst, force_uncompressed=header.sharding.force_uncompressed_tiff)

    entry: Dict[str, object] = {
        "dataset_id": item.dataset_id,
        "item_id": item.item_id,
        "raw_split": item.raw_split,
        "split": item.assigned_split or target_split,
        "input": str(input_dst.relative_to(shard_dir)),
        "has_target": bool(item.target_path),
        "slic": str((slic_dir / f"{item.item_id}.npz").relative_to(shard_dir)),
        "slic_meta": slic_result["meta"],
    }
    if target_dst is not None:
        entry["target"] = str(target_dst.relative_to(shard_dir))

    preview_path: Optional[Path] = None
    if generate_preview:
        preview_payload = slic_result.get("preview")
        if preview_payload:
            preview_path = _render_preview(
                rgb=preview_payload["rgb"],
                labels_main=preview_payload["labels_main"],
                labels_fine=preview_payload.get("labels_fine"),
                labels_coarse=preview_payload.get("labels_coarse"),
                edges_main=preview_payload["edges_main"],
                edges_fine=preview_payload.get("edges_fine"),
                edges_coarse=preview_payload.get("edges_coarse"),
                labels_seeds=preview_payload.get("labels_seeds"),
                edges_seeds=preview_payload.get("edges_seeds"),
                preview_dir=preview_root / shard_name,
                item_id=item.item_id,
                seed=preview_payload["seed"],
            )

    return entry, preview_path


def _shard_items(
    assignments: Dict[str, List[DatasetItem]],
    header: DatasetHeader,
    output_root: Path,
    *,
    shard_size: int,
    max_items: Optional[int],
    overwrite: bool,
    viz_interval: float = 0.0,
    threads: int = 4,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {"train": {"items": 0, "shards": 0}, "val": {"items": 0, "shards": 0}, "test": {"items": 0, "shards": 0}}

    worker_count = max(1, int(threads) if threads is not None else 1)

    for split, items in assignments.items():
        target_split = _normalize_split_name(split)
        if max_items is not None:
            items = items[:max_items]
        summary.setdefault(target_split, {"items": 0, "shards": 0})
        if not items:
            continue
        kept_count = 0

        split_dir = output_root / target_split / header.dataset_id
        if split_dir.exists():
            if overwrite:
                shutil.rmtree(split_dir)
            elif any(split_dir.iterdir()):
                raise FileExistsError(f"Output split directory already exists: {split_dir}. Use --overwrite to replace it.")
        split_dir.mkdir(parents=True, exist_ok=True)

        preview_root = output_root / "previews" / header.dataset_id / target_split

        for shard_index in range(0, len(items), shard_size):
            shard_items = items[shard_index : shard_index + shard_size]
            shard_name = f"shard-{shard_index // shard_size:05d}"
            shard_dir = split_dir / shard_name

            entries_by_order: List[Optional[Dict[str, object]]] = [None] * len(shard_items)
            previews: List[Tuple[str, Optional[Path]]] = []
            preview_orders = _preview_orders_for_threads(len(shard_items), worker_count)

            # Process preview-designated items synchronously to preserve ordering and avoid thread scheduling skew.
            pending: List[Tuple[int, DatasetItem]] = []
            for order, item in enumerate(shard_items):
                if order in preview_orders:
                    result = _process_item(
                        item,
                        header,
                        shard_dir,
                        target_split,
                        preview_root,
                        shard_name,
                        generate_preview=True,
                    )
                    if result is None:
                        continue
                    entry, preview_path = result
                    entries_by_order[order] = entry
                    if preview_path is not None:
                        previews.append((item.item_id, preview_path))
                else:
                    pending.append((order, item))

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for order, item in pending:
                    futures.append(
                        (
                            order,
                            item,
                            executor.submit(
                                _process_item,
                                item,
                                header,
                                shard_dir,
                                target_split,
                                preview_root,
                                shard_name,
                                generate_preview=False,
                            ),
                        )
                    )

                for order, item, future in futures:
                    result = future.result()
                    if result is None:
                        continue
                    entry, preview_path = result
                    entries_by_order[order] = entry
                    if preview_path is not None:
                        previews.append((item.item_id, preview_path))

            entries: List[Dict[str, object]] = [entry for entry in entries_by_order if entry is not None]
            if not entries:
                continue
            kept_count += len(entries)

            for item_id, preview_path in previews:
                if preview_path is not None:
                    print(f"[viz] {header.dataset_id}/{target_split}/{item_id} -> {preview_path}")

            _write_index(shard_dir, entries)
            summary[target_split]["shards"] = summary[target_split].get("shards", 0) + 1

        summary[target_split]["items"] = kept_count

    for name in ("train", "val", "test"):
        summary.setdefault(name, {"items": 0, "shards": 0})
    return summary


def _build_for_header(
    header: DatasetHeader,
    *,
    extracted_root: Path,
    output_root: Path,
    shard_size: Optional[int],
    seed_override: Optional[int],
    max_items: Optional[int],
    overwrite: bool,
    dry_run: bool,
    viz_interval: float = 0.0,
    threads: int = 4,
) -> Dict[str, Dict[str, int]]:
    source_root = header.resolve_source_root(extracted_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    items = _collect_items(header, source_root)
    assignments = _assign_splits(items, header, seed_override)
    effective_shard_size = shard_size or header.sharding.shard_size

    if dry_run:
        counts = {}
        for split, split_items in assignments.items():
            counts[split] = min(len(split_items), max_items) if max_items is not None else len(split_items)
        print(f"[dry-run] {header.dataset_id} assignments: {counts}")
        return counts  # type: ignore[return-value]

    summary = _shard_items(
        assignments,
        header,
        output_root,
        shard_size=effective_shard_size,
        max_items=max_items,
        overwrite=overwrite,
        viz_interval=viz_interval,
        threads=threads,
    )

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / f"{header.dataset_id}_summary.json"
    summary_payload = {
        "dataset_id": header.dataset_id,
        "version": header.version,
        "layout_version": header.sharding.layout_version,
        "seed": seed_override if seed_override is not None else header.splits.seed,
        "train_metric_fraction_of_labeled": header.splits.train_metric_fraction_of_labeled,
        "split_ratios": header.splits.ratios,
        "splits": summary,
        "slic": {
            "region_size": (header.slic.region_size if header.slic else None),
            "ruler": (header.slic.ruler if header.slic else None),
            "iterations": (header.slic.iterations if header.slic else None),
            "algorithm": (header.slic.algorithm if header.slic else None),
            "min_size": (header.slic.min_size if header.slic else None),
            "max_size": (header.slic.max_size if header.slic else None),
            "scales": (header.slic.scales if header.slic else None),
            "enable_seeds": (header.slic.enable_seeds if header.slic else None),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic shard layout from dataset headers.")
    parser.add_argument("--headers-dir", default="training/datasets/headers", help="Directory containing dataset headers")
    parser.add_argument("--header", help="Path to a single header file")
    parser.add_argument("--dataset-id", help="Only process this dataset id")
    parser.add_argument("--extracted-root", default="training/datasets/extracted", help="Root of extracted datasets")
    parser.add_argument("--output-root", default="training/datasets/processed", help="Where to write shards")
    parser.add_argument("--shard-size", type=int, help="Override shard size")
    parser.add_argument("--seed", type=int, help="Override seed for split assignment")
    parser.add_argument("--max-items", type=int, help="Limit items per split for smoke runs")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing processed data")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing output; print summary only")
    parser.add_argument("--viz-interval", type=float, default=10.0, help="Seconds between preview saves (0 to disable)")
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads for shard building")
    return parser


def _load_header_sources(args: argparse.Namespace) -> List[DatasetHeader]:
    if args.header:
        header = load_header(Path(args.header))
        if args.dataset_id and header.dataset_id != args.dataset_id:
            raise ValueError(f"Header dataset_id {header.dataset_id} does not match filter {args.dataset_id}")
        return [header]
    headers_dir = Path(args.headers_dir)
    if not headers_dir.exists():
        raise FileNotFoundError(f"Headers directory not found: {headers_dir}")
    return load_headers(headers_dir, dataset_id=args.dataset_id)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    headers = _load_header_sources(args)
    extracted_root = Path(args.extracted_root).expanduser().resolve()
    if not extracted_root.exists():
        fallback = Path(__file__).resolve().parent / "data" / "extracted"
        if fallback.exists():
            extracted_root = fallback
            print(f"Using fallback extracted root: {extracted_root}")
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists() and args.output_root == "training/datasets/processed":
        fallback_out = Path(__file__).resolve().parent / "data" / "processed"
        if fallback_out.exists():
            output_root = fallback_out
            print(f"Using fallback output root: {output_root}")

    if not args.dry_run and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for header in headers:
        print(f"Processing dataset {header.dataset_id}...")
        summary = _build_for_header(
            header,
            extracted_root=extracted_root,
            output_root=output_root,
            shard_size=args.shard_size,
            seed_override=args.seed,
            max_items=args.max_items,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            viz_interval=args.viz_interval,
            threads=args.threads,
        )
        if args.dry_run:
            print(f"[dry-run] summary for {header.dataset_id}: {summary}")
        else:
            print(f"Finished {header.dataset_id}; summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
