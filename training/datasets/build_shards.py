# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import rasterio
import shutil

from .header_schema import DatasetHeader, ModalitySpec, load_header, load_headers


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


def _shard_items(assignments: Dict[str, List[DatasetItem]], header: DatasetHeader, output_root: Path, *, shard_size: int, max_items: Optional[int], overwrite: bool) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {"train": {"items": 0, "shards": 0}, "val": {"items": 0, "shards": 0}, "test": {"items": 0, "shards": 0}}

    for split, items in assignments.items():
        target_split = _normalize_split_name(split)
        if max_items is not None:
            items = items[:max_items]
        summary.setdefault(target_split, {"items": 0, "shards": 0})
        summary[target_split]["items"] = len(items)
        if not items:
            continue

        split_dir = output_root / target_split / header.dataset_id
        if split_dir.exists():
            if overwrite:
                shutil.rmtree(split_dir)
            elif any(split_dir.iterdir()):
                raise FileExistsError(f"Output split directory already exists: {split_dir}. Use --overwrite to replace it.")
        split_dir.mkdir(parents=True, exist_ok=True)

        for shard_index in range(0, len(items), shard_size):
            shard_items = items[shard_index : shard_index + shard_size]
            shard_name = f"shard-{shard_index // shard_size:05d}"
            shard_dir = split_dir / shard_name
            inputs_dir = shard_dir / "inputs"
            targets_dir = shard_dir / "targets"
            entries: List[Dict[str, object]] = []
            for item in shard_items:
                input_dst = inputs_dir / f"{item.item_id}.tif"
                _copy_uncompressed(item.input_path, input_dst, force_uncompressed=header.sharding.force_uncompressed_tiff)
                entry: Dict[str, object] = {
                    "dataset_id": item.dataset_id,
                    "item_id": item.item_id,
                    "raw_split": item.raw_split,
                    "split": item.assigned_split or target_split,
                    "input": str(input_dst.relative_to(shard_dir)),
                    "has_target": bool(item.target_path),
                }
                if item.target_path is not None:
                    target_dst = targets_dir / f"{item.item_id}.tif"
                    _copy_uncompressed(item.target_path, target_dst, force_uncompressed=header.sharding.force_uncompressed_tiff)
                    entry["target"] = str(target_dst.relative_to(shard_dir))
                entries.append(entry)
            _write_index(shard_dir, entries)
            summary[target_split]["shards"] = summary[target_split].get("shards", 0) + 1
    for name in ("train", "val", "test"):
        summary.setdefault(name, {"items": 0, "shards": 0})
    return summary


def _build_for_header(header: DatasetHeader, *, extracted_root: Path, output_root: Path, shard_size: Optional[int], seed_override: Optional[int], max_items: Optional[int], overwrite: bool, dry_run: bool) -> Dict[str, Dict[str, int]]:
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

    summary = _shard_items(assignments, header, output_root, shard_size=effective_shard_size, max_items=max_items, overwrite=overwrite)

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
        )
        if args.dry_run:
            print(f"[dry-run] summary for {header.dataset_id}: {summary}")
        else:
            print(f"Finished {header.dataset_id}; summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
