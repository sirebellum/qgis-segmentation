# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import rasterio
import yaml

from .header_schema import DatasetHeader, ModalitySpec, PairingSpec, ShardingSpec, SplitSpec, ValidationSpec


def _scan_sample(path: Path) -> Dict[str, object]:
    with rasterio.open(path) as src:
        data = src.read()
        unique_values = np.unique(data)
        compress = src.profile.get("compress")
        return {
            "shape": (src.count, src.height, src.width),
            "dtype": src.dtypes[0],
            "compress": compress,
            "crs": str(src.crs) if src.crs else None,
            "unique_values": unique_values.tolist() if unique_values.size <= 8 else None,
        }


def _collect_pairs(root: Path, split_names: Iterable[str]) -> Tuple[List[Path], List[Path]]:
    sat_files: List[Path] = []
    map_files: List[Path] = []
    for split in split_names:
        sat_files.extend(sorted((root / split / "sat").glob("*.tiff")))
        map_files.extend(sorted((root / split / "map").glob("*.tif")))
    return sat_files, map_files


def _ms_buildings_header(source_root: Path) -> Tuple[DatasetHeader, Dict[str, object]]:
    split_names = [d.name for d in sorted(source_root.iterdir()) if d.is_dir() and (d / "sat").exists() and (d / "map").exists()]
    if not split_names:
        raise FileNotFoundError(f"No splits with sat/map subdirs found under {source_root}")

    sat_files, map_files = _collect_pairs(source_root, split_names)
    if not sat_files or not map_files:
        raise FileNotFoundError("ms_buildings requires sat and map files; none found")

    sat_sample = _scan_sample(sat_files[0])
    map_sample = _scan_sample(map_files[0])

    sat_channels = sat_sample["shape"][0]
    map_channels = map_sample["shape"][0]
    map_values = map_sample.get("unique_values") or []

    header = DatasetHeader(
        dataset_id="ms_buildings",
        version="0.1.0",
        description="Microsoft building footprints tiles (Massachusetts); sat imagery paired with binary building masks.",
        source_root=source_root.name,
        modalities=[
            ModalitySpec(
                name="sat",
                role="input",
                kind="raster",
                pattern="{split}/sat/*.tiff",
                channels=int(sat_channels),
                dtype=str(sat_sample.get("dtype", "uint8")),
                georef_required=False,
            ),
            ModalitySpec(
                name="map",
                role="target",
                kind="raster",
                pattern="{split}/map/*.tif",
                channels=int(map_channels),
                dtype=str(map_sample.get("dtype", "uint8")),
                georef_required=False,
                label_values=[int(v) for v in map_values] if map_values else None,
            ),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(
            raw_splits=split_names,
            seed=123,
            labeled_policy="validation_only_with_metrics_subset",
            train_metric_fraction_of_labeled=0.25,
            ratios={"train": 0.6, "val": 0.3, "test": 0.1},
        ),
        sharding=ShardingSpec(shard_size=512, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    stats = {
        "splits": {split: {"sat": len(list((source_root / split / "sat").glob("*.tiff"))), "map": len(list((source_root / split / "map").glob("*.tif")))} for split in split_names},
        "sat_sample": sat_sample,
        "map_sample": map_sample,
    }
    return header, stats


def _whu_building_header(source_root: Path) -> Tuple[DatasetHeader, Dict[str, object]]:
    preferred_order = ["train", "validation", "test"]
    split_names = [name for name in preferred_order if (source_root / name).exists()]
    if not split_names:
        split_names = [d.name for d in sorted(source_root.iterdir()) if d.is_dir()]
    if not split_names:
        raise FileNotFoundError(f"No splits found under {source_root}")

    files: List[Path] = []
    for split in split_names:
        files.extend(sorted((source_root / split).glob("*.TIF")))
    if not files:
        raise FileNotFoundError("whu_building requires imagery files; none found")

    sample = _scan_sample(files[0])
    channels = sample["shape"][0]

    header = DatasetHeader(
        dataset_id="whu_building",
        version="0.1.0",
        description="WHU building dataset tiles; RGB imagery only (labels not bundled here).",
        source_root=source_root.name,
        modalities=[
            ModalitySpec(
                name="sat",
                role="input",
                kind="raster",
                pattern="{split}/*.TIF",
                channels=int(channels),
                dtype=str(sample.get("dtype", "uint8")),
                georef_required=False,
            ),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality=None),
        splits=SplitSpec(
            raw_splits=split_names,
            seed=123,
            labeled_policy="validation_only_with_metrics_subset",
            train_metric_fraction_of_labeled=0.0,
            ratios={"train": 0.6, "val": 0.3, "test": 0.1},
        ),
        sharding=ShardingSpec(shard_size=512, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    stats = {
        "splits": {split: {"sat": len(list((source_root / split).glob("*.TIF")))} for split in split_names},
        "sat_sample": sample,
    }
    return header, stats


def _openearth_header(source_root: Path) -> Tuple[DatasetHeader, Dict[str, object]]:
    manifest_map = {
        "train": source_root / "train.txt",
        "val": source_root / "val.txt",
        "test": source_root / "test.txt",
    }
    for name, path in manifest_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest for split '{name}': {path}")

    def _load_names(path: Path) -> List[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_list = _load_names(manifest_map["train"])
    val_list = _load_names(manifest_map["val"])
    test_list = _load_names(manifest_map["test"])
    sample_name = (train_list or val_list or test_list)[0]
    city_prefix = sample_name.split("_")[0]
    sample_input = source_root / city_prefix / "images" / sample_name
    if not sample_input.exists():
        raise FileNotFoundError(f"Sample input not found: {sample_input}")
    sat_sample = _scan_sample(sample_input)

    sample_label = source_root / city_prefix / "labels" / sample_name
    label_sample = _scan_sample(sample_label) if sample_label.exists() else None

    header = DatasetHeader(
        dataset_id="openearth",
        version="0.1.0",
        description="OpenEarth imagery/label tiles with explicit train/val/test manifests across multiple cities.",
        source_root=source_root.name,
        modalities=[
            ModalitySpec(
                name="sat",
                role="input",
                kind="raster",
                pattern="{split}/images/*.tif",
                channels=int(sat_sample.get("shape", (3,))[0]),
                dtype=str(sat_sample.get("dtype", "uint8")),
                georef_required=False,
            ),
            ModalitySpec(
                name="map",
                role="target",
                kind="raster",
                pattern="{split}/labels/*.tif",
                channels=int((label_sample or {"shape": (1,)}).get("shape", (1,))[0]),
                dtype=str((label_sample or {"dtype": "uint8"}).get("dtype", "uint8")),
                georef_required=False,
                label_values=label_sample.get("unique_values") if isinstance(label_sample, dict) else None,
            ),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(
            raw_splits=["all"],
            seed=123,
            labeled_policy="validation_only_with_metrics_subset",
            train_metric_fraction_of_labeled=0.0,
            ratios=None,
            manifest_files={k: v.name for k, v in manifest_map.items()},
        ),
        sharding=ShardingSpec(shard_size=512, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    stats = {
        "counts": {"train": len(train_list), "val": len(val_list), "test": len(test_list)},
        "sat_sample": sat_sample,
        "map_sample": label_sample,
    }
    return header, stats


def _inria_header(source_root: Path) -> Tuple[DatasetHeader, Dict[str, object]]:
    split_names = []
    for name in ("train", "val", "test"):
        split_path = source_root / name
        if split_path.exists():
            split_names.append(name)
    if not split_names:
        split_names = [d.name for d in sorted(source_root.iterdir()) if d.is_dir()]
    if not split_names:
        raise FileNotFoundError(f"No splits found under {source_root}")

    train_dir = source_root / "train"
    sample_img = next((train_dir / "images").glob("*.tif"), None)
    sample_map = next((train_dir / "gt").glob("*.tif"), None)
    if sample_img is None:
        raise FileNotFoundError(f"No imagery found under {train_dir}/images")
    sat_sample = _scan_sample(sample_img)
    map_sample = _scan_sample(sample_map) if sample_map is not None else None

    header = DatasetHeader(
        dataset_id="inria",
        version="0.1.0",
        description="Inria aerial building dataset; imagery under images/, labels under gt/ (train split labeled, test split unlabeled).",
        source_root=source_root.name,
        modalities=[
            ModalitySpec(
                name="sat",
                role="input",
                kind="raster",
                pattern="{split}/images/*.tif",
                channels=int(sat_sample.get("shape", (3,))[0]),
                dtype=str(sat_sample.get("dtype", "uint8")),
                georef_required=False,
            ),
            ModalitySpec(
                name="map",
                role="target",
                kind="raster",
                pattern="{split}/gt/*.tif",
                channels=int((map_sample or {"shape": (1,)}).get("shape", (1,))[0]),
                dtype=str((map_sample or {"dtype": "uint8"}).get("dtype", "uint8")),
                georef_required=False,
                label_values=map_sample.get("unique_values") if isinstance(map_sample, dict) else None,
            ),
        ],
        pairing=PairingSpec(strategy="by_stem", input_modality="sat", target_modality="map"),
        splits=SplitSpec(
            raw_splits=split_names,
            seed=123,
            labeled_policy="validation_only_with_metrics_subset",
            train_metric_fraction_of_labeled=0.0,
            ratios=None,
            manifest_files=None,
            preserve_raw_split=True,
        ),
        sharding=ShardingSpec(shard_size=1, layout_version=1, copy_mode="copy", force_uncompressed_tiff=True),
        validation=ValidationSpec(iou_ignore_label_leq=0, iou_average="macro_over_present_labels"),
        header_path=None,
    )

    stats = {
        "splits": {
            split: {
                "sat": len(list((source_root / split / "images").glob("*.tif"))),
                "map": len(list((source_root / split / "gt").glob("*.tif"))),
            }
            for split in split_names
        },
        "sat_sample": sat_sample,
        "map_sample": map_sample,
    }
    return header, stats


_GENERATOR_MAP: Dict[str, Callable[[Path], Tuple[DatasetHeader, Dict[str, object]]]] = {
    "ms_buildings": _ms_buildings_header,
    "whu_building": _whu_building_header,
    "openearth": _openearth_header,
    "inria": _inria_header,
}


def _discover_datasets(extracted_root: Path) -> List[str]:
    found: List[str] = []
    for name in sorted(_GENERATOR_MAP):
        if (extracted_root / name).exists():
            found.append(name)
    return found


def _write_header(path: Path, header: DatasetHeader) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(header)
    payload.pop("header_path", None)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# SPDX-License-Identifier: BSD-3-Clause\n")
        handle.write("# Generated by training/datasets/generate_headers.py\n")
        yaml.safe_dump(payload, handle, sort_keys=False)


def _print_stats(stats: Dict[str, object]) -> None:
    print("Split counts:")
    splits = stats.get("splits", {}) if isinstance(stats, dict) else {}
    for name, counts in splits.items():
        sat = counts.get("sat", 0)
        map_count = counts.get("map", 0)
        print(f"  {name}: sat={sat}, map={map_count}")
    sat_sample = stats.get("sat_sample") if isinstance(stats, dict) else None
    map_sample = stats.get("map_sample") if isinstance(stats, dict) else None
    if sat_sample:
        print("Sample sat metadata:", sat_sample)
    if map_sample:
        print("Sample map metadata:", map_sample)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset header YAML files from extracted data.")
    parser.add_argument("--dataset", help="Dataset id to generate (e.g., ms_buildings). If omitted, process all supported datasets found under extracted-root.")
    parser.add_argument("--extracted-root", default="training/datasets/extracted", help="Root containing extracted datasets")
    parser.add_argument("--headers-dir", default="training/datasets/headers", help="Where to write generated headers")
    parser.add_argument("--dry-run", action="store_true", help="Print inferred stats without writing header")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = (args.dataset or "").strip().lower()
    extracted_root = Path(args.extracted_root).expanduser().resolve()
    if not extracted_root.exists():
        fallback = Path(__file__).resolve().parent / "data" / "extracted"
        if fallback.exists():
            extracted_root = fallback
            print(f"Using fallback extracted root: {extracted_root}")
    headers_dir = Path(args.headers_dir).expanduser().resolve()
    candidates: List[str]
    if dataset:
        if dataset not in _GENERATOR_MAP:
            parser.error(f"Unsupported dataset '{dataset}'. Extend generate_headers.py to add a generator.")
        candidates = [dataset]
    else:
        candidates = _discover_datasets(extracted_root)
        if not candidates:
            parser.error(f"No supported datasets found under {extracted_root}. Supported: {', '.join(sorted(_GENERATOR_MAP))}")

    for ds in candidates:
        source_root = extracted_root / ds
        if not source_root.exists():
            print(f"Skipping {ds}: source root not found at {source_root}")
            continue
        generator = _GENERATOR_MAP.get(ds)
        if generator is None:
            print(f"Skipping {ds}: no generator available.")
            continue

        header, stats = generator(source_root)
        print(f"=== {ds} ===")
        _print_stats(stats)
        if args.dry_run:
            continue
        header_path = headers_dir / f"{header.dataset_id}.yaml"
        _write_header(header_path, header)
        print(f"Wrote header to {header_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
