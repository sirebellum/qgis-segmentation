# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ALLOWED_ROLES = {"input", "target"}
_ALLOWED_PAIRING = {"by_stem", "none"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    role: str
    kind: str
    pattern: str
    channels: Optional[int] = None
    dtype: Optional[str] = None
    nodata: Optional[float] = None
    georef_required: bool = False
    label_values: Optional[List[int]] = None


@dataclass(frozen=True)
class PairingSpec:
    strategy: str
    input_modality: Optional[str]
    target_modality: Optional[str]


@dataclass(frozen=True)
class SplitSpec:
    raw_splits: List[str]
    seed: int = 123
    labeled_policy: str = "validation_only_with_metrics_subset"
    train_metric_fraction_of_labeled: float = 0.25
    ratios: Optional[Dict[str, float]] = None
    manifest_files: Optional[Dict[str, str]] = None
    preserve_raw_split: bool = False


@dataclass(frozen=True)
class ShardingSpec:
    shard_size: int = 512
    layout_version: int = 1
    copy_mode: str = "copy"
    force_uncompressed_tiff: bool = True


@dataclass(frozen=True)
class SlicSpec:
    region_size: int = 32
    ruler: float = 10.0
    iterations: int = 10
    algorithm: str = "slico"
    min_size: int = 8
    max_size: int = 64
    scales: Optional[List[float]] = None
    enable_seeds: bool = True


@dataclass(frozen=True)
class ValidationSpec:
    iou_ignore_label_leq: int = 0
    iou_average: str = "macro_over_present_labels"


@dataclass(frozen=True)
class DatasetHeader:
    dataset_id: str
    version: str
    description: str
    source_root: str
    modalities: List[ModalitySpec]
    pairing: PairingSpec
    splits: SplitSpec
    sharding: ShardingSpec
    validation: ValidationSpec
    has_target_maps: bool = False
    target_path_pattern: Optional[str] = None
    target_description: Optional[str] = None
    slic: Optional[SlicSpec] = None
    header_path: Optional[Path] = None

    def resolve_source_root(self, extracted_root: Optional[Path] = None) -> Path:
        root = Path(self.source_root)
        if not root.is_absolute() and extracted_root is not None:
            root = Path(extracted_root) / root
        return root


def _parse_modality(raw: Dict[str, Any]) -> ModalitySpec:
    name = str(raw.get("name", "")).strip()
    role = str(raw.get("role", "")).strip().lower()
    kind = str(raw.get("kind", "")).strip().lower()
    pattern = str(raw.get("pattern", "")).strip()
    _require(name, "modality.name is required")
    _require(role in _ALLOWED_ROLES, f"modality.role must be one of {_ALLOWED_ROLES}")
    _require(kind == "raster", "only raster modalities are supported in v0")
    _require(pattern, "modality.pattern is required")
    channels = raw.get("channels")
    if channels is not None:
        channels = int(channels)
    if role == "input" and channels is not None and channels != 3:
        raise ValueError("input modality must declare exactly 3 channels (RGB-only)")
    dtype = raw.get("dtype")
    nodata = raw.get("nodata")
    if nodata is not None:
        nodata = float(nodata)
    georef_required = bool(raw.get("georef_required", False))
    label_values = raw.get("label_values")
    if label_values is not None:
        if not isinstance(label_values, (list, tuple)):
            raise ValueError("modality.label_values must be a list when provided")
        label_values = [int(v) for v in label_values]
    return ModalitySpec(
        name=name,
        role=role,
        kind=kind,
        pattern=pattern,
        channels=channels,
        dtype=dtype,
        nodata=nodata,
        georef_required=georef_required,
        label_values=label_values,
    )


def _parse_pairing(raw: Dict[str, Any], modalities: List[ModalitySpec]) -> PairingSpec:
    names = {m.name for m in modalities}
    strategy = str(raw.get("strategy", "by_stem") or "by_stem").strip().lower()
    _require(strategy in _ALLOWED_PAIRING, f"pairing.strategy must be one of {_ALLOWED_PAIRING}")
    input_modality = raw.get("input_modality") or raw.get("input")
    target_modality = raw.get("target_modality") or raw.get("target")
    if input_modality:
        _require(input_modality in names, f"pairing.input_modality '{input_modality}' not found in modalities")
    if target_modality:
        _require(target_modality in names, f"pairing.target_modality '{target_modality}' not found in modalities")
    if strategy == "by_stem":
        _require(input_modality, "pairing.input_modality is required for by_stem")
    return PairingSpec(strategy=strategy, input_modality=input_modality, target_modality=target_modality)


def _parse_splits(raw: Dict[str, Any]) -> SplitSpec:
    raw_splits = raw.get("raw") or raw.get("raw_splits") or ["training", "val", "test"]
    if not isinstance(raw_splits, (list, tuple)):
        raise ValueError("splits.raw must be a list of split names")
    raw_splits = [str(s).strip() for s in raw_splits if str(s).strip()]
    _require(raw_splits, "splits.raw must include at least one split name")
    seed = int(raw.get("seed", 123))
    labeled_policy = str(raw.get("labeled_policy", "validation_only_with_metrics_subset") or "validation_only_with_metrics_subset")
    fraction = float(raw.get("train_metric_fraction_of_labeled", raw.get("metrics_fraction", 0.25)))
    _require(0.0 <= fraction <= 1.0, "splits.train_metric_fraction_of_labeled must be between 0 and 1")
    ratios_raw = raw.get("ratios")
    ratios: Optional[Dict[str, float]] = None
    if ratios_raw is not None:
        if not isinstance(ratios_raw, dict):
            raise ValueError("splits.ratios must be a mapping of split name to fraction")
        ratios = {}
        for key, value in ratios_raw.items():
            name = str(key).strip()
            if not name:
                continue
            val = float(value)
            _require(val >= 0.0, "splits.ratios values must be non-negative")
            ratios[name] = val
        _require(ratios and sum(ratios.values()) > 0.0, "splits.ratios must include at least one positive value")
    manifests_raw = raw.get("manifests") or raw.get("manifest_files")
    manifests: Optional[Dict[str, str]] = None
    if manifests_raw is not None:
        if not isinstance(manifests_raw, dict):
            raise ValueError("splits.manifests must be a mapping of split name to manifest path")
        manifests = {}
        for key, value in manifests_raw.items():
            name = str(key).strip()
            path_str = str(value).strip()
            if name and path_str:
                manifests[name] = path_str
        _require(bool(manifests), "splits.manifests must include at least one entry when provided")
    preserve_raw = bool(raw.get("preserve_raw_split", False))
    return SplitSpec(
        raw_splits=raw_splits,
        seed=seed,
        labeled_policy=labeled_policy,
        train_metric_fraction_of_labeled=fraction,
        ratios=ratios,
        manifest_files=manifests,
        preserve_raw_split=preserve_raw,
    )


def _parse_sharding(raw: Dict[str, Any]) -> ShardingSpec:
    shard_size = int(raw.get("shard_size", 512))
    _require(shard_size > 0, "sharding.shard_size must be positive")
    layout_version = int(raw.get("layout_version", 1))
    copy_mode = str(raw.get("copy_mode", "copy") or "copy")
    force_uncompressed = bool(raw.get("force_uncompressed_tiff", True))
    return ShardingSpec(
        shard_size=shard_size,
        layout_version=layout_version,
        copy_mode=copy_mode,
        force_uncompressed_tiff=force_uncompressed,
    )


def _parse_slic(raw: Optional[Dict[str, Any]]) -> SlicSpec:
    data = raw or {}
    region_size = int(data.get("region_size", 32))
    ruler = float(data.get("ruler", 10.0))
    iterations = int(data.get("iterations", 10))
    algorithm = str(data.get("algorithm", "slico") or "slico").strip().lower()
    min_size = int(data.get("min_size", 8))
    max_size = int(data.get("max_size", 64))
    scales_raw = data.get("scales")
    scales: Optional[List[float]] = None
    if scales_raw is not None:
        if not isinstance(scales_raw, (list, tuple)):
            raise ValueError("slic.scales must be a list of scale multipliers when provided")
        scales = [float(s) for s in scales_raw]
    enable_seeds = bool(data.get("enable_seeds", True))
    _require(region_size > 0, "slic.region_size must be positive")
    _require(iterations > 0, "slic.iterations must be positive")
    _require(algorithm in {"slic", "slico", "mslico"}, "slic.algorithm must be slic|slico|mslico")
    _require(min_size > 0, "slic.min_size must be positive")
    _require(max_size >= min_size, "slic.max_size must be >= min_size")
    return SlicSpec(
        region_size=region_size,
        ruler=ruler,
        iterations=iterations,
        algorithm=algorithm,
        min_size=min_size,
        max_size=max_size,
        scales=scales,
        enable_seeds=enable_seeds,
    )


def _parse_validation(raw: Dict[str, Any]) -> ValidationSpec:
    ignore = int(raw.get("iou_ignore_label_leq", 0))
    average = str(raw.get("iou_average", "macro_over_present_labels") or "macro_over_present_labels")
    return ValidationSpec(iou_ignore_label_leq=ignore, iou_average=average)


def load_header(path: Path, *, extracted_root: Optional[Path] = None) -> DatasetHeader:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Header file {path} must contain a mapping at the top level.")

    dataset_id = str(data.get("dataset_id", "")).strip()
    version = str(data.get("version", "")).strip() or "0"
    description = str(data.get("description", "")).strip()
    source_root = str(data.get("source_root", "")).strip()

    _require(dataset_id, "dataset_id is required")
    _require(description, "description is required")
    _require(source_root, "source_root is required")

    modalities_raw = data.get("modalities") or []
    if not isinstance(modalities_raw, (list, tuple)):
        raise ValueError("modalities must be a list")
    modalities = [_parse_modality(raw_mod) for raw_mod in modalities_raw]
    _require(modalities, "at least one modality is required")

    pairing_raw = data.get("pairing") or {}
    pairing = _parse_pairing(pairing_raw, modalities)

    splits_raw = data.get("splits") or {}
    splits = _parse_splits(splits_raw)

    sharding_raw = data.get("sharding") or {}
    sharding = _parse_sharding(sharding_raw)

    validation_raw = data.get("validation") or data.get("validation_metrics") or {}
    validation = _parse_validation(validation_raw)

    has_target_maps = bool(data.get("has_target_maps", False))
    target_path_pattern = data.get("target_path_pattern")
    if target_path_pattern is not None:
        target_path_pattern = str(target_path_pattern).strip() or None
    target_description = data.get("target_description")
    if target_description is not None:
        target_description = str(target_description).strip() or None

    slic_raw = data.get("slic") or {}
    slic = _parse_slic(slic_raw)

    # Enforce RGB-only inputs and validate target presence when declared.
    input_modalities = [m for m in modalities if m.role == "input"]
    target_modalities = [m for m in modalities if m.role == "target"]
    if not input_modalities:
        raise ValueError("at least one input modality is required")
    for mod in input_modalities:
        if mod.channels is not None and mod.channels != 3:
            raise ValueError("input modality must be 3-band RGB")

    if target_modalities:
        has_target_maps = True if data.get("has_target_maps", None) is None else has_target_maps

    if has_target_maps and not target_modalities:
        raise ValueError("has_target_maps=true requires a target modality with role 'target'")
    if not has_target_maps and target_modalities:
        raise ValueError("target modality present but has_target_maps is false; set has_target_maps=true")

    return DatasetHeader(
        dataset_id=dataset_id,
        version=version,
        description=description,
        source_root=source_root,
        modalities=modalities,
        pairing=pairing,
        splits=splits,
        sharding=sharding,
        validation=validation,
        has_target_maps=has_target_maps,
        target_path_pattern=target_path_pattern,
        target_description=target_description,
        slic=slic,
        header_path=Path(path),
    )


def load_headers(headers_dir: Path, *, dataset_id: Optional[str] = None) -> List[DatasetHeader]:
    headers: List[DatasetHeader] = []
    for path in sorted(Path(headers_dir).glob("*.yaml")):
        header = load_header(path)
        if dataset_id is None or header.dataset_id == dataset_id:
            headers.append(header)
    if dataset_id is not None and not headers:
        raise ValueError(f"No header found for dataset_id={dataset_id} in {headers_dir}")
    return headers
