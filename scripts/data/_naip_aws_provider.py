# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""NAIP-on-AWS discovery helpers.

This module isolates NAIP bucket/index knowledge so dataset prep scripts can
swap sources without touching tiling/warping logic. It intentionally uses GDAL
CLIs (via _gdal_utils) plus lightweight HTTP requests; no heavy geospatial
Python deps are required.

Key facts from AWS Open Data docs (https://registry.opendata.aws/naip/):
- Buckets (Requester Pays, us-west-2):
  - naip-visualization (RGB COG, JPEG YCbCr, 512x512 blocks)
  - naip-analytic (4-band COG/MRF)
  - naip-source (raw 4-band GeoTIFF)
- Directory structure: <state>/<year>/<res_cm>/<band_variant>/.../file.tif
  with index shapefiles under .../index/*.shp and manifests under manifest.txt.
- Requester Pays: use AWS_REQUEST_PAYER=requester (and AWS_REGION=us-west-2) for
  vsis3/HTTPS access.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import requests

from ._gdal_utils import build_vrt, subset_vector_by_bbox

Logger = Optional[Callable[[str], None]]

NAIP_BUCKET_VIS = "naip-visualization"
NAIP_BUCKET_ANALYTIC = "naip-analytic"
NAIP_BUCKET_SOURCE = "naip-source"

DEFAULT_INDEX_URLS = [
    # GeoPackage footprint published alongside visualization COGs.
    f"https://{NAIP_BUCKET_VIS}.s3.amazonaws.com/index/naip_footprint.gpkg",
    f"https://{NAIP_BUCKET_ANALYTIC}.s3.amazonaws.com/index/naip_footprint.gpkg",
]


@dataclass
class NaipAwsConfig:
    bucket: str = NAIP_BUCKET_VIS
    requester_pays: bool = True
    region: str = "us-west-2"
    index_url: Optional[str] = None
    allow_https: bool = True


@dataclass
class NaipTileRef:
    tile_id: str
    url: str
    state: Optional[str]
    year: Optional[str]
    resolution_cm: Optional[int]
    bbox_wgs84: Sequence[float]

    def gdal_source(self, use_vsis3: bool = True) -> str:
        return to_gdal_url(self.url, use_vsis3=use_vsis3)


class NaipIndex:
    """Index wrapper that clips footprint GeoPackage/GeoJSON via ogr2ogr."""

    def __init__(self, cache_dir: Path, cfg: NaipAwsConfig, index_path: Optional[Path] = None, refresh: bool = False, logger: Logger = None):
        self.cache_dir = cache_dir
        self.cfg = cfg
        self.logger = logger
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self._ensure_index(index_path, refresh)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)

    def _ensure_index(self, index_path: Optional[Path], refresh: bool) -> Path:
        if index_path:
            path = index_path.expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"NAIP index not found: {path}")
            return path
        urls = [self.cfg.index_url] if self.cfg.index_url else DEFAULT_INDEX_URLS
        first_error: Optional[Exception] = None
        for url in urls:
            if not url:
                continue
            filename = url.split("/")[-1]
            target = self.cache_dir / filename
            if target.exists() and not refresh:
                self._log(f"Using cached NAIP index: {target}")
                return target
            try:
                self._download(url, target)
                return target
            except Exception as exc:  # pragma: no cover - best-effort fetch
                first_error = exc
                self._log(f"Failed to download index {url}: {exc}")
                continue
        raise RuntimeError(
            "Unable to obtain NAIP index. Provide --naip-index-path or set NAIP index URL."
        ) from first_error

    def _download(self, url: str, target: Path) -> None:
        self._log(f"Downloading NAIP index -> {target}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def query(self, bbox_wgs84: Sequence[float]) -> List[NaipTileRef]:
        subset = subset_vector_by_bbox(self.index_path, tuple(bbox_wgs84), logger=self.logger)
        with subset.open() as f:
            data = json.load(f)
        features = data.get("features", [])
        refs: List[NaipTileRef] = []
        for feat in features:
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            bbox = geom.get("bbox") or bbox_wgs84
            url = _extract_url(props)
            tile_id = str(props.get("name") or props.get("objectid") or props.get("OBJECTID") or props.get("title") or len(refs))
            state = props.get("state") or props.get("STATE")
            year = props.get("year") or props.get("YEAR") or props.get("acqYear")
            res_cm = _extract_resolution_cm(props)
            if not url:
                continue
            refs.append(NaipTileRef(tile_id=tile_id, url=url, state=state, year=str(year) if year else None, resolution_cm=res_cm, bbox_wgs84=bbox))
        return refs


def to_gdal_url(url: str, use_vsis3: bool = True) -> str:
    if url.startswith("s3://"):
        if not use_vsis3:
            return url
        without = url[len("s3://") :]
        return f"/vsis3/{without}"
    if url.startswith("http"):
        return url
    # Assume bucket/key
    if use_vsis3:
        return f"/vsis3/{url}"
    return f"s3://{url}"


def build_naip_vrt(urls: Iterable[str], vrt_path: Path, requester_pays: bool = True, logger: Logger = None) -> None:
    env = None
    if requester_pays:
        env = os.environ.copy()
        env.setdefault("AWS_REQUEST_PAYER", "requester")
    sources = [to_gdal_url(u, use_vsis3=True) for u in urls]
    build_vrt(sources, vrt_path, env=env, logger=logger)


def _extract_url(props: dict) -> Optional[str]:
    candidates = [
        props.get("location"),
        props.get("Location"),
        props.get("download_url"),
        props.get("downloadURL"),
        props.get("url"),
        props.get("URL"),
        props.get("path"),
        props.get("s3_path"),
        props.get("S3Path"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _extract_resolution_cm(props: dict) -> Optional[int]:
    res_keys = ["resolution", "RESOLUTION", "res_cm", "resolution_cm"]
    for key in res_keys:
        if key in props:
            try:
                val = props[key]
                if isinstance(val, str) and val.endswith("cm"):
                    val = val.replace("cm", "")
                return int(round(float(val)))
            except Exception:
                continue
    return None
