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
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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


STUB_INDEX_FEATURES = [
    {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "bbox": [-125.0, 24.0, -66.0, 50.0],
            "coordinates": [
                [
                    [-125.0, 24.0],
                    [-66.0, 24.0],
                    [-66.0, 50.0],
                    [-125.0, 50.0],
                    [-125.0, 24.0],
                ]
            ],
        },
        "properties": {
            "name": "naip_stub_conus",
            "location": f"s3://{NAIP_BUCKET_VIS}/stub/conus_stub.tif",
            "state": "CONUS",
            "year": "stub",
            "resolution": 100,
        },
    }
]
STUB_INDEX_FILENAME = "naip_index_stub.geojson"


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

    def __init__(
        self,
        cache_dir: Path,
        cfg: NaipAwsConfig,
        index_path: Optional[Path] = None,
        refresh: bool = False,
        logger: Logger = None,
        allow_stub: bool = False,
    ):
        self.cache_dir = cache_dir
        self.cfg = cfg
        self.logger = logger
        self.allow_stub = allow_stub
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
        cached = self._first_cached(urls)
        if cached is not None and not refresh:
            self._log(f"Using cached NAIP index: {cached}")
            return cached
        if self.allow_stub:
            stub = self._write_stub_index(self.cache_dir / STUB_INDEX_FILENAME)
            self._log(f"Dry-run: using embedded NAIP index stub at {stub}")
            return stub
        first_error: Optional[Exception] = None
        for url in urls:
            if not url:
                continue
            filename = url.split("/")[-1]
            target = self.cache_dir / filename
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

    def _first_cached(self, urls: Iterable[str]) -> Optional[Path]:
        for url in urls:
            if not url:
                continue
            filename = url.split("/")[-1]
            target = self.cache_dir / filename
            if target.exists():
                return target
        return None

    def _write_stub_index(self, target: Path) -> Path:
        doc = {"type": "FeatureCollection", "features": STUB_INDEX_FEATURES}
        target.write_text(json.dumps(doc))
        return target

    def _download(self, url: str, target: Path) -> None:
        self._log(f"Downloading NAIP index -> {target}")
        headers = {}
        if self.cfg.requester_pays:
            headers["x-amz-request-payer"] = "requester"
            if self.cfg.region:
                headers["x-amz-bucket-region"] = self.cfg.region
        try:
            resp = requests.get(url, stream=True, timeout=60, headers=headers or None)
            resp.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network required
            status = exc.response.status_code if exc.response is not None else None
            if status == 403 and self.cfg.requester_pays:
                raise RuntimeError(
                    "NAIP index download failed with 403 (Requester Pays). "
                    "Provide AWS credentials (AWS_ACCESS_KEY_ID/SECRET + AWS_REQUEST_PAYER=requester) "
                    "or pass --naip-index-path/--naip-index-source/--sample-data."
                ) from exc
            raise
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def query(self, bbox_wgs84: Sequence[float]) -> List[NaipTileRef]:
        suffix = self.index_path.suffix.lower()
        if suffix == ".geojson":
            return self._query_geojson(self.index_path, bbox_wgs84)

        subset = subset_vector_by_bbox(self.index_path, tuple(bbox_wgs84), logger=self.logger)
        try:
            return self._query_geojson(subset, bbox_wgs84)
        finally:
            if subset.exists() and subset != self.index_path:
                subset.unlink(missing_ok=True)

    def _query_geojson(self, path: Path, bbox_wgs84: Sequence[float]) -> List[NaipTileRef]:
        with path.open() as f:
            data = json.load(f)
        features = data.get("features", [])
        refs: List[NaipTileRef] = []
        for feat in features:
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            feat_bbox = _bbox_from_geom(geom) or bbox_wgs84
            if not _bbox_overlaps(tuple(feat_bbox), tuple(bbox_wgs84)):
                continue
            url = _extract_url(props)
            tile_id = str(
                props.get("name")
                or props.get("objectid")
                or props.get("OBJECTID")
                or props.get("title")
                or len(refs)
            )
            state = props.get("state") or props.get("STATE")
            year = props.get("year") or props.get("YEAR") or props.get("acqYear")
            res_cm = _extract_resolution_cm(props)
            if not url:
                continue
            refs.append(
                NaipTileRef(
                    tile_id=tile_id,
                    url=url,
                    state=state,
                    year=str(year) if year else None,
                    resolution_cm=res_cm,
                    bbox_wgs84=tuple(feat_bbox),
                )
            )
        return refs


def to_gdal_url(url: str, use_vsis3: bool = True, force_vsicurl: bool = False) -> str:
    if url.startswith("s3://"):
        if not use_vsis3:
            return url
        without = url[len("s3://") :]
        return f"/vsis3/{without}"
    if url.startswith("http"):
        if force_vsicurl and not url.startswith("/vsicurl/"):
            return f"/vsicurl/{url}"
        return url
    # Assume bucket/key
    if use_vsis3:
        return f"/vsis3/{url}"
    return f"s3://{url}"


def build_naip_vrt(urls: Iterable[str], vrt_path: Path, requester_pays: bool = True, logger: Logger = None, force_vsicurl: bool = False) -> None:
    env = None
    if requester_pays:
        env = os.environ.copy()
        env.setdefault("AWS_REQUEST_PAYER", "requester")
    sources = [to_gdal_url(u, use_vsis3=True, force_vsicurl=force_vsicurl) for u in urls]
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


def _bbox_from_geom(geom: dict) -> Optional[Tuple[float, float, float, float]]:
    if not geom:
        return None
    bbox = geom.get("bbox")
    if bbox and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    coords = geom.get("coordinates")
    if not coords:
        return None
    xs: List[float] = []
    ys: List[float] = []
    def _walk(points):
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt):
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
            elif isinstance(pt, (list, tuple)):
                _walk(pt)
    _walk(coords)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
