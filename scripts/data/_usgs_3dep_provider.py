# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""USGS 3DEP discovery/download via TNM Access API.

The National Map TNM Access REST API base: https://tnmaccess.usgs.gov/api/v1/
Products endpoint: https://tnmaccess.usgs.gov/api/v1/products

We query by bbox (WGS84) and select DEM tiers in order of preference:
1) 1 m ("3DEP 1 meter DEM", "USGS 1 meter x 1 meter Resolution DEM")
2) 1/3 arc-second (~10 m)
3) 1 arc-second (~30 m)

References: https://apps.nationalmap.gov/tnmaccess/ (latest news: endpoint moved
under /api/v1/)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import requests

TNM_API_PRODUCTS = "https://tnmaccess.usgs.gov/api/v1/products"


@dataclass
class Product:
    id: str
    title: str
    download_url: str
    dataset: str
    bbox: Sequence[float]
    spatial_reference: Optional[str]
    resolution: Optional[float]
    tier_label: Optional[str] = None


@dataclass
class DemSelection:
    product: Product
    native_gsd: Optional[float]


DEM_TIERS: List[Dict[str, object]] = [
    {"label": "1m", "datasets": ["3DEP 1 meter DEM", "USGS 1 meter x 1 meter Resolution DEM"], "native_gsd": 1.0},
    {"label": "1/3 arc-second", "datasets": ["3DEP 1/3 arc-second DEM", "USGS 1/3 arc-second DEM"], "native_gsd": 10.0},
    {"label": "1 arc-second", "datasets": ["3DEP 1 arc-second DEM"], "native_gsd": 30.0},
]

Logger = Optional[Callable[[str], None]]


def _log(logger: Logger, msg: str) -> None:
    if logger:
        logger(msg)


def query_products(dataset: str, bbox: Sequence[float], max_items: int = 40, logger: Logger = None) -> List[Product]:
    params: Dict[str, str] = {
        "datasets": dataset,
        "bbox": ",".join(str(x) for x in bbox),
        "max": str(max_items),
    }
    _log(logger, f"Querying TNM {dataset} for bbox {bbox}")
    resp = requests.get(TNM_API_PRODUCTS, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    products: List[Product] = []
    for item in data.get("items", []):
        url = item.get("downloadURL") or item.get("url") or item.get("s3path") or ""
        if not url:
            continue
        bbox_item = item.get("bbox") or item.get("boundingBox") or bbox
        spatial = item.get("spatialReference") or item.get("spatialReferenceSystem")
        res_val = item.get("resolution") or item.get("resolutionX")
        res = None
        try:
            if res_val is not None:
                res = float(res_val)
        except Exception:
            res = None
        products.append(
            Product(
                id=str(item.get("id") or item.get("entityId") or item.get("title")),
                title=str(item.get("title") or item.get("name") or dataset),
                download_url=url,
                dataset=dataset,
                bbox=bbox_item,
                spatial_reference=spatial,
                resolution=res,
            )
        )
    return products


def select_best_dem(products: Iterable[Product], prefer_1m: bool = True) -> Optional[DemSelection]:
    prod_list = list(products)
    if not prod_list:
        return None
    tiers = DEM_TIERS[:] if prefer_1m else DEM_TIERS[1:] + DEM_TIERS[:1]
    for tier in tiers:
        label = tier["label"]
        datasets = tier["datasets"]
        native_gsd = tier.get("native_gsd")
        for prod in prod_list:
            title = prod.title.lower()
            if any(key.lower() in title for key in [label, *datasets]):
                return DemSelection(product=Product(**{**prod.__dict__, "tier_label": label}), native_gsd=native_gsd if isinstance(native_gsd, (int, float)) else None)
    best = prod_list[0]
    return DemSelection(product=best, native_gsd=None)


def download_product(product: Product, cache_dir: Path, logger: Logger = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{product.id}.tif"
    if target.exists():
        _log(logger, f"Using cached {product.dataset} product: {target}")
        return target
    _log(logger, f"Downloading {product.dataset} -> {target}")
    with requests.get(product.download_url, stream=True, timeout=90) as resp:
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return target
