# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Minimal USGS TNM provider wrapper for NAIP + 3DEP discovery.

This module uses the public TNM API endpoint and keeps logic isolated so the
endpoint can be swapped without touching the main script. Responses are kept
lightweight; callers should cache downloads.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import requests

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"


@dataclass
class Product:
    id: str
    title: str
    download_url: str
    dataset: str
    bbox: Sequence[float]
    spatial_reference: Optional[str]
    resolution: Optional[float]


def _log(logger: Optional[Callable[[str], None]], msg: str) -> None:
    if logger:
        logger(msg)


def query_products(dataset: str, bbox: Sequence[float], max_items: int = 30, logger: Optional[Callable[[str], None]] = None) -> List[Product]:
    params: Dict[str, str] = {
        "datasets": dataset,
        "bbox": ",".join(str(x) for x in bbox),
        "max": str(max_items),
    }
    _log(logger, f"Querying TNM {dataset} for bbox {bbox}")
    resp = requests.get(TNM_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    products = []
    for item in data.get("items", []):
        url = item.get("downloadURL") or item.get("url") or ""
        if not url:
            continue
        bbox_item = item.get("bbox") or item.get("boundingBox") or bbox
        spatial = None
        res = None
        try:
            spatial = item.get("spatialReference") or item.get("spatialReferenceSystem")
            res_val = item.get("resolution") or item.get("resolutionX")
            if res_val is not None:
                res = float(res_val)
        except Exception:
            pass
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


NAIP_DATASET = "USDA National Agriculture Imagery Program (NAIP)"
DEM_DATASETS = [
    "3DEP 1 meter DEM",
    "USGS 1 meter x 1 meter Resolution DEM",
    "3DEP 1/3 arc-second DEM",
    "USGS 1/3 arc-second DEM",
    "3DEP 1 arc-second DEM",
]


def select_best_naip(products: Iterable[Product]) -> Optional[Product]:
    ranked = sorted(products, key=lambda p: (p.resolution or math.inf))
    return ranked[0] if ranked else None


def select_best_dem(products: Iterable[Product], prefer_1m: bool = True) -> Optional[Product]:
    tiers: List[List[str]] = []
    if prefer_1m:
        tiers.append(["1 meter", "1 meter x 1 meter"])
    tiers.append(["1/3", "0.33", "10 meter"])
    tiers.append(["1 arc-second", "30 meter", "arc-second"])

    products_list = list(products)
    for tier in tiers:
        for prod in products_list:
            title = prod.title.lower()
            if any(key in title for key in tier):
                return prod
    return products_list[0] if products_list else None


def download_product(product: Product, cache_dir: Path, logger: Optional[Callable[[str], None]] = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{product.id}.tif"
    if target.exists():
        _log(logger, f"Using cached {product.dataset} product: {target}")
        return target
    _log(logger, f"Downloading {product.dataset} -> {target}")
    with requests.get(product.download_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return target
