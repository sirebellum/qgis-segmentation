# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import sys
from pathlib import Path

import scripts.data.prepare_naip_aws_3dep_dataset as prep
from scripts.data._naip_aws_provider import NaipAwsConfig, NaipIndex
from scripts.data._usgs_3dep_provider import Product


def test_dry_run_completes_without_gdal(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"
    argv = [
        "prepare_naip_aws_3dep_dataset.py",
        "--output-dir",
        str(out_dir),
        "--dry-run",
        "--seed",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    prep.main()
    captured = capsys.readouterr().out
    assert "Dry-run: using embedded NAIP index stub" in captured
    assert "Dry-run complete" in captured
    assert "selected 1 NAIP tiles" in captured


def test_naip_index_geojson_query_without_gdal(tmp_path):
    cfg = NaipAwsConfig()
    fixture = Path(__file__).parent / "fixtures" / "naip_index_min.geojson"
    index = NaipIndex(tmp_path / "cache", cfg, index_path=fixture, refresh=False, logger=None, allow_stub=False)
    refs = index.query([-88.0, 32.0, -82.0, 38.0])
    assert len(refs) == 1
    ref = refs[0]
    assert ref.tile_id == "naip_fixture_tile"
    assert ref.gdal_source(use_vsis3=True).startswith("/vsis3/")


def test_select_dem_override_skips_tnm(monkeypatch):
    calls = {"query": 0}

    def _boom(*args, **kwargs):
        calls["query"] += 1
        raise AssertionError("TNM query should not run when override provided")

    monkeypatch.setattr(prep, "query_products", _boom)
    override = Product(
        id="dem_override",
        title="Override DEM",
        download_url="https://example.com/dem.tif",
        dataset="custom_dem",
        bbox=(0, 0, 1, 1),
        spatial_reference="EPSG:4326",
        resolution=5.0,
        tier_label="custom",
    )
    selection = prep._select_dem((0, 0, 1, 1), prefer_1m=True, logger=None, allow_stub=False, override_product=override, override_native_gsd=5.0)
    assert selection.product.id == "dem_override"
    assert selection.native_gsd == 5.0
    assert calls["query"] == 0


def test_naip_index_requester_pays_header(monkeypatch, tmp_path):
    cfg = NaipAwsConfig()
    dummy_index = tmp_path / "idx.geojson"
    dummy_index.write_text("{\"type\":\"FeatureCollection\",\"features\":[]}")
    index = NaipIndex(tmp_path / "cache", cfg, index_path=dummy_index, refresh=False, logger=None, allow_stub=True)

    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield b"123"

    def fake_get(url, stream=False, timeout=None, headers=None):
        captured["headers"] = headers or {}
        return _Resp()

    monkeypatch.setattr("scripts.data._naip_aws_provider.requests.get", fake_get)
    index._download("https://example.com/index.gpkg", tmp_path / "downloaded.gpkg")
    assert captured["headers"].get("x-amz-request-payer") == "requester"
