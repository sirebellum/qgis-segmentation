# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import io
from contextlib import redirect_stdout

import pytest

try:
    from scripts.datasets_ingest import cli, manifest
    from scripts.datasets_ingest.config import IngestConfig
    from scripts.datasets_ingest.providers import PlaceholderProvider
except ModuleNotFoundError:
    pytest.skip("scripts package not available", allow_module_level=True)


def test_validate_manifest_rejects_missing_fields():
    with pytest.raises(ValueError):
        manifest.validate_manifest([{"id": "only"}])


def test_plan_manifest_respects_sample_size():
    provider = PlaceholderProvider()
    cfg = IngestConfig(dataset="stub", sample_size=3)
    entries = cli.plan_manifest(provider, cfg)
    assert len(entries) == 3
    assert all(entry.id.startswith("placeholder-") for entry in entries)


def test_cli_lists_providers(capsys):
    code = cli.main(["--list-providers"])
    assert code == 0
    out = capsys.readouterr().out
    assert "placeholder" in out and "naip_aws" in out


def test_cli_rejects_unknown_provider():
    with pytest.raises(KeyError):
        cli._select_provider("does-not-exist")


def test_cli_plan_stays_offline():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main(["--provider", "naip_aws", "--dataset", "demo", "--sample-size", "1", "--allow-network"])
    assert code == 0
    output = buffer.getvalue()
    assert "Planned 1 entries" in output
