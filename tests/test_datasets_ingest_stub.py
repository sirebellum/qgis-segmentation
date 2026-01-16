# SPDX-License-Identifier: BSD-3-Clause
import io
from contextlib import redirect_stdout

import pytest

from scripts.datasets_ingest import cli, manifest
from scripts.datasets_ingest.config import IngestConfig
from scripts.datasets_ingest.providers import NAIPAWSProvider, PlaceholderProvider


def test_manifest_validation_requires_fields():
    with pytest.raises(ValueError):
        manifest.validate_entry({})
    entry = manifest.validate_entry({"id": "demo", "source": "stub://path", "split": "train"})
    assert entry.id == "demo"
    assert entry.source == "stub://path"


def test_placeholder_provider_plan_respects_sample_size():
    provider = PlaceholderProvider()
    cfg = IngestConfig(dataset="demo", sample_size=2)
    entries = provider.plan(cfg)
    assert len(entries) == 2
    assert all(entry.source.startswith("placeholder://noop/") for entry in entries)


def test_naip_stub_provider_offline_notes():
    provider = NAIPAWSProvider()
    cfg = IngestConfig(dataset="demo", sample_size=1, allow_network=True)
    entries = provider.plan(cfg)
    assert entries and "allow_network" in (entries[0].notes or "")


def test_cli_help_runs(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Dataset ingestion scaffold" in out


def test_cli_plan_outputs_manifest():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main(["--provider", "placeholder", "--dataset", "demo", "--sample-size", "1"])
    assert code == 0
    stdout = buffer.getvalue()
    assert "Planned 1 entries" in stdout
