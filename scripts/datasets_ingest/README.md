<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Dataset Ingestion Scaffold (Stub)

Purpose: future ingestion pipeline for raster datasets. The scaffold is offline-only, import-safe, and does not perform network or GDAL calls. It defines config, interfaces, manifest validation, provider stubs, and a CLI shell for planning manifests.

## Layout
- config.py: dataclasses for ingestion metadata and flags.
- interfaces.py: abstract provider/ingestor contracts.
- manifest.py: minimal manifest entry schema + validation helpers.
- providers/: stub provider implementations (`placeholder`, `naip_aws`) returning deterministic placeholder manifests.
- cli.py: argparse entrypoint for listing providers and planning stub manifests.

## Usage (stub only)
```
python -m scripts.datasets_ingest.cli --help
python -m scripts.datasets_ingest.cli --list-providers
python -m scripts.datasets_ingest.cli --provider placeholder --dataset demo --sample-size 2
```

Notes: No downloads are performed; outputs are in-memory manifest entries suitable for downstream wiring once real ingestion steps are added.
