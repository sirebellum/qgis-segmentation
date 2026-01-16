<!--
SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2026 Quant Civil
-->
# Datasets (Rewrite In Progress)

- Status: Dataset ingestion is being redesigned; historical NAIP/DEM loaders were removed. Training defaults to synthetic RGB tiles until the new pipeline is implemented.
- New scaffold: see [scripts/datasets_ingest/README.md](../../scripts/datasets_ingest/README.md) for the planned ingestion interface, manifest validation stubs, and CLI entrypoint. The scaffold is offline-only and does not perform network or GDAL calls.
- Legacy artifacts: sample configs under [configs/datasets](../../configs/datasets) remain for reference but are not currently wired into the training loaders. GDAL helpers persist in [scripts/data/_gdal_utils.py](../../scripts/data/_gdal_utils.py) for future reuse.
- Tests: default pytest remains QGIS-free and offline; new ingestion tests validate only stub behaviors.
