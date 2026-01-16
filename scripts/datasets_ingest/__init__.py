# SPDX-License-Identifier: BSD-3-Clause
"""Stub dataset ingestion package.

Offline scaffolding for future dataset ingestion. Providers return deterministic
placeholder manifests; no IO occurs at import time.
"""

from .config import IngestConfig  # noqa: F401
from .interfaces import BaseProvider, Ingestor  # noqa: F401
from .manifest import ManifestEntry, validate_entry, validate_manifest  # noqa: F401
