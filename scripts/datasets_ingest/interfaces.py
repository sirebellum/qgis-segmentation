# SPDX-License-Identifier: BSD-3-Clause
from abc import ABC, abstractmethod
from typing import List

from .config import IngestConfig
from .manifest import ManifestEntry


class BaseProvider(ABC):
    """Abstract provider that plans manifest entries for a dataset."""

    name: str = "base"

    @abstractmethod
    def plan(self, config: IngestConfig) -> List[ManifestEntry]:
        """Return a manifest plan without performing downloads."""


class Ingestor(ABC):
    """Abstract ingestor that would realize a manifest into storage."""

    @abstractmethod
    def ingest(self, entries: List[ManifestEntry]) -> None:
        raise NotImplementedError
