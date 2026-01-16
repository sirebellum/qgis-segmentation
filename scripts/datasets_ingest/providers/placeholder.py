# SPDX-License-Identifier: BSD-3-Clause
from typing import List

from ..config import IngestConfig
from ..interfaces import BaseProvider
from ..manifest import ManifestEntry


class PlaceholderProvider(BaseProvider):
    name = "placeholder"

    def plan(self, config: IngestConfig) -> List[ManifestEntry]:
        sample_size = max(0, int(config.sample_size)) if config.sample_size else 1
        entries = [
            ManifestEntry(
                id=f"placeholder-{idx}",
                source=f"placeholder://noop/{idx}",
                split="train",
                notes="stub entry; no IO performed",
            )
            for idx in range(sample_size)
        ]
        return entries
