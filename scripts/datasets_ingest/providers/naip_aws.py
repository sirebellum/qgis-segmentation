# SPDX-License-Identifier: BSD-3-Clause
from typing import List

from ..config import IngestConfig
from ..interfaces import BaseProvider
from ..manifest import ManifestEntry


class NAIPAWSProvider(BaseProvider):
    name = "naip_aws"

    def plan(self, config: IngestConfig) -> List[ManifestEntry]:
        """Return a deterministic stub manifest for NAIP-on-AWS.

        Network and GDAL work are intentionally omitted; this serves as a
        contract placeholder while the real ingestion path is rewritten.
        """

        count = 1 if config.sample_size == 0 else max(1, int(config.sample_size))
        entries = [
            ManifestEntry(
                id=f"naip-aws-stub-{idx}",
                source="s3://naip-requester-pays/stub",  # stub URI; not fetched
                split="train",
                notes="stub NAIP AWS entry; offline-only",
            )
            for idx in range(count)
        ]
        if config.allow_network:
            # Document the flag but still avoid IO to keep tests offline.
            entries = [
                ManifestEntry(
                    id=entry.id,
                    source=entry.source,
                    split=entry.split,
                    notes=f"{entry.notes} (allow_network flag observed)",
                )
                for entry in entries
            ]
        return entries
