# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IngestConfig:
    dataset: str
    input_root: str = ""
    output_root: str = ""
    sample_size: int = 0
    allow_network: bool = False
    notes: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "input_root": self.input_root,
            "output_root": self.output_root,
            "sample_size": self.sample_size,
            "allow_network": self.allow_network,
            "notes": self.notes,
        }
