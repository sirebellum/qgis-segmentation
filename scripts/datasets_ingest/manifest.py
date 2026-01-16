# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    source: str
    split: str = "train"
    notes: Optional[str] = None


def validate_entry(raw: Mapping[str, Any]) -> ManifestEntry:
    if "id" not in raw or "source" not in raw:
        raise ValueError("Manifest entry requires 'id' and 'source'.")
    entry_id = str(raw.get("id", "")).strip()
    source = str(raw.get("source", "")).strip()
    if not entry_id:
        raise ValueError("Manifest entry id must be non-empty.")
    if not source:
        raise ValueError("Manifest entry source must be non-empty.")
    split = str(raw.get("split", "train") or "train").strip()
    notes = raw.get("notes")
    notes_str = None if notes is None else str(notes)
    return ManifestEntry(id=entry_id, source=source, split=split, notes=notes_str)


def validate_manifest(raw_entries: Iterable[Mapping[str, Any]]) -> List[ManifestEntry]:
    return [validate_entry(entry) for entry in raw_entries]
