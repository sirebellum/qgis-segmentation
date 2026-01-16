# SPDX-License-Identifier: BSD-3-Clause
"""Stub CLI for dataset ingestion planning.

This CLI is offline-only and produces placeholder manifest entries to verify the
API surface. Use `--help` or `--list-providers` to inspect capabilities.
"""

import argparse
from typing import Dict, List

from .config import IngestConfig
from .providers import NAIPAWSProvider, PlaceholderProvider
from .interfaces import BaseProvider
from .manifest import ManifestEntry, validate_manifest


def _providers() -> Dict[str, BaseProvider]:
    return {
        "placeholder": PlaceholderProvider(),
        "naip_aws": NAIPAWSProvider(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset ingestion scaffold (offline)")
    parser.add_argument("--provider", default="placeholder", choices=sorted(_providers().keys()))
    parser.add_argument("--dataset", default="stub", help="Logical dataset name for the manifest plan")
    parser.add_argument("--output-root", default="./data/ingest_stub", help="Target output root (not used in stub)")
    parser.add_argument("--sample-size", type=int, default=1, help="Number of placeholder entries to plan")
    parser.add_argument("--allow-network", action="store_true", help="Document intent; no network calls are made")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    return parser


def _select_provider(name: str) -> BaseProvider:
    providers = _providers()
    if name not in providers:
        raise KeyError(f"Unknown provider: {name}")
    return providers[name]


def plan_manifest(provider: BaseProvider, config: IngestConfig) -> List[ManifestEntry]:
    entries = provider.plan(config)
    return validate_manifest({"id": entry.id, "source": entry.source, "split": entry.split, "notes": entry.notes} for entry in entries)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_providers:
        print("Available providers:")
        for name in sorted(_providers().keys()):
            print(f"- {name}")
        return 0

    config = IngestConfig(
        dataset=args.dataset,
        input_root="",
        output_root=args.output_root,
        sample_size=args.sample_size,
        allow_network=bool(args.allow_network),
    )
    provider = _select_provider(args.provider)
    entries = plan_manifest(provider, config)
    print(f"Planned {len(entries)} entries for dataset '{config.dataset}' using provider '{provider.name}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
