# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import importlib.util
import sys
from pathlib import Path

import pytest


_GPU_METRICS = []
_PACKAGE_NAME = "segmenter"
_ROOT = Path(__file__).resolve().parents[1]


def _ensure_segmenter_package_loaded():
    """Load the plugin package from the working tree so relative imports resolve."""

    if _PACKAGE_NAME in sys.modules:
        return

    init_path = _ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(_ROOT)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE_NAME] = module
        spec.loader.exec_module(module)


_ensure_segmenter_package_loaded()


@pytest.fixture(scope="session")
def gpu_metric_recorder():
    def _record(entry):
        _GPU_METRICS.append(entry)

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _GPU_METRICS:
        return
    terminalreporter.section("GPU throughput", sep="-")
    baseline_record = next((record for record in _GPU_METRICS if record.get("is_baseline")), _GPU_METRICS[0])
    baseline_tp = baseline_record.get("throughput", 0.0) or 0.0
    baseline_elapsed = baseline_record.get("elapsed", 0.0) or 0.0
    for record in _GPU_METRICS:
        throughput = record.get("throughput", 0.0)
        elapsed = record.get("elapsed", 0.0)
        prefetch = record.get("prefetch")
        label = record.get("label", "run")
        device = record.get("device", "unknown")
        mp_per_s = throughput / 1_000_000.0
        if baseline_tp > 0:
            throughput_delta = ((throughput / baseline_tp) - 1.0) * 100.0
        else:
            throughput_delta = 0.0
        if baseline_elapsed > 0:
            latency_delta = ((baseline_elapsed - elapsed) / baseline_elapsed) * 100.0
        else:
            latency_delta = 0.0
        if record is baseline_record:
            delta_label = "(baseline)"
        else:
            delta_label = f"({throughput_delta:+.1f}% throughput, {latency_delta:+.1f}% latency)"
        terminalreporter.line(
            f"{label} [{device}] prefetch={prefetch}: {throughput:,.0f} px/s ({mp_per_s:.2f} MP/s, {elapsed*1000:.1f} ms) {delta_label}"
        )
