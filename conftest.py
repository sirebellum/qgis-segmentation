import urllib.request

import pytest

try:  # Optional requests dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests may be absent
    requests = None

try:  # Torch may be unavailable in some environments
    import torch.hub as torch_hub  # type: ignore
except Exception:  # pragma: no cover - torch may be absent
    torch_hub = None

try:  # Training module may be absent in plugin-only deployments
    from training.utils.seed import set_seed
except ImportError:  # pragma: no cover - training may be absent
    set_seed = None


_GPU_METRICS = []


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
            f"{label} [{device}] prefetch={prefetch}: {throughput:,.0f} px/s ({mp_per_s:.2f} MP/s, {elapsed * 1000:.1f} ms) {delta_label}"
        )


@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    if set_seed is not None:
        set_seed(1234)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def _deny(*args, **kwargs):  # pragma: no cover - behavior-only
        raise RuntimeError("Network access disabled during tests")

    monkeypatch.setattr(urllib.request, "urlopen", _deny)
    if requests is not None:
        monkeypatch.setattr(requests, "get", _deny, raising=False)
        monkeypatch.setattr(requests, "post", _deny, raising=False)
    if torch_hub is not None:
        monkeypatch.setattr(torch_hub, "load", _deny, raising=False)


@pytest.fixture(autouse=True)
def _segmenter_env(monkeypatch):
    monkeypatch.setenv("SEGMENTER_SKIP_AUTO_INSTALL", "1")
    # Keep deterministic sampling defaults across tests
    monkeypatch.setenv("PYTHONHASHSEED", "0")
