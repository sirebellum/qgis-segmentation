import pytest


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
            f"{label} [{device}] prefetch={prefetch}: {throughput:,.0f} px/s ({mp_per_s:.2f} MP/s, {elapsed*1000:.1f} ms) {delta_label}"
        )
