"""Device-specific profiling for adaptive batching and prefetch parameters."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch

try:
    from .funcs import AdaptiveSettings, set_adaptive_settings, predict_cnn
except ImportError:  # pragma: no cover
    from funcs import AdaptiveSettings, set_adaptive_settings, predict_cnn

PROFILE_FILENAME = "perf_profile.json"
SAFETY_CHOICES = (4, 6, 8, 10)
PREFETCH_CHOICES = (1, 2, 3, 4)
DEFAULT_BUDGET_BYTES = 64 * 1024 * 1024


def load_or_profile_settings(
    plugin_dir: str,
    device: torch.device,
    status_callback: Callable[[str], None] | None = None,
    benchmark_runner: Callable[[torch.device, Callable[[str], None] | None], AdaptiveSettings] | None = None,
) -> Tuple[AdaptiveSettings, bool]:
    """Load cached adaptive settings or profile the current device.

    Returns (settings, created_flag).
    """

    if os.environ.get("SEGMENTER_SKIP_PROFILING", "").lower() in {"1", "true"}:
        settings = AdaptiveSettings()
        set_adaptive_settings(settings)
        return settings, False

    profile_path = Path(plugin_dir) / PROFILE_FILENAME
    data = _read_profile_file(profile_path)
    key = _device_key(device)
    if key in data:
        settings = AdaptiveSettings(**data[key])
        set_adaptive_settings(settings)
        return settings, False

    runner = benchmark_runner or _run_profile
    settings = runner(device, status_callback)
    data[key] = asdict(settings)
    _write_profile_file(profile_path, data)
    set_adaptive_settings(settings)
    return settings, True


def _run_profile(device: torch.device, status_callback: Callable[[str], None] | None = None) -> AdaptiveSettings:
    """Profile device using a two-phase search to reduce iterations.
    
    Phase 1: Find best prefetch_depth with default safety_factor
    Phase 2: Find best safety_factor with the best prefetch_depth
    
    This reduces iterations from 16 (4x4) to 8 (4+4).
    """
    array = np.random.randint(0, 255, size=(3, 256, 256), dtype=np.uint8)
    dummy_model = _ProfilerModel().to(device)
    
    default_safety = SAFETY_CHOICES[0]  # Start with lowest safety factor
    total_steps = len(PREFETCH_CHOICES) + len(SAFETY_CHOICES)
    current_step = 0
    
    # Phase 1: Find best prefetch_depth with default safety factor
    best_prefetch = PREFETCH_CHOICES[0]
    best_prefetch_score = -float("inf")
    
    for prefetch in PREFETCH_CHOICES:
        current_step += 1
        settings = AdaptiveSettings(safety_factor=default_safety, prefetch_depth=prefetch)
        score = _benchmark_settings(
            settings, dummy_model, array, device, status_callback,
            phase=1, step=current_step, total=total_steps
        )
        if score is not None and score > best_prefetch_score:
            best_prefetch_score = score
            best_prefetch = prefetch
    
    # Phase 2: Find best safety_factor with the best prefetch_depth
    best_settings = AdaptiveSettings(safety_factor=default_safety, prefetch_depth=best_prefetch)
    best_score = best_prefetch_score
    
    for safety in SAFETY_CHOICES:
        current_step += 1
        settings = AdaptiveSettings(safety_factor=safety, prefetch_depth=best_prefetch)
        score = _benchmark_settings(
            settings, dummy_model, array, device, status_callback,
            phase=2, step=current_step, total=total_steps
        )
        if score is not None and score > best_score:
            best_score = score
            best_settings = settings
    
    set_adaptive_settings(best_settings)
    return best_settings


def _benchmark_settings(
    settings: AdaptiveSettings,
    model: torch.nn.Module,
    array: np.ndarray,
    device: torch.device,
    status_callback: Callable[[str], None] | None,
    phase: int,
    step: int,
    total: int,
) -> float | None:
    """Run a single benchmark iteration and return the score, or None on failure."""
    set_adaptive_settings(settings)
    start = time.perf_counter()
    try:
        predict_cnn(
            model,
            array,
            num_segments=3,
            tile_size=128,
            device=device,
            memory_budget=DEFAULT_BUDGET_BYTES,
            prefetch_depth=settings.prefetch_depth,
            status_callback=None,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - start, 1e-6)
        score = (array.shape[1] * array.shape[2]) / elapsed
        score /= settings.prefetch_depth  # penalize large prefetch depths slightly
        if status_callback:
            progress = int(100 * step / total)
            status_callback(
                f"[{progress}%] Phase {phase}: safety={settings.safety_factor}, "
                f"prefetch={settings.prefetch_depth}: {score:.0f} px/s"
            )
        return score
    except RuntimeError as e:
        # Catch OOM or other runtime errors and skip this settings combo
        if status_callback:
            progress = int(100 * step / total)
            status_callback(
                f"[{progress}%] Phase {phase}: safety={settings.safety_factor}, "
                f"prefetch={settings.prefetch_depth}: FAILED ({str(e).splitlines()[0]})"
            )
        return None


def _device_key(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx}:{name}"
    if device.type == "mps":
        return "mps"
    return "cpu"


def _read_profile_file(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_profile_file(path: Path, data: Dict[str, Dict[str, int]]) -> None:
    path.write_text(json.dumps(data, indent=2))


class _ProfilerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.net(x.float() / 255.0)
        return None, out