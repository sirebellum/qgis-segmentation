"""Device-specific profiling for adaptive batching and prefetch parameters."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Optional, Union, Any, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - resolved at runtime via bootstrap
    try:
        from .dependency_manager import ensure_dependencies
    except ImportError:  # pragma: no cover
        from dependency_manager import ensure_dependencies  # type: ignore
    ensure_dependencies()
    import torch  # type: ignore

logger = logging.getLogger(__name__)

try:  # UI feedback when available
    from qgis.PyQt.QtCore import QThread  # type: ignore
    from qgis.PyQt.QtWidgets import QApplication, QMessageBox  # type: ignore
except Exception:  # pragma: no cover - PyQt not available in tests
    QApplication = None
    QMessageBox = None
    QThread = None

try:
    from .funcs import (
        AdaptiveSettings,
        set_adaptive_settings,
        predict_cnn,
        DEFAULT_MEMORY_BUDGET,
    )
except ImportError:  # pragma: no cover
    from funcs import AdaptiveSettings, set_adaptive_settings, predict_cnn, DEFAULT_MEMORY_BUDGET

@dataclass
class ProfilePayload:
    settings: Dict[str, AdaptiveSettings]
    options: Dict[str, List[Tuple[int, AdaptiveSettings]]]
    default_tier: str
    metrics: Dict[str, Dict[str, float]]

    def primary(self) -> AdaptiveSettings:
        if self.default_tier in self.settings:
            return self.settings[self.default_tier]
        if self.settings:
            return next(iter(self.settings.values()))
        return AdaptiveSettings()


ProfileRunner = Callable[
    [torch.device, Optional[Callable[[str], None]]],
    Union[
        ProfilePayload,
        AdaptiveSettings,
        Tuple[AdaptiveSettings, List[Tuple[int, AdaptiveSettings]]],
        Tuple[Dict[str, AdaptiveSettings], Dict[str, List[Tuple[int, AdaptiveSettings]]]],
    ],
]

MB = 1024 * 1024
PROFILE_FILENAME = "perf_profile.json"
PREFETCH_CHOICES = (1, 3, 5)
SPATIAL_CHOICES = (4, 8, 12)
DEFAULT_BUDGET_BYTES = 64 * MB
PROFILE_TIERS = ("high", "medium", "low")
PROFILE_TILE_SIZES = {
    "high": 256,
    "medium": 192,
    "low": 160,
}
PROFILE_BUDGETS = {
    "high": DEFAULT_BUDGET_BYTES,
    "medium": int(DEFAULT_BUDGET_BYTES * 0.75),
    "low": int(DEFAULT_BUDGET_BYTES * 0.5),
}
PROFILE_TILE_STACK = (256, 512, 768, 1024)
BASE_BACKEND_BUDGETS = {
    "cuda": 320 * MB,
    "mps": 224 * MB,
    "cpu": 160 * MB,
}


def _build_max_safe_matrix() -> Dict[int, Dict[int, Dict[str, int]]]:
    matrix: Dict[int, Dict[int, Dict[str, int]]] = {}
    base_scale = 0.55
    for pref_idx, prefetch in enumerate(PREFETCH_CHOICES):
        matrix[prefetch] = {}
        for spatial_idx, spatial in enumerate(SPATIAL_CHOICES):
            scale = base_scale + 0.06 * pref_idx + 0.04 * spatial_idx
            entry: Dict[str, int] = {}
            for backend, budget in BASE_BACKEND_BUDGETS.items():
                backend_scale = 1.0
                if backend == "mps":
                    backend_scale = 0.85
                elif backend == "cpu":
                    backend_scale = 0.6
                multiplier = min(1.6, max(0.35, scale * backend_scale))
                entry[backend] = int(budget * multiplier)
            matrix[prefetch][spatial] = entry
    return matrix


MAX_SAFE_VRAM = _build_max_safe_matrix()


def _backend_key(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _max_safe_budget(prefetch: int, spatial: int, backend: str, fallback: int) -> int:
    entry = MAX_SAFE_VRAM.get(prefetch, {}).get(spatial)
    if not entry:
        return fallback
    budget = entry.get(backend)
    if not budget:
        return fallback
    return max(32 * MB, budget)
PROFILE_TILE_STACK = (256, 512, 768, 1024)


def _synthetic_tile_batch(seed: int, tile_sizes: Sequence[int] = PROFILE_TILE_STACK) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)
        for size in tile_sizes
    ]


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _clone_setting(settings: AdaptiveSettings) -> AdaptiveSettings:
    return AdaptiveSettings(safety_factor=settings.safety_factor, prefetch_depth=settings.prefetch_depth)


def _normalize_profile_result(
    result: Union[
        ProfilePayload,
        AdaptiveSettings,
        Tuple[AdaptiveSettings, List[Tuple[int, AdaptiveSettings]]],
        Tuple[Dict[str, AdaptiveSettings], Dict[str, List[Tuple[int, AdaptiveSettings]]]],
    ]
) -> ProfilePayload:
    if isinstance(result, ProfilePayload):
        return result

    if isinstance(result, tuple):
        raw_settings, raw_options = result
    else:
        raw_settings, raw_options = result, []

    settings_map: Dict[str, AdaptiveSettings]
    default_tier: str

    if isinstance(raw_settings, dict):
        settings_map = {tier: _clone_setting(cfg) for tier, cfg in raw_settings.items()}
        default_tier = next(iter(settings_map), "default")
    else:
        settings_map = {"default": _clone_setting(raw_settings)}  # type: ignore[arg-type]
        default_tier = "default"

    if isinstance(raw_options, dict):
        options_map = {
            tier: [
                (int(max(0, threshold)), _clone_setting(cfg))
                for threshold, cfg in entries
            ]
            for tier, entries in raw_options.items()
        }
    else:
        option_list = list(raw_options) if raw_options else []  # type: ignore[arg-type]
        options_map = {
            default_tier: [
                (int(max(0, threshold)), _clone_setting(cfg))
                for threshold, cfg in option_list
            ]
        }

    for tier in settings_map:
        options_map.setdefault(tier, [])

    return ProfilePayload(settings=settings_map, options=options_map, default_tier=default_tier, metrics={})


def _report_profile_metrics(
    payload: ProfilePayload,
    status_callback: Optional[Callable[[str], None]],
    cached: bool,
) -> None:
    if not status_callback or not payload.metrics:
        return
    label = "Cached" if cached else "Measured"
    for tier in PROFILE_TIERS:
        summary = payload.metrics.get(tier)
        if not summary:
            continue
        best = summary.get("best_px_per_s", 0.0)
        speedup = summary.get("speedup_vs_prefetch1", 0.0)
        status_callback(f"[{tier}] {label} peak {best:,.0f} px/s (x{speedup:.2f} vs baseline).")


def load_or_profile_settings(
    plugin_dir: str,
    device: torch.device,
    status_callback: Optional[Callable[[str], None]] = None,
    benchmark_runner: Optional[ProfileRunner] = None,
) -> Tuple[AdaptiveSettings, bool]:
    """Load cached adaptive settings or profile the current device."""

    if os.environ.get("SEGMENTER_SKIP_PROFILING", "").lower() in {"1", "true"}:
        settings = AdaptiveSettings()
        set_adaptive_settings(settings)
        return settings, False

    profile_path = Path(plugin_dir) / PROFILE_FILENAME
    data = _read_profile_file(profile_path)
    key = _device_key(device)
    if key in data:
        payload = _deserialize_profile_entry(data[key])
        set_adaptive_settings(payload.settings, payload.options, default_tier=payload.default_tier)
        _report_profile_metrics(payload, status_callback, cached=True)
        return payload.primary(), False

    runner = benchmark_runner or _run_profile
    dialog = _start_profiling_popup()
    try:
        result = runner(device, status_callback)
    finally:
        _close_profiling_popup(dialog)
        payload = _normalize_profile_result(result)
        data[key] = _serialize_profile_entry(payload)
        _write_profile_file(profile_path, data)
        set_adaptive_settings(payload.settings, payload.options, default_tier=payload.default_tier)
        return payload.primary(), True


def _run_profile(
    device: torch.device,
    status_callback: Optional[Callable[[str], None]] = None,
    tile_stack: Optional[Sequence[int]] = None,
    tiers: Optional[Sequence[str]] = None,
) -> ProfilePayload:
    """Profile each resolution tier and capture throughput metrics per tier."""

    model = _ProfilerModel().to(device)
    tier_settings: Dict[str, AdaptiveSettings] = {}
    tier_options: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {}
    tier_metrics: Dict[str, Dict[str, float]] = {}
    active_tiers: Sequence[str] = tuple(tiers) if tiers else PROFILE_TIERS
    tile_stack = tuple(tile_stack) if tile_stack else PROFILE_TILE_STACK

    for tier in active_tiers:
        tile_size = PROFILE_TILE_SIZES.get(tier, PROFILE_TILE_SIZES["high"])
        memory_budget = PROFILE_BUDGETS.get(tier, DEFAULT_BUDGET_BYTES)
        seed = (hash((tier, tuple(tile_stack))) & 0xFFFF) or 1
        tile_batch = _synthetic_tile_batch(seed, tile_stack)
        best_settings, options, metrics = _profile_single_tier(
            tier,
            model,
            tile_batch,
            device,
            status_callback,
            memory_budget,
            tile_size,
        )
        tier_settings[tier] = _clone_setting(best_settings)
        tier_options[tier] = options
        tier_metrics[tier] = metrics
        if status_callback:
            best_px = metrics.get("best_px_per_s", 0.0)
            speedup = metrics.get("speedup_vs_prefetch1", 0.0)
            status_callback(
                f"[{tier}] Peak throughput {best_px:,.0f} px/s (x{speedup:.2f} vs baseline)."
            )

    default_tier = PROFILE_TIERS[0]
    return ProfilePayload(
        settings=tier_settings,
        options=tier_options,
        default_tier=default_tier,
        metrics=tier_metrics,
    )


def _profile_single_tier(
    tier: str,
    model: torch.nn.Module,
    arrays: Sequence[np.ndarray],
    device: torch.device,
    status_callback: Optional[Callable[[str], None]],
    memory_budget: int,
    tile_size: int,
) -> Tuple[AdaptiveSettings, List[Tuple[int, AdaptiveSettings]], Dict[str, float]]:
    total_steps = len(PREFETCH_CHOICES) * len(SPATIAL_CHOICES)
    step = 0
    candidates: list[tuple[int, float, AdaptiveSettings]] = []
    score_map: Dict[Tuple[int, int], float] = {}
    backend = _backend_key(device)

    def _record_candidate(settings: AdaptiveSettings, score: Optional[float]):
        if score is None:
            return
        est_bytes = _estimate_memory_for_settings(settings)
        copy = AdaptiveSettings(safety_factor=settings.safety_factor, prefetch_depth=settings.prefetch_depth)
        candidates.append((est_bytes, score, copy))
        score_map[(settings.prefetch_depth, settings.safety_factor)] = score

    best_settings = AdaptiveSettings(safety_factor=SPATIAL_CHOICES[0], prefetch_depth=PREFETCH_CHOICES[0])
    best_score = -float("inf")

    for prefetch in PREFETCH_CHOICES:
        for spatial in SPATIAL_CHOICES:
            step += 1
            settings = AdaptiveSettings(safety_factor=spatial, prefetch_depth=prefetch)
            budget_hint = _max_safe_budget(prefetch, spatial, backend, memory_budget)
            score = _benchmark_settings(
                settings,
                model,
                arrays,
                device,
                status_callback,
                phase=None,
                step=step,
                total=total_steps,
                tier=tier,
                memory_budget=budget_hint,
                tile_size=tile_size,
            )
            _record_candidate(settings, score)
            if score is not None and score > best_score:
                best_score = score
                best_settings = settings

    options = _build_adaptive_options(candidates)
    metrics = _summarize_tier_metrics(score_map, best_settings, best_score)
    return best_settings, options, metrics


def _summarize_tier_metrics(
    scores: Dict[Tuple[int, int], float],
    best_settings: AdaptiveSettings,
    best_score: Optional[float],
) -> Dict[str, float]:
    if best_score is None or not math.isfinite(best_score):
        best = 0.0
    else:
        best = float(best_score)
    baseline_key = (PREFETCH_CHOICES[0], SPATIAL_CHOICES[0])
    baseline = float(scores.get(baseline_key, 0.0))
    speedup = best / baseline if baseline > 0 else 0.0
    return {
        "best_px_per_s": best,
        "baseline_px_per_s": baseline,
        "speedup_vs_prefetch1": speedup,
        "prefetch_depth": best_settings.prefetch_depth,
        "safety_factor": best_settings.safety_factor,
    }


def _benchmark_settings(
    settings: AdaptiveSettings,
    model: torch.nn.Module,
    arrays: Sequence[np.ndarray],
    device: torch.device,
    status_callback: Optional[Callable[[str], None]],
    phase: Optional[int] = None,
    step: Optional[int] = None,
    total: Optional[int] = None,
    tier: str = "default",
    memory_budget: int = DEFAULT_BUDGET_BYTES,
    tile_size: int = 256,
) -> Optional[float]:
    """Run predict_cnn with the supplied settings and return throughput."""

    if not arrays:
        return None

    set_adaptive_settings({tier: settings}, default_tier=tier)
    start = time.perf_counter()
    total_pixels = 0
    sample_count = 0
    effective_budget = max(32 * MB, int(memory_budget))
    try:
        for array in arrays:
            predict_cnn(
                model,
                array,
                num_segments=3,
                tile_size=tile_size,
                device=device,
                memory_budget=effective_budget,
                prefetch_depth=settings.prefetch_depth,
                status_callback=None,
                profile_tier=tier,
            )
            _synchronize_device(device)
            total_pixels += array.shape[1] * array.shape[2]
            sample_count += 1
        elapsed_total = max(time.perf_counter() - start, 1e-6)
        avg_elapsed = elapsed_total / max(sample_count, 1)
        avg_pixels = total_pixels / max(sample_count, 1)
        score = avg_pixels / max(avg_elapsed, 1e-6)
        if status_callback:
            progress = int(100 * step / total) if step is not None and total else None
            prefix = f"[{tier}] "
            if phase is not None:
                prefix = f"{prefix}[Phase {phase}] "
            suffix = f" ({progress}% complete)" if progress is not None else ""
            status_callback(
                f"{prefix}safety={settings.safety_factor}, prefetch={settings.prefetch_depth}, budget={effective_budget // (1024 * 1024)}MB: {score:,.0f} px/s{suffix}"
            )
        return score
    except RuntimeError as exc:
        if status_callback:
            progress = int(100 * step / total) if step is not None and total else None
            prefix = f"[{tier}] "
            if phase is not None:
                prefix = f"{prefix}[Phase {phase}] "
            suffix = f" ({progress}% complete)" if progress is not None else ""
            status_callback(
                f"{prefix}safety={settings.safety_factor}, prefetch={settings.prefetch_depth}, budget={effective_budget // (1024 * 1024)}MB: FAILED {suffix} ({exc})"
            )
        logger.debug("Profiling settings failed", exc_info=True)
        return None


def _device_key(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx}:{name}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if device.type == "mps":
            return "mps"
    return "cpu"


def _read_profile_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse profile file %s: %s", path, exc)
        return {}
    except OSError as exc:
        logger.warning("Failed to read profile file %s: %s", path, exc)
        return {}


def _write_profile_file(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2))


def _serialize_profile_entry(payload: ProfilePayload) -> Dict[str, object]:
    return {
        "default_tier": payload.default_tier,
        "settings": {tier: asdict(cfg) for tier, cfg in payload.settings.items()},
        "options": {
            tier: [
                {
                    "min_bytes": int(max(0, threshold)),
                    **asdict(opt_settings),
                }
                for threshold, opt_settings in entries
            ]
            for tier, entries in payload.options.items()
        },
        "metrics": payload.metrics,
    }


def _deserialize_profile_entry(entry: Dict[str, object]) -> ProfilePayload:
    if "settings" in entry:
        settings_block = entry.get("settings", {})  # type: ignore[arg-type]
        settings = {
            str(tier): AdaptiveSettings(**data)  # type: ignore[arg-type]
            for tier, data in settings_block.items()
        }
        options_data = entry.get("options", {})  # type: ignore[arg-type]
        options: Dict[str, List[Tuple[int, AdaptiveSettings]]] = {}
        for tier, entries in options_data.items():  # type: ignore[arg-type]
            parsed: List[Tuple[int, AdaptiveSettings]] = []
            for option in entries:
                min_bytes = int(option.get("min_bytes", 0))  # type: ignore[arg-type]
                opt_settings = AdaptiveSettings(
                    safety_factor=int(option.get("safety_factor", settings.get(str(tier), AdaptiveSettings()).safety_factor)),
                    prefetch_depth=int(option.get("prefetch_depth", settings.get(str(tier), AdaptiveSettings()).prefetch_depth)),
                )
                parsed.append((min_bytes, opt_settings))
            parsed.sort(key=lambda item: item[0])
            options[str(tier)] = parsed
        for tier in settings:
            options.setdefault(tier, [])
        default_tier = entry.get("default_tier") or next(iter(settings), "default")
        metrics_block = entry.get("metrics", {})  # type: ignore[arg-type]
        metrics: Dict[str, Dict[str, float]] = {}
        for tier, data in metrics_block.items():  # type: ignore[arg-type]
            metrics[str(tier)] = {
                str(k): float(v)
                for k, v in data.items()
                if isinstance(v, (int, float))
            }
        default_tier_str = str(default_tier)
        if default_tier_str not in settings and settings:
            default_tier_str = next(iter(settings))
        return ProfilePayload(
            settings=settings,
            options=options,
            default_tier=default_tier_str,
            metrics=metrics,
        )

    default_data = entry.get("default", entry)
    options_data = entry.get("options", [])
    settings = AdaptiveSettings(**default_data)  # type: ignore[arg-type]
    option_list: List[Tuple[int, AdaptiveSettings]] = []
    for option in options_data:
        min_bytes = int(option.get("min_bytes", 0))  # type: ignore[arg-type]
        opt_settings = AdaptiveSettings(
            safety_factor=int(option.get("safety_factor", settings.safety_factor)),
            prefetch_depth=int(option.get("prefetch_depth", settings.prefetch_depth)),
        )
        option_list.append((min_bytes, opt_settings))
    option_list.sort(key=lambda item: item[0])
    return ProfilePayload(
        settings={"default": settings},
        options={"default": option_list},
        default_tier="default",
        metrics={},
    )


def _estimate_memory_for_settings(settings: AdaptiveSettings) -> int:
    safety = max(1, settings.safety_factor)
    prefetch = max(1, settings.prefetch_depth)
    safety_scale = 8 / safety
    estimate = DEFAULT_MEMORY_BUDGET * prefetch * safety_scale
    return int(max(16 * 1024 * 1024, estimate))


def _is_gui_thread() -> bool:
    if QApplication is None:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    if QThread is None:
        return False
    current = QThread.currentThread()
    return current is not None and current == app.thread()


def _start_profiling_popup() -> Optional[Any]:
    if QMessageBox is None:
        return None

    if not _is_gui_thread():
        logger.debug("Skipping profiling popup: not on GUI/main thread")
        return None

    box = QMessageBox()
    box.setWindowTitle("Segmenter Profiling")
    box.setText(
        "Profiling this device for optimal batching...\n"
        "This runs once and typically takes 5–10 minutes."
    )
    box.setStandardButtons(QMessageBox.NoButton)
    box.setIcon(QMessageBox.Information)
    box.show()
    if QApplication is not None:
        QApplication.processEvents()
    return box


def _close_profiling_popup(box: Optional[Any]) -> None:
    if box is None:
        return
    box.hide()
    box.deleteLater()


def _build_adaptive_options(
    candidates: list[tuple[int, float, AdaptiveSettings]]
) -> list[tuple[int, AdaptiveSettings]]:
    best_by_threshold: Dict[int, tuple[float, AdaptiveSettings]] = {}
    for threshold, score, settings in candidates:
        if score is None:
            continue
        record = best_by_threshold.get(threshold)
        if record is None or score > record[0]:
            best_by_threshold[threshold] = (score, settings)
    ordered = sorted(best_by_threshold.items(), key=lambda item: item[0])
    return [(threshold, AdaptiveSettings(s.safety_factor, s.prefetch_depth)) for threshold, (score, s) in ordered]


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