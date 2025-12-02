"""Utilities for on-demand dependency installation inside QGIS."""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from qgis.PyQt.QtWidgets import QApplication, QMessageBox  # type: ignore
except Exception:  # pragma: no cover - PyQt unavailable in tests
    QApplication = None
    QMessageBox = None

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_VENDOR_DIR.mkdir(exist_ok=True)
_vendor_str = str(_VENDOR_DIR)
while _vendor_str in sys.path:
    sys.path.remove(_vendor_str)
sys.path.append(_vendor_str)

_ENSURED = False


def ensure_dependencies() -> None:
    """Install runtime dependencies into the plugin's vendor folder if needed."""
    global _ENSURED
    if _ENSURED:
        return

    if _skip_requested():
        _ENSURED = True
        return

    for spec in _package_specs():
        _ensure_package(spec)

    _ENSURED = True


def _skip_requested() -> bool:
    flag = os.environ.get("SEGMENTER_SKIP_AUTO_INSTALL", "").strip().lower()
    return flag in {"1", "true", "yes"}


def _package_specs() -> Iterable[Dict[str, object]]:
    torch_spec = os.environ.get("SEGMENTER_TORCH_SPEC")
    if not torch_spec:
        torch_spec = "torch==2.2.2"

    specs: List[Dict[str, object]] = [
        {
            "import": "torch",
            "pip": torch_spec,
            "label": "PyTorch",
            "extra_args": _torch_index_args(),
        },
        {
            "import": "numpy",
            "pip": "numpy>=1.23,<2.0",
            "label": "NumPy",
        },
        {
            "import": "sklearn",
            "pip": "scikit-learn>=1.1,<2.0",
            "label": "scikit-learn",
        },
    ]
    return specs


def _torch_index_args() -> List[str]:
    custom_index = os.environ.get("SEGMENTER_TORCH_INDEX_URL")
    if custom_index:
        return ["--index-url", custom_index]

    system = platform.system().lower()
    if system == "darwin":
        return ["--index-url", "https://download.pytorch.org/whl/cpu"]
    return ["--index-url", "https://download.pytorch.org/whl/cu121"]


def _ensure_package(spec: Dict[str, object]) -> None:
    import_name = spec["import"]  # type: ignore[index]
    try:
        importlib.import_module(import_name)  # type: ignore[arg-type]
        return
    except ImportError:
        pass

    pip_name = spec["pip"]  # type: ignore[index]
    raw_args = spec.get("extra_args")
    if raw_args is None:
        extra_args: List[str] = []
    elif isinstance(raw_args, str):
        extra_args = [raw_args]
    else:
        extra_args = list(raw_args)  # type: ignore[arg-type]

    _bootstrap_pip()
    command = [
        _python_executable(),
        "-m",
        "pip",
        "install",
        pip_name,
        "--target",
        str(_VENDOR_DIR),
        "--upgrade",
        "--no-cache-dir",
    ]
    command.extend(extra_args)

    label = spec.get("label", pip_name)
    dialog = _start_install_popup(str(label))
    try:
        subprocess.check_call(command)
    except (subprocess.CalledProcessError, OSError) as exc:
        _close_install_popup(dialog)
        raise ImportError(
            "Segmenter could not install dependency '" + str(pip_name) + "'. "
            "Run QGIS Python console and install it manually."
        ) from exc
    _close_install_popup(dialog)


def _bootstrap_pip() -> None:
    try:
        import pip  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import ensurepip
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("pip is not available in this Python runtime") from exc

    ensurepip.bootstrap(upgrade=True)


def _python_executable() -> str:
    override = os.environ.get("SEGMENTER_PYTHON")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return str(override_path)

    exe = Path(sys.executable)
    if "python" in exe.name.lower():
        return str(exe)

    candidates = []

    # Common locations relative to the current executable and sys.prefix
    candidates.append(exe.parent / "python3")
    candidates.append(exe.parent / "python")
    prefix_bin = Path(sys.prefix) / "bin"
    candidates.append(prefix_bin / "python3")
    candidates.append(prefix_bin / "python")

    # macOS app bundle layout
    candidates.append(Path("/Applications/QGIS.app/Contents/MacOS/bin/python3"))
    candidates.append(Path("/Applications/QGIS.app/Contents/MacOS/bin/python"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def _start_install_popup(package_label: str) -> Optional[Any]:
    if QMessageBox is None:
        return None

    box = QMessageBox()
    box.setWindowTitle("Segmenter Dependencies")
    box.setText(
        "Installing "
        + package_label
        + "...\nThis may take a few minutes. QGIS will continue once this finishes."
    )
    box.setStandardButtons(QMessageBox.NoButton)
    box.setIcon(QMessageBox.Information)
    box.show()
    if QApplication is not None:
        QApplication.processEvents()
    return box


def _close_install_popup(box: Optional[Any]) -> None:
    if box is None:
        return
    box.hide()
    box.deleteLater()
