# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Utilities for on-demand dependency installation inside QGIS."""
from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - required for pip dependency installation
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from qgis.PyQt.QtCore import QThread  # type: ignore
    from qgis.PyQt.QtWidgets import QApplication, QMessageBox  # type: ignore
except Exception:  # pragma: no cover - PyQt unavailable in tests
    QApplication = None
    QMessageBox = None
    QThread = None

_PLUGIN_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PLUGIN_DIR / "vendor"
_VENDOR_DIR.mkdir(exist_ok=True)
_vendor_str = str(_VENDOR_DIR)
while _vendor_str in sys.path:
    sys.path.remove(_vendor_str)
sys.path.append(_vendor_str)

_plugin_str = str(_PLUGIN_DIR)
if _plugin_str not in sys.path:
    sys.path.append(_plugin_str)

_PIP_BOOTSTRAP_DIR = _VENDOR_DIR / "_pip_bootstrap"
_PIP_BOOTSTRAP_DIR.mkdir(exist_ok=True)

_PIP_COMMAND: Optional[Tuple[Tuple[str, ...], Dict[str, str]]] = None
_EXTERNAL_PIP_CMD: Optional[List[str]] = None

_ENSURED = False
_LOGGER = logging.getLogger(__name__)

_system = platform.system().lower()
if _system == "darwin":
    # macOS needs these guards to prevent libomp crashes inside the QGIS host.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


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
        torch_spec = _default_torch_spec()

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
    ]
    return specs


def _default_torch_spec() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    if (major, minor) >= (3, 13):
        return "torch>=2.5.1,<3.0"
    if (major, minor) >= (3, 12):
        return "torch>=2.3.1,<3.0"
    return "torch==2.2.2"


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

    pip_command, pip_env = _pip_command()

    def _build_command(args: List[str]) -> List[str]:
        cmd = pip_command + [
            "install",
            str(pip_name),
            "--target",
            str(_VENDOR_DIR),
            "--upgrade",
            "--no-cache-dir",
        ]
        cmd.extend(args)
        return cmd

    command = _build_command(extra_args)

    label = str(spec.get("label", pip_name))
    _log_dependency_status(f"Installing dependency: {label} ({pip_name})")
    dialog = _start_install_popup(label)
    try:
        subprocess.check_call(command, env=pip_env)  # nosec B603 - pip command with controlled args
        _log_dependency_status(f"Dependency installed: {label}")
    except (subprocess.CalledProcessError, OSError) as exc:
        if import_name == "torch":
            cpu_args = ["--index-url", "https://download.pytorch.org/whl/cpu"]
            if extra_args != cpu_args:
                _LOGGER.warning(
                    "CUDA torch install failed (%s); retrying with CPU wheels.",
                    exc,
                )
                _log_dependency_status("PyTorch CUDA install failed; retrying with CPU wheels.")
                try:
                    subprocess.check_call(_build_command(cpu_args), env=pip_env)  # nosec B603
                    _log_dependency_status("PyTorch CPU wheel installed successfully.")
                    _close_install_popup(dialog)
                    return
                except (subprocess.CalledProcessError, OSError) as cpu_exc:
                    exc = cpu_exc
        _close_install_popup(dialog)
        raise ImportError(
            "Segmenter could not install dependency '" + str(pip_name) + "'."
            "Run QGIS Python console and install it manually."
        ) from exc
    _close_install_popup(dialog)


def _pip_command() -> Tuple[List[str], Dict[str, str]]:
    global _PIP_COMMAND
    if _PIP_COMMAND is not None:
        command, env = _PIP_COMMAND
        return list(command), dict(env)

    python_exe = _python_executable()
    env = _pip_environment()
    external = _ensure_pip_available(python_exe, env)
    if external:
        command = tuple(external)
        _PIP_COMMAND = (command, dict(env))
        return list(command), dict(env)

    command = (python_exe, "-m", "pip")
    _PIP_COMMAND = (command, dict(env))
    return [python_exe, "-m", "pip"], dict(env)


def _ensure_pip_available(python_exe: str, env: Dict[str, str]) -> Optional[List[str]]:
    global _EXTERNAL_PIP_CMD
    if _pip_available(python_exe, env):
        return None

    if _run_module(python_exe, "ensurepip", ["--default-pip"], env):
        if _pip_available(python_exe, env):
            return None

    if _download_and_install_get_pip(python_exe, env):
        if _pip_available(python_exe, env):
            return None

    if _EXTERNAL_PIP_CMD:
        return _EXTERNAL_PIP_CMD

    external = _system_pip_invocation()
    if external:
        _EXTERNAL_PIP_CMD = external
        _LOGGER.warning(
            "Using fallback pip executable: %s",
            " ".join(external),
        )
        return external

    raise ImportError(
        "pip is not available in this Python runtime. "
        "Install pip for the Python that launches QGIS, expose a pip3 executable on PATH, "
        "or set SEGMENTER_PYTHON/SEGMENTER_PIP_EXECUTABLE to a runtime that ships pip."
    )


def _system_pip_invocation() -> Optional[List[str]]:
    candidates: List[str] = []
    override = os.environ.get("SEGMENTER_PIP_EXECUTABLE")
    if override:
        candidates.append(override)
    candidates.extend(["pip3", "pip"])
    for entry in candidates:
        if not entry:
            continue
        path = shutil.which(entry)
        if path and _pip_command_works([path]):
            return [path]

    for python_name in ("python3", "python"):
        path = shutil.which(python_name)
        if not path:
            continue
        cmd = [path, "-m", "pip"]
        if _pip_command_works(cmd):
            return cmd
    return None


def _pip_command_works(command: List[str]) -> bool:
    try:
        subprocess.check_call(  # nosec B603 - version check with controlled command
            command + ["--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _log_dependency_status(message: str) -> None:
    try:
        from qgis.core import QgsMessageLog, Qgis  # type: ignore
    except Exception:  # pragma: no cover - qgis not available in tests
        QgsMessageLog = None  # type: ignore
        Qgis = None  # type: ignore

    logged = False
    if QgsMessageLog and Qgis:
        try:
            QgsMessageLog.logMessage(message, "Segmenter", level=Qgis.Info)
            logged = True
        except Exception:  # nosec B110 - best effort logging
            pass
    if not logged:
        try:
            print(f"[Segmenter] {message}")
        except Exception:  # nosec B110 - best effort logging
            pass


def _pip_available(python_exe: str, env: Dict[str, str]) -> bool:
    try:
        subprocess.check_call(  # nosec B603 - pip version check
            [python_exe, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _run_module(python_exe: str, module: str, args: List[str], env: Dict[str, str]) -> bool:
    command = [python_exe, "-m", module]
    command.extend(args)
    try:
        subprocess.check_call(  # nosec B603 - module invocation with controlled args
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _download_and_install_get_pip(python_exe: str, env: Dict[str, str]) -> bool:
    url = os.environ.get(
        "SEGMENTER_GET_PIP_URL", "https://bootstrap.pypa.io/get-pip.py"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310 - official bootstrap URL
            script_bytes = response.read()
    except Exception:
        return False

    tmp_dir = tempfile.mkdtemp(prefix="segmenter_get_pip_")
    script_path = Path(tmp_dir) / "get-pip.py"
    try:
        script_path.write_bytes(script_bytes)
        command = [
            python_exe,
            str(script_path),
            "--target",
            str(_PIP_BOOTSTRAP_DIR),
            "--no-warn-script-location",
        ]
        subprocess.check_call(command, env=env)  # nosec B603 - get-pip.py installation
        return True
    except (subprocess.CalledProcessError, OSError):
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _pip_environment() -> Dict[str, str]:
    env = os.environ.copy()
    bootstrap_path = str(_PIP_BOOTSTRAP_DIR)
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = bootstrap_path + os.pathsep + existing
    else:
        env["PYTHONPATH"] = bootstrap_path
    return env


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


def _start_install_popup(package_label: str) -> Optional[Any]:
    if QMessageBox is None:
        return None

    if not _is_gui_thread():
        _LOGGER.debug("Skipping dependency popup for %s: not on GUI/main thread", package_label)
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
