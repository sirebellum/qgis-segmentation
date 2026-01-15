# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Config loader with optional module import overrides."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import dataclasses

from .config import Config, default_config


def load_config(path: Optional[str] = None) -> Config:
    """Load a Config from a python file path or return the default config.

    The loader treats the file as a Python module that must expose a
    ``get_config() -> Config`` function. This avoids extra dependencies
    on YAML/JSON while keeping override flexibility.
    """
    if not path:
        return default_config()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config path not found: {cfg_path}")

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML required for YAML configs. Install with `pip install pyyaml` or use a python config file.") from exc
        with cfg_path.open() as f:
            data = yaml.safe_load(f) or {}
        return _config_from_mapping(data)



    spec = importlib.util.spec_from_file_location("_training_config_override", cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import config from {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    getter = getattr(module, "get_config", None)
    if callable(getter):
        cfg = getter()
        if not isinstance(cfg, Config):
            raise TypeError("get_config() must return training.config.Config")
        return cfg

    if hasattr(module, "CONFIG") and isinstance(module.CONFIG, Config):
        return module.CONFIG

    raise AttributeError(
        "Config file must define get_config() -> Config or CONFIG: Config"
    )


def _config_from_mapping(mapping: dict) -> Config:
    cfg = default_config()
    return _update_dataclass(cfg, mapping)


def _update_dataclass(obj, mapping: dict):
    for field in dataclasses.fields(obj):
        name = field.name
        if name not in mapping:
            continue
        value = mapping[name]
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, name, value)
    return obj
