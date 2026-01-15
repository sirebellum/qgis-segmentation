# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Runtime artifacts for the next-gen numpy-based model."""

from .runtime_numpy import NumpySegmenter, RuntimeMeta, load_runtime_model

__all__ = ["NumpySegmenter", "RuntimeMeta", "load_runtime_model"]
