# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Tests for runtime/common.py - shared utilities."""

import threading

import pytest
import torch

from runtime.common import (
    SegmentationCanceled,
    _coerce_torch_device,
    _distance_compute_dtype,
    _emit_status,
    _maybe_raise_cancel,
    _quantization_device,
    _runtime_float_dtype,
)


class TestSegmentationCanceled:
    def test_is_exception(self):
        with pytest.raises(SegmentationCanceled):
            raise SegmentationCanceled()


class TestCoerceTorchDevice:
    def test_none_returns_none(self):
        device = _coerce_torch_device(None)
        assert device is None

    def test_string_device(self):
        device = _coerce_torch_device("cpu")
        assert device.type == "cpu"

    def test_torch_device_passthrough(self):
        original = torch.device("cpu")
        result = _coerce_torch_device(original)
        assert result == original


class TestQuantizationDevice:
    def test_returns_torch_device(self):
        device = _quantization_device(torch.device("cpu"))
        assert isinstance(device, torch.device)


class TestRuntimeFloatDtype:
    def test_cpu_returns_float32(self):
        dtype = _runtime_float_dtype(torch.device("cpu"))
        assert dtype == torch.float32


class TestDistanceComputeDtype:
    def test_returns_valid_dtype(self):
        dtype = _distance_compute_dtype(torch.device("cpu"))
        assert dtype in (torch.float16, torch.float32, torch.bfloat16)


class TestMaybeRaiseCancel:
    def test_none_token_does_not_raise(self):
        _maybe_raise_cancel(None)  # Should not raise

    def test_cancelled_token_raises(self):
        from funcs import SegmentationCanceled

        class MockToken:
            def is_cancelled(self):
                return True

        with pytest.raises(SegmentationCanceled):
            _maybe_raise_cancel(MockToken())

    def test_active_token_does_not_raise(self):
        class MockToken:
            def is_cancelled(self):
                return False

        _maybe_raise_cancel(MockToken())  # Should not raise


class TestEmitStatus:
    def test_none_callback_does_not_raise(self):
        _emit_status(None, "test message")

    def test_callback_receives_message(self):
        messages = []
        _emit_status(messages.append, "hello")
        assert messages == ["hello"]
