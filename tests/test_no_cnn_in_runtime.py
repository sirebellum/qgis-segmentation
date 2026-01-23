# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""Guard tests to ensure CNN code paths are not imported by the runtime segmentation pipeline."""

from pathlib import Path


def test_segmenter_no_cnn_imports():
    """Segmenter.py must not import CNN-related functions."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "segmenter.py").read_text(encoding="utf-8")

    # No CNN execution function
    assert "execute_cnn_segmentation" not in text, "CNN segmentation function must not be imported"
    # No model loading (TorchScript CNN weights)
    assert "def load_model" not in text, "load_model function must not exist"
    assert "torch.jit.load" not in text, "TorchScript loading must not be present"
    # No model dropdown references
    assert "inputLoadModel" not in text, "Model dropdown must be removed"
    assert "set_model" not in text or "def set_model" not in text, "set_model method must be removed"


def test_no_model_dropdown_in_ui():
    """UI file must not contain model selection dropdown."""
    root = Path(__file__).resolve().parents[1]
    ui_file = root / "segmenter_dialog_base.ui"
    text = ui_file.read_text(encoding="utf-8")

    assert 'name="inputLoadModel"' not in text, "Model dropdown widget must be removed"
    assert 'name="labelLoadModel"' not in text, "Model dropdown label must be removed"


def test_pipeline_no_cnn_imports():
    """Pipeline module must not import CNN functions."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "runtime" / "pipeline.py").read_text(encoding="utf-8")

    # No CNN imports
    assert "from .cnn import" not in text, "CNN imports must be removed from pipeline"
    assert "fit_global_cnn_centers" not in text
    assert "predict_cnn_with_centers" not in text
    assert "predict_cnn" not in text
    # No model materialization (used for CNN models)
    assert "_materialize_model" not in text, "Model materialization must be removed"


def test_funcs_facade_no_cnn():
    """Funcs facade must not export CNN symbols."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "funcs.py").read_text(encoding="utf-8")

    # Check __all__ does not contain CNN exports
    assert '"predict_cnn"' not in text
    assert '"execute_cnn_segmentation"' not in text
    assert '"tile_raster"' not in text
    assert '"_materialize_model"' not in text


def test_no_cnn_model_artifacts_required():
    """Runtime must not require model .pth files to function."""
    # This test validates that the K-Means path doesn't load any model files.
    # The models/ directory may still exist for legacy reasons but should not be loaded.
    root = Path(__file__).resolve().parents[1]
    segmenter_text = (root / "segmenter.py").read_text(encoding="utf-8")

    # No reference to models/ directory for loading
    assert "models/model_" not in segmenter_text, "No model file paths should be in segmenter.py"
