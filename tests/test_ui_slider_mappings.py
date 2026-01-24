# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil
"""QGIS-free tests for slider level mappings, smoothing toggle defaults, and UI styling."""

import pytest
import importlib
import sys
import re


def test_slider_level_constants_exist():
    """Verify slider level constants are defined in segmenter module."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    assert "SLIDER_LEVEL_LOW = 0" in content
    assert "SLIDER_LEVEL_MEDIUM = 1" in content
    assert "SLIDER_LEVEL_HIGH = 2" in content
    assert "SLIDER_LEVEL_NAMES" in content
    assert "SMOOTHING_BASE_KERNELS" in content


def test_smoothing_base_kernels_has_three_levels():
    """Verify SMOOTHING_BASE_KERNELS has three levels with base_kernel and iterations."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    # Find SMOOTHING_BASE_KERNELS block
    assert "SMOOTHING_BASE_KERNELS" in content, "SMOOTHING_BASE_KERNELS not found"
    # Should have base_kernel and iterations
    assert "base_kernel" in content
    assert "iterations" in content
    # Verify all three levels are present
    assert "SLIDER_LEVEL_LOW" in content
    assert "SLIDER_LEVEL_MEDIUM" in content
    assert "SLIDER_LEVEL_HIGH" in content
    # Should have base resolution reference
    assert "SMOOTHING_BASE_RESOLUTION" in content


def test_smoothing_checkbox_default_unchecked():
    """Verify the UI file has smoothing checkbox default to unchecked."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    assert 'name="checkSmoothing"' in content
    checkbox_match = re.search(
        r'<widget class="QCheckBox" name="checkSmoothing">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert checkbox_match, "checkSmoothing widget not found"
    checkbox_block = checkbox_match.group(0)
    assert "<bool>false</bool>" in checkbox_block, "Checkbox should default to unchecked"


def test_smoothness_slider_exists():
    """Verify sliderSmoothness exists with correct range."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    slider_match = re.search(
        r'<widget class="QSlider" name="sliderSmoothness">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert slider_match, "sliderSmoothness widget not found"
    slider_block = slider_match.group(0)
    assert "<number>0</number>" in slider_block
    assert "<number>2</number>" in slider_block


def test_speed_accuracy_sliders_removed():
    """Verify sliderSpeed and sliderAccuracy are removed from UI."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    assert 'name="sliderSpeed"' not in content, "sliderSpeed should be removed"
    assert 'name="sliderAccuracy"' not in content, "sliderAccuracy should be removed"
    assert 'name="infoSpeed"' not in content, "infoSpeed should be removed"
    assert 'name="infoAccuracy"' not in content, "infoAccuracy should be removed"
    assert 'name="labelSpeed"' not in content, "labelSpeed should be removed"
    assert 'name="labelAccuracy"' not in content, "labelAccuracy should be removed"


def test_info_smoothing_exists():
    """Verify infoSmoothing icon exists with tooltip."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    info_match = re.search(
        r'<widget class="QToolButton" name="infoSmoothing">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert info_match, "infoSmoothing info icon not found"
    info_block = info_match.group(0)
    assert "toolTip" in info_block
    assert "&lt;b&gt;" in info_block, "Tooltip should have rich text"
    # Verify palette-aware styling (inherited from global QToolButton style)
    assert 'QToolButton { color: palette(text); }' in content, "QToolButton should use palette-aware styling"


def test_blur_config_returns_none_when_smoothing_disabled():
    """Verify _blur_config logic returns None when smoothing is disabled."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    assert "_is_smoothing_enabled" in content
    assert "if not self._is_smoothing_enabled():" in content
    assert "return None" in content


def test_smoothing_checkbox_has_text():
    """Verify the checkbox has 'Smoothing' text (not a separate label)."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    # Checkbox should have the text property
    checkbox_match = re.search(
        r'<widget class="QCheckBox" name="checkSmoothing">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert checkbox_match, "checkSmoothing widget not found"
    checkbox_block = checkbox_match.group(0)
    assert "<string>Smoothing</string>" in checkbox_block, "Checkbox should have Smoothing text"
    # Separate label should NOT exist
    assert 'name="labelSmoothing"' not in content
    assert 'name="labelSmoothness"' not in content


def test_smoothness_slider_default_value():
    """Verify sliderSmoothness defaults to 1 (medium)."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    smooth_match = re.search(
        r'<widget class="QSlider" name="sliderSmoothness">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert smooth_match
    slider_block = smooth_match.group(0)
    value_match = re.search(r'<property name="value">\s*<number>(\d+)</number>', slider_block)
    assert value_match, "Smoothness slider should have a value property"
    assert value_match.group(1) == "1", "Smoothness slider should default to 1 (medium)"


def test_dropdown_text_palette_aware():
    """Verify QComboBox uses palette-aware text color for dark/light mode support."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    # Check for QComboBox palette-aware stylesheet
    assert "QComboBox" in content and "color: palette(text)" in content


def test_checkbox_indicator_palette_aware():
    """Verify checkbox indicator uses palette-aware colors for dark/light mode support."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    assert "QCheckBox::indicator" in content
    assert "background-color: palette(base)" in content


def test_input_layer_label_centered():
    """Verify Input Layer label has centered alignment."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    label_match = re.search(
        r'<widget class="QLabel" name="labelInputLayer">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert label_match, "labelInputLayer not found"
    label_block = label_match.group(0)
    assert "Qt::AlignCenter" in label_block, "Input Layer label should be centered"


def test_resolution_label_centered():
    """Verify Resolution label has centered alignment."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    label_match = re.search(
        r'<widget class="QLabel" name="labelResolution">.*?</widget>',
        content,
        re.DOTALL,
    )
    assert label_match, "labelResolution not found"
    label_block = label_match.group(0)
    assert "Qt::AlignCenter" in label_block, "Resolution label should be centered"


def test_window_height_reduced():
    """Verify dialog height is reasonable (not excessively large)."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    # Find the main dialog geometry
    geom_match = re.search(
        r'<widget class="QDialog" name="SegmenterDialogBase">.*?<property name="geometry">.*?<height>(\d+)</height>',
        content,
        re.DOTALL,
    )
    assert geom_match, "Could not find dialog geometry"
    height = int(geom_match.group(1))
    assert height <= 480, f"Dialog height should be <= 480, got {height}"


def test_blur_config_passed_to_task():
    """Verify blur_config is added to kwargs for the task."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    # Check that blur_config is passed in kwargs
    assert '"blur_config": blur_config' in content or "'blur_config': blur_config" in content


def test_apply_optional_blur_imported():
    """Verify _apply_optional_blur is imported in segmenter.py."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    assert "_apply_optional_blur" in content
    assert "from .funcs import" in content or "from funcs import" in content


def test_blur_config_scales_with_resolution():
    """Verify _blur_config uses SMOOTHING_BASE_KERNELS and scales with resolution."""
    spec_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter.py")
    with open(spec_file, encoding="utf-8") as f:
        content = f.read()

    # Find the _blur_config method and check it uses scaled kernels
    assert "def _blur_config" in content
    assert "SMOOTHING_BASE_KERNELS" in content
    assert "SMOOTHING_BASE_RESOLUTION" in content
    # Should scale kernel based on resolution
    assert "scale" in content or "resolution" in content.lower()


def test_all_text_widgets_palette_aware():
    """Verify all text-bearing widgets use palette-aware styling for dark/light mode."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    # All these widgets should use palette(text) for proper dark/light mode support
    required_palette_styles = [
        "QComboBox { color: palette(text); }",
        "QLabel { color: palette(text); }",
        "QLineEdit { color: palette(text);",
        "QPlainTextEdit { color: palette(text);",
        "QToolButton { color: palette(text); }",
        "QProgressBar { color: palette(text); }",
        "QPushButton { color: palette(text); }",
        "QDialog { background-color: palette(window); }",
    ]

    for style in required_palette_styles:
        assert style in content, f"Missing palette-aware style: {style}"


def test_no_hardcoded_white_text_color():
    """Verify no hardcoded 'color: white' styling in UI file."""
    ui_file = str(__import__("pathlib").Path(__file__).parents[1] / "segmenter_dialog_base.ui")
    with open(ui_file, encoding="utf-8") as f:
        content = f.read()

    # Should not have hardcoded white color for text
    import re
    # Look for color: white but not in comments or documentation
    hardcoded_white = re.search(r'styleSheet.*?color:\s*white', content, re.IGNORECASE | re.DOTALL)
    assert hardcoded_white is None, "UI should not have hardcoded 'color: white' - use palette(text) instead"
