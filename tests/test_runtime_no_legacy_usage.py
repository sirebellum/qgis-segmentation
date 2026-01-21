# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import funcs
from runtime import pipeline


def test_funcs_no_legacy_entrypoints():
    assert not hasattr(funcs, "legacy_cnn_segmentation")
    assert not hasattr(funcs, "legacy_kmeans_segmentation")


def test_pipeline_no_legacy_entrypoints():
    assert not hasattr(pipeline, "legacy_cnn_segmentation")
    assert not hasattr(pipeline, "legacy_kmeans_segmentation")
