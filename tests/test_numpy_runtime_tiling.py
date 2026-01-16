# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import pytest


@pytest.mark.skip(reason="Next-gen numpy runtime is deferred; legacy plugin path is TorchScript/K-Means")
def test_predict_nextgen_numpy_stitches_tiles():
    ...