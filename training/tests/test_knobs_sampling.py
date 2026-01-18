# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

import random

from training.config import default_config
from training.models.model import MonolithicSegmenter


def test_sample_knobs_within_bounds():
    cfg = default_config()
    cfg.model.embed_dim = 16
    cfg.model.max_k = 12
    model = MonolithicSegmenter(cfg.model)

    random.seed(123)
    smoothing_seen = set()
    downsample_seen = set()
    for _ in range(50):
        knobs = model.sample_knobs()
        assert cfg.model.cluster_iters[0] <= knobs["cluster_iters"] <= cfg.model.cluster_iters[1]
        assert cfg.model.smoothing_iters[0] <= knobs["smooth_iters"] <= cfg.model.smoothing_iters[1]
        assert knobs["downsample"] in cfg.model.downsample_choices
        assert knobs["smoothing_lane"] in cfg.model.smoothing_lanes
        assert 2 <= knobs["k"] <= cfg.model.max_k and knobs["k"] % 2 == 0
        smoothing_seen.add(knobs["smoothing_lane"])
        downsample_seen.add(knobs["downsample"])

    assert smoothing_seen.issuperset(set(cfg.model.smoothing_lanes))
    assert downsample_seen.issuperset(set(cfg.model.downsample_choices))
