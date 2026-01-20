# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.losses_distill import geometric_mean_normalized, normalize_with_ema, scale_neutral_merge


def test_scale_neutral_merge_permutation_invariant():
    losses_a = {"patch_256": torch.tensor(2.0), "patch_512": torch.tensor(1.0), "patch_1024": torch.tensor(0.5)}
    merged_a, _ = scale_neutral_merge(losses_a, {}, decay=0.9, eps=1e-4)
    losses_b = {"patch_1024": torch.tensor(0.5), "patch_512": torch.tensor(1.0), "patch_256": torch.tensor(2.0)}
    merged_b, _ = scale_neutral_merge(losses_b, {}, decay=0.9, eps=1e-4)
    assert torch.allclose(merged_a, merged_b)


def test_scale_neutral_merge_balances_scale():
    # With decay=0, EMA equals the current losses so normalized terms are ~1
    losses = {"patch_256": torch.tensor(10.0), "patch_512": torch.tensor(1.0), "patch_1024": torch.tensor(0.1)}
    merged, state = scale_neutral_merge(losses, {}, decay=0.0, eps=1e-6)
    assert torch.allclose(merged, torch.tensor(1.0), atol=1e-5)
    assert set(state.keys()) == {"patch_256", "patch_512", "patch_1024"}


def test_geometric_mean_normalized_respects_all_scales():
    normalized = {
        "patch_256": torch.tensor(0.8),
        "patch_512": torch.tensor(1.2),
        "patch_1024": torch.tensor(1.0),
    }
    geo = geometric_mean_normalized(normalized, eps=1e-6)
    expected = torch.exp(torch.stack([torch.log(v) for v in normalized.values()]).mean())
    assert torch.allclose(geo, expected)


def test_normalize_with_ema_tracks_scale_device():
    ema_state = {}
    value = torch.tensor(4.0)
    norm, ema_state = normalize_with_ema(value, ema_state, "patch_512", decay=0.5, eps=1e-4)
    assert "patch_512" in ema_state
    assert norm.device == value.device
