# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.augmentations import apply_augmentations, make_rng
from training.config import AugConfig


def _quad_sample():
    labels = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    rgb = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    rgb[0, 0] = labels.float() / 10.0
    return rgb, labels


def test_geometry_alignment_and_targets_follow_rgb():
    cfg = AugConfig(
        enabled=True,
        rotate_choices=(90,),
        flip_h_prob=0.0,
        flip_v_prob=0.0,
        photometric_prob=0.0,
        noise_std_range=(0.0, 0.0),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
    )
    rgb, labels = _quad_sample()
    slic = labels.unsqueeze(0).unsqueeze(0)
    rng = make_rng(7, sample_index=0, item_id="item", view_id=0)

    result = apply_augmentations(rgb, slic=slic, target=labels, aug_cfg=cfg, rng=rng)

    expected_rgb = torch.rot90(rgb, 1, dims=(-2, -1))
    expected_slic = torch.rot90(slic, 1, dims=(-2, -1))
    expected_target = torch.rot90(labels, 1, dims=(0, 1))

    assert torch.allclose(result["rgb"], expected_rgb)
    assert torch.equal(result["slic"], expected_slic)
    assert torch.equal(result["target"], expected_target)
    assert torch.allclose(result["rgb"][0, 0] * 10.0, result["slic"].float().squeeze(0).squeeze(0))


def test_photometric_aug_clamps_rgb_and_leaves_labels():
    cfg = AugConfig(
        enabled=True,
        rotate_choices=(0,),
        flip_h_prob=0.0,
        flip_v_prob=0.0,
        photometric_prob=1.0,
        noise_std_range=(0.05, 0.05),
        contrast_range=(1.5, 1.5),
        saturation_range=(0.5, 0.5),
    )
    rgb = torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32)
    slic = torch.arange(16, dtype=torch.int64).reshape(1, 4, 4).unsqueeze(0)
    target = slic.squeeze(0)
    rng = make_rng(11, sample_index=0, view_id=0)

    result = apply_augmentations(rgb, slic=slic, target=target, aug_cfg=cfg, rng=rng)

    assert torch.equal(result["slic"], slic)
    assert torch.equal(result["target"], target)
    assert result["rgb"].max() <= 1.0 + 1e-6
    assert result["rgb"].min() >= 0.0 - 1e-6
    assert not torch.equal(result["rgb"], rgb)


def test_deterministic_rng_per_view():
    cfg = AugConfig(
        enabled=True,
        rotate_choices=(0,),
        flip_h_prob=0.0,
        flip_v_prob=0.0,
        photometric_prob=1.0,
        noise_std_range=(0.1, 0.1),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
    )
    rgb = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    rng_a = make_rng(123, sample_index=5, item_id="abc", view_id=0)
    out_a = apply_augmentations(rgb, aug_cfg=cfg, rng=rng_a)["rgb"]

    rng_b = make_rng(123, sample_index=5, item_id="abc", view_id=0)
    out_b = apply_augmentations(rgb, aug_cfg=cfg, rng=rng_b)["rgb"]

    rng_c = make_rng(123, sample_index=5, item_id="abc", view_id=1)
    out_c = apply_augmentations(rgb, aug_cfg=cfg, rng=rng_c)["rgb"]

    assert torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_c)
