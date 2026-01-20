# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.losses_distill import boundary_prior_losses
from training.teachers.dinov2 import Dinov2Teacher


def test_boundary_prior_losses_respects_boundaries():
    emb = torch.ones(1, 4, 2, 2)
    slic = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64)
    rgb = torch.zeros(1, 3, 2, 2)
    out = boundary_prior_losses(emb, slic, rgb, lambda_boundary=1.0, lambda_antimerge=1.0, lambda_within=0.0)
    assert torch.allclose(out["boundary"], torch.tensor(1.0))
    assert torch.allclose(out["antimerge"], torch.tensor(0.0))
    assert torch.allclose(out["smooth_within"], torch.tensor(0.0), atol=1e-4)


def test_boundary_prior_losses_zero_when_homogeneous_labels():
    emb = torch.ones(1, 4, 2, 2)
    slic = torch.zeros(1, 2, 2, dtype=torch.int64)
    rgb = torch.zeros(1, 3, 2, 2)
    out = boundary_prior_losses(emb, slic, rgb, lambda_boundary=1.0, lambda_antimerge=1.0, lambda_within=0.2)
    assert torch.allclose(out["boundary"], torch.tensor(0.0))
    assert torch.allclose(out["antimerge"], torch.tensor(0.0))
    assert torch.allclose(out["smooth_within"], torch.tensor(0.0), atol=1e-4)


def test_dinov2_teacher_uses_registers(monkeypatch):
    calls = {}

    class DummyBackbone:
        def __init__(self):
            self.embed_dim = 12

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

        def get_intermediate_layers(self, x, n=1, reshape=True):
            b, _, h, w = x.shape
            h_tokens = max(1, h // 14)
            w_tokens = max(1, w // 14)
            return [torch.zeros(b, self.embed_dim, h_tokens, w_tokens, device=x.device)]

    def fake_load(repo, model_name):
        calls["args"] = (repo, model_name)
        return DummyBackbone()

    monkeypatch.setattr(torch.hub, "load", fake_load)

    teacher = Dinov2Teacher()
    rgb = torch.rand(1, 3, 224, 224)
    out = teacher(rgb)

    assert calls.get("args") == ("facebookresearch/dinov2", "dinov2_vitl14_reg")
    assert out.features.shape == (1, teacher.proj_dim, 56, 56)
    assert out.stride == 4
