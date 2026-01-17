# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.teachers.teacher_base import FakeTeacher


def test_fake_teacher_forward():
    teacher = FakeTeacher(embed_dim=64, stride=8)
    x = torch.rand(1, 3, 256, 256)
    out = teacher.extract(x)
    assert out.features.shape[1] == 64
    assert out.features.shape[-1] == 32
    assert out.stride == 8
    assert torch.isfinite(out.features).all()
