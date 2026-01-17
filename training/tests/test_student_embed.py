# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.models.student_cnn import StudentEmbeddingNet


def test_student_stride_and_params():
    model = StudentEmbeddingNet(embed_dim=128)
    params = StudentEmbeddingNet.param_count(model)
    assert params < 10_000_000
    x = torch.rand(1, 3, 512, 512)
    emb = model(x)
    assert emb.shape == (1, 128, 128, 128)
    assert torch.isfinite(emb).all()


def test_batched_kmeans_determinism():
    from training.models.student_cnn import batched_kmeans

    torch.manual_seed(0)
    emb = torch.rand(1, 32, 8, 8)
    assign1, proto1 = batched_kmeans(emb, k=4, iters=3, temperature=0.8)
    torch.manual_seed(0)
    assign2, proto2 = batched_kmeans(emb, k=4, iters=3, temperature=0.8)
    assert torch.allclose(assign1, assign2)
    assert torch.allclose(proto1, proto2)
