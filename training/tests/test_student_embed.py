# SPDX-License-Identifier: BSD-3-Clause
import torch

from training.models.student_cnn import StudentEmbeddingNet, VGGBlock3x3


def test_student_stride_and_params():
    model = StudentEmbeddingNet(embed_dim=64, depth=2)
    params = StudentEmbeddingNet.param_count(model)
    assert params < 10_000_000
    for patch in (256, 512, 1024):
        x = torch.rand(1, 3, patch, patch)
        emb = model(x)
        assert emb.shape == (1, model.embed_dim, patch // model.output_stride, patch // model.output_stride)
        assert torch.isfinite(emb).all()


def test_vgg_block_preserves_spatial_shape():
    block = VGGBlock3x3(64, depth=3, norm="group", groups=8)
    x = torch.rand(2, 64, 16, 16)
    y = block(x)
    assert y.shape == x.shape


def test_batched_kmeans_determinism():
    from training.models.student_cnn import batched_kmeans

    torch.manual_seed(0)
    emb = torch.rand(1, 32, 8, 8)
    assign1, proto1 = batched_kmeans(emb, k=4, iters=3, temperature=0.8)
    torch.manual_seed(0)
    assign2, proto2 = batched_kmeans(emb, k=4, iters=3, temperature=0.8)
    assert torch.allclose(assign1, assign2)
    assert torch.allclose(proto1, proto2)
