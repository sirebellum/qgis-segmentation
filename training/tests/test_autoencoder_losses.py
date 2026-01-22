# SPDX-License-Identifier: BSD-3-Clause
"""Tests for autoencoder reconstruction loss (training-only).

Covers:
A) Unit: reconstruction target correctness (downsample+blur shapes/values)
B) Unit: Sobel gradient implementation (finite, deterministic)
C) Unit: decoder forward shape
D) Training step integration (finite losses with autoencoder enabled)
E) Training-only separation (decoder excluded from student artifact)
"""
from __future__ import annotations

import torch
import pytest

from training.losses_recon import (
    TinyReconDecoder,
    build_recon_targets,
    reconstruction_loss,
    sobel_gradients,
    gaussian_blur_2d,
    downsample_area,
    update_recon_ema,
    apply_recon_loss,
)
from training.models.student_cnn import StudentEmbeddingNet


class TestReconTargetCorrectness:
    """A) Unit: reconstruction target correctness."""

    def test_downsample_area_shapes(self):
        rgb = torch.rand(2, 3, 512, 512)
        ds = downsample_area(rgb, factor=4)
        assert ds.shape == (2, 3, 128, 128)
        
    def test_downsample_area_identity(self):
        rgb = torch.rand(2, 3, 64, 64)
        ds = downsample_area(rgb, factor=1)
        assert ds.shape == rgb.shape
        assert torch.allclose(ds, rgb)

    def test_gaussian_blur_shapes(self):
        rgb = torch.rand(2, 3, 128, 128)
        blurred = gaussian_blur_2d(rgb, sigma=1.0, kernel_size=5)
        assert blurred.shape == rgb.shape

    def test_gaussian_blur_bounds(self):
        rgb = torch.rand(2, 3, 128, 128)
        blurred = gaussian_blur_2d(rgb, sigma=1.0, kernel_size=5)
        # Blurred output should still be in [0, 1] for [0,1] input
        assert blurred.min() >= 0.0
        assert blurred.max() <= 1.0

    def test_build_recon_targets_shapes(self):
        rgb = torch.rand(2, 3, 512, 512)
        targets = build_recon_targets(rgb, stride=4, blur_sigma=1.0, blur_kernel=5)
        
        assert "rgb_ds" in targets
        assert "rgb_blur" in targets
        assert "grad_target" in targets
        
        assert targets["rgb_ds"].shape == (2, 3, 128, 128)
        assert targets["rgb_blur"].shape == (2, 3, 128, 128)
        assert targets["grad_target"].shape == (2, 1, 128, 128)

    def test_build_recon_targets_deterministic(self):
        torch.manual_seed(42)
        rgb = torch.rand(2, 3, 256, 256)
        t1 = build_recon_targets(rgb, stride=4)
        t2 = build_recon_targets(rgb, stride=4)
        
        assert torch.allclose(t1["rgb_blur"], t2["rgb_blur"])
        assert torch.allclose(t1["grad_target"], t2["grad_target"])


class TestSobelGradients:
    """B) Unit: Sobel gradient implementation."""

    def test_sobel_gradients_finite(self):
        rgb = torch.rand(2, 3, 64, 64)
        grad = sobel_gradients(rgb)
        assert torch.isfinite(grad).all()

    def test_sobel_gradients_shape(self):
        rgb = torch.rand(2, 3, 64, 64)
        grad = sobel_gradients(rgb)
        assert grad.shape == (2, 1, 64, 64)

    def test_sobel_gradients_deterministic(self):
        torch.manual_seed(123)
        rgb = torch.rand(2, 3, 64, 64)
        g1 = sobel_gradients(rgb)
        g2 = sobel_gradients(rgb)
        assert torch.allclose(g1, g2)

    def test_sobel_gradients_non_negative(self):
        rgb = torch.rand(2, 3, 64, 64)
        grad = sobel_gradients(rgb)
        # Gradient magnitude should always be non-negative
        assert (grad >= 0).all()

    def test_sobel_on_constant_image(self):
        # Constant image should have near-zero gradients
        rgb = torch.full((1, 3, 32, 32), 0.5)
        grad = sobel_gradients(rgb, eps=1e-6)
        # Interior gradients should be very small
        interior = grad[:, :, 2:-2, 2:-2]
        assert interior.max() < 0.01


class TestDecoderForwardShape:
    """C) Unit: decoder forward shape."""

    def test_decoder_output_shape(self):
        decoder = TinyReconDecoder(in_channels=96, hidden_channels=64, num_blocks=2)
        feat = torch.rand(2, 96, 128, 128)
        out = decoder(feat)
        assert out.shape == (2, 3, 128, 128)

    def test_decoder_output_range(self):
        decoder = TinyReconDecoder(in_channels=64, hidden_channels=32, num_blocks=2)
        feat = torch.randn(2, 64, 64, 64)  # can be negative
        out = decoder(feat)
        # Sigmoid output should be in [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_decoder_different_input_channels(self):
        for in_ch in [32, 64, 128, 256]:
            decoder = TinyReconDecoder(in_channels=in_ch, hidden_channels=64, num_blocks=2)
            feat = torch.rand(1, in_ch, 32, 32)
            out = decoder(feat)
            assert out.shape == (1, 3, 32, 32)

    def test_decoder_training_only_marker(self):
        decoder = TinyReconDecoder(in_channels=64)
        assert hasattr(decoder, "_training_only")
        assert decoder._training_only is True


class TestReconstructionLoss:
    """Test reconstruction loss computation."""

    def test_reconstruction_loss_finite(self):
        pred = torch.rand(2, 3, 128, 128)
        targets = {
            "rgb_ds": torch.rand(2, 3, 128, 128),
            "rgb_blur": torch.rand(2, 3, 128, 128),
            "grad_target": torch.rand(2, 1, 128, 128),
        }
        losses = reconstruction_loss(pred, targets, grad_weight=0.2)
        
        assert torch.isfinite(losses["recon_blur"])
        assert torch.isfinite(losses["recon_grad"])
        assert torch.isfinite(losses["recon_raw"])

    def test_reconstruction_loss_shape_mismatch_handled(self):
        # Pred slightly different size should be resized
        pred = torch.rand(2, 3, 64, 64)
        targets = {
            "rgb_ds": torch.rand(2, 3, 128, 128),
            "rgb_blur": torch.rand(2, 3, 128, 128),
            "grad_target": torch.rand(2, 1, 128, 128),
        }
        losses = reconstruction_loss(pred, targets)
        assert torch.isfinite(losses["recon_raw"])


class TestEMANormalization:
    """Test EMA-normalized loss weighting."""

    def test_update_ema_initialization(self):
        ema_state = {}
        loss = torch.tensor(1.0)
        norm, ema_state = update_recon_ema(ema_state, loss, key="test", decay=0.9)
        
        assert "test" in ema_state
        assert torch.isfinite(norm)

    def test_update_ema_decay(self):
        ema_state = {"test": torch.tensor(1.0)}
        loss = torch.tensor(2.0)
        norm, ema_state = update_recon_ema(ema_state, loss, key="test", decay=0.9)
        
        # EMA should move toward 2.0
        expected_ema = 0.9 * 1.0 + 0.1 * 2.0
        assert torch.isclose(ema_state["test"], torch.tensor(expected_ema))

    def test_apply_recon_loss_coefficient(self):
        norm_loss = torch.tensor(1.0)
        contrib = apply_recon_loss(norm_loss, lambda_recon=0.01)
        assert torch.isclose(contrib, torch.tensor(0.01))


class TestStudentFeatureReturn:
    """Test student model return_features option."""

    def test_student_default_returns_embeddings_only(self):
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        rgb = torch.rand(1, 3, 64, 64)
        out = student(rgb)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 64, 16, 16)

    def test_student_return_features(self):
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        rgb = torch.rand(1, 3, 64, 64)
        emb, feat = student(rgb, return_features=True)
        
        assert emb.shape == (1, 64, 16, 16)
        assert feat.shape == (1, student.pre_proj_channels, 16, 16)

    def test_student_pre_proj_channels_matches_decoder(self):
        student = StudentEmbeddingNet(embed_dim=96, depth=2)
        decoder = TinyReconDecoder(in_channels=student.pre_proj_channels)
        
        rgb = torch.rand(1, 3, 128, 128)
        emb, feat = student(rgb, return_features=True)
        pred_rgb = decoder(feat)
        
        assert pred_rgb.shape == (1, 3, 32, 32)


class TestTrainingOnlySeparation:
    """E) Training-only separation: decoder excluded from student artifact."""

    def test_decoder_params_not_in_student_state_dict(self):
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        decoder = TinyReconDecoder(in_channels=student.pre_proj_channels)
        
        student_keys = set(student.state_dict().keys())
        decoder_keys = set(decoder.state_dict().keys())
        
        # No overlap should exist
        assert student_keys.isdisjoint(decoder_keys)

    def test_save_student_only_artifact(self):
        import tempfile
        import os
        
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        decoder = TinyReconDecoder(in_channels=student.pre_proj_channels)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save student only (as done in train_distill.py)
            artifact_path = os.path.join(tmpdir, "student.pt")
            torch.save({"student": student.state_dict()}, artifact_path)
            
            # Load and verify decoder params are absent
            loaded = torch.load(artifact_path, weights_only=True)
            assert "student" in loaded
            assert "decoder" not in loaded
            
            # Verify decoder keys are not in student state dict
            decoder_param_names = [n for n, _ in decoder.named_parameters()]
            for name in decoder_param_names:
                assert name not in loaded["student"]


class TestIntegrationTrainingStep:
    """D) Training step integration with autoencoder."""

    def test_forward_backward_with_autoencoder(self):
        """Run a mini training step with AE enabled and verify losses are finite."""
        torch.manual_seed(42)
        
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        decoder = TinyReconDecoder(in_channels=student.pre_proj_channels)
        
        rgb = torch.rand(2, 3, 64, 64)
        
        # Forward with features
        emb, feat = student(rgb, return_features=True)
        
        # Decoder forward
        pred_rgb = decoder(feat)
        
        # Build targets
        targets = build_recon_targets(rgb, stride=4, blur_sigma=1.0)
        
        # Compute loss
        losses = reconstruction_loss(pred_rgb, targets, grad_weight=0.2)
        
        # EMA normalization
        ema_state = {}
        norm_loss, ema_state = update_recon_ema(ema_state, losses["recon_raw"])
        contrib = apply_recon_loss(norm_loss, lambda_recon=0.01)
        
        # Verify finite
        assert torch.isfinite(losses["recon_raw"])
        assert torch.isfinite(losses["recon_blur"])
        assert torch.isfinite(losses["recon_grad"])
        assert torch.isfinite(norm_loss)
        assert torch.isfinite(contrib)
        
        # Backward should work
        contrib.backward()
        
        # Check gradients exist
        assert any(p.grad is not None for p in decoder.parameters())

    def test_detach_backbone_no_student_grad(self):
        """When detach_backbone=True, student should not receive grad from AE."""
        torch.manual_seed(42)
        
        student = StudentEmbeddingNet(embed_dim=64, depth=1)
        decoder = TinyReconDecoder(in_channels=student.pre_proj_channels)
        
        rgb = torch.rand(2, 3, 64, 64)
        emb, feat = student(rgb, return_features=True)
        
        # Detach features
        feat_detached = feat.detach()
        pred_rgb = decoder(feat_detached)
        
        targets = build_recon_targets(rgb, stride=4)
        losses = reconstruction_loss(pred_rgb, targets)
        losses["recon_raw"].backward()
        
        # Student should have no gradients from this path
        for p in student.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
