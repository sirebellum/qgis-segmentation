import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_status(callback, message):
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


class PatchAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 24, patch_size: int = 96):
        super().__init__()
        full_kernel = max(3, int(patch_size))
        padding = full_kernel // 2
        self.base_grid = max(4, patch_size // 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=full_kernel, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=full_kernel, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=full_kernel, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(latent_dim, latent_dim * self.base_grid * self.base_grid),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (latent_dim, self.base_grid, self.base_grid)),
            nn.Conv2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


@dataclass
class _Schedule:
    steps: int
    lr: float
    batch_size: int
    betas: Tuple[float, float]
    weight_decay: float


class TextureAutoencoderManager:
    def __init__(
        self,
        storage_dir: str,
        latent_dim: int = 24,
        patch_size: int = 96,
        tile_size: int = 640,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.device = torch.device("cpu")
        self._state_path = self.storage_dir / "workspace_autoencoder_state.json"
        self._weights_path = self.storage_dir / "workspace_autoencoder.pt"
        self._state = self._load_state()
        self._model: Optional[PatchAutoencoder] = None
        self._prior_weight = 0.4
        self._prior_min_weight = 0.05
        self._variance_template: Optional[torch.Tensor] = None
        self._smoothness_weight = 0.12
        self._fib_cache = {}
        self._local_emphasis = 0.65
        self._permutation_chunk_scale = 32

    def set_device(self, device: torch.device) -> None:
        if device != self.device:
            self.device = device
            if self._model is not None:
                self._model.to(self.device)

    def refresh_and_remap(
        self,
        raster: np.ndarray,
        label_map: np.ndarray,
        status_callback=None,
    ) -> Optional[np.ndarray]:
        if raster is None or label_map is None:
            return None
        if raster.ndim != 3:
            return None
        model = self._load_model()
        schedule = self._training_schedule()
        if schedule.steps > 0:
            _safe_status(status_callback, f"Autoencoder tuning for {schedule.steps} steps (lr={schedule.lr:.4e}).")
            self._train_model(raster, schedule, status_callback)
            self._state["runs"] = int(self._state.get("runs", 0)) + 1
            self._save_state()
            if model is not None:
                torch.save(model.state_dict(), self._weights_path)
        grayscale = self._remap_labels_with_texture(raster, label_map, status_callback)
        return grayscale

    def _load_model(self):
        if self._model is not None:
            return self._model
        model = PatchAutoencoder(latent_dim=self.latent_dim, patch_size=self.patch_size)
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
            except Exception:
                pass
        model.to(self.device)
        self._model = model
        return model

    def _load_state(self) -> Dict[str, int]:
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                pass
        return {"runs": 0}

    def _save_state(self) -> None:
        with open(self._state_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle)

    def _training_schedule(self) -> _Schedule:
        runs = int(self._state.get("runs", 0))
        base_steps = 768
        decay = 0.8
        min_steps = 128
        steps = int(max(min_steps, base_steps * (decay ** runs)))
        lr_min = 1.5e-4
        base_lr = 2.5e-3
        lr = max(lr_min, base_lr * (0.6 ** runs))
        if runs == 0:
            batch = 32
        elif runs < 4:
            batch = 24
        else:
            batch = 16
        betas = (0.9, 0.93) if runs < 2 else (0.9, 0.985)
        weight_decay = 0.012 if runs < 3 else 0.007
        return _Schedule(steps=steps, lr=lr, batch_size=batch, betas=betas, weight_decay=weight_decay)

    def _train_model(self, raster: np.ndarray, sched: _Schedule, status_callback=None) -> None:
        if self._model is None:
            return
        model = self._model
        array = raster.astype(np.float32, copy=False)
        channels, height, width = array.shape
        patch = int(min(self.patch_size, height, width))
        if patch < 16:
            _safe_status(status_callback, "Raster too small for autoencoder training. Skipping update.")
            return
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=sched.lr,
            betas=sched.betas,
            weight_decay=sched.weight_decay,
        )
        loss_fn = nn.MSELoss()
        batches = max(1, sched.batch_size)
        for step in range(sched.steps):
            batch = self._sample_batch(array, patch, batches)
            if batch is None:
                break
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            latent = model.encoder(batch)
            recon = model.decoder(latent)
            base_loss = loss_fn(recon, batch)
            prior_loss = self._latent_prior_penalty(latent)
            smooth_loss = self._multiscale_total_variation(recon)
            prior_weight = self._scheduled_prior_weight(step, sched.steps)
            loss = base_loss + (prior_weight * prior_loss) + (self._smoothness_weight * smooth_loss)
            should_log = step % max(1, sched.steps // 6) == 0 or step == sched.steps - 1
            grad_norms = None
            if should_log:
                grad_norms = self._collect_grad_norms(model, base_loss, prior_loss, smooth_loss)
            loss_value = float(loss.detach())
            loss.backward()
            optimizer.step()
            if should_log:
                recon_grad, prior_grad, smooth_grad = grad_norms if grad_norms else (float("nan"),) * 3
                _safe_status(
                    status_callback,
                    (
                        "Autoencoder step {step}/{total} — total {total_loss:.4f}, "
                        "recon {recon_loss:.4f} (grad {recon_grad:.3e}), "
                        "prior {prior_loss:.4f} w={prior_w:.3f} (grad {prior_grad:.3e}), "
                        "smooth {smooth_loss:.4f} (grad {smooth_grad:.3e})"
                    ).format(
                        step=step + 1,
                        total=sched.steps,
                        total_loss=loss_value,
                        recon_loss=base_loss.item(),
                        prior_loss=prior_loss.item(),
                        smooth_loss=smooth_loss.item(),
                        prior_w=prior_weight,
                        recon_grad=recon_grad,
                        prior_grad=prior_grad,
                        smooth_grad=smooth_grad,
                    ),
                )

    def _sample_batch(self, raster: np.ndarray, patch: int, batch: int) -> Optional[torch.Tensor]:
        channels, height, width = raster.shape
        if height < 1 or width < 1:
            return None
        ph = min(patch, height)
        pw = min(patch, width)
        patches = np.empty((batch, channels, ph, pw), dtype=np.float32)
        for idx in range(batch):
            y0 = 0 if height == ph else np.random.randint(0, height - ph + 1)
            x0 = 0 if width == pw else np.random.randint(0, width - pw + 1)
            patch_view = raster[:, y0 : y0 + ph, x0 : x0 + pw]
            patches[idx] = patch_view / 255.0
        chunk_estimate = self._estimate_tile_count(height, width, patch)
        copies = self._permutation_copies_for_chunks(chunk_estimate)
        patches = self._augment_patch_batch(patches, copies)
        tensor = torch.from_numpy(patches)
        return tensor

    def _remap_labels_with_texture(
        self,
        raster: np.ndarray,
        label_map: np.ndarray,
        status_callback=None,
    ) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        centroids = self._accumulate_latent_centroids(raster, label_map, status_callback)
        if not centroids:
            return None
        lookup = self._build_grayscale_lookup(centroids)
        result = np.zeros_like(label_map, dtype=np.uint8)
        for label, value in lookup.items():
            mask = label_map == label
            result[mask] = value
        return result

    def _accumulate_latent_centroids(
        self,
        raster: np.ndarray,
        label_map: np.ndarray,
        status_callback=None,
    ) -> Dict[int, np.ndarray]:
        channels, height, width = raster.shape
        tile = max(128, self.tile_size)
        sums: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        local_weighted: Dict[int, np.ndarray] = {}
        local_weight_mass: Dict[int, float] = {}
        total_tiles = max(1, ((height + tile - 1) // tile) * ((width + tile - 1) // tile))
        processed = 0
        for y0 in range(0, height, tile):
            y1 = min(height, y0 + tile)
            for x0 in range(0, width, tile):
                x1 = min(width, x0 + tile)
                patch = raster[:, y0:y1, x0:x1]
                labels = label_map[y0:y1, x0:x1]
                latent = self._encode_patch(patch)
                if latent is None:
                    continue
                latent_height, latent_width = latent.shape[1:]
                label_tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    label_low = F.interpolate(
                        label_tensor,
                        size=(latent_height, latent_width),
                        mode="nearest",
                    )
                label_low = label_low.squeeze().cpu().numpy().astype(np.int64)
                latent_flat = latent.reshape(self.latent_dim, -1).T
                label_flat = label_low.reshape(-1)
                grad_energy = self._compute_patch_gradient_energy(patch)
                local_weight = 1.0 + (grad_energy * 4.0)
                unique_labels = np.unique(label_flat)
                for label in unique_labels:
                    mask = label_flat == label
                    if not np.any(mask):
                        continue
                    pixel_count = int(mask.sum())
                    vec_sum = latent_flat[mask].sum(axis=0, dtype=np.float64)
                    sums[label] = sums.get(label, np.zeros(self.latent_dim, dtype=np.float64)) + vec_sum
                    counts[label] = counts.get(label, 0) + pixel_count
                    tile_mean = vec_sum / max(pixel_count, 1)
                    weighted = tile_mean * local_weight
                    local_weighted[label] = local_weighted.get(label, np.zeros_like(tile_mean)) + weighted
                    local_weight_mass[label] = local_weight_mass.get(label, 0.0) + local_weight
                processed += 1
                if processed % 4 == 0:
                    percent = int((processed / total_tiles) * 100)
                    _safe_status(status_callback, f"Encoding textures {percent}% complete.")
        centroids = {}
        for label, total in sums.items():
            count = counts.get(label, 0)
            if count <= 0:
                continue
            global_vec = (total / count).astype(np.float32)
            local_sum = local_weighted.get(label)
            weight_mass = local_weight_mass.get(label, 0.0)
            local_vec = None
            if local_sum is not None and weight_mass > 0:
                local_vec = (local_sum / weight_mass).astype(np.float32)
            centroids[label] = self._blend_local_global_vector(global_vec, local_vec, weight_mass, count)
        return centroids

    def _blend_local_global_vector(
        self,
        global_vec: np.ndarray,
        local_vec: Optional[np.ndarray],
        local_weight: float,
        global_count: int,
    ) -> np.ndarray:
        if local_vec is None:
            return global_vec
        emphasis = float(np.clip(self._local_emphasis, 0.0, 1.0))
        denom = max(local_weight + float(global_count), 1e-6)
        local_ratio = np.clip(local_weight / denom, 0.0, 1.0)
        alpha = float(np.clip(emphasis * local_ratio, 0.05, 0.9))
        return ((1.0 - alpha) * global_vec) + (alpha * local_vec)

    def _encode_patch(self, patch: np.ndarray) -> Optional[np.ndarray]:
        if patch.size == 0 or self._model is None:
            return None
        model = self._model
        tensor = torch.from_numpy(patch[np.newaxis].astype(np.float32) / 255.0).to(self.device)
        with torch.no_grad():
            latent = model.encoder(tensor)
        return latent.cpu().numpy()[0]

    def _build_grayscale_lookup(self, centroids: Dict[int, np.ndarray]) -> Dict[int, int]:
        labels = sorted(centroids.keys())
        vectors = np.stack([centroids[label] for label in labels], axis=0)
        if vectors.shape[0] == 1:
            return {labels[0]: 128}
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, np.argmax(eigvals)]
        except np.linalg.LinAlgError:
            axis = np.ones(vectors.shape[1], dtype=np.float32)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm
        scores = centered @ axis
        min_score = scores.min()
        max_score = scores.max()
        denom = max(max_score - min_score, 1e-6)
        normalized = (scores - min_score) / denom
        lookup = {label: int(np.clip(value, 0.0, 1.0) * 255) for label, value in zip(labels, normalized)}
        return lookup

    def _latent_prior_penalty(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            return torch.zeros((), device=latent.device)
        dims = (0, 2, 3)
        channel_mean = latent.mean(dim=dims)
        channel_var = latent.var(dim=dims, unbiased=False)
        mean_penalty = torch.mean(channel_mean ** 2)
        template = self._adaptive_variance_template(channel_var)
        var_penalty = torch.mean((channel_var - template) ** 2)
        return mean_penalty + 0.5 * var_penalty

    def _multiscale_total_variation(self, recon: torch.Tensor) -> torch.Tensor:
        if recon.ndim != 4:
            return torch.zeros((), device=recon.device)
        loss = self._total_variation(recon)
        pooled = recon
        for _ in range(2):
            pooled = F.avg_pool2d(pooled, kernel_size=2, stride=2, ceil_mode=True)
            loss = loss + 0.5 * self._total_variation(pooled)
        return loss

    def _total_variation(self, tensor: torch.Tensor) -> torch.Tensor:
        dh = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        dw = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return dh.abs().mean() + dw.abs().mean()

    def _fibonacci_variance_template(self, channels: int, device: torch.device) -> torch.Tensor:
        if channels <= 0:
            return torch.ones(1, device=device)
        cache_key = (channels, device.type)
        cached = self._fib_cache.get(cache_key)
        if cached is not None and cached.device == device:
            return cached
        seq = [1.0, 1.0]
        while len(seq) < channels:
            seq.append(seq[-1] + seq[-2])
        seq = seq[:channels]
        template = torch.tensor(seq, dtype=torch.float32, device=device)
        template = template / template.mean().clamp(min=1e-6)
        self._fib_cache[cache_key] = template
        return template

    def _adaptive_variance_template(self, channel_var: torch.Tensor) -> torch.Tensor:
        latent_dim = channel_var.shape[-1]
        device = channel_var.device
        base = self._fibonacci_variance_template(latent_dim, device)
        if self._variance_template is None or self._variance_template.shape[0] != latent_dim:
            self._variance_template = base.detach().clone()
        if self._variance_template.device != device:
            self._variance_template = self._variance_template.to(device)
        momentum = 0.9
        self._variance_template = (
            momentum * self._variance_template + (1.0 - momentum) * channel_var.detach()
        )
        self._variance_template = self._variance_template / self._variance_template.mean().clamp(min=1e-6)
        return self._variance_template.detach()

    def _scheduled_prior_weight(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self._prior_weight
        progress = min(max(step / (total_steps - 1), 0.0), 1.0)
        return float(self._prior_min_weight + (self._prior_weight - self._prior_min_weight) * progress)

    def _collect_grad_norms(self, model: PatchAutoencoder, *terms: torch.Tensor) -> Tuple[float, float, float]:
        norms = []
        params = tuple(model.parameters())
        for term in terms:
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            total = 0.0
            for grad in grads:
                if grad is None:
                    continue
                total += float(torch.sum(grad * grad))
            norms.append(math.sqrt(total) if total > 0 else 0.0)
        return tuple(norms)

    @staticmethod
    def _compute_patch_gradient_energy(patch: np.ndarray) -> float:
        if patch.size == 0:
            return 0.0
        gray = patch.astype(np.float32, copy=False).mean(axis=0)
        if gray.size == 0:
            return 0.0
        gy, gx = np.gradient(gray)
        energy = np.sqrt(gy * gy + gx * gx)
        return float(np.mean(energy))

    def _augment_patch_batch(self, patches: np.ndarray, copies: int) -> np.ndarray:
        copies = max(0, int(copies))
        if patches.size == 0:
            return patches
        permuted_batches = [patches]
        base = patches.copy()
        for _ in range(copies):
            permuted = np.empty_like(base)
            for idx in range(base.shape[0]):
                permuted[idx] = self._permute_patch(base[idx])
            permuted_batches.append(permuted)
        augmented = []
        for batch in permuted_batches:
            augmented.extend(self._geometric_permutations(batch))
        return np.concatenate(augmented, axis=0)

    @staticmethod
    def _geometric_permutations(batch: np.ndarray) -> list:
        variants = []
        if batch.size == 0:
            return [batch]
        for flip in (False, True):
            working = np.flip(batch, axis=3) if flip else batch
            for rotation in range(4):
                rotated = np.rot90(working, k=rotation, axes=(2, 3))
                variants.append(rotated.copy())
        return variants

    @staticmethod
    def _permute_patch(patch: np.ndarray) -> np.ndarray:
        result = patch.copy()
        if result.shape[1] > 1:
            perm_y = np.random.permutation(result.shape[1])
            result = result[:, perm_y, :]
        if result.shape[2] > 1:
            perm_x = np.random.permutation(result.shape[2])
            result = result[:, :, perm_x]
        if result.shape[0] > 1 and np.random.rand() < 0.5:
            perm_c = np.random.permutation(result.shape[0])
            result = result[perm_c, :, :]
        if np.random.rand() < 0.3:
            axis = 1 if np.random.rand() < 0.5 else 2
            result = np.flip(result, axis=axis)
        return result

    @staticmethod
    def _estimate_tile_count(height: int, width: int, tile: int) -> int:
        tile = max(1, int(tile))
        rows = max(1, math.ceil(height / tile))
        cols = max(1, math.ceil(width / tile))
        return rows * cols

    def _permutation_copies_for_chunks(self, chunk_count: int) -> int:
        chunk_count = max(1, int(chunk_count))
        scale = max(1, int(self._permutation_chunk_scale))
        copies = int(max(0, math.ceil(chunk_count / scale) - 1))
        return min(copies, 8)


def default_autoencoder_path(plugin_dir: str) -> str:
    base = Path(os.path.expanduser("~")) / ".quantcivil" / "segmenter"
    base.mkdir(parents=True, exist_ok=True)
    if plugin_dir:
        safe_name = Path(plugin_dir).name
        root = base / safe_name
        root.mkdir(parents=True, exist_ok=True)
        return str(root)
    return str(base)
