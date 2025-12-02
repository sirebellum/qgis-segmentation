import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
    def __init__(self, in_channels: int = 3, latent_dim: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
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
        self._load_model()
        schedule = self._training_schedule()
        if schedule.steps > 0:
            _safe_status(status_callback, f"Autoencoder tuning for {schedule.steps} steps (lr={schedule.lr:.4e}).")
            self._train_model(raster, schedule, status_callback)
            self._state["runs"] = int(self._state.get("runs", 0)) + 1
            self._save_state()
            torch.save(self._model.state_dict(), self._weights_path)
        grayscale = self._remap_labels_with_texture(raster, label_map, status_callback)
        return grayscale

    def _load_model(self):
        if self._model is not None:
            return self._model
        model = PatchAutoencoder(latent_dim=self.latent_dim)
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
        base_steps = 480
        decay = 0.65
        min_steps = 64
        steps = int(max(min_steps, base_steps * (decay ** runs)))
        lr_min = 2e-4
        lr = max(lr_min, 1e-3 * (0.75 ** runs))
        batch = 24 if runs < 4 else 16
        return _Schedule(steps=steps, lr=lr, batch_size=batch)

    def _train_model(self, raster: np.ndarray, sched: _Schedule, status_callback=None) -> None:
        if self._model is None:
            return
        array = raster.astype(np.float32, copy=False)
        channels, height, width = array.shape
        patch = int(min(self.patch_size, height, width))
        if patch < 16:
            _safe_status(status_callback, "Raster too small for autoencoder training. Skipping update.")
            return
        optimizer = torch.optim.Adam(self._model.parameters(), lr=sched.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        batches = max(1, sched.batch_size)
        for step in range(sched.steps):
            batch = self._sample_batch(array, patch, batches)
            if batch is None:
                break
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            recon = self._model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            if step % max(1, sched.steps // 6) == 0 or step == sched.steps - 1:
                _safe_status(status_callback, f"Autoencoder step {step + 1}/{sched.steps} — loss {loss.item():.4f}")

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
                unique_labels = np.unique(label_flat)
                for label in unique_labels:
                    mask = label_flat == label
                    if not np.any(mask):
                        continue
                    vec_sum = latent_flat[mask].sum(axis=0, dtype=np.float64)
                    sums[label] = sums.get(label, np.zeros(self.latent_dim, dtype=np.float64)) + vec_sum
                    counts[label] = counts.get(label, 0) + int(mask.sum())
                processed += 1
                if processed % 4 == 0:
                    percent = int((processed / total_tiles) * 100)
                    _safe_status(status_callback, f"Encoding textures {percent}% complete.")
        centroids = {}
        for label, total in sums.items():
            count = counts.get(label, 0)
            if count <= 0:
                continue
            centroids[label] = (total / count).astype(np.float32)
        return centroids

    def _encode_patch(self, patch: np.ndarray) -> Optional[np.ndarray]:
        if patch.size == 0:
            return None
        tensor = torch.from_numpy(patch[np.newaxis].astype(np.float32) / 255.0).to(self.device)
        with torch.no_grad():
            latent = self._model.encoder(tensor)
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


def default_autoencoder_path(plugin_dir: str) -> str:
    base = Path(os.path.expanduser("~")) / ".quantcivil" / "segmenter"
    base.mkdir(parents=True, exist_ok=True)
    if plugin_dir:
        safe_name = Path(plugin_dir).name
        root = base / safe_name
        root.mkdir(parents=True, exist_ok=True)
        return str(root)
    return str(base)
