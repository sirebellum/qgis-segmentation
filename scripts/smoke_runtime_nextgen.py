# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Quick smoke test for the numpy runtime path (no torch required)."""
from __future__ import annotations

import argparse
import numpy as np

from model import load_runtime_model


def main():
    parser = argparse.ArgumentParser(description="Smoke test for next-gen numpy runtime")
    parser.add_argument("--model-dir", type=str, default="model/best", help="Directory containing meta.json/model.npz")
    parser.add_argument("--K", type=int, default=6, help="Number of segments")
    parser.add_argument("--size", type=int, default=128, help="Square image size for synthetic input")
    args = parser.parse_args()

    runtime = load_runtime_model(args.model_dir)
    rgb = np.random.randint(0, 255, size=(3, args.size, args.size), dtype=np.uint8)
    probs = runtime.forward(rgb.astype(np.float32), k=args.K)
    labels = np.argmax(probs, axis=0)
    print({
        "probs_shape": tuple(probs.shape),
        "labels_shape": tuple(labels.shape),
        "labels_unique": np.unique(labels).tolist(),
    })


if __name__ == "__main__":
    main()
