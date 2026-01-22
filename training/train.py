# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Quant Civil

"""Legacy train.py entrypoint — forwards to train_distill.py.

This module exists for backwards compatibility. All training should use
train_distill.py directly. Running `python -m training.train` will invoke
the distillation trainer with the same CLI arguments.
"""
from __future__ import annotations

import sys


def main():
    """Forward to train_distill.main() for unified training entrypoint."""
    from .train_distill import main as distill_main
    distill_main()


if __name__ == "__main__":
    main()
