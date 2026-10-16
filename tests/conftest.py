"""Shared pytest fixtures for the test suite.

Fixtures here are available to every test module without importing. Add new
fixtures below; do not remove existing ones, other modules rely on them.

The repository root is put on ``sys.path`` so the flat top-level modules
(``bitutils``, ``payload``, ``carriers``, ...) import the same way whether the
suite is started with ``python -m pytest`` or plain ``pytest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_synthetic_rgb(seed: int = 0, size: int = 256) -> np.ndarray:
    """Build a natural-looking RGB test image.

    Smooth gradients (so neighbouring pixels are correlated, as in real
    photos) plus mild Gaussian noise (sigma 2), so the LSB plane is not
    trivially structured. Deterministic for a given ``seed``.

    Args:
        seed: Seed for the noise.
        size: Width and height in pixels.

    Returns:
        np.ndarray: ``(size, size, 3)`` ``uint8`` array.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float) * (255.0 / max(size - 1, 1))
    base = np.stack([x, y, (x + y) / 2.0], axis=-1)
    noisy = base + rng.normal(0.0, 2.0, base.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


@pytest.fixture
def synthetic_rgb():
    """Factory fixture: ``synthetic_rgb(seed=0, size=256)`` -> RGB uint8 array."""
    return make_synthetic_rgb


@pytest.fixture
def save_png(tmp_path):
    """Factory fixture: ``save_png(arr, name="image.png")`` -> ``Path`` in tmp_path.

    The array is written with Pillow (mode inferred from its shape), so a
    ``(H, W, 3)`` array becomes an RGB PNG and ``(H, W, 4)`` an RGBA PNG.
    """
    def _save(arr: np.ndarray, name: str = "image.png") -> Path:
        path = tmp_path / name
        Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
        return path

    return _save


@pytest.fixture
def rgb_png(synthetic_rgb, save_png) -> Path:
    """A 256x256 synthetic RGB cover saved as ``cover.png`` in tmp_path."""
    return save_png(synthetic_rgb(0), "cover.png")
