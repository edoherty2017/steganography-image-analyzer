"""LSB carrier for still images (PNG, BMP, TIFF output).

Samples are the R, G and B values of every pixel in row-major order (left to
right, top to bottom), and within a pixel R, then G, then B. With one bit per
sample, no key and the same bits, this writes exactly the same pixels as the
legacy ``encoder.encode()``, so old and new code agree on the layout.

Covers:
    ``load()`` accepts anything Pillow can open, including lossy formats
    such as JPEG, because the cover only has to be *read*. Palette ("P"),
    grayscale ("L", "1", "I", ...) and other modes are converted to RGB;
    modes with transparency ("RGBA", "LA", "PA", or a palette with a
    transparent entry) become RGBA. For animated files only the first frame
    is used.

Alpha:
    The alpha channel is never used for payload and is written back
    unchanged. BMP cannot round-trip alpha through Pillow (it reloads as
    RGB), so saving a carrier that has alpha to ``.bmp`` raises
    ``LossyFormatError``; use PNG or TIFF for those.

Output:
    ``save()`` writes PNG, BMP or TIFF (uncompressed), chosen by extension.
    JPEG, WebP, GIF and unknown extensions raise ``LossyFormatError``: JPEG
    and lossy WebP re-quantize the pixels, and GIF reduces them to a 256-
    colour palette, either of which destroys the LSBs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import register
from .base import Carrier, LossyFormatError

# Pillow format name for each output extension.
_SAVE_FORMATS = {".png": "PNG", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}

# Modes that already carry an alpha channel (converted to RGBA).
_ALPHA_MODES = {"RGBA", "RGBa", "LA", "La", "PA"}


def _has_transparency(img: Image.Image) -> bool:
    """Return True if ``img`` has alpha or a transparent palette entry."""
    if img.mode in _ALPHA_MODES:
        return True
    check = getattr(img, "has_transparency_data", None)  # Pillow >= 10.1
    if isinstance(check, bool):
        return check
    return "transparency" in img.info


def _to_array(img: Image.Image) -> np.ndarray:
    """Convert a Pillow image to an (H, W, 3) or (H, W, 4) uint8 array."""
    if img.mode in ("RGB", "RGBA"):
        rgb = img
    elif _has_transparency(img):
        rgb = img.convert("RGBA")
    else:
        rgb = img.convert("RGB")
    # np.array (not np.asarray) so the result is a writable copy.
    return np.array(rgb, dtype=np.uint8)


@register
class ImageLSBCarrier(Carrier):
    """Hide bits in the low bits of an image's R, G and B values.

    Args:
        pixels: ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA ``uint8`` array. It
            is copied, so the caller's array is never modified.
        bits_per_sample: Low bits used per channel value, 1 or 2.

    Attributes:
        rgb: ``(H, W, 3)`` C-contiguous ``uint8`` array of colour values.
            embed() modifies it in place.
        alpha: ``(H, W)`` ``uint8`` alpha channel, or ``None`` for RGB.
    """

    name = "image-lsb"
    description = "LSB replacement in the R, G, B channels of PNG/BMP/TIFF images"
    extensions = (".png", ".bmp", ".tif", ".tiff")
    cover_extensions = (
        ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".webp",
        ".ppm", ".pgm", ".pbm", ".pnm", ".tga",
    )

    def __init__(self, pixels: np.ndarray, bits_per_sample: int = 1) -> None:
        super().__init__(bits_per_sample)
        arr = np.asarray(pixels)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                "pixels must be a uint8 array of shape (H, W, 3) or (H, W, 4), "
                f"got {arr.dtype} {arr.shape}"
            )
        # Keep colour and alpha in separate contiguous arrays so the colour
        # samples can be exposed as a flat view without copying.
        self.rgb = np.ascontiguousarray(arr[..., :3]).copy()
        self.alpha = arr[..., 3].copy() if arr.shape[2] == 4 else None

    @classmethod
    def load(cls, path: str | os.PathLike, bits_per_sample: int = 1) -> "ImageLSBCarrier":
        """Open an image file as a cover.

        Args:
            path: Any image Pillow can read (PNG, BMP, TIFF, JPEG, GIF, ...).
            bits_per_sample: Low bits used per channel value, 1 or 2.

        Returns:
            ImageLSBCarrier: A carrier holding a copy of the pixels.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            PIL.UnidentifiedImageError: If the file is not a readable image.
        """
        with Image.open(path) as img:
            img.load()
            arr = _to_array(img)
        return cls(arr, bits_per_sample=bits_per_sample)

    @classmethod
    def from_image(cls, img: Image.Image, bits_per_sample: int = 1) -> "ImageLSBCarrier":
        """Build a carrier from an in-memory Pillow image (same conversions as load())."""
        return cls(_to_array(img), bits_per_sample=bits_per_sample)

    @property
    def mode(self) -> str:
        """``"RGBA"`` if the image has alpha, else ``"RGB"``."""
        return "RGB" if self.alpha is None else "RGBA"

    @property
    def size(self) -> tuple[int, int]:
        """Image size as ``(width, height)``, like ``PIL.Image.size``."""
        return self.rgb.shape[1], self.rgb.shape[0]

    @property
    def pixels(self) -> np.ndarray:
        """A copy of the full pixel array, ``(H, W, 3)`` or ``(H, W, 4)``."""
        if self.alpha is None:
            return self.rgb.copy()
        return np.dstack([self.rgb, self.alpha])

    def to_image(self) -> Image.Image:
        """Return the current pixels as a Pillow image (RGB or RGBA)."""
        return Image.fromarray(self.pixels, self.mode)

    def _samples(self) -> np.ndarray:
        # reshape of a C-contiguous array is a view: R, G, B per pixel,
        # pixels in row-major order.
        return self.rgb.reshape(-1)

    def _write(self, path: Path) -> None:
        fmt = _SAVE_FORMATS[path.suffix.lower()]
        if fmt == "BMP" and self.alpha is not None:
            raise LossyFormatError(
                "BMP output would drop this image's alpha channel; save as .png or .tiff"
            )
        # TIFF is written uncompressed (Pillow's default), PNG with its usual
        # lossless deflate; both preserve every bit.
        self.to_image().save(path, fmt)
