"""LSB carrier for still images (PNG, BMP, TIFF output).

Samples are the R, G and B values of every pixel in row-major order (left to
right, top to bottom), and within a pixel R, then G, then B. With one bit per
sample, no key and the same bits, this writes exactly the same pixels as the
legacy ``encoder.encode()``, so old and new code agree on the layout.

Covers:
    ``load()`` accepts anything Pillow can open, including lossy formats
    such as JPEG, because the cover only has to be *read*. Every cover is
    turned into 8-bit RGB or RGBA by ``image_to_array()``:

    * Palette ("P"), 8-bit grayscale ("L", "1") and other modes ("CMYK",
      "YCbCr", ...) are converted to RGB.
    * Modes with transparency become RGBA: "RGBA", "LA", "PA", a palette
      with a transparent entry, and RGB or grayscale images whose
      transparency is a PNG ``tRNS`` colour key (one RGB or gray value that
      means "transparent").
    * High-bit-depth grayscale ("I;16" from 16-bit PNG/TIFF, "I" from 16-bit
      PGM, 32-bit "I" and float "F") is *scaled* to 8 bits and then
      repeated into R, G and B. Pillow's own ``convert("RGB")`` clips these
      values at 255 instead, which turns a 16-bit image almost entirely
      white. The scaling rules are listed in ``_scale_to_8bit()``. The stego
      output is 8 bits per channel; the cover's extra precision is not kept.

    For animated files only the first frame is used.

Alpha:
    The alpha channel is never used for payload and is written back
    unchanged. A ``tRNS`` colour key is written back as an explicit alpha
    channel rather than as a key, because embedding changes pixel values:
    a keyed pixel whose LSB changed would no longer match the key (and turn
    opaque), and an opaque pixel could start to match it (and vanish).
    BMP cannot round-trip alpha through Pillow (it reloads as RGB), so
    saving a carrier that has alpha to ``.bmp`` raises ``LossyFormatError``;
    use PNG or TIFF for those.

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


def _is_high_depth(mode: str) -> bool:
    """Return True for grayscale modes whose values do not fit in a byte.

    "I;16", "I;16B", "I;16L" and "I;16N" hold unsigned 16-bit values, "I"
    signed 32-bit integers and "F" 32-bit floats.
    """
    return mode in ("I", "F") or mode.startswith("I;16")


def _scale_to_8bit(img: Image.Image) -> np.ndarray:
    """Scale a high-bit-depth grayscale image to ``uint8`` gray values.

    "I;16*" images are known to be 16-bit, so they keep their high byte
    (``value >> 8``). "I" and "F" images do not record the bit depth they
    came from, so their range is guessed from the values (a heuristic,
    applied in this order):

    1. "F" with every value in 0..1: a normalised float image, scaled by 255.
    2. Every value in 0..255: already 8-bit data, used as is (rounded).
    3. Every value in 0..65535: 16-bit data (for example a 16-bit PGM, which
       Pillow opens as "I"), ``value >> 8``.
    4. Anything else (negative or larger values): stretched linearly so the
       minimum maps to 0 and the maximum to 255.

    NaN and infinite values (only possible in "F") count as 0.

    Args:
        img: A Pillow image whose mode satisfies ``_is_high_depth()``.

    Returns:
        np.ndarray: ``(H, W)`` ``uint8`` array.
    """
    raw = np.asarray(img)
    if img.mode.startswith("I;16"):
        # astype() also turns big-endian ("I;16B") data into native order.
        return (raw.astype(np.uint16) >> 8).astype(np.uint8)
    values = raw.astype(np.float64)
    values[~np.isfinite(values)] = 0.0
    if values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = float(values.min()), float(values.max())
    if img.mode == "F" and lo >= 0.0 and hi <= 1.0:
        scaled = values * 255.0
    elif lo >= 0.0 and hi <= 255.0:
        scaled = values
    elif lo >= 0.0 and hi <= 65535.0:
        scaled = np.floor(values / 256.0)  # same as >> 8 for integers
    else:
        scaled = (values - lo) * (255.0 / (hi - lo))  # hi > lo here
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert a Pillow image to the 8-bit RGB or RGBA array a carrier uses.

    This is the conversion ``ImageLSBCarrier.load()`` applies to every
    cover (see the module docstring). Other code that shows or analyses an
    image should use it too, so that it sees the same pixels the carrier
    embeds into; a plain ``img.convert("RGB")`` clips 16-bit grayscale to
    white and drops ``tRNS`` colour-key transparency.

    Args:
        img: Any Pillow image (it is not modified).

    Returns:
        np.ndarray: A new, writable ``uint8`` array of shape ``(H, W, 3)``,
        or ``(H, W, 4)`` when the image has any kind of transparency.
    """
    if _is_high_depth(img.mode):
        gray = _scale_to_8bit(img)
        channels = [gray, gray, gray]
        key = img.info.get("transparency")
        # A 16-bit PNG's tRNS key is a gray value in the original depth, so
        # it is matched against the unscaled values.
        if isinstance(key, (int, float)) and not isinstance(key, bool):
            channels.append(np.where(np.asarray(img) == key, 0, 255).astype(np.uint8))
        return np.dstack(channels)
    if img.mode == "RGBA":
        rgb = img
    elif _has_transparency(img):
        # Checked before the RGB shortcut: an RGB (or L) PNG can carry a
        # tRNS colour key, which convert("RGBA") turns into explicit alpha.
        rgb = img.convert("RGBA")
    elif img.mode == "RGB":
        rgb = img
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
            arr = image_to_array(img)
        return cls(arr, bits_per_sample=bits_per_sample)

    @classmethod
    def from_image(cls, img: Image.Image, bits_per_sample: int = 1) -> "ImageLSBCarrier":
        """Build a carrier from an in-memory Pillow image (same conversions as load())."""
        return cls(image_to_array(img), bits_per_sample=bits_per_sample)

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
