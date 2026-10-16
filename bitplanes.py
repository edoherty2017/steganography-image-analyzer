#!/usr/bin/env python3
# bitplanes.py — bit-plane slicing and LSB visualization for steganalysis
"""Bit-plane views of an image for visual LSB steganalysis.

Each 8-bit channel value v can be split into eight bit planes, one per bit
position b = 0..7, where plane b holds ``(v >> b) & 1`` for every pixel.
Bit 7 (the MSB) carries the coarse shape of the picture, and each lower plane
looks progressively more like noise. In a natural photo the LSB plane (bit 0)
is mostly noise but often still shows faint structure in flat, saturated or
dark regions. Sequential LSB embedding (encoder.py, the image carrier)
overwrites that plane with message bits, so the embedded region tends to look
uniformly noise-like, typically as a band at the top of the image because
samples are used in row-major order.

Looking at the LSB plane for that kind of change is the "visual attack" of
Westfeld & Pfitzmann (1999, "Attacks on Steganographic Systems"). It is a
heuristic aid for a human analyst, not a statistical test: a noise-like LSB
plane is suggestive, not evidence, because many clean photos (noisy sensors,
heavy texture) already have random-looking LSBs, while key-scattered or
low-rate embedding leaves no visible band at all. The numeric detectors in the
``detectors`` package are the quantitative counterpart.

Files are read exactly as the image carrier reads a cover
(``carriers.image_lsb.image_to_array``), so the planes are the 8-bit samples
an embedder writes into. For a 16-bit file that is the high byte of each
sample: the planes are correctly scaled (not clipped to white), but the
file's own low byte is not shown, and the grid's title says so.

Public functions:

* ``bitplane``: one bit of one channel as a 0/255 image.
* ``save_bitplane_grid``: a PNG grid of planes, rows = channels, columns =
  bits 7..0.
* ``lsb_amplify``: every channel's LSB scaled to 0/255, as a viewable RGB
  image.
* ``lsb_diff`` / ``save_lsb_diff``: a mask of pixels whose LSBs differ
  between a cover and a stego image (white = changed).

Figures are drawn with Matplotlib's object-oriented API on an Agg canvas. This
module never imports ``matplotlib.pyplot`` and never calls ``plt.show``, so it
does not change the global backend that the Tkinter GUI may be using and does
not leak open figures.

Usage (from the repository root)::

    # Bit-plane grid of all RGB bits.
    python bitplanes.py stego.png --grid results/stego_bitplanes.png

    # Amplified LSB image and a cover-vs-stego LSB difference mask.
    python bitplanes.py stego.png --amplify results/stego_lsb.png \\
        --diff cover.png --diff-out results/lsb_diff.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from carriers.image_lsb import image_to_array

# --- Grid geometry -----------------------------------------------------------
# All sizes are chosen so that inches * GRID_DPI is a whole number of pixels.
# Agg truncates fractional pixel sizes, so integral products make the output
# PNG's size exactly predictable:
#     width  = n_cols * GRID_CELL_INCHES + GRID_LABEL_INCHES   (inches)
#     height = n_rows * GRID_CELL_INCHES + GRID_TITLE_INCHES   (inches)
# The figure is never saved with bbox_inches="tight", which would resize it.
GRID_DPI = 120
GRID_CELL_INCHES = 1.5     # one bit-plane panel (180 px at 120 dpi)
GRID_LABEL_INCHES = 0.25   # left margin for the channel row labels (30 px)
GRID_TITLE_INCHES = 0.5    # top margin for the figure title (60 px)

# Panels larger than this many pixels per side are stride-subsampled before
# drawing. A panel is only 180 px wide in the PNG, so this is a display-only
# shortcut that keeps a 12-megapixel cover from pushing 24 full-size planes
# through imshow. It never affects bitplane(), lsb_amplify() or lsb_diff().
GRID_MAX_PANEL_PX = 512

# Channel letter -> index along the last axis of an (H, W, 3 or 4) array.
CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2, "A": 3}

# Pillow raw modes that decode 16 bits per sample into an 8-bit mode by
# keeping only the high byte (48/64-bit RGB(A) PNG and TIFF). Same rule as the
# detectors package uses.
_SIXTEEN_BIT_RAWMODE = re.compile(r";16[BLN]")


def _depth_note(img: Image.Image) -> str | None:
    """Describe how a file with more than 8 bits per sample was reduced.

    The bit planes are always drawn from the 8-bit view that the image
    carrier embeds into (``carriers.image_lsb.image_to_array``). For a
    16-bit file that view is the high byte of each sample, so the file's own
    low-order bits, where a 16-bit LSB payload would sit, are not shown. This
    note lets the grid say so instead of silently labelling bit 8 of the
    file "bit 0 (LSB)".

    Must be called on a freshly opened image, before it is loaded: Pillow
    clears ``img.tile`` on load, and the tile's raw mode is the only sign
    that a 48-bit RGB or 64-bit RGBA file was decoded to 8-bit "RGB" or
    "RGBA".

    Args:
        img: A Pillow image that has been opened but not yet loaded.

    Returns:
        A short sentence for the figure title, or None when the file has 8
        bits per sample (nothing is hidden by the 8-bit view).
    """
    mode = img.mode
    rawmodes = []
    for tile in getattr(img, "tile", None) or []:
        args = tile[3] if len(tile) > 3 else None
        rawmodes.append(args[0] if isinstance(args, tuple) and args else args)
    # ";16B"/";16L"/";16N" raw modes carry 16 bits per sample. Packed-pixel
    # modes such as "BGR;16" (an RGB565 BMP) have 8 or fewer bits per sample
    # and are deliberately not matched.
    if mode.startswith("I;16") or any(
            isinstance(r, str) and _SIXTEEN_BIT_RAWMODE.search(r) for r in rawmodes):
        return "16-bit samples: planes show the high byte; the low byte is not shown"
    if mode in ("I", "F"):
        values = np.asarray(img)
        if values.size and not (
                np.all(np.isfinite(values)) and values.min() >= 0 and values.max() <= 255
                and np.all(values == np.floor(values))):
            return (f"mode {mode} samples scaled to 8 bits; "
                    "the low-order bits are not shown")
    return None


def _read_file(path) -> tuple[np.ndarray, str | None]:
    """Open an image file as the 8-bit array the image carrier sees.

    Args:
        path: Path (``str`` or ``os.PathLike``) to any image Pillow can read.

    Returns:
        ``(arr, note)``: ``arr`` is a uint8 (H, W, 3) or (H, W, 4) array from
        ``carriers.image_lsb.image_to_array``; ``note`` is ``_depth_note``'s
        sentence for a file with more than 8 bits per sample, else None.

    Raises:
        FileNotFoundError: If the path does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    with Image.open(path) as img:
        note = _depth_note(img)  # before image_to_array() loads the pixels
        return image_to_array(img), note


def _load(image) -> np.ndarray:
    """Return ``image`` as a uint8 array, loading it from disk if needed.

    Paths are opened with Pillow and converted exactly as the image carrier
    converts a cover (``carriers.image_lsb.image_to_array``), so the planes
    shown are the pixels an embedder writes into: images with any
    transparency (alpha, a transparent palette entry or a PNG ``tRNS``
    colour key) become RGBA, everything else becomes RGB. Samples wider than
    8 bits are scaled, not clipped: a 16-bit file keeps its high byte
    (``value >> 8``), so bit 7 here is bit 15 of the file and the file's low
    byte is not visible (``save_bitplane_grid`` says so in its title).
    Pillow's plain ``convert("RGB")`` would instead clip every 16-bit value
    above 255 to 255 and turn every plane almost entirely white.
    Arrays are passed through after validation, so a 2-D grayscale array stays
    2-D.

    Args:
        image: A path (``str`` or ``os.PathLike``) to any image Pillow can
            read, or a numpy array of shape (H, W), (H, W, 3) or (H, W, 4).

    Returns:
        numpy.ndarray of dtype uint8 with shape (H, W), (H, W, 3) or
        (H, W, 4).

    Raises:
        FileNotFoundError: If a path does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
        TypeError: If ``image`` is neither a path nor an integer array.
        ValueError: If the array has an unsupported shape or values outside
            0..255.
    """
    if isinstance(image, (str, os.PathLike)):
        return _read_file(image)[0]

    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected a path or numpy array, got {type(image).__name__}")
    if image.dtype == np.uint8:
        arr = image
    elif np.issubdtype(image.dtype, np.integer) or image.dtype == np.bool_:
        # Accept other integer dtypes as long as the values fit in a byte.
        if image.size and (image.min() < 0 or image.max() > 255):
            raise ValueError("array values must be in 0..255")
        arr = image.astype(np.uint8)
    else:
        raise TypeError(f"expected an integer array, got dtype {image.dtype}")

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1, 3, 4)):
        return arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr
    raise ValueError(f"expected shape (H, W), (H, W, 3) or (H, W, 4), got {arr.shape}")


def _check_bit(bit: int) -> int:
    """Validate a bit position (0 = LSB, 7 = MSB) and return it as an int.

    Raises:
        ValueError: If ``bit`` is not an integer in 0..7.
    """
    if isinstance(bit, bool) or not isinstance(bit, (int, np.integer)) or not 0 <= bit <= 7:
        raise ValueError(f"bit must be an integer in 0..7, got {bit!r}")
    return int(bit)


def bitplane(arr: np.ndarray, channel: int, bit: int) -> np.ndarray:
    """Extract one bit plane of one channel as a black/white image.

    Args:
        arr: Image array of shape (H, W, C) or (H, W) with values in 0..255.
            A path is also accepted and loaded as in ``save_bitplane_grid``.
        channel: Channel index along the last axis (0 = R, 1 = G, 2 = B,
            3 = A). Must be 0 for a 2-D grayscale array.
        bit: Bit position, 0 = least significant bit, 7 = most significant.

    Returns:
        numpy.ndarray of shape (H, W) and dtype uint8 whose pixels are 255
        where the chosen bit is 1 and 0 where it is 0.

    Raises:
        ValueError: If ``channel`` or ``bit`` is out of range.
    """
    arr = _load(arr)
    bit = _check_bit(bit)
    n_channels = 1 if arr.ndim == 2 else arr.shape[2]
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) \
            or not 0 <= channel < n_channels:
        raise ValueError(f"channel must be in 0..{n_channels - 1}, got {channel!r}")
    plane = arr if arr.ndim == 2 else arr[..., int(channel)]
    return np.where((plane >> bit) & 1, 255, 0).astype(np.uint8)


def _grid_rows(arr: np.ndarray, channels: str) -> list[tuple[str, int]]:
    """Resolve a channel string into (label, index) rows for the grid.

    Letters are case-insensitive and duplicates are dropped. "A" is skipped
    silently when the image has no alpha channel, so ``channels="RGBA"`` works
    for both RGB and RGBA inputs. A 2-D grayscale array always yields a single
    "L" row, whatever ``channels`` says.

    Raises:
        ValueError: If ``channels`` contains a letter other than R, G, B, A,
            or resolves to no rows.
    """
    if arr.ndim == 2:
        return [("L", 0)]
    rows: list[tuple[str, int]] = []
    for letter in channels.upper():
        if letter not in CHANNEL_INDEX:
            raise ValueError(f"unknown channel {letter!r}; use letters from 'RGBA'")
        idx = CHANNEL_INDEX[letter]
        if idx >= arr.shape[2] or (letter, idx) in rows:
            continue  # no alpha in this image, or a repeated letter
        rows.append((letter, idx))
    if not rows:
        raise ValueError(f"no requested channels ({channels!r}) exist in this image")
    return rows


def grid_size_px(n_rows: int, n_cols: int, dpi: int = GRID_DPI) -> tuple[int, int]:
    """Pixel size (width, height) of a bit-plane grid PNG.

    Args:
        n_rows: Number of channel rows.
        n_cols: Number of bit columns.
        dpi: Resolution the grid is saved at.

    Returns:
        (width, height) in pixels, matching what ``save_bitplane_grid``
        writes for that many rows and columns.
    """
    width = (n_cols * GRID_CELL_INCHES + GRID_LABEL_INCHES) * dpi
    height = (n_rows * GRID_CELL_INCHES + GRID_TITLE_INCHES) * dpi
    return int(width), int(height)


def save_bitplane_grid(image, out_path, channels: str = "RGB",
                       bits: Iterable[int] = range(8)) -> str:
    """Draw every requested bit plane in one labeled grid and save it as PNG.

    Rows are channels in the order given (R, G, B, and A when requested and
    present). Columns are the requested bits from most to least significant,
    so the default shows bit 7 (MSB) on the left and bit 0 (LSB) on the right.
    Each panel is black where the bit is 0 and white where it is 1. Panels for
    large images are stride-subsampled for display (see
    ``GRID_MAX_PANEL_PX``); the planes themselves are exact.

    A file with more than 8 bits per sample is shown as its 8-bit view (the
    high byte of a 16-bit file, see ``_load``), and a second title line says
    that the file's low-order bits are not shown, so bit 0 in the grid is not
    mistaken for the file's real LSB.

    Args:
        image: Path to an image file, or a uint8 array of shape (H, W),
            (H, W, 3) or (H, W, 4).
        out_path: Where to write the PNG. Parent directories are created.
        channels: Channel letters from "RGBA", e.g. "RGB", "B" or "RGBA".
            "A" is ignored when the image has no alpha channel.
        bits: Bit positions to show (0 = LSB). Order and duplicates do not
            matter; columns always run from the highest bit to the lowest.

    Returns:
        ``out_path`` as a string. The PNG is ``grid_size_px(rows, cols)``
        pixels at ``GRID_DPI`` dots per inch.

    Raises:
        ValueError: If a channel letter or bit is invalid, or ``bits`` is empty.
    """
    if isinstance(image, (str, os.PathLike)):
        arr, note = _read_file(image)
    else:
        arr, note = _load(image), None
    rows = _grid_rows(arr, channels)
    cols = sorted({_check_bit(b) for b in bits}, reverse=True)
    if not cols:
        raise ValueError("bits must name at least one bit position")

    height, width = arr.shape[:2]
    step = max(1, -(-max(height, width) // GRID_MAX_PANEL_PX))  # ceil division
    shown = arr[::step, ::step]  # display-only subsample; a view, not a copy

    fig_w = len(cols) * GRID_CELL_INCHES + GRID_LABEL_INCHES
    fig_h = len(rows) * GRID_CELL_INCHES + GRID_TITLE_INCHES
    # Figure + FigureCanvasAgg instead of pyplot: no global backend switch and
    # no figure registry, so nothing has to be closed afterwards.
    fig = Figure(figsize=(fig_w, fig_h), dpi=GRID_DPI, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(rows), len(cols), squeeze=False)

    name = os.path.basename(os.fspath(image)) if isinstance(image, (str, os.PathLike)) else "image"
    title = f"Bit planes of {name} ({width}x{height}), white = 1"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=11)

    for r, (label, ch) in enumerate(rows):
        for c, bit in enumerate(cols):
            ax = axes[r][c]
            plane = bitplane(shown, ch, bit)
            # nearest keeps planes crisp 0/255; the default antialiasing would
            # average noisy low planes into flat gray.
            ax.imshow(plane, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            # Hide ticks but keep the axes, so the row label and column title
            # remain visible (ax.axis("off") would hide them too).
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#999999")
            if r == 0:
                suffix = " (MSB)" if bit == 7 else " (LSB)" if bit == 0 else ""
                ax.set_title(f"bit {bit}{suffix}", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=10, va="center")

    directory = os.path.dirname(os.fspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(out_path, dpi=GRID_DPI, format="png")
    return os.fspath(out_path)


def lsb_amplify(arr) -> np.ndarray:
    """Scale each color channel's least significant bit up to a visible 0/255.

    The result shows the LSB planes of R, G and B together as one RGB image:
    a pixel is white where all three LSBs are 1, black where all are 0, and
    colored otherwise. Alpha, if present, is dropped (it is never used for
    embedding). A 2-D grayscale input is repeated into three equal channels.

    Args:
        arr: Image array of shape (H, W), (H, W, 3) or (H, W, 4), or a path.

    Returns:
        numpy.ndarray of shape (H, W, 3) and dtype uint8 with values 0 or 255.
    """
    arr = _load(arr)
    rgb = np.repeat(arr[..., None], 3, axis=2) if arr.ndim == 2 else arr[..., :3]
    return np.where(rgb & 1, 255, 0).astype(np.uint8)


def lsb_diff(cover, stego, channel: int | None = None, *, bits: int = 1) -> np.ndarray:
    """Mark the pixels whose least significant bit(s) differ between two images.

    Only the low bits are compared, so changes in higher bits (e.g. from
    lossy re-encoding or editing) do not show up, and rewriting an LSB with
    the value it already had is correctly invisible. With a random-looking
    payload, about half of the embedded samples keep their original LSB, so
    inside an embedded region a single channel's mask lights up about 1/2 of
    the pixels and the any-channel mask over R, G and B about 7/8
    (1 - (1/2)**3), rather than all of them.

    A 2-D grayscale array compared with a color image is first repeated into
    R, G and B (the image carrier stores a grayscale cover as RGB and embeds
    in all three channels). When one image has alpha and the other does
    not, only R, G and B are compared.

    Args:
        cover: The original image, as a path or uint8 array.
        stego: The suspected stego image, same size as ``cover``.
        channel: Channel index to compare (0 = R, 1 = G, 2 = B, 3 = A), or
            None to flag a pixel if any channel differs.
        bits: Keyword-only. Number of low bits to compare, 1..8. Use 2 for
            the two-bits-per-sample image carrier.

    Returns:
        numpy.ndarray of shape (H, W) and dtype uint8: 255 (white) where the
        compared bits differ, 0 (black) where they match.

    Raises:
        ValueError: If the images differ in size, ``channel`` is out of range,
            or ``bits`` is not in 1..8.
    """
    a, b = _load(cover), _load(stego)
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 8:
        raise ValueError(f"bits must be an integer in 1..8, got {bits!r}")
    if a.ndim == 2:
        a = a[..., None]
    if b.ndim == 2:
        b = b[..., None]
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"image sizes differ: {a.shape[1]}x{a.shape[0]} "
                         f"vs {b.shape[1]}x{b.shape[0]}")
    # Grayscale vs color (e.g. an "L" cover array against the RGB stego the
    # image carrier writes for it): repeat the gray channel into R, G and B,
    # as lsb_amplify does, so changes in G and B are compared too instead of
    # being cut off by the truncation below.
    if a.shape[2] == 1 and b.shape[2] > 1:
        a = np.repeat(a, 3, axis=2)
    elif b.shape[2] == 1 and a.shape[2] > 1:
        b = np.repeat(b, 3, axis=2)
    if a.shape[2] != b.shape[2]:
        # RGB vs RGBA (e.g. a JPEG cover against an RGBA stego): compare only
        # the channels both images have.
        n = min(a.shape[2], b.shape[2])
        a, b = a[..., :n], b[..., :n]

    mask = np.uint8((1 << bits) - 1)
    changed = (a & mask) != (b & mask)
    if channel is None:
        hit = changed.any(axis=2)
    else:
        if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) \
                or not 0 <= channel < changed.shape[2]:
            raise ValueError(f"channel must be in 0..{changed.shape[2] - 1}, got {channel!r}")
        hit = changed[..., int(channel)]
    return np.where(hit, 255, 0).astype(np.uint8)


def save_lsb_diff(cover_path, stego_path, out_path) -> str:
    """Write the cover-vs-stego LSB difference mask as a grayscale PNG.

    The mask is saved at the images' native resolution (one output pixel per
    image pixel, white = an RGB(A) LSB changed), so it is exact and can be
    reloaded and compared pixel for pixel with ``lsb_diff``.

    Args:
        cover_path: Path to the original image.
        stego_path: Path to the stego image, same size as the cover.
        out_path: Where to write the mask. Use a lossless format such as
            .png; parent directories are created.

    Returns:
        ``out_path`` as a string.

    Raises:
        ValueError: If the images differ in size.
    """
    mask = lsb_diff(cover_path, stego_path)
    directory = os.path.dirname(os.fspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(mask).save(out_path)  # 2-D uint8 -> mode "L"
    return os.fspath(out_path)


def _main() -> None:
    """Command-line entry point; see the module docstring for examples."""
    parser = argparse.ArgumentParser(
        description="Bit-plane grid, amplified LSB image and LSB diff for an image.")
    parser.add_argument("image", help="image to inspect (any format Pillow reads)")
    parser.add_argument("--grid", metavar="PNG", help="write a bit-plane grid here")
    parser.add_argument("--channels", default="RGB",
                        help="channel letters for the grid, from RGBA (default: RGB)")
    parser.add_argument("--amplify", metavar="PNG",
                        help="write the amplified RGB LSB image here")
    parser.add_argument("--diff", metavar="COVER",
                        help="cover image to compare LSBs against (needs --diff-out)")
    parser.add_argument("--diff-out", metavar="PNG",
                        help="where to write the LSB difference mask")
    args = parser.parse_args()

    if args.diff and not args.diff_out:
        parser.error("--diff needs --diff-out")
    if not (args.grid or args.amplify or args.diff):
        parser.error("nothing to do: pass --grid, --amplify and/or --diff")

    if args.grid:
        print(save_bitplane_grid(args.image, args.grid, channels=args.channels))
    if args.amplify:
        directory = os.path.dirname(args.amplify)
        if directory:
            os.makedirs(directory, exist_ok=True)
        Image.fromarray(lsb_amplify(args.image)).save(args.amplify)
        print(args.amplify)
    if args.diff:
        out = save_lsb_diff(args.diff, args.image, args.diff_out)
        changed = int((np.asarray(Image.open(out)) > 0).sum())
        print(f"{out} ({changed} pixels with changed LSBs)")


# Run the CLI only when executed as a script, not when imported.
if __name__ == "__main__":
    _main()
