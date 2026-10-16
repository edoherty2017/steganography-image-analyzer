# tests/test_bitplanes.py
"""Tests for bitplanes.py: plane extraction, grid PNGs, LSB amplify and diff.

The module is self-contained (no conftest fixtures) so it does not depend on
fixture names other test modules may define.

Run from the repository root:

    python -m pytest -q tests/test_bitplanes.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from bitplanes import (
    GRID_DPI,
    bitplane,
    grid_size_px,
    lsb_amplify,
    lsb_diff,
    save_bitplane_grid,
    save_lsb_diff,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOOTH_BLUE = os.path.join(REPO_ROOT, "test-images", "smoothBlue.png")


def _synthetic(seed: int = 0, size: int = 64) -> np.ndarray:
    """Natural-looking RGB test image: smooth gradients plus mild noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float) * (255.0 / (size - 1))
    base = np.stack([x, y, (x + y) / 2], axis=-1)
    noisy = base + rng.normal(0.0, 2.0, base.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def _save(arr: np.ndarray, path) -> str:
    """Save an array as a PNG and return the path as a string."""
    Image.fromarray(arr).save(path)
    return str(path)


# --- bitplane ----------------------------------------------------------------

def test_bitplane_values_are_only_0_or_255():
    arr = _synthetic(1)
    for channel in range(3):
        for bit in range(8):
            plane = bitplane(arr, channel, bit)
            assert plane.dtype == np.uint8
            assert plane.shape == arr.shape[:2]
            assert set(np.unique(plane)) <= {0, 255}


def test_bitplane_extracts_known_bits():
    # One pixel per row; each channel value has a distinct, hand-chosen bit
    # pattern so every (channel, bit) combination is checked.
    values = np.array([
        [[0b10000001, 0b01000010, 0b00100100]],
        [[0b11111111, 0b00000000, 0b10101010]],
        [[0b01010101, 0b00011000, 0b11110000]],
    ], dtype=np.uint8)
    for channel in range(3):
        for bit in range(8):
            expected = ((values[..., channel] >> bit) & 1) * 255
            np.testing.assert_array_equal(bitplane(values, channel, bit), expected)

    # Spot checks written out by hand.
    assert bitplane(values, 0, 0)[0, 0] == 255   # 0b10000001, bit 0
    assert bitplane(values, 0, 7)[0, 0] == 255   # 0b10000001, bit 7
    assert bitplane(values, 0, 1)[0, 0] == 0
    assert bitplane(values, 1, 1)[0, 0] == 255   # 0b01000010, bit 1
    assert bitplane(values, 2, 2)[0, 0] == 255   # 0b00100100, bit 2
    assert bitplane(values, 2, 0)[1, 0] == 0     # 0b10101010, bit 0
    assert bitplane(values, 2, 1)[1, 0] == 255


def test_bitplane_grayscale_and_path_input(tmp_path):
    gray = np.array([[0, 1, 2, 3], [128, 129, 254, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(
        bitplane(gray, 0, 0), [[0, 255, 0, 255], [0, 255, 0, 255]])
    np.testing.assert_array_equal(
        bitplane(gray, 0, 7), [[0, 0, 0, 0], [255, 255, 255, 255]])

    arr = _synthetic(2)
    path = _save(arr, tmp_path / "img.png")
    np.testing.assert_array_equal(bitplane(path, 2, 0), bitplane(arr, 2, 0))


@pytest.mark.parametrize("channel, bit", [(3, 0), (-1, 0), (0, 8), (0, -1)])
def test_bitplane_rejects_out_of_range(channel, bit):
    with pytest.raises(ValueError):
        bitplane(_synthetic(3), channel, bit)


# --- save_bitplane_grid ------------------------------------------------------

def test_grid_file_has_expected_size_and_dpi(tmp_path):
    out = tmp_path / "nested" / "grid.png"
    returned = save_bitplane_grid(_synthetic(4), out)
    assert returned == str(out)
    assert out.is_file()
    with Image.open(out) as img:
        assert img.format == "PNG"
        # 3 channel rows x 8 bit columns.
        assert img.size == grid_size_px(3, 8)
        dpi_x, dpi_y = img.info["dpi"]
    assert dpi_x == pytest.approx(GRID_DPI, abs=0.5)
    assert dpi_y == pytest.approx(GRID_DPI, abs=0.5)
    # Pin the geometry too, so an accidental bbox_inches="tight" or a layout
    # change that alters the size is caught: 8*180+30 by 3*180+60 pixels.
    assert grid_size_px(3, 8) == (1470, 600)


def test_grid_columns_follow_requested_bits(tmp_path):
    out = save_bitplane_grid(_synthetic(5), tmp_path / "g.png",
                             channels="B", bits=[0, 1, 1, 7])
    with Image.open(out) as img:
        assert img.size == grid_size_px(1, 3)  # duplicates dropped


def test_grid_rejects_bad_arguments(tmp_path):
    arr = _synthetic(6)
    with pytest.raises(ValueError):
        save_bitplane_grid(arr, tmp_path / "a.png", bits=[])
    with pytest.raises(ValueError):
        save_bitplane_grid(arr, tmp_path / "b.png", bits=[8])
    with pytest.raises(ValueError):
        save_bitplane_grid(arr, tmp_path / "c.png", channels="RGX")
    with pytest.raises(ValueError):
        save_bitplane_grid(arr, tmp_path / "d.png", channels="A")  # no alpha


def test_grid_downsamples_large_images(tmp_path):
    big = np.random.default_rng(7).integers(0, 256, (1200, 900, 3), dtype=np.uint8)
    out = save_bitplane_grid(big, tmp_path / "big.png", bits=[0])
    with Image.open(out) as img:
        assert img.size == grid_size_px(3, 1)


@pytest.mark.skipif(not os.path.exists(SMOOTH_BLUE),
                    reason="test-images/smoothBlue.png not present (gitignored)")
def test_grid_from_real_image(tmp_path):
    out = save_bitplane_grid(SMOOTH_BLUE, tmp_path / "blue.png")
    with Image.open(out) as img:
        assert img.size == grid_size_px(3, 8)


# --- RGBA handling -----------------------------------------------------------

def _rgba(seed: int = 8) -> np.ndarray:
    rgb = _synthetic(seed)
    alpha = np.full(rgb.shape[:2], 0b11001001, dtype=np.uint8)  # LSB = 1
    alpha[: rgb.shape[0] // 2] = 0b11001000                     # LSB = 0
    return np.dstack([rgb, alpha])


def test_rgba_file_keeps_alpha_and_grid_adds_row(tmp_path):
    arr = _rgba()
    path = _save(arr, tmp_path / "rgba.png")

    # The alpha plane is readable from the file (index 3).
    np.testing.assert_array_equal(bitplane(path, 3, 0), bitplane(arr, 3, 0))
    assert bitplane(path, 3, 0)[0, 0] == 0
    assert bitplane(path, 3, 0)[-1, 0] == 255

    with Image.open(save_bitplane_grid(path, tmp_path / "g4.png", channels="RGBA")) as img:
        assert img.size == grid_size_px(4, 8)
    # Without "A" in channels, alpha is not drawn.
    with Image.open(save_bitplane_grid(path, tmp_path / "g3.png")) as img:
        assert img.size == grid_size_px(3, 8)


def test_rgba_request_on_rgb_image_skips_alpha(tmp_path):
    out = save_bitplane_grid(_synthetic(9), tmp_path / "g.png", channels="RGBA")
    with Image.open(out) as img:
        assert img.size == grid_size_px(3, 8)


def test_palette_image_with_transparency_loads_as_rgba(tmp_path):
    pal = Image.fromarray(_synthetic(10)).convert("P")
    pal.info["transparency"] = 0
    path = tmp_path / "pal.png"
    pal.save(path, transparency=0)
    with Image.open(save_bitplane_grid(path, tmp_path / "g.png", channels="RGBA")) as img:
        assert img.size == grid_size_px(4, 8)


# --- lsb_amplify -------------------------------------------------------------

def test_lsb_amplify_rgb_matches_lsbs():
    arr = _synthetic(11)
    amp = lsb_amplify(arr)
    assert amp.shape == arr.shape and amp.dtype == np.uint8
    np.testing.assert_array_equal(amp, (arr & 1) * 255)


def test_lsb_amplify_drops_alpha_and_tiles_grayscale():
    arr = _rgba()
    amp = lsb_amplify(arr)
    assert amp.shape == arr.shape[:2] + (3,)
    np.testing.assert_array_equal(amp, (arr[..., :3] & 1) * 255)

    gray = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    amp = lsb_amplify(gray)
    assert amp.shape == (2, 2, 3)
    for c in range(3):
        np.testing.assert_array_equal(amp[..., c], [[0, 255], [0, 255]])


# --- lsb_diff / save_lsb_diff ------------------------------------------------

def _known_modification():
    """Cover plus a stego copy with hand-placed changes; returns expected masks."""
    cover = _synthetic(12)
    stego = cover.copy()
    expected = np.zeros(cover.shape, dtype=bool)  # per-channel LSB changes

    # Flipped LSBs: these must show up.
    for (r, c, ch) in [(0, 0, 0), (5, 7, 1), (20, 3, 2), (63, 63, 0), (63, 63, 2)]:
        stego[r, c, ch] ^= 1
        expected[r, c, ch] = True

    # Rewriting the LSB with the value it already has: no visible change.
    stego[10, 10, 0] = (stego[10, 10, 0] & 0xFE) | (cover[10, 10, 0] & 1)
    # Flipping bit 1 only: the LSB is unchanged, so it must not show up.
    stego[30, 30, 1] ^= 0b10
    stego[31, 31, 2] ^= 0b10
    return cover, stego, expected


def test_lsb_diff_exact_on_known_modification():
    cover, stego, expected = _known_modification()

    mask = lsb_diff(cover, stego)
    assert mask.dtype == np.uint8 and mask.shape == cover.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    np.testing.assert_array_equal(mask, np.where(expected.any(axis=2), 255, 0))
    assert int((mask == 255).sum()) == 4  # (63, 63) has two changed channels

    for ch in range(3):
        np.testing.assert_array_equal(
            lsb_diff(cover, stego, channel=ch), np.where(expected[..., ch], 255, 0))


def test_lsb_diff_two_bits_sees_bit1_changes():
    cover, stego, expected = _known_modification()
    mask = lsb_diff(cover, stego, bits=2)
    assert mask[30, 30] == 255 and mask[31, 31] == 255
    assert int((mask == 255).sum()) == 6


def test_lsb_diff_identical_images_is_all_black():
    arr = _synthetic(13)
    assert not lsb_diff(arr, arr.copy()).any()


def test_lsb_diff_rgba_alpha_changes():
    cover = _rgba()
    stego = cover.copy()
    stego[2, 3, 3] ^= 1  # alpha LSB only
    assert lsb_diff(cover, stego)[2, 3] == 255
    assert lsb_diff(cover, stego, channel=3)[2, 3] == 255
    assert not lsb_diff(cover, stego, channel=0).any()
    # RGB vs RGBA: only the shared R, G, B channels are compared.
    assert not lsb_diff(cover[..., :3], stego).any()


def test_lsb_diff_rejects_bad_input():
    a = _synthetic(14, size=32)
    b = _synthetic(14, size=16)
    with pytest.raises(ValueError):
        lsb_diff(a, b)
    with pytest.raises(ValueError):
        lsb_diff(a, a, channel=3)
    with pytest.raises(ValueError):
        lsb_diff(a, a, bits=0)


def test_save_lsb_diff_round_trip(tmp_path):
    cover, stego, expected = _known_modification()
    cover_path = _save(cover, tmp_path / "cover.png")
    stego_path = _save(stego, tmp_path / "stego.png")
    out = tmp_path / "out" / "diff.png"

    assert save_lsb_diff(cover_path, stego_path, out) == str(out)
    with Image.open(out) as img:
        assert img.mode == "L"
        saved = np.asarray(img)
    np.testing.assert_array_equal(saved, lsb_diff(cover, stego))
    np.testing.assert_array_equal(saved, np.where(expected.any(axis=2), 255, 0))


def test_lsb_diff_grayscale_array_against_rgb_compares_all_channels():
    # Regression: a 2-D cover array against an RGB stego used to be cut down
    # to channel 0, so LSB changes in G and B were never reported.
    gray = np.random.default_rng(0).integers(0, 256, (10, 10), dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    stego = rgb.copy()
    stego[..., 1:] ^= 1  # flip the G and B LSBs only

    assert int((lsb_diff(gray, stego) > 0).sum()) == 100
    assert int((lsb_diff(stego, gray) > 0).sum()) == 100  # either argument order
    np.testing.assert_array_equal(lsb_diff(gray, stego), lsb_diff(rgb, stego))
    assert not lsb_diff(gray, stego, channel=0).any()
    assert (lsb_diff(gray, stego, channel=1) == 255).all()
    assert (lsb_diff(gray, stego, channel=2) == 255).all()

    # Grayscale against RGBA: R, G and B are compared, alpha is ignored.
    stego_rgba = np.dstack([stego, np.full(gray.shape, 255, np.uint8)])
    assert int((lsb_diff(gray, stego_rgba) > 0).sum()) == 100
    # Two grayscale arrays still compare their single channel.
    assert not lsb_diff(gray, gray.copy()).any()
    assert int((lsb_diff(gray, gray ^ 1) > 0).sum()) == 100


# --- 16-bit sources ----------------------------------------------------------

def _gradient16(height: int = 128) -> np.ndarray:
    """A 16-bit horizontal ramp whose high bytes are exactly 0..255."""
    return np.linspace(0, 65535, 256)[None, :].repeat(height, 0).astype(np.uint16)


def _write_png48(path, arr16: np.ndarray) -> str:
    """Write an (H, W, 3) uint16 array as a 48-bit RGB PNG.

    Pillow cannot save 48-bit RGB, so the chunks are written by hand.
    """
    import struct
    import zlib

    h, w, _ = arr16.shape
    raw = b"".join(b"\x00" + arr16[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit, truecolor
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return str(path)


def test_16bit_grayscale_file_is_scaled_not_clipped(tmp_path):
    # Regression: Pillow's convert("RGB") clips 16-bit values above 255 to
    # 255, which made 99.6% of every plane white. The planes must be those
    # of the high byte (value >> 8) instead.
    g16 = _gradient16()
    path = str(tmp_path / "g16.png")
    Image.fromarray(g16).save(path)
    with Image.open(path) as img:
        assert img.mode == "I;16"

    # The high bytes are 0..255 once per row, so every plane is exactly half
    # white; bit 7 is the left/right split of the ramp.
    assert (bitplane(path, 0, 7) == 255).mean() == 0.5
    high = (g16 >> 8).astype(np.uint8)
    for channel in range(3):
        for bit in range(8):
            np.testing.assert_array_equal(bitplane(path, channel, bit),
                                          bitplane(high, 0, bit))
    np.testing.assert_array_equal(lsb_amplify(path)[..., 0], (high & 1) * 255)


def test_16bit_rgb_png_shows_high_byte(tmp_path):
    g16 = _gradient16(8)
    arr16 = np.dstack([g16, g16[:, ::-1], np.full_like(g16, 0x8001)])
    path = _write_png48(tmp_path / "rgb48.png", arr16)
    high = (arr16 >> 8).astype(np.uint8)
    for channel in range(3):
        for bit in (0, 7):
            np.testing.assert_array_equal(bitplane(path, channel, bit),
                                          bitplane(high, channel, bit))


def test_file_planes_match_image_carrier_pixels(tmp_path):
    # The planes must be drawn from the pixels the image carrier embeds into,
    # whatever the file's mode, so the grid and lsb_diff line up with stego.
    from carriers.image_lsb import ImageLSBCarrier

    paths = [str(tmp_path / "g16.png"), _save(_synthetic(15), tmp_path / "rgb.png"),
             _save(_rgba(16), tmp_path / "rgba.png")]
    Image.fromarray(_gradient16(16)).save(paths[0])
    Image.fromarray(_synthetic(17)[..., 0]).save(tmp_path / "l.png")
    paths.append(str(tmp_path / "l.png"))
    for path in paths:
        pixels = ImageLSBCarrier.load(path).pixels
        for channel in range(pixels.shape[2]):
            for bit in (0, 7):
                np.testing.assert_array_equal(bitplane(path, channel, bit),
                                              bitplane(pixels, channel, bit))


def test_grid_title_says_when_low_bits_are_not_shown(tmp_path):
    from bitplanes import _read_file

    g16_path = str(tmp_path / "g16.png")
    Image.fromarray(_gradient16(16)).save(g16_path)
    png48 = _write_png48(tmp_path / "rgb48.png", np.dstack([_gradient16(8)] * 3))
    png8 = _save(_synthetic(18), tmp_path / "rgb8.png")

    for path in (g16_path, png48):
        note = _read_file(path)[1]
        assert note is not None and "high byte" in note and "not shown" in note
    assert _read_file(png8)[1] is None

    # The two-line title must not change the figure's pixel size.
    with Image.open(save_bitplane_grid(g16_path, tmp_path / "g.png")) as img:
        assert img.size == grid_size_px(3, 8)


def test_hide_in_16bit_cover_changes_only_payload_lsbs(tmp_path):
    # End to end: with the cover read the same way as the carrier, the LSB
    # diff marks only the few pixels the payload went into, not ~99.6% of
    # the image as when the cover was clipped to white.
    import payload
    import stego

    cover = str(tmp_path / "g16.png")
    Image.fromarray(_gradient16()).save(cover)
    out = str(tmp_path / "stego.png")
    stego.hide(cover, out, "hi")

    changed = int((lsb_diff(cover, out) > 0).sum())
    frame_bits = (payload.HEADER_LEN + len(b"hi")) * 8
    assert 0 < changed <= -(-frame_bits // 3)  # at most one pixel per 3 bits


# --- module hygiene ----------------------------------------------------------

def test_import_does_not_load_pyplot():
    # The GUI relies on whatever backend pyplot picks; bitplanes must not
    # import pyplot or switch the global backend as a side effect.
    code = ("import sys, bitplanes; "
            "assert 'matplotlib.pyplot' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_rejects_non_image_input():
    with pytest.raises(TypeError):
        bitplane([[1, 2], [3, 4]], 0, 0)
    with pytest.raises(TypeError):
        lsb_amplify(np.zeros((4, 4, 3), dtype=float))
    with pytest.raises(ValueError):
        lsb_amplify(np.zeros((4, 4, 2), dtype=np.uint8))
