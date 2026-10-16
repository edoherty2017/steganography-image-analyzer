"""Regression tests for the input handling and score combination in detectors/__init__.py.

* Grayscale images are analysed as one channel. Interleaving three identical
  channels repeated every LSB three times, so the runs test called a fully
  random LSB plane "structured" and the chi-square counts were tripled.
* Images with more than 8 bits per sample are rejected instead of being
  clipped or cut to their high byte (which made every detector say clean).
* ``combine_results`` no longer adds a bonus when SPA and RS agree: the two
  estimators are strongly correlated, so the bonus counted the same evidence
  twice and pushed clean covers to "likely_stego".
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import detectors
import report
from conftest import make_synthetic_rgb
from test_detectors import EXPECTED, embed_lsb, natural_cover


def _gray_stego() -> np.ndarray:
    """Smooth low-contrast grayscale cover with every LSB replaced by random bits."""
    g = natural_cover(2, noise=0, contrast=0.3)[:, :, 1]
    bits = np.random.default_rng(0).integers(0, 2, g.shape, dtype=np.uint8)
    return (g & 0xFE) | bits


def _write_png16(path: Path, arr: np.ndarray, color_type: int) -> Path:
    """Write a 16-bit-per-sample PNG (Pillow cannot save 48/64-bit RGB PNGs)."""
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


# --------------------------------------------------------------------------- #
# Grayscale input
# --------------------------------------------------------------------------- #

def test_gray_png_random_lsb_is_seen_like_a_single_channel(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(_gray_stego(), "L").save(path)
    rgb = detectors.run("lsb_randomness", path, channels="RGB")
    one = detectors.run("lsb_randomness", path, channels="R")
    # The equality is the regression guard for the tripled stream; "not clean"
    # is the reported symptom and also depends on randomness.py thresholds.
    assert rgb["score"] == pytest.approx(one["score"])
    assert rgb["verdict"] == one["verdict"] != "clean", rgb["summary"]
    # The runs test sees each LSB once: no huge negative z from repeated bits.
    assert abs(rgb["details"]["stream"]["runs_z"]) < 4.0
    assert rgb["details"]["input"]["grayscale"] is True
    assert rgb["details"]["input"]["analysed_channels"] == detectors.GRAY_LABEL


@pytest.mark.parametrize("form", ["2d", "rgb_copies", "l_png", "rgb_png"])
def test_every_detector_analyses_gray_input_once(tmp_path, form):
    gray = _gray_stego()
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if form == "2d":
        image = gray
    elif form == "rgb_copies":
        image = rgb
    elif form == "l_png":
        image = tmp_path / "g.png"
        Image.fromarray(gray, "L").save(image)
    else:  # a grayscale picture saved as an RGB file
        image = tmp_path / "g_rgb.png"
        Image.fromarray(rgb, "RGB").save(image)
    res = detectors.run_all(image)
    single = detectors.run_all(gray, channels="G")
    assert list(res) == [d["name"] for d in detectors.list_detectors()]
    for name in EXPECTED:
        assert res[name]["score"] == pytest.approx(single[name]["score"]), name
        assert res[name]["details"]["input"]["requested_channels"] == "RGB"
        assert "input" not in single[name]["details"], name
    for name in ("sample_pair_analysis", "rs_analysis"):
        assert set(res[name]["details"]["per_channel_rate"]) == {detectors.GRAY_LABEL}
    # One chi-square stream (no tripled interleaved stream), same statistic.
    curves = res["chi_square"]["details"]["chi2_curve"]
    assert list(curves) == [detectors.GRAY_LABEL]
    assert curves[detectors.GRAY_LABEL] == pytest.approx(
        single["chi_square"]["details"]["chi2_curve"]["G"])


def test_colour_images_keep_all_channels():
    cover = natural_cover(0)
    res = detectors.run_all(cover)
    assert set(res["sample_pair_analysis"]["details"]["per_channel_rate"]) == {"R", "G", "B"}
    assert all("input" not in r["details"] for r in res.values())
    # A grey picture with a payload interleaved over R, G and B is no longer
    # grayscale and must get the three-channel analysis.
    grey_rgb = np.repeat(cover[:, :, 1:2], 3, axis=2)
    stego = embed_lsb(grey_rgb, 0.5, "seq")
    assert not detectors.is_grayscale(stego)
    res = detectors.run_all(stego)
    assert set(res["rs_analysis"]["details"]["per_channel_rate"]) == {"R", "G", "B"}


def test_is_grayscale():
    gray = _gray_stego()
    assert detectors.is_grayscale(gray)
    assert detectors.is_grayscale(np.repeat(gray[:, :, None], 3, axis=2))
    rgb = natural_cover(0)
    assert not detectors.is_grayscale(rgb)
    almost = np.repeat(gray[:, :, None], 3, axis=2)
    almost[5, 7, 2] ^= 1
    assert not detectors.is_grayscale(almost)


# --------------------------------------------------------------------------- #
# More than 8 bits per sample
# --------------------------------------------------------------------------- #

def _gradient16() -> np.ndarray:
    return (np.add.outer(np.arange(128), np.arange(128)) * 200).astype(np.uint16)


def test_16bit_gray_png_is_rejected_not_clipped(tmp_path):
    path = tmp_path / "g16.png"
    Image.fromarray(_gradient16()).save(path)
    assert Image.open(path).mode == "I;16"
    with pytest.raises(ValueError, match="I;16"):
        detectors.load_rgb(path)
    with pytest.raises(ValueError, match="8-bit samples"):
        detectors.run_all(path)
    with pytest.raises(ValueError):
        detectors.run("lsb_randomness", str(path))
    with Image.open(path) as im:  # a loaded PIL image, as report.analyze_image passes
        im.load()
        with pytest.raises(ValueError):
            detectors.run_all(im)
    with pytest.raises(ValueError):
        report.analyze_image(path, run_bruteforce=False)


def test_16bit_container_with_8bit_values_is_accepted(tmp_path):
    g8 = (_gradient16() // 256).astype(np.uint8)
    path = tmp_path / "g8_in_16.png"
    Image.fromarray(g8.astype(np.uint16)).save(path)
    assert Image.open(path).mode == "I;16"
    arr = detectors.load_rgb(path)
    assert arr.dtype == np.uint8 and np.array_equal(arr[:, :, 0], g8)
    from_16 = detectors.run_all(path)
    from_8 = detectors.run_all(g8)
    for name in EXPECTED:
        assert from_16[name]["score"] == pytest.approx(from_8[name]["score"]), name


@pytest.mark.parametrize("color_type,channels", [(2, 3), (6, 4), (4, 2)])
def test_48_and_64_bit_png_are_rejected(tmp_path, color_type, channels):
    # Pillow opens these as plain RGB / RGBA holding only the high byte.
    arr = np.random.default_rng(1).integers(0, 65536, (32, 32, channels)).astype(np.uint16)
    path = _write_png16(tmp_path / f"deep{color_type}.png", arr, color_type)
    assert Image.open(path).mode in ("RGB", "RGBA")
    with pytest.raises(ValueError, match="16-bit-per-sample"):
        detectors.run_all(path)
    with Image.open(path) as im:
        im.load()  # clears im.tile; the file header is read again
        with pytest.raises(ValueError, match="16-bit-per-sample"):
            detectors.run_all(im)


def test_float_and_int32_images_are_rejected(tmp_path):
    path = tmp_path / "f.tif"
    Image.fromarray(np.random.default_rng(2).random((32, 32)).astype(np.float32)).save(path)
    with pytest.raises(ValueError, match="mode F"):
        detectors.run_all(path)
    wide = Image.fromarray((np.arange(64 * 64, dtype=np.int32) * 50).reshape(64, 64))
    assert wide.mode == "I"
    with pytest.raises(ValueError, match="mode I"):
        detectors.load_rgb(wide)
    narrow = Image.fromarray((np.arange(64 * 64, dtype=np.int32) % 256).reshape(64, 64))
    assert detectors.load_rgb(narrow).shape == (64, 64, 3)


def test_packed_16bit_pixel_modes_are_not_mistaken_for_16bit_samples():
    pattern = detectors._SIXTEEN_BIT_RAWMODE
    for raw in ("RGB;16B", "RGB;16L", "RGBA;16B", "LA;16B", "RGBX;16L", "I;16B", "I;16N"):
        assert pattern.search(raw), raw
    for raw in ("BGR;16", "BGR;15", "BGRA;15Z", "RGB", "RGBX", "L"):
        assert not pattern.search(raw), raw


# --------------------------------------------------------------------------- #
# combine_results
# --------------------------------------------------------------------------- #

def _fake(name: str, score: float) -> dict:
    return {"name": name, "score": score, "verdict": detectors.verdict_for(score),
            "summary": "", "details": {}}


def test_combine_results_does_not_add_a_bonus_for_spa_and_rs_agreeing():
    scores = {"signatures": 0.0, "chi_square": 0.1, "sample_pair_analysis": 0.566,
              "rs_analysis": 0.5, "histogram_pairs": 0.2, "lsb_randomness": 0.3}
    out = detectors.combine_results({k: _fake(k, v) for k, v in scores.items()})
    assert out["score"] == pytest.approx(0.566)
    assert out["verdict"] == "suspicious"
    assert out["driver"] == "sample_pair_analysis"
    assert set(out["flagged"]) == {"sample_pair_analysis", "rs_analysis"}


@pytest.mark.parametrize("seed,size", [(8, 64), (0, 128), (3, 256)])
def test_combine_results_is_the_largest_weighted_score(seed, size):
    res = detectors.run_all(make_synthetic_rgb(seed, size))
    best = max(r["score"] * detectors.COMBINE_WEIGHTS[k] for k, r in res.items())
    assert detectors.combine_results(res)["score"] == pytest.approx(best)
