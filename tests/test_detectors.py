"""Tests for the detectors package (blind LSB steganalysis).

Covers are synthetic and built inline so the expected behaviour does not hinge
on one image:

* ``gradient_cover`` (like the conftest ``synthetic_rgb`` fixture, which the
  schema test also runs on): wide linear gradients plus Gaussian noise
  (sigma 2). Its histogram is smooth and its LSB plane already random, the
  hard case for histogram and randomness tests.
* ``natural_cover``: smooth gradients, blurred blobs and a soft sinusoid plus
  mild noise, like a photo of a smooth scene.
* ``natural_cover(stretch=True)``: the same after a contrast stretch through a
  rounded lookup table, which leaves the comb-shaped histogram real edited
  photos have (pair-scale histogram structure).
* ``natural_cover(noise=0, contrast=0.3)``: noiseless low-contrast content
  whose LSB plane is spatially structured (like a rendered graphic or a very
  clean sky).
* ``test-images/smoothBlue.png`` when present (the folder is git-ignored).

Stego versions are made with numpy: random (or text) bits replace the LSBs
of a fraction of the samples, either sequentially from the first sample
(row-major, R, G, B interleaved, the order of this project's encoder) or at
scattered random positions.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

import detectors
from detectors import chi_square as chi_mod
from detectors import randomness as rand_mod
from detectors import signatures as sig_mod
from detectors._common import VERDICTS

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOOTH_BLUE = REPO_ROOT / "test-images" / "smoothBlue.png"

EXPECTED = {"signatures", "chi_square", "sample_pair_analysis", "rs_analysis",
            "histogram_pairs", "lsb_randomness"}


# --------------------------------------------------------------------------- #
# Cover and stego builders
# --------------------------------------------------------------------------- #

def natural_cover(seed: int = 0, size: int = 256, noise: float = 1.0,
                  stretch: bool = False, contrast: float = 1.0) -> np.ndarray:
    """Smooth natural-looking RGB cover (see module docstring).

    ``contrast`` scales the content around grey level 100; a low value makes
    neighbouring pixels share values, so the clean LSB plane is structured.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float) / (size - 1)
    chans = []
    for _ in range(3):
        blobs = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 12)
        blobs /= blobs.std()
        wx, wy = rng.uniform(0.3, 1.0, 2)
        base = (30 + 140 * (wx * x + wy * y) / (wx + wy) + 22 * blobs
                + 12 * np.sin(2 * np.pi * (rng.uniform(1, 3) * x + rng.uniform(0.5, 2) * y)
                              + rng.uniform(0, 2 * np.pi)))
        chans.append(base)
    img = 100 + contrast * (np.stack(chans, -1) - 100)
    if noise:
        img = img + rng.normal(0, noise, img.shape)
    if stretch:
        img = np.rint(np.clip(np.rint(img * 180 / 255), 0, 180) * 255 / 180)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def gradient_cover(seed: int = 0, size: int = 256, noise: float = 2.0) -> np.ndarray:
    """Wide linear gradients plus Gaussian noise: smooth, flat histogram."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float) * (255.0 / (size - 1))
    base = np.stack([x, y, (x + y) / 2.0], axis=-1)
    return np.clip(np.rint(base + rng.normal(0.0, noise, base.shape)), 0, 255).astype(np.uint8)


def embed_lsb(arr: np.ndarray, rate: float, mode: str = "seq", seed: int = 1,
              bits: np.ndarray | None = None) -> np.ndarray:
    """Replace the LSB of ``rate`` of all samples with payload bits.

    ``mode="seq"`` uses the first samples in row-major R, G, B order;
    ``mode="scat"`` uses random positions. ``bits`` defaults to fair coin
    flips (an encrypted payload) and is repeated as needed.
    """
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1).copy()
    n = int(round(rate * flat.size))
    idx = np.arange(n) if mode == "seq" else rng.permutation(flat.size)[:n]
    payload = rng.integers(0, 2, n, dtype=np.uint8) if bits is None else np.resize(bits, n)
    flat[idx] = (flat[idx] & 0xFE) | payload
    return flat.reshape(arr.shape)


def write_bytes_lsb(arr: np.ndarray, data: bytes, channel: int | None = None,
                    bits_per_sample: int = 1) -> np.ndarray:
    """Write ``data`` (MSB-first) into the low bits from the first sample on."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    out = arr.copy()
    if channel is None:
        flat = out.reshape(-1)
    else:
        flat = out[:, :, channel].reshape(-1)  # copy; written back below
    if bits_per_sample == 1:
        flat[:bits.size] = (flat[:bits.size] & 0xFE) | bits
    else:
        pairs = bits.reshape(-1, 2)
        n = pairs.shape[0]
        flat[:n] = (flat[:n] & 0xFC) | (pairs[:, 0] << 1) | pairs[:, 1]
    if channel is not None:
        out[:, :, channel] = flat.reshape(out.shape[:2])
    return out


def text_bits() -> np.ndarray:
    txt = ("The quick brown fox jumps over the lazy dog while the analyst watches "
           "the histogram comb grow. ") * 200
    return np.unpackbits(np.frombuffer(txt.encode("ascii"), dtype=np.uint8))


def stga_frame(body: bytes, flags: int = 0) -> bytes:
    """Build a 14-byte-header STGA frame inline (independent of payload.py)."""
    return (b"STGA" + bytes([1, flags]) + len(body).to_bytes(4, "big")
            + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big") + body)


def check_schema(name: str, result: dict) -> None:
    assert set(result) == {"name", "score", "verdict", "summary", "details"}
    assert result["name"] == name
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0
    assert result["verdict"] in VERDICTS
    assert result["verdict"] == detectors.verdict_for(result["score"])
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["details"], dict)
    json.dumps(result, allow_nan=False)


@pytest.fixture(scope="module")
def covers() -> dict[str, np.ndarray]:
    """Named clean covers used across tests."""
    out = {
        "gradient_noise2": gradient_cover(0),
        "natural": natural_cover(0, noise=1.0),
        "natural_lownoise": natural_cover(1, noise=0.5),
        "stretched": natural_cover(3, noise=1.0, stretch=True),
        "smooth": natural_cover(4, noise=0.0, contrast=0.3),
    }
    if SMOOTH_BLUE.exists():
        out["smoothBlue"] = np.asarray(Image.open(SMOOTH_BLUE).convert("RGB"))
    return out


@pytest.fixture(scope="module")
def structured(covers) -> dict[str, np.ndarray]:
    """Covers whose histogram has pair-scale structure (chi-square applies)."""
    keys = [k for k in ("stretched", "smoothBlue") if k in covers]
    return {k: covers[k] for k in keys}


# --------------------------------------------------------------------------- #
# Registry, schema and run_all
# --------------------------------------------------------------------------- #

def test_registry_lists_all_detectors():
    listed = detectors.list_detectors()
    assert {d["name"] for d in listed} == EXPECTED
    assert all(d["description"] for d in listed)
    for d in listed:
        assert callable(detectors.get_detector(d["name"]))
    with pytest.raises(KeyError):
        detectors.get_detector("no_such_detector")


@pytest.mark.parametrize("rate,mode", [(0.0, "seq"), (0.25, "seq"), (0.5, "scat"), (1.0, "scat")])
def test_every_result_matches_schema(covers, synthetic_rgb, rate, mode):
    for cover in [*covers.values(), synthetic_rgb(0)]:
        results = detectors.run_all(embed_lsb(cover, rate, mode))
        assert list(results) == [d["name"] for d in detectors.list_detectors()]
        for name, res in results.items():
            check_schema(name, res)


def test_run_all_on_path_matches_array(covers, save_png):
    arr = embed_lsb(covers["natural"], 0.5, "scat")
    path = save_png(arr, "stego.png")
    from_path = detectors.run_all(path)
    from_str = detectors.run_all(str(path))
    from_array = detectors.run_all(arr)
    from_pil = detectors.run_all(Image.open(path))
    for name in EXPECTED:
        assert from_path[name]["score"] == pytest.approx(from_array[name]["score"])
        assert from_str[name]["score"] == pytest.approx(from_array[name]["score"])
        assert from_pil[name]["score"] == pytest.approx(from_array[name]["score"])


def test_run_all_accepts_rgba_and_gray(covers):
    rgb = covers["natural"]
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 200, np.uint8)], axis=2)
    a = detectors.run_all(rgb)
    b = detectors.run_all(rgba)
    for name in EXPECTED:
        assert a[name]["score"] == pytest.approx(b[name]["score"])
    gray = detectors.run_all(rgb[:, :, 1])
    for name, res in gray.items():
        check_schema(name, res)


def test_channel_selection(covers):
    res = detectors.run_all(covers["natural"], channels="B")
    assert set(res["sample_pair_analysis"]["details"]["per_channel_rate"]) == {"B"}
    assert set(res["rs_analysis"]["details"]["per_channel_rate"]) == {"B"}
    one = detectors.run("sample_pair_analysis", covers["natural"], channels="gb")
    assert set(one["details"]["per_channel_rate"]) == {"G", "B"}
    with pytest.raises(ValueError):
        detectors.run_all(covers["natural"], channels="RGX")


def test_failing_detector_is_contained(covers):
    @detectors.register("always_fails", "test detector that raises")
    def _boom(arr):
        raise RuntimeError("boom")
    try:
        res = detectors.run_all(covers["natural"])
        check_schema("always_fails", res["always_fails"])
        assert "boom" in res["always_fails"]["details"]["error"]
        assert res["always_fails"]["score"] == 0.0
    finally:
        detectors.unregister("always_fails")
    assert "always_fails" not in {d["name"] for d in detectors.list_detectors()}


def test_tiny_and_constant_images_are_inconclusive():
    rng = np.random.default_rng(0)
    for arr in (rng.integers(0, 256, (5, 5, 3), dtype=np.uint8),
                np.full((64, 64, 3), 128, np.uint8)):
        res = detectors.run_all(arr)
        for name, r in res.items():
            check_schema(name, r)
            assert r["verdict"] == "clean", (name, r["summary"])


# --------------------------------------------------------------------------- #
# Sample pair analysis and RS analysis
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["sample_pair_analysis", "rs_analysis"])
def test_rate_estimators_near_zero_on_clean(covers, name):
    for key, cover in covers.items():
        r = detectors.run(name, cover)
        assert r["details"]["estimated_rate"] < 0.08, key
        assert r["verdict"] == "clean", key


@pytest.mark.parametrize("name", ["sample_pair_analysis", "rs_analysis"])
@pytest.mark.parametrize("mode", ["seq", "scat"])
def test_rate_estimators_at_full_embedding(covers, name, mode):
    for key, cover in covers.items():
        r = detectors.run(name, embed_lsb(cover, 1.0, mode))
        for ch, est in r["details"]["per_channel_rate"].items():
            assert est == pytest.approx(1.0, abs=0.15), (key, ch)
        assert r["verdict"] == "likely_stego", key


@pytest.mark.parametrize("name", ["sample_pair_analysis", "rs_analysis"])
@pytest.mark.parametrize("rate", [0.25, 0.5])
def test_rate_estimators_track_partial_scattered_embedding(covers, name, rate):
    for key, cover in covers.items():
        r = detectors.run(name, embed_lsb(cover, rate, "scat"))
        assert r["details"]["mean_rate"] == pytest.approx(rate, abs=0.1), key
        assert r["verdict"] != "clean", key


def test_spa_counts_and_degenerate_cases():
    from detectors.spa import spa_estimate
    flat = np.full((32, 32), 100, np.uint8)
    est = spa_estimate(flat)
    assert est["X"] == est["Y"] == 0 and est["Z"] == est["pairs"]
    assert est["rate"] == pytest.approx(0.0)


def test_rs_solver_handles_full_embedding_noise():
    # Near p = 1 the x-space roots are complex; the p-space vertex must sit near 1.
    from detectors.rs import _solve_rate
    p, _, case = _solve_rate(-0.0042, 0.0032, -0.3651)
    assert case == "no_real_root"
    assert p == pytest.approx(1.0, abs=0.01)
    p0, _, _ = _solve_rate(0.05, -0.2, 0.0)
    assert p0 == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Chi-square attack
# --------------------------------------------------------------------------- #

def test_chi_square_clean_structured_cover_has_low_p(structured):
    for key, cover in structured.items():
        r = detectors.run("chi_square", cover)
        d = r["details"]
        assert d["p_full"] < 0.01, key
        assert d["estimated_sequential_fraction"] == 0.0, key
        assert r["verdict"] == "clean", key


def test_chi_square_full_sequential_has_high_p(structured):
    for key, cover in structured.items():
        r = detectors.run("chi_square", embed_lsb(cover, 1.0, "seq"))
        d = r["details"]
        assert d["p_full"] > 0.9, key
        assert min(d["p_curve"]["RGB"]) > 0.5, key
        assert d["estimated_sequential_fraction"] == 1.0, key
        assert r["verdict"] == "likely_stego", key


@pytest.mark.parametrize("rate", [0.25, 0.5])
def test_chi_square_estimates_sequential_length(structured, rate):
    for key, cover in structured.items():
        r = detectors.run("chi_square", embed_lsb(cover, rate, "seq"))
        f = r["details"]["estimated_sequential_fraction"]
        assert f == pytest.approx(rate, abs=0.1), key
        assert r["verdict"] == "likely_stego", key


def test_chi_square_smooth_histogram_is_inconclusive_not_alarm(covers):
    # Flat gradient histogram: pairs are balanced with no payload at all.
    r = detectors.run("chi_square", covers["gradient_noise2"])
    assert r["details"]["inconclusive"] is True
    assert r["verdict"] == "clean"


def test_pair_chi_square_merges_sparse_bins():
    h = np.zeros(256, np.int64)
    h[100], h[101] = 3, 1        # expected 2 < 5: merged with following pairs
    h[102], h[103] = 50, 50
    h[200], h[201] = 80, 20
    res = chi_mod.pair_chi_square(h)
    assert res["categories"] == 2 and res["dof"] == 1
    assert 0.0 <= res["p"] <= 1.0
    assert np.isnan(chi_mod.pair_chi_square(np.zeros(256))["p"])


# --------------------------------------------------------------------------- #
# Histogram pairs
# --------------------------------------------------------------------------- #

def test_histogram_pairs_clean_vs_full_random(structured, covers):
    for key, cover in {**structured, "natural": covers["natural"]}.items():
        assert detectors.run("histogram_pairs", cover)["verdict"] == "clean", key
        r = detectors.run("histogram_pairs", embed_lsb(cover, 1.0, "scat"))
        assert r["verdict"] == "likely_stego", key
        assert abs(r["details"]["beta"]) < 0.03, key


def test_histogram_pairs_sees_text_comb(structured):
    bits = text_bits()
    ones = bits.mean()
    for key, cover in structured.items():
        r = detectors.run("histogram_pairs", embed_lsb(cover, 1.0, "seq", bits=bits))
        assert r["verdict"] == "likely_stego", key
        assert r["details"]["estimated_one_bit_fraction"] == pytest.approx(ones, abs=0.03), key
        assert "comb" in r["summary"]


def test_histogram_pairs_keeps_legacy_stats(covers):
    r = detectors.run("histogram_pairs", covers["stretched"])
    legacy = r["details"]["channels"]["B"]["legacy"]
    assert set(legacy) == {"L1", "L2", "median_abs", "max_abs", "max_k"}


# --------------------------------------------------------------------------- #
# LSB randomness
# --------------------------------------------------------------------------- #

def test_bit_stream_stats_random_vs_constant():
    rng = np.random.default_rng(5)
    rnd = rand_mod.bit_stream_stats(rng.integers(0, 2, 80_000))
    assert abs(rnd["monobit_z"]) < 4 and abs(rnd["runs_z"]) < 4
    assert rnd["entropy"] > 7.9 and rnd["byte_chi2_p"] > 1e-4
    assert rand_mod.looks_random(rnd)
    const = rand_mod.bit_stream_stats(np.zeros(80_000, np.uint8))
    assert const["entropy"] == pytest.approx(0.0)
    assert not rand_mod.looks_random(const)


def test_randomness_flags_fully_random_lsbs(covers):
    smooth = [covers["smooth"]] + ([covers["smoothBlue"]] if "smoothBlue" in covers else [])
    for cover in smooth:
        clean = detectors.run("lsb_randomness", cover)
        assert clean["details"]["random_lsb"] is False
        assert clean["verdict"] == "clean"
        r = detectors.run("lsb_randomness", embed_lsb(cover, 1.0, "scat"))
        assert r["details"]["random_lsb"] is True
        assert r["verdict"] == "likely_stego"
        assert "encrypted or compressed" in r["details"]["payload_hint"]
    # On a noisy cover random LSBs are expected; still flagged, but not an alarm.
    r = detectors.run("lsb_randomness", embed_lsb(covers["gradient_noise2"], 1.0, "scat"))
    assert r["details"]["random_lsb"] is True
    assert r["verdict"] == "clean"


def test_randomness_finds_sequential_random_region(covers):
    r = detectors.run("lsb_randomness", embed_lsb(covers["smooth"], 0.25, "seq"))
    d = r["details"]
    assert d["sequential_random_region"] is True
    assert d["leading_random_fraction"] == pytest.approx(0.25, abs=1 / 16)
    assert r["verdict"] == "likely_stego"


def test_randomness_does_not_call_text_random(covers):
    r = detectors.run("lsb_randomness", embed_lsb(covers["smooth"], 1.0, "seq", bits=text_bits()))
    assert r["details"]["random_lsb"] is False
    assert r["verdict"] == "clean"


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #

def test_signatures_clean_covers_have_no_findings(covers):
    for key, cover in covers.items():
        r = detectors.run("signatures", cover)
        assert r["details"]["findings"] == [], key
        assert r["score"] == 0.0


def test_signatures_finds_plain_stga_frame(covers):
    body = b"Hidden in plain sight: meet at the north gate."
    arr = write_bytes_lsb(covers["natural"], stga_frame(body))
    r = detectors.run("signatures", arr)
    assert r["score"] == 1.0 and r["verdict"] == "likely_stego"
    stga = r["details"]["stga"]
    assert stga["order"] == "RGB" and stga["offset"] == 0
    assert stga["version"] == 1 and stga["flags"] == 0 and stga["flag_names"] == []
    assert stga["body_len"] == len(body)
    assert stga["crc_ok"] is True
    assert stga["preview"] == body.decode()


def test_signatures_parses_flags_without_reading_encrypted_body(covers):
    body = np.random.default_rng(9).integers(0, 256, 300, dtype=np.uint8).tobytes()
    arr = write_bytes_lsb(covers["smooth"], stga_frame(body, flags=0x01 | 0x04))
    stga = detectors.run("signatures", arr)["details"]["stga"]
    assert stga["flag_names"] == ["encrypted", "compressed"]
    assert stga["crc_ok"] is True
    assert "preview" not in stga


def test_signatures_crc_checked_beyond_scan_window(covers):
    body = bytes(range(256)) * 40          # 10 KB body, longer than the 4 KB scan
    arr = write_bytes_lsb(covers["natural"], stga_frame(body))
    stga = detectors.run("signatures", arr)["details"]["stga"]
    assert stga["crc_ok"] is True


def test_signatures_other_orders(covers):
    frame = stga_frame(b"two bits per sample")
    two = detectors.run("signatures", write_bytes_lsb(covers["natural"], frame, bits_per_sample=2))
    assert two["details"]["stga"]["order"] == "RGB/2bit"
    blue = detectors.run("signatures", write_bytes_lsb(covers["natural"], frame, channel=2))
    assert blue["details"]["stga"]["order"] == "B"
    assert blue["score"] == 1.0


def test_signatures_finds_png_magic(covers):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(64)
    r = detectors.run("signatures", write_bytes_lsb(covers["natural"], png))
    best = r["details"]["findings"][0]
    assert best["kind"] == "file" and best["label"] == "PNG image" and best["offset"] == 0
    assert best["extension"] == "png"
    assert r["verdict"] == "likely_stego"


def test_signatures_reads_legacy_text_message(covers):
    msg = "Meet me at the old bridge at midnight"
    arr = write_bytes_lsb(covers["natural"], msg.encode() + b"\0" * 8)
    r = detectors.run("signatures", arr)
    best = r["details"]["findings"][0]
    assert best["kind"] == "text" and best["end_marker"] is True
    assert best["preview"] == msg
    assert r["verdict"] == "likely_stego"


def test_parse_stga_header_rejects_bad_input():
    assert sig_mod.parse_stga_header(b"STGA") is None
    assert sig_mod.parse_stga_header(b"XXXX" + bytes(10)) is None
    bad = sig_mod.parse_stga_header(b"STGA" + bytes([9, 0x80]) + bytes(8))
    assert bad["plausible"] is False


# --------------------------------------------------------------------------- #
# Combined verdict
# --------------------------------------------------------------------------- #

def test_combine_results(covers):
    clean = detectors.combine_results(detectors.run_all(covers["natural"]))
    assert clean["verdict"] == "clean"
    stego = detectors.combine_results(detectors.run_all(embed_lsb(covers["natural"], 1.0, "scat")))
    assert stego["verdict"] == "likely_stego"
    assert stego["driver"] in EXPECTED


@pytest.mark.skipif(not SMOOTH_BLUE.exists(), reason="test-images/smoothBlue.png not present")
def test_smooth_blue_png_is_clean_on_every_detector():
    res = detectors.run_all(SMOOTH_BLUE)
    for name, r in res.items():
        check_schema(name, r)
        assert r["verdict"] == "clean", (name, r["summary"])
    assert detectors.combine_results(res)["verdict"] == "clean"
