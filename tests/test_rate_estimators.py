"""Regression tests for the two rate estimators, SPA and RS.

Two problems are covered:

* Size scaling. The score knots were calibrated at 256x256, but the spread
  of the estimate on a clean cover grows as ``1 / sqrt(pairs)``. Before the
  fix, 16 of 20 clean 64x64 synthetic covers had an overall verdict of
  suspicious or likely_stego (SPA alone read 20.2% on seed 2). The knots now
  scale with ``sqrt(reference units / units)`` (``detectors._rates``).
* Sequential payloads. Both estimators assume scattered embedding and read
  low on a payload written from the top of the image (0.31 for a true 0.50
  on smoothBlue.png). They now compare the top and bottom halves and label
  the estimate a lower bound when the payload looks sequential.

Covers come from ``conftest.make_synthetic_rgb`` (wide gradients plus noise,
the noisiest case) and ``test_detectors.natural_cover``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import detectors
from conftest import make_synthetic_rgb
from detectors import rs as rs_mod
from detectors import spa as spa_mod
from detectors._common import interp_score
from detectors._rates import knot_scale, scale_knots
from test_detectors import embed_lsb, natural_cover

ESTIMATORS = ["sample_pair_analysis", "rs_analysis"]
SMOOTH_BLUE = Path(__file__).resolve().parent.parent / "test-images" / "smoothBlue.png"


def _units(name: str, size: int) -> tuple[int, int]:
    """(units per channel, reference units) for a square image of ``size``."""
    if name == "sample_pair_analysis":
        return size * (size - 1), spa_mod.REF_PAIRS
    return size * (size // len(rs_mod.MASK)), rs_mod.REF_GROUPS


def _module(name: str):
    """The detector module behind a registered estimator name."""
    return spa_mod if name == "sample_pair_analysis" else rs_mod


# --------------------------------------------------------------------------- #
# Size-scaled thresholds
# --------------------------------------------------------------------------- #

def test_issue_repro_small_clean_cover_is_clean():
    """The 64x64 cover from the bug report: SPA read 20.2% -> likely_stego."""
    res = detectors.run_all(make_synthetic_rgb(2, 64))
    for name in ESTIMATORS:
        r = res[name]
        assert r["verdict"] == "clean", r["summary"]
        assert r["details"]["knot_scale"] == pytest.approx(4.0, abs=0.05)
        assert "thresholds widened x4.0" in r["summary"]
    # The estimate itself is unchanged (about 0.20); only its interpretation scales.
    assert res["sample_pair_analysis"]["details"]["estimated_rate"] > 0.15


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("size", [64, 128])
def test_small_clean_covers_are_not_flagged(name, size):
    """20 seeds of two cover families: no likely_stego, at most one suspicious.

    With the old fixed knots SPA flagged 14 of the 20 wide-gradient covers at
    64x64 (8 of them likely_stego), and RS 9 of the 20 natural ones.
    """
    verdicts = []
    for seed in range(20):
        for cover in (make_synthetic_rgb(seed, size), natural_cover(seed, size)):
            verdicts.append(detectors.run(name, cover)["verdict"])
    assert verdicts.count("likely_stego") == 0
    assert verdicts.count("suspicious") <= (0 if size == 64 else 1)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_knots_scale_with_image_size(name):
    for size in (64, 128, 256, 320):
        units, ref = _units(name, size)
        r = detectors.run(name, make_synthetic_rgb(0, size))
        d = r["details"]
        expected = max(1.0, math.sqrt(ref / units))
        assert d["knot_scale"] == pytest.approx(expected)
        base = _module(name).SCORE_KNOTS
        assert [tuple(k) for k in d["base_score_knots"]] == list(base)
        assert [tuple(k) for k in d["score_knots"]] == pytest.approx(list(scale_knots(base, expected)))
        assert r["score"] == pytest.approx(interp_score(d["estimated_rate"], d["score_knots"]))
        assert ("thresholds widened" in r["summary"]) == (size < 256)
    # 256x256 is the calibration size: the knots there are the documented ones.
    assert knot_scale(*_units(name, 256)) == 1.0
    assert knot_scale(*_units(name, 128)) == pytest.approx(2.0, abs=0.02)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_small_images_still_detect_heavy_payloads(name):
    for seed in range(5):
        full64 = detectors.run(name, embed_lsb(natural_cover(seed, 64), 1.0, "scat", seed=seed + 1))
        assert full64["verdict"] == "likely_stego", full64["summary"]
        half128 = detectors.run(name, embed_lsb(make_synthetic_rgb(seed, 128), 0.5, "scat", seed=seed + 1))
        assert half128["verdict"] == "likely_stego", half128["summary"]


@pytest.mark.parametrize("name", ESTIMATORS)
def test_below_minimum_size_is_inconclusive(name):
    """48x48 is below the new minimum (about 64x64): too noisy to report."""
    r = detectors.run(name, embed_lsb(make_synthetic_rgb(0, 48), 1.0, "scat"))
    assert r["details"]["inconclusive"] is True
    assert r["verdict"] == "clean" and r["summary"].startswith("Inconclusive")
    assert detectors.run(name, make_synthetic_rgb(0, 64))["details"].get("inconclusive") is None


# --------------------------------------------------------------------------- #
# Sequential payloads: lower-bound label
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def smooth_cover() -> np.ndarray:
    """Noiseless low-contrast 256x256 cover, where the sequential bias is large."""
    return natural_cover(4, noise=0.0, contrast=0.3)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("rate", [0.25, 0.5, 0.9])
def test_sequential_payload_is_labelled_lower_bound(smooth_cover, name, rate):
    r = detectors.run(name, embed_lsb(smooth_cover, rate, "seq"))
    d = r["details"]
    assert d["lower_bound"] is True
    assert d["row_halves"]["sequential"] is True
    assert d["row_halves"]["top"] > d["row_halves"]["bottom"]
    assert "lower bound" in r["summary"]
    # The bias this label warns about: the whole-image mean reads low.
    assert d["mean_rate"] < rate - 0.08
    # The label does not change the score.
    assert r["score"] == pytest.approx(interp_score(d["estimated_rate"], d["score_knots"]))


@pytest.mark.parametrize("name", ESTIMATORS)
def test_no_lower_bound_label_without_sequential_pattern(smooth_cover, name):
    cases = {
        "clean": smooth_cover,
        "scattered 0.5": embed_lsb(smooth_cover, 0.5, "scat"),
        "scattered 0.9": embed_lsb(smooth_cover, 0.9, "scat"),
        "sequential 1.0": embed_lsb(smooth_cover, 1.0, "seq"),  # whole image, no gap
        "noisy clean": make_synthetic_rgb(3),
        "noisy scattered 0.5": embed_lsb(make_synthetic_rgb(3), 0.5, "scat"),
    }
    for label, arr in cases.items():
        r = detectors.run(name, arr)
        assert r["details"]["lower_bound"] is False, label
        assert "lower bound" not in r["summary"], label


@pytest.mark.skipif(not SMOOTH_BLUE.exists(), reason="test-images/smoothBlue.png not present")
@pytest.mark.parametrize("name", ESTIMATORS)
def test_smooth_blue_sequential_half_payload(name):
    """The bug report's case: a true 0.50 sequential payload read about 0.31."""
    cover = np.asarray(Image.open(SMOOTH_BLUE).convert("RGB"))
    seq = detectors.run(name, embed_lsb(cover, 0.5, "seq"))
    scat = detectors.run(name, embed_lsb(cover, 0.5, "scat"))
    assert seq["details"]["mean_rate"] < 0.4
    assert seq["details"]["lower_bound"] is True and "lower bound" in seq["summary"]
    assert scat["details"]["mean_rate"] == pytest.approx(0.5, abs=0.05)
    assert scat["details"]["lower_bound"] is False
