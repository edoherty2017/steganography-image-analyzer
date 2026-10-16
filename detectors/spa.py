"""Sample pair analysis (SPA) of LSB embedding.

Method: S. Dumitrescu, X. Wu and Z. Wang, "Detection of LSB steganography via
sample pair analysis", IEEE Transactions on Signal Processing 51(7), 2003.

For every pair of horizontally adjacent samples ``(u, v)`` in one channel:

* ``Z`` (= D_0): pairs with ``u == v``.
* ``C_0``: pairs that agree in their upper 7 bits (``u >> 1 == v >> 1``).
* ``X``: pairs whose difference is odd and whose larger value is even.
* ``Y``: pairs whose difference is odd and whose larger value is odd.

In a natural image ``|X|`` and ``|Y|`` are statistically equal, because which
of two neighbours is odd has nothing to do with which one is larger. LSB
replacement breaks that symmetry in a predictable way: if a fraction ``p`` of
the samples carries message bits (each one then flips with probability 1/2),
the expected stego counts satisfy, for every ``m``, the quadratic of
Dumitrescu et al. (their equations for ``C_m``). Summing those equations over
all ``m`` telescopes to::

    (|C_0| / 2) p^2 - (|Z| + |Y| - |X|) p + (|Y| - |X|) = 0

where every count is measured on the image under test. The estimate is the
root closer to 0, clipped to [0, 1]. Degenerate cases: if ``|C_0| == 0`` the
equation is linear; if the discriminant is negative (happens from sampling
noise when nearly every sample is embedded) the vertex ``-b / 2a`` is used.

The estimate is the fraction of samples that carry message bits (1.0 means
every LSB was replaced with a payload bit), not the fraction that changed.
It is accurate when the payload bits are random-looking and spread over the
image at random positions, which is what the model assumes.

Sequential payloads. When the payload fills the first rows (this project's
default without a password) the model does not hold: the two samples of
almost every pair are either both embedded or both untouched, and the
pooled quadratic reads low. The bias is large on smooth covers. On
``test-images/smoothBlue.png`` (480x270) payloads filling 25%, 50% and 90%
of capacity read 0.14, 0.31 and 0.69 (mean over channels) when written
sequentially, against about 0.25, 0.50 and 0.90 when scattered. On the
noisier 256x256 synthetic test covers a sequential 0.50 reads 0.44 to 0.49.
The detector therefore also estimates the top and bottom halves of the image
(``_rates.row_halves``). When the top half reads much higher it reports a
sequential pattern and labels the estimate a lower bound (``details
["lower_bound"]``). This label does not change the score. The check is a
heuristic that needs a large gap between the halves. It misses short
payloads (10% of capacity on smoothBlue.png), and it can never fire at
64x64, where a half is too small for any gap to count.

Score (heuristic, applied to the largest per-channel estimate), for images
of 256x256 or more:
    estimate <= 0.05 ->  score 0.0
    estimate 0.12    ->  score 0.35         (suspicious from here)
    estimate 0.20    ->  score 0.65         (likely_stego from here)
    estimate >= 0.40 ->  score 1.0
For a smaller image every knot is multiplied by ``sqrt(REF_PAIRS / pairs)``
(``_rates.knot_scale``: x2 at 128x128, x4 at 64x64), because the spread of
the estimate on a clean cover grows as ``1 / sqrt(pairs)``.

Calibration (heuristic; synthetic covers only). There were 20 seeds of each of
five cover families from the test suite: wide gradients plus noise (conftest
``synthetic_rgb``), smooth natural-looking scenes, a low-noise variant, a
contrast-stretched variant, and noiseless low-contrast content. On these, the
standard deviation of the clean per-channel estimate was about
``c / sqrt(pairs)``. ``c`` was 5 to 8.5 for the wide-gradient covers, which are
the noisiest, and at most 5 for the others. The largest clean per-channel
estimate was 0.09 at 256x256, 0.21 at 128x128 and 0.41 at 64x64. With
unscaled knots, 14 of the 20 wide-gradient covers at 64x64 read suspicious or
likely_stego. With the scaled knots, none of the clean covers of any family
reads above clean at 64, 96, 128, 192 or 256 pixels square. The cost is
sensitivity on small images: at 64x64 a scattered payload must cover about
half of the samples to read suspicious, and about 80% to read likely_stego.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import register
from ._common import as_channels, channel_labels, interp_score, make_result, solve_quadratic_small_root
from ._rates import REFERENCE_SIDE, knot_scale, rate_summary, row_halves, scale_knots

NAME = "sample_pair_analysis"

# Estimated embedding rate -> score at 256x256 and larger (heuristic
# thresholds, see docstring); scaled up for smaller images.
SCORE_KNOTS = ((0.05, 0.0), (0.12, 0.35), (0.20, 0.65), (0.40, 1.0))
# Rate at which the unscaled score reaches "likely_stego".
LIKELY_RATE = 0.20
# Horizontal pairs per channel of a 256x256 image, where SCORE_KNOTS hold.
REF_PAIRS = REFERENCE_SIDE * (REFERENCE_SIDE - 1)
# Below this many horizontal pairs per channel (about a 64x64 image, which
# has 64 * 63 = 4032) the knot scale passes 4, so the "likely_stego" knot
# would sit above an estimate of 0.8 and the result says inconclusive
# instead.
MIN_PAIRS = 4000


def spa_counts(channel: np.ndarray) -> dict[str, int]:
    """Count the SPA pair classes for one channel (horizontal neighbours).

    Args:
        channel: ``(H, W)`` array of 8-bit samples.

    Returns:
        Dict with ``pairs`` (total), ``Z``, ``C0``, ``X`` and ``Y`` counts.
    """
    ch = np.asarray(channel, dtype=np.int16)
    u = ch[:, :-1].ravel()
    v = ch[:, 1:].ravel()
    diff = v - u
    odd = (diff & 1).astype(bool)
    larger_odd = (np.maximum(u, v) & 1).astype(bool)
    return {
        "pairs": int(u.size),
        "Z": int(np.count_nonzero(diff == 0)),
        "C0": int(np.count_nonzero((u >> 1) == (v >> 1))),
        "X": int(np.count_nonzero(odd & ~larger_odd)),
        "Y": int(np.count_nonzero(odd & larger_odd)),
    }


def spa_estimate(channel: np.ndarray) -> dict[str, Any]:
    """Estimate the LSB embedding rate of one channel with SPA.

    Args:
        channel: ``(H, W)`` array of 8-bit samples (at least 2 columns).

    Returns:
        Dict with ``rate`` (clipped estimate in [0, 1], NaN if undefined),
        ``raw_root`` (unclipped root), ``case`` (how the quadratic was solved)
        and the pair counts.
    """
    counts = spa_counts(channel)
    if counts["pairs"] == 0:
        return {"rate": float("nan"), "raw_root": float("nan"), "case": "no_pairs", **counts}
    a = counts["C0"] / 2.0
    b = -(counts["Z"] + counts["Y"] - counts["X"])
    c = counts["Y"] - counts["X"]
    root, case = solve_quadratic_small_root(a, b, c)
    rate = float(np.clip(root, 0.0, 1.0)) if np.isfinite(root) else float("nan")
    return {"rate": rate, "raw_root": float(root), "case": case, **counts}


@register(NAME, "Sample pair analysis (Dumitrescu, Wu & Wang 2003): estimates the "
                "fraction of samples carrying LSB payload per channel.")
def sample_pair_analysis(arr: np.ndarray,
                         channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Run SPA on every channel and combine into one result.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details`` holds the
        per-channel estimates, their mean and max, the pair counts, the
        size-scaled score knots (``score_knots``, with ``knot_scale`` and
        the unscaled ``base_score_knots``), the top/bottom half estimates
        (``row_halves``) and ``lower_bound`` (True when the payload looks
        sequential, so the estimate is likely low).
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    n_pairs = a.shape[0] * max(a.shape[1] - 1, 0)
    if n_pairs < MIN_PAIRS:
        return make_result(NAME, 0.0,
                           f"Inconclusive: {n_pairs} pixel pairs per channel is too few for sample "
                           f"pair analysis (needs {MIN_PAIRS}, about a 64x64 image).",
                           {"inconclusive": True, "pairs": n_pairs, "min_pairs": MIN_PAIRS})
    per_channel = {}
    for i, lab in enumerate(labels):
        per_channel[lab] = spa_estimate(a[:, :, i])
    rates = [r["rate"] for r in per_channel.values() if np.isfinite(r["rate"])]
    if not rates:
        return make_result(NAME, 0.0, "Sample pair analysis was undefined for every channel.",
                           {"inconclusive": True, "channels": per_channel})
    max_rate = max(rates)
    mean_rate = float(np.mean(rates))
    top = max(per_channel, key=lambda k: per_channel[k]["rate"] if np.isfinite(per_channel[k]["rate"]) else -1)
    factor = knot_scale(n_pairs, REF_PAIRS)
    knots = scale_knots(SCORE_KNOTS, factor)
    score = interp_score(max_rate, knots)
    halves = row_halves(a, lambda ch: spa_estimate(ch)["rate"],
                        lambda rows, width: rows * max(width - 1, 0), REF_PAIRS, LIKELY_RATE)
    lower_bound = bool(halves is not None and halves["sequential"])
    summary = rate_summary(max_rate, top, mean_rate, factor, n_pairs, "pairs", halves)
    details = {
        "estimated_rate": max_rate,
        "mean_rate": mean_rate,
        "max_channel": top,
        "per_channel_rate": {k: v["rate"] for k, v in per_channel.items()},
        "channels": per_channel,
        "pairs": "horizontal neighbours",
        "pairs_per_channel": n_pairs,
        "knot_scale": factor,
        "score_knots": knots,
        "base_score_knots": SCORE_KNOTS,
        "row_halves": halves,
        "lower_bound": lower_bound,
    }
    return make_result(NAME, score, summary, details)
