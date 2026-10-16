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
Natural covers give estimates scattered around 0 with a spread of a few
hundredths; the estimate is accurate when the payload bits are random-looking
and spread over the image, and still tracks the overall fraction for a
sequential (top-of-image) embedding.

Score (heuristic, applied to the largest per-channel estimate):
    estimate <= 0.05 ->  score 0.0
    estimate 0.12    ->  score 0.35         (suspicious from here)
    estimate 0.20    ->  score 0.65         (likely_stego from here)
    estimate >= 0.40 ->  score 1.0
On 120 clean synthetic covers (gradients, smooth scenes, stretched and noisy
variants) the largest per-channel estimate was at most 0.10; covers with
strong noise on wide gradients give the largest clean estimates.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import register
from ._common import as_channels, channel_labels, interp_score, make_result, solve_quadratic_small_root

NAME = "sample_pair_analysis"

# Estimated embedding rate -> score (heuristic thresholds, see docstring).
SCORE_KNOTS = ((0.05, 0.0), (0.12, 0.35), (0.20, 0.65), (0.40, 1.0))
# Below this many horizontal pairs per channel the estimate is too noisy to
# report (about a 32x32 image).
MIN_PAIRS = 1000


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
        per-channel estimates, their mean and max, and the pair counts.
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    n_pairs = a.shape[0] * max(a.shape[1] - 1, 0)
    if n_pairs < MIN_PAIRS:
        return make_result(NAME, 0.0,
                           f"Inconclusive: {n_pairs} pixel pairs per channel is too few for sample "
                           f"pair analysis (needs {MIN_PAIRS}).",
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
    score = interp_score(max_rate, SCORE_KNOTS)
    summary = (f"Estimated LSB embedding rate {max_rate:.1%} (channel {top}); "
               f"mean over channels {mean_rate:.1%}.")
    details = {
        "estimated_rate": max_rate,
        "mean_rate": mean_rate,
        "max_channel": top,
        "per_channel_rate": {k: v["rate"] for k, v in per_channel.items()},
        "channels": per_channel,
        "pairs": "horizontal neighbours",
        "score_knots": SCORE_KNOTS,
    }
    return make_result(NAME, score, summary, details)
