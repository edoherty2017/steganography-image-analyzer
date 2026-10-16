"""Single-image even/odd histogram pair statistic.

This is the blind (no cover image needed) version of the pair statistic in
``pairwise_stats.py``. For one channel's 256-bin histogram ``h`` it looks at
the within-pair differences ``d_k = h[2k] - h[2k+1]`` (the pairs LSB
replacement moves samples inside) and, as a control, the shifted differences
``h[2k+1] - h[2k+2]`` (pairs LSB replacement never mixes).

Model (pairs-of-values argument of Westfeld & Pfitzmann 1999): if every LSB
of a channel is replaced by payload bits that are 1 with probability ``q``,
each pair's ``s_k = h[2k] + h[2k+1]`` samples split binomially, so
``d_k = (1 - 2q) s_k`` plus binomial noise of variance about ``s_k``. A random
payload (``q = 1/2``, encrypted or compressed data) flattens the pairs
(``d_k ~ 0``); a text payload (``q ~ 0.45``) makes every pair lean the same
way, the "comb" seen in this project's committed results. Either way the
within-pair differences become *proportional to the pair totals*, which a
natural image's differences are not.

Statistic (heuristic, built for this tool): fit ``d_k = beta * s_k`` by
weighted least squares (``beta = sum d / sum s``) and measure what is left::

    resid_in  = sum_k (d_k - beta s_k)^2 / s_k  -  (K - 1)      # noise-corrected
    resid_out = the same for the shifted control pairs

    ratio = sqrt(resid_in / resid_out)

A natural image has comparable structure in both, so ``ratio`` is near 1.
Embedding removes the within-pair structure, so ``ratio`` falls toward 0 as
the embedded fraction grows. It is a detection statistic, not a calibrated
rate estimate: how fast it falls depends on the cover's histogram (it stays
near 1 up to about half capacity on some covers and drops to ~0 at full
capacity on all tested ones); use SPA or RS for the rate. Only pairs with
``s_k >= 10`` are used. When the control residual is not clearly above
its noise level (the histogram is smooth at the scale of single values),
the channel is marked inconclusive.

``beta`` also hints at the payload type once the ratio is low: ``beta ~ 0``
means balanced (random-looking) bits, ``beta`` clearly above 0 means more
0-bits than 1-bits, as in ASCII text (estimated one-bit fraction
``(1 - beta) / 2``).

Score (heuristic): the larger of two piecewise-linear maps, one of the
ratio pooled over all conclusive channels and a stricter one of the single
lowest channel ratio (so a payload in one channel is not diluted):

    pooled ratio:   >= 0.70 -> 0.0, 0.50 -> 0.35, 0.30 -> 0.65, <= 0.20 -> 1.0
    lowest channel: >= 0.55 -> 0.0, 0.38 -> 0.35, 0.22 -> 0.65, <= 0.10 -> 1.0

On 96 conclusive clean synthetic covers the pooled ratio was at least 0.56
and the lowest channel ratio at least 0.48 (scores <= 0.25); on 100 fully
embedded ones they were at most 0.25 and 0.22.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import register
from ._common import as_channels, channel_labels, interp_score, make_result

NAME = "histogram_pairs"

MIN_PAIR_COUNT = 10
# Control residual must exceed this many standard errors of pure noise.
CONTROL_SIGMAS = 4.0
# Imbalance ratio -> score (decreasing; heuristic thresholds, see docstring).
POOLED_KNOTS = ((0.20, 1.0), (0.30, 0.65), (0.50, 0.35), (0.70, 0.0))
CHANNEL_KNOTS = ((0.10, 1.0), (0.22, 0.65), (0.38, 0.35), (0.55, 0.0))


def legacy_pair_stats(h: np.ndarray) -> dict[str, Any]:
    """The L1/L2/median/max summary of ``d_k`` used by ``pairwise_stats.py``."""
    d = h[0::2].astype(np.int64) - h[1::2].astype(np.int64)
    a = np.abs(d)
    return {"L1": int(a.sum()), "L2": float(np.sqrt((d.astype(np.float64) ** 2).sum())),
            "median_abs": float(np.median(a)), "max_abs": int(a.max()), "max_k": int(a.argmax())}


def _residual(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Noise-corrected residual of ``d = first - second`` after fitting ``d = beta * s``."""
    s = (first + second).astype(np.float64)
    d = (first - second).astype(np.float64)
    keep = s >= MIN_PAIR_COUNT
    s, d = s[keep], d[keep]
    k = int(s.size)
    if k < 3:
        return {"pairs": k, "beta": float("nan"), "resid": float("nan"), "noise_se": float("nan")}
    beta = float(d.sum() / s.sum())
    chi = float(np.sum((d - beta * s) ** 2 / s))
    return {"pairs": k, "beta": beta, "resid": chi - (k - 1), "noise_se": float(np.sqrt(2.0 * (k - 1)))}


def channel_pair_stats(channel: np.ndarray) -> dict[str, Any]:
    """Pair imbalance statistics for one channel.

    Args:
        channel: Array of 8-bit samples (any shape).

    Returns:
        Dict with ``ratio`` (NaN when inconclusive), ``conclusive``, ``beta``
        (within-pair parity lean), the within and control residuals and the
        legacy L1/L2/median/max summary.
    """
    h = np.bincount(np.asarray(channel, dtype=np.uint8).ravel(), minlength=256).astype(np.int64)
    inner = _residual(h[0::2], h[1::2])
    control = _residual(h[1:255:2], h[2:256:2])
    conclusive = bool(np.isfinite(control["resid"]) and np.isfinite(inner["resid"])
                      and control["resid"] > CONTROL_SIGMAS * control["noise_se"])
    if conclusive:
        ratio = float(np.sqrt(max(inner["resid"], 0.0) / control["resid"]))
    else:
        ratio = float("nan")
    return {
        "ratio": ratio,
        "conclusive": conclusive,
        "beta": inner["beta"],
        "inner": inner,
        "control": control,
        "legacy": legacy_pair_stats(h),
    }


@register(NAME, "Single-image even/odd histogram pair imbalance against shifted control "
                "pairs; detects flattened pairs (random payload) and the text-payload comb.")
def histogram_pairs(arr: np.ndarray,
                    channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Run the pair imbalance statistic per channel and pooled.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details`` holds the
        per-channel ratio, parity lean ``beta``, residuals and the legacy
        L1/L2/median/max pair summary.
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    per_channel = {lab: channel_pair_stats(a[:, :, i]) for i, lab in enumerate(labels)}
    conclusive = {k: v for k, v in per_channel.items() if v["conclusive"]}
    if not conclusive:
        return make_result(
            NAME, 0.0,
            "Inconclusive: the histogram is too smooth at the scale of single values for the "
            "pair statistic to separate payload from cover.",
            {"inconclusive": True, "channels": per_channel,
             "pooled_knots": POOLED_KNOTS, "channel_knots": CHANNEL_KNOTS})

    pooled_in = sum(v["inner"]["resid"] for v in conclusive.values())
    pooled_out = sum(v["control"]["resid"] for v in conclusive.values())
    pooled_ratio = float(np.sqrt(max(pooled_in, 0.0) / pooled_out))
    worst = min(conclusive, key=lambda k: conclusive[k]["ratio"])
    pooled_score = interp_score(pooled_ratio, POOLED_KNOTS)
    channel_score = interp_score(conclusive[worst]["ratio"], CHANNEL_KNOTS)
    score = max(pooled_score, channel_score)
    ratio = pooled_ratio if pooled_score >= channel_score else conclusive[worst]["ratio"]
    beta = conclusive[worst]["beta"]
    ones = (1.0 - beta) / 2.0
    if score < 0.35:
        summary = (f"Even/odd pairs keep their natural imbalance (ratio {ratio:.2f} against "
                   f"control pairs); no sign of LSB replacement.")
    elif abs(beta) < 0.03:
        summary = (f"Even/odd pairs are flattened (ratio {ratio:.2f}, channel {worst}); consistent "
                   f"with LSB replacement by random-looking (encrypted or compressed) bits.")
    else:
        summary = (f"Even/odd pairs lean one way in proportion to their size (ratio {ratio:.2f}, "
                   f"channel {worst}, about {ones:.0%} one-bits); the comb a text payload leaves.")
    details = {
        "ratio": ratio,
        "pooled_ratio": pooled_ratio,
        "worst_channel": worst,
        "beta": beta,
        "estimated_one_bit_fraction": ones,
        "per_channel_ratio": {k: v["ratio"] for k, v in per_channel.items()},
        "channels": per_channel,
        "min_pair_count": MIN_PAIR_COUNT,
        "pooled_knots": POOLED_KNOTS,
        "channel_knots": CHANNEL_KNOTS,
    }
    return make_result(NAME, score, summary, details)
