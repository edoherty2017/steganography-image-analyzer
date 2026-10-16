"""Chi-square attack on pairs of values (sequential LSB embedding).

Method: A. Westfeld and A. Pfitzmann, "Attacks on steganographic systems",
Information Hiding (IH'99), LNCS 1768, 1999.

LSB replacement only moves a sample between the two values of a pair
``(2k, 2k+1)``. When the payload bits are close to 50/50, the embedded
samples split evenly between the two values, so in the embedded region the
counts ``h[2k]`` and ``h[2k+1]`` become nearly equal, while a natural image
usually has visible pair imbalances. The attack measures how well the pair
counts fit "equal within each pair"::

    expected_k = (h[2k] + h[2k+1]) / 2
    chi2       = sum_k (h[2k] - expected_k)^2 / expected_k
    p          = P(Chi2_{K-1} > chi2)          # scipy.stats.chi2.sf

As in the original paper only the even value of each pair enters the sum
and the degrees of freedom are (number of categories - 1). That statistic
is about half of the two-cell Pearson statistic, so for fully equalized
pairs ``p`` sits near 1 instead of being uniform; ``p`` near 1 means "looks
embedded", ``p`` near 0 means "pair imbalances like a natural image". Pairs
whose expected count is below 5 are merged with the following pairs until
the merged category reaches 5 (the usual rule for a chi-square test); any
remainder joins the last category.

The test is run on growing prefixes of the sample stream in embedding order
(row-major pixels, channels interleaved R, G, B, as written by this
project's encoder) and of each channel alone. A sequential payload shows up
as ``p`` near 1 for every prefix inside the payload and a drop to near 0
once the prefix runs past it, so the length of the leading run of prefixes
with ``p > 0.5`` estimates the embedded fraction.

Control pairs (an addition to the original attack, heuristic): a cover whose
histogram is smooth at the scale of single values (wide gradients, strong
noise) already has ``h[2k] ~ h[2k+1]`` and gives a high ``p`` with no
payload at all, a known source of false alarms for this attack. The same
test is therefore also run on the shifted pairs ``(2k+1, 2k+2)``, which LSB
replacement does not equalize. A prefix only counts as embedded when its
LSB pairs look equalized *and* its control pairs are clearly unbalanced
(``p_control < CONTROL_ALPHA``). On a cover that is smooth everywhere the
detector reports "inconclusive" with score 0 rather than a false alarm.

Limits: the attack needs the payload to be packed at the start of the
stream. A payload scattered over the image with a key barely changes any
prefix until nearly every sample is used. Biased payloads (plain text is
about 45% one-bits) equalize pairs less and weaken the signal. On covers
with smooth histograms even full embedding can go undetected (see above);
the SPA and RS estimators do not share that weakness.

Score (heuristic, from the estimated sequential fraction f of the
strongest stream):
    f = 0          ->  0.0   (clean)
    f = 5%         ->  0.25  (one prefix only, can be chance)
    f = 10%        ->  0.45  (suspicious)
    f >= 20%       ->  0.70+ (likely_stego), 1.0 at f = 100%
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2 as chi2_dist

from . import register
from ._common import as_channels, channel_labels, interp_score, make_result

NAME = "chi_square"

# Prefix fractions of the sample stream at which the test is evaluated.
PREFIX_FRACTIONS = tuple(float(f) for f in np.round(np.linspace(0.05, 1.0, 20), 4))
# p above this counts as "pairs equalized": a midpoint between the p ~ 1 of
# an embedded region and the p ~ 0 of natural pair imbalances (heuristic).
P_THRESHOLD = 0.5
# A prefix only counts as "embedded" when the shifted control pairs reject
# equality at this level, i.e. the histogram has pair-scale structure that
# LSB replacement would have had to remove (heuristic guard, see docstring).
CONTROL_ALPHA = 0.01
# Minimum expected count per category (standard chi-square rule of thumb).
MIN_EXPECTED = 5.0
# Estimated sequential fraction -> score (heuristic thresholds, see docstring).
SCORE_KNOTS = ((0.0, 0.0), (0.05, 0.25), (0.10, 0.45), (0.20, 0.70), (0.50, 0.90), (1.0, 1.0))


def pair_chi_square(hist: np.ndarray, offset: int = 0) -> dict[str, Any]:
    """Westfeld-Pfitzmann chi-square statistic for one 256-bin histogram.

    Args:
        hist: Length-256 array of value counts.
        offset: 0 tests the LSB pairs ``(2k, 2k+1)``. 1 tests the shifted
            control pairs ``(2k+1, 2k+2)``, which LSB replacement never
            equalizes; a high control ``p`` means the histogram is too smooth
            for the attack to say anything.

    Returns:
        Dict with ``chi2`` (statistic), ``dof``, ``p`` (``chi2.sf``; NaN when
        fewer than two categories survive merging) and ``categories``.
    """
    h = np.asarray(hist, dtype=np.float64)
    if offset:
        h = h[1:255]
    even = h[0::2]
    expected = (h[0::2] + h[1::2]) / 2.0
    obs_cats: list[float] = []
    exp_cats: list[float] = []
    acc_o = acc_e = 0.0
    for o, e in zip(even, expected):
        acc_o += o
        acc_e += e
        if acc_e >= MIN_EXPECTED:
            obs_cats.append(acc_o)
            exp_cats.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        if obs_cats:
            obs_cats[-1] += acc_o
            exp_cats[-1] += acc_e
        else:
            obs_cats.append(acc_o)
            exp_cats.append(acc_e)
    k = len(obs_cats)
    if k < 2:
        return {"chi2": float("nan"), "dof": 0, "p": float("nan"), "categories": k}
    o = np.asarray(obs_cats)
    e = np.asarray(exp_cats)
    stat = float(np.sum((o - e) ** 2 / e))
    dof = k - 1
    return {"chi2": stat, "dof": dof, "p": float(chi2_dist.sf(stat, dof)), "categories": k}


def prefix_curve(stream: np.ndarray,
                 fractions: Iterable[float] = PREFIX_FRACTIONS) -> list[dict[str, Any]]:
    """Chi-square p-values on growing prefixes of a sample stream.

    Each sample is counted once: histograms of consecutive segments are
    accumulated, so the cost is one pass over the stream.

    Args:
        stream: 1-D uint8 array in embedding order.
        fractions: Increasing prefix fractions in (0, 1].

    Returns:
        One dict per fraction: ``fraction``, ``samples``, ``chi2``, ``dof``,
        ``p`` (NaN if undefined) and ``p_control`` (the same test on the
        shifted control pairs).
    """
    s = np.asarray(stream, dtype=np.uint8).ravel()
    n = s.size
    out = []
    hist = np.zeros(256, dtype=np.int64)
    start = 0
    for f in fractions:
        end = int(round(float(f) * n))
        if end > start:
            hist += np.bincount(s[start:end], minlength=256)
            start = end
        res = pair_chi_square(hist)
        ctl = pair_chi_square(hist, offset=1)
        out.append({"fraction": float(f), "samples": end, "chi2": res["chi2"],
                    "dof": res["dof"], "p": res["p"], "p_control": ctl["p"]})
    return out


def leading_fraction(curve: list[dict[str, Any]], threshold: float = P_THRESHOLD,
                     control_alpha: float = CONTROL_ALPHA) -> float:
    """Length of the leading run of prefixes that look embedded.

    A prefix counts when its LSB pairs look equalized (``p > threshold``)
    while its shifted control pairs are clearly unbalanced
    (``p_control < control_alpha``). The control condition is what keeps a
    cover with a smooth histogram, whose pairs are balanced with no payload
    at all, from counting as embedded. Undefined p-values end the run.

    Returns:
        The largest prefix fraction such that every prefix up to it counts,
        or 0.0 if the first prefix does not.
    """
    best = 0.0
    for pt in curve:
        p, pc = pt["p"], pt.get("p_control", 0.0)
        if p is None or not np.isfinite(p) or p <= threshold:
            break
        if pc is None or not np.isfinite(pc) or pc >= control_alpha:
            break
        best = pt["fraction"]
    return best


@register(NAME, "Chi-square attack on pairs of values (Westfeld & Pfitzmann 1999) over "
                "growing prefixes of the sample stream; detects sequential LSB embedding.")
def chi_square(arr: np.ndarray,
               channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Run the chi-square attack on the interleaved stream and on each channel.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details`` holds the
        p-value curve per stream, the estimated sequential fraction per
        stream and the stream that produced the score.
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    streams: dict[str, np.ndarray] = {}
    if a.shape[2] > 1:
        streams["".join(labels)] = a.reshape(-1)
    for i, lab in enumerate(labels):
        streams[lab] = a[:, :, i].reshape(-1)

    curves = {}
    fractions = {}
    for name, stream in streams.items():
        curve = prefix_curve(stream)
        curves[name] = curve
        fractions[name] = leading_fraction(curve)

    best = max(fractions, key=lambda k: fractions[k])
    f_hat = fractions[best]
    score = interp_score(f_hat, SCORE_KNOTS)
    primary = next(iter(streams))
    p_full = curves[primary][-1]["p"]
    p_control_full = curves[primary][-1]["p_control"]
    informative = {k: bool(c[-1]["p_control"] is not None and np.isfinite(c[-1]["p_control"])
                           and c[-1]["p_control"] < CONTROL_ALPHA) for k, c in curves.items()}
    # Inconclusive when the run stopped only because the control pairs were
    # balanced too (smooth histogram), not because the LSB pairs were unbalanced.
    first = curves[primary][0]
    inconclusive = bool(f_hat <= 0 and first["p"] is not None and np.isfinite(first["p"])
                        and first["p"] > P_THRESHOLD)
    if inconclusive:
        summary = ("Inconclusive: the histogram is so smooth that even the control pairs are "
                   "balanced, so the chi-square attack cannot separate payload from cover here.")
    elif f_hat <= 0:
        summary = ("Pairs of values are unbalanced from the first prefix on, as in a natural "
                   "image; no sequential LSB payload found.")
    elif f_hat >= 1.0:
        summary = (f"Pairs of values are equalized over the whole {best} stream "
                   f"(p > {P_THRESHOLD} at every prefix); consistent with LSB embedding in every sample.")
    else:
        summary = (f"Pairs of values are equalized over the first {f_hat:.0%} of the {best} stream, "
                   f"then unbalanced; consistent with a sequential payload of about that size.")
    details = {
        "estimated_sequential_fraction": f_hat,
        "stream": best,
        "per_stream_fraction": fractions,
        "p_full": p_full,
        "p_control_full": p_control_full,
        "informative": informative,
        "inconclusive": inconclusive,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "p_curve": {k: [pt["p"] for pt in c] for k, c in curves.items()},
        "p_control_curve": {k: [pt["p_control"] for pt in c] for k, c in curves.items()},
        "chi2_curve": {k: [pt["chi2"] for pt in c] for k, c in curves.items()},
        "dof_curve": {k: [pt["dof"] for pt in c] for k, c in curves.items()},
        "p_threshold": P_THRESHOLD,
        "control_alpha": CONTROL_ALPHA,
        "score_knots": SCORE_KNOTS,
    }
    return make_result(NAME, score, summary, details)
