"""RS (Regular/Singular groups) analysis of LSB embedding.

Method: J. Fridrich, M. Goljan and R. Du, "Reliable detection of LSB
steganography in color and grayscale images", Proc. ACM Workshop on
Multimedia and Security, 2001.

Each channel is cut into non-overlapping groups of 4 horizontally adjacent
samples ``G = (x1, x2, x3, x4)``. The discrimination function
``f(G) = sum |x_{i+1} - x_i|`` measures how noisy a group is. Two flipping
functions act on single values:

* ``F1``: 0 <-> 1, 2 <-> 3, ... (flip the LSB, ``x ^ 1``).
* ``F-1``: -1 <-> 0, 1 <-> 2, 3 <-> 4, ... (``((x + 1) ^ 1) - 1``).

With the mask ``M = [0, 1, 1, 0]`` (apply ``F1`` to the middle two samples)
a group is *regular* if flipping makes it noisier (``f(F_M(G)) > f(G)``),
*singular* if flipping makes it smoother, and unusable otherwise. ``R_M`` and
``S_M`` are the fractions of regular and singular groups; ``R_-M`` and
``S_-M`` use ``F-1`` with the negated mask.

In a natural image ``R_M ~ R_-M`` and ``S_M ~ S_-M``. LSB replacement pushes
``R_M`` and ``S_M`` together while ``R_-M`` and ``S_-M`` drift apart. Measuring
the four fractions on the image (``d0``, ``d-0``) and again on the same image
with every LSB flipped (``d1``, ``d-1``), where ``d = R - S``, Fridrich et
al. derive::

    2 (d1 + d0) x^2 + (d-0 - d-1 - d1 - 3 d0) x + d0 - d-0 = 0
    p = x / (x - 1/2)

using the root ``x`` with the smaller absolute value. ``p`` is the fraction
of samples that carry message bits (1.0 = every LSB replaced). Because
``x`` diverges as the embedding approaches 100%, this module solves the
equivalent quadratic in ``p`` instead (see ``_solve_rate``), which stays
well conditioned there; ``p`` is clipped to [0, 1], and a negative
discriminant (sampling noise near full embedding) uses the vertex.

``F-1`` maps 0 to -1 and 255 to 256; the groups are evaluated on signed
integers so those out-of-range values only enter ``f`` and never wrap.

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
from ._common import as_channels, channel_labels, interp_score, make_result

NAME = "rs_analysis"
MASK = (0, 1, 1, 0)

# Estimated embedding rate -> score (heuristic thresholds, see docstring).
SCORE_KNOTS = ((0.05, 0.0), (0.12, 0.35), (0.20, 0.65), (0.40, 1.0))
# Below this many 4-pixel groups per channel the estimate is too noisy to
# report (about a 32x32 image).
MIN_GROUPS = 250


def _flip_pos(x: np.ndarray) -> np.ndarray:
    """F1: swap 2k <-> 2k+1."""
    return x ^ 1


def _flip_neg(x: np.ndarray) -> np.ndarray:
    """F-1: swap 2k-1 <-> 2k (so 0 -> -1 and 255 -> 256)."""
    return ((x + 1) ^ 1) - 1


def _discrimination(groups: np.ndarray) -> np.ndarray:
    """f(G) = sum of absolute differences of neighbours within each group."""
    return np.abs(np.diff(groups, axis=-1)).sum(axis=-1)


def rs_fractions(groups: np.ndarray, mask: tuple[int, ...] = MASK) -> dict[str, float]:
    """Regular and singular group fractions for the mask and its negation.

    Args:
        groups: ``(N, n)`` signed integer array of pixel groups.
        mask: 0/1 mask of length ``n`` (positions to flip).

    Returns:
        Dict with ``R_M``, ``S_M``, ``R_negM`` and ``S_negM`` (fractions).
    """
    m = np.asarray(mask, dtype=bool)
    f0 = _discrimination(groups)
    pos = groups.copy()
    pos[:, m] = _flip_pos(pos[:, m])
    neg = groups.copy()
    neg[:, m] = _flip_neg(neg[:, m])
    fp = _discrimination(pos)
    fn = _discrimination(neg)
    n = max(groups.shape[0], 1)
    return {
        "R_M": float(np.count_nonzero(fp > f0)) / n,
        "S_M": float(np.count_nonzero(fp < f0)) / n,
        "R_negM": float(np.count_nonzero(fn > f0)) / n,
        "S_negM": float(np.count_nonzero(fn < f0)) / n,
    }


def rs_estimate(channel: np.ndarray, mask: tuple[int, ...] = MASK) -> dict[str, Any]:
    """Estimate the LSB embedding rate of one channel with RS analysis.

    Args:
        channel: ``(H, W)`` array of 8-bit samples (at least ``len(mask)``
            columns).
        mask: Flipping mask; the standard ``[0, 1, 1, 0]`` by default.

    Returns:
        Dict with ``rate`` (clipped estimate in [0, 1], NaN if undefined),
        ``raw_rate`` (before clipping), ``root`` (the ``x`` of the
        quadratic), ``case`` and the RS fractions for the image (``stego``)
        and for its LSB-flipped copy (``flipped``).
    """
    n = len(mask)
    ch = np.asarray(channel, dtype=np.int32)
    h, w = ch.shape
    usable = (w // n) * n
    if usable == 0 or h == 0:
        return {"rate": float("nan"), "raw_rate": float("nan"), "root": float("nan"),
                "case": "no_groups", "groups": 0}
    groups = ch[:, :usable].reshape(-1, n)
    stego = rs_fractions(groups, mask)
    flipped = rs_fractions(_flip_pos(groups), mask)
    d0 = stego["R_M"] - stego["S_M"]
    dn0 = stego["R_negM"] - stego["S_negM"]
    d1 = flipped["R_M"] - flipped["S_M"]
    dn1 = flipped["R_negM"] - flipped["S_negM"]
    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    raw, x, case = _solve_rate(a, b, c)
    rate = float(np.clip(raw, 0.0, 1.0)) if np.isfinite(raw) else float("nan")
    return {"rate": rate, "raw_rate": raw, "root": x, "case": case,
            "groups": int(groups.shape[0]), "stego": stego, "flipped": flipped,
            "d": {"d0": d0, "d1": d1, "d_neg0": dn0, "d_neg1": dn1}}


def _solve_rate(a: float, b: float, c: float) -> tuple[float, float, str]:
    """Solve the RS quadratic ``a x^2 + b x + c = 0`` for the rate ``p``.

    Substituting ``x = p / (2 (p - 1))`` (the inverse of ``p = x / (x - 1/2)``)
    and multiplying by ``4 (p - 1)^2`` gives an equivalent quadratic in ``p``::

        (a + 2b + 4c) p^2 - (2b + 8c) p + 4c = 0

    Its roots are the images of the roots in ``x``, so picking the root whose
    ``x`` has the smaller magnitude is the rule of Fridrich et al. unchanged.
    Solving in ``p`` matters near full embedding: there ``a`` and ``b`` shrink
    to sampling noise, the roots in ``x`` run off to infinity (often as a
    complex pair) and the ``x``-space vertex is meaningless, while in ``p``
    the equation tends to ``4c (p - 1)^2 = 0`` and its vertex sits near 1.

    Returns:
        ``(p, x, case)``: the unclipped rate, the matching ``x`` (inf when
        ``p == 1``) and ``"two_roots"``, ``"linear"``, ``"no_real_root"``
        (vertex used) or ``"degenerate"``.
    """
    qa = a + 2.0 * b + 4.0 * c
    qb = -(2.0 * b + 8.0 * c)
    qc = 4.0 * c
    scale = max(abs(qa), abs(qb), abs(qc), 1e-300)
    if abs(qa) <= 1e-12 * scale:
        if abs(qb) <= 1e-12 * scale:
            return float("nan"), float("nan"), "degenerate"
        p, case = -qc / qb, "linear"
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0:
            p, case = -qb / (2.0 * qa), "no_real_root"
        else:
            sq = np.sqrt(disc)
            roots = ((-qb + sq) / (2.0 * qa), (-qb - sq) / (2.0 * qa))
            p = min(roots, key=lambda r: abs(_x_of(r)))
            case = "two_roots"
    return float(p), _x_of(p), case


def _x_of(p: float) -> float:
    """The RS variable ``x`` for a rate ``p`` (inverse of ``p = x/(x - 1/2)``)."""
    return float("inf") if abs(p - 1.0) < 1e-12 else p / (2.0 * (p - 1.0))


@register(NAME, "RS analysis (Fridrich, Goljan & Du 2001), mask [0,1,1,0] on groups of 4 "
                "horizontal pixels: estimates the fraction of samples carrying LSB payload.")
def rs_analysis(arr: np.ndarray,
                channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Run RS analysis on every channel and combine into one result.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details`` holds the
        per-channel estimates, their mean and max, and the R/S fractions.
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    n_groups = a.shape[0] * (a.shape[1] // len(MASK))
    if n_groups < MIN_GROUPS:
        return make_result(NAME, 0.0,
                           f"Inconclusive: {n_groups} pixel groups per channel is too few for RS "
                           f"analysis (needs {MIN_GROUPS}).",
                           {"inconclusive": True, "groups": n_groups, "min_groups": MIN_GROUPS})
    per_channel = {lab: rs_estimate(a[:, :, i]) for i, lab in enumerate(labels)}
    rates = [r["rate"] for r in per_channel.values() if np.isfinite(r["rate"])]
    if not rates:
        return make_result(NAME, 0.0, "RS analysis was undefined for every channel.",
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
        "mask": list(MASK),
        "score_knots": SCORE_KNOTS,
    }
    return make_result(NAME, score, summary, details)
