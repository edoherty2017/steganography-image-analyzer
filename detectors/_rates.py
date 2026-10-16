"""Helpers shared by the two rate estimators, SPA and RS (not a detector).

``spa.py`` (Dumitrescu, Wu & Wang 2003) and ``rs.py`` (Fridrich, Goljan &
Du 2001) both estimate, per channel, the fraction of samples that carry
LSB payload. They have the same two practical limits, handled here so the
two detectors treat them the same way.

Image size. The spread of the estimate on a clean cover shrinks roughly as
``1 / sqrt(n)``, where ``n`` is the number of units the estimator counts
per channel (horizontal pixel pairs for SPA, 4-pixel groups for RS). The
score knots of both detectors were calibrated at 256x256, so for a smaller
image ``knot_scale`` widens them by ``sqrt(n_ref / n)``, where ``n_ref`` is
the unit count of a 256x256 channel. The knots then sit at about the same
multiple of the noise level at every size. Images at least as large as
256x256 keep the unscaled knots. This is a heuristic correction calibrated
on synthetic covers (see the module docstrings of ``spa`` and ``rs`` for
the measurements), not a formal significance test.

Sequential payloads. Both estimators assume the embedded samples are spread
at random over the image. A sequential payload (this project's default when
no password is given: the first rows carry it, the rest is untouched) breaks
that assumption, and the whole-image estimate then reads low, by a lot on
smooth covers. ``row_halves`` estimates the top and the bottom half of the
image separately; a top half that reads much higher than the bottom half is
the signature of a payload written from the top, and the detector then
labels its estimate as a lower bound.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

# Side of the square image at which the score knots were calibrated.
REFERENCE_SIDE = 256


def knot_scale(n_units: int, n_ref: int) -> float:
    """Factor by which a rate estimator's score knots are widened.

    Args:
        n_units: Pairs (SPA) or groups (RS) per channel in the image.
        n_ref: The same count for a ``REFERENCE_SIDE`` square channel.

    Returns:
        ``max(1, sqrt(n_ref / n_units))``; ``inf`` when ``n_units`` is 0.
    """
    if n_units <= 0:
        return math.inf
    return max(1.0, math.sqrt(n_ref / n_units))


def scale_knots(knots: Sequence[tuple[float, float]], factor: float) -> tuple[tuple[float, float], ...]:
    """Multiply the statistic coordinate of every ``(statistic, score)`` knot."""
    return tuple((float(x) * factor, float(y)) for x, y in knots)


def row_halves(arr: np.ndarray, channel_rate: Callable[[np.ndarray], float],
               units: Callable[[int, int], int], n_ref: int,
               likely_rate: float) -> dict[str, object] | None:
    """Estimate the top and bottom halves of the image separately.

    The gap ``top - bottom`` (each the mean of the clipped per-channel
    estimates over that half) is compared with the "likely" knot of the
    rate estimator, scaled for the size of one half with ``knot_scale``:
    a gap that large would be a "likely_stego" reading for a half-size image
    on its own. A sequential payload that fills part of the image produces
    such a gap; a scattered payload or a clean cover does not, since both
    halves then read alike. A payload filling the whole image gives no gap,
    and its whole-image estimate is accurate anyway. Heuristic.

    Args:
        arr: ``(H, W, C)`` uint8 image array.
        channel_rate: Function from one ``(h, W)`` channel slice to its
            clipped rate estimate (NaN when undefined).
        units: Function ``(rows, width) -> units per channel`` (pairs for
            SPA, groups for RS).
        n_ref: Units per channel at the reference size (see ``knot_scale``).
        likely_rate: Unscaled rate at which the estimator's score reaches
            "likely_stego".

    Returns:
        ``{"top", "bottom", "gap", "threshold", "sequential"}`` or ``None``
        when the image has fewer than 2 rows or a half is undefined.
    """
    h, w = arr.shape[:2]
    if h < 2:
        return None
    mid = h // 2
    rates = []
    for part in (arr[:mid], arr[mid:]):
        vals = [channel_rate(part[:, :, i]) for i in range(part.shape[2])]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return None
        rates.append(float(np.mean(vals)))
    top, bottom = rates
    threshold = likely_rate * knot_scale(units(h - mid, w), n_ref)
    gap = top - bottom
    return {"top": top, "bottom": bottom, "gap": gap, "threshold": threshold,
            "sequential": bool(gap >= threshold)}


def rate_summary(max_rate: float, top_channel: str, mean_rate: float, factor: float,
                 n_units: int, unit_name: str, halves: dict[str, object] | None) -> str:
    """Summary sentence shared by SPA and RS.

    Starts with the estimate (the wording the README and the reports quote),
    then adds a clause when the knots were widened for a small image and one
    when the payload looks sequential, so the estimate is a lower bound.
    """
    text = (f"Estimated LSB embedding rate {max_rate:.1%} (channel {top_channel}); "
            f"mean over channels {mean_rate:.1%}.")
    if halves is not None and halves["sequential"]:
        text += (f" Top half reads {halves['top']:.0%}, bottom half {halves['bottom']:.0%}: the "
                 f"payload looks sequential (written from the top), which this estimator reads "
                 f"low, so treat the rate as a lower bound.")
    if factor > 1.0:
        text += (f" Small image ({n_units} {unit_name} per channel): thresholds widened "
                 f"x{factor:.1f} for estimator noise.")
    return text
