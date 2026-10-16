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

How the run ends matters (heuristic interpretation, recorded per stream as
``ended_by`` and ``estimate_kind``):

* The LSB pairs of the next prefix are unbalanced again. Only this
  *locates* the end of the payload, and the run length is reported as an
  estimate of the payload size (``"located"``). The cumulative test lags
  behind the true end, more so on covers whose natural pair imbalances are
  weak, so the estimate tends to be too high.
* The control pairs of the next prefix are balanced as well (or a p-value
  is undefined). The test is merely inconclusive beyond that point: the
  payload may continue past it, and on a smooth cover the run may also
  have overshot the payload (see the next point), so the run length is not
  a size estimate (``"open"``).
* Every prefix counts. The control guard can be fooled by the payload
  itself: equalizing the pairs of a smooth histogram leaves a staircase at
  the pair boundaries that unbalances the shifted control pairs, so on a
  smooth cover the cumulative control stays "informative" long after the
  payload has ended, while the clean remainder's pairs are balanced anyway;
  a 25% payload can then run on to the end of the stream. Such a run is
  only called a full-stream payload (``"whole_stream"``) when the last
  ``TAIL_FRACTION`` of the stream, tested on its own, also looks embedded;
  otherwise it is ``"open"`` too.

When choosing the stream that reports the estimate, located and
whole-stream runs win over open ones, whatever their length.

Limits: the attack needs the payload to be packed at the start of the
stream. A payload scattered over the image with a key barely changes any
prefix until nearly every sample is used. Biased payloads (plain text is
about 45% one-bits) equalize pairs less and weaken the signal. On covers
with smooth histograms even full embedding can go undetected, and the
length of a sequential payload often cannot be estimated (see above); the
SPA and RS estimators do not share that weakness.

Score (heuristic, from the run length f of the selected stream, chosen as
described above):
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
# Fraction of the stream, at its end, tested on its own to certify a run that
# reaches the end of the stream (heuristic, see docstring). A sequential
# payload that does not fill the stream never reaches this tail.
TAIL_FRACTION = 0.2
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


def _finite(x: Any) -> bool:
    """True for a real, finite number (False for None and NaN)."""
    return x is not None and bool(np.isfinite(x))


def leading_run(curve: list[dict[str, Any]], threshold: float = P_THRESHOLD,
                control_alpha: float = CONTROL_ALPHA) -> dict[str, Any]:
    """Leading run of prefixes that look embedded, and what ended it.

    A prefix counts when its LSB pairs look equalized (``p > threshold``)
    while its shifted control pairs are clearly unbalanced
    (``p_control < control_alpha``). The control condition is what keeps a
    cover with a smooth histogram, whose pairs are balanced with no payload
    at all, from counting as embedded.

    Args:
        curve: Output of ``prefix_curve``.
        threshold: ``p`` above which the LSB pairs count as equalized.
        control_alpha: ``p_control`` below which the control pairs count as
            unbalanced.

    Returns:
        Dict with ``fraction`` (the largest prefix fraction such that every
        prefix up to it counts; 0.0 if the first prefix does not) and
        ``ended_by``, the reason the run stopped:

        * ``"pairs_unbalanced"``: the LSB pairs of the next prefix are
          unbalanced again, as in a clean image. Only this locates the end
          of a sequential payload.
        * ``"control_balanced"``: the LSB pairs still look equalized but the
          control pairs are balanced too, so the test cannot tell payload
          from a smooth cover past this point (inconclusive, not an end).
        * ``"undefined"``: a p-value of the next prefix is undefined (too
          few categories).
        * ``"end_of_stream"``: every prefix counts.
    """
    best = 0.0
    for pt in curve:
        p, pc = pt["p"], pt.get("p_control", 0.0)
        if not _finite(p):
            return {"fraction": best, "ended_by": "undefined"}
        if p <= threshold:
            return {"fraction": best, "ended_by": "pairs_unbalanced"}
        if not _finite(pc):
            return {"fraction": best, "ended_by": "undefined"}
        if pc >= control_alpha:
            return {"fraction": best, "ended_by": "control_balanced"}
        best = pt["fraction"]
    return {"fraction": best, "ended_by": "end_of_stream"}


def leading_fraction(curve: list[dict[str, Any]], threshold: float = P_THRESHOLD,
                     control_alpha: float = CONTROL_ALPHA) -> float:
    """Length of the leading run of prefixes that look embedded.

    Shorthand for ``leading_run(curve, threshold, control_alpha)["fraction"]``;
    use ``leading_run`` to also learn why the run ended.

    Returns:
        The largest prefix fraction such that every prefix up to it counts,
        or 0.0 if the first prefix does not.
    """
    return leading_run(curve, threshold, control_alpha)["fraction"]


def tail_test(stream: np.ndarray, tail_fraction: float = TAIL_FRACTION,
              threshold: float = P_THRESHOLD,
              control_alpha: float = CONTROL_ALPHA) -> dict[str, Any]:
    """Pair test on the last part of a stream on its own (not cumulative).

    A run of embedded-looking prefixes that reaches the end of the stream
    can be an artefact: the payload's own staircase keeps the cumulative
    control pairs unbalanced while a smooth clean remainder keeps the LSB
    pairs balanced. The end of the stream, which a sequential payload that
    does not fill the stream never reaches, is therefore tested separately.

    Args:
        stream: 1-D uint8 array in embedding order.
        tail_fraction: Fraction of the stream, at its end, to test.
        threshold: As in ``leading_run``.
        control_alpha: As in ``leading_run``.

    Returns:
        Dict with ``samples``, ``p``, ``p_control`` (NaN when undefined) and
        ``state``: ``"embedded"`` (LSB pairs equalized, control pairs
        unbalanced), ``"unbalanced"`` (LSB pairs unbalanced, as in a clean
        image), ``"smooth"`` (both balanced: the tail cannot be judged) or
        ``"undefined"``.
    """
    s = np.asarray(stream, dtype=np.uint8).ravel()
    start = s.size - int(round(float(tail_fraction) * s.size))
    hist = np.bincount(s[start:], minlength=256)
    p = pair_chi_square(hist)["p"]
    pc = pair_chi_square(hist, offset=1)["p"]
    if not _finite(p):
        state = "undefined"
    elif p <= threshold:
        state = "unbalanced"
    elif not _finite(pc):
        state = "undefined"
    elif pc < control_alpha:
        state = "embedded"
    else:
        state = "smooth"
    return {"samples": int(s.size - start), "p": p, "p_control": pc, "state": state}


def estimate_kind(run: dict[str, Any], tail: dict[str, Any]) -> str:
    """How far a stream's run length can be read as a payload size.

    Args:
        run: Output of ``leading_run``.
        tail: Output of ``tail_test`` for the same stream.

    Returns:
        ``"none"`` (no run), ``"located"`` (the LSB pairs became unbalanced
        again: the run length estimates the payload size), ``"whole_stream"``
        (every prefix counts and the tail looks embedded on its own) or
        ``"open"`` (the run stopped for another reason, or reached the end
        with a tail that does not look embedded: the test is inconclusive
        beyond the run and the run length is not a size estimate).
    """
    if run["fraction"] <= 0:
        return "none"
    if run["ended_by"] == "pairs_unbalanced":
        return "located"
    if run["ended_by"] == "end_of_stream" and tail["state"] == "embedded":
        return "whole_stream"
    return "open"


def _summary(f_hat: float, stream: str, kind: str, ended_by: str,
             tail: dict[str, Any]) -> str:
    """One-sentence summary for a non-empty run (``f_hat > 0``)."""
    tail_pct = f"{TAIL_FRACTION:.0%}"
    if kind == "located":
        return (f"Pairs of values are equalized over the first {f_hat:.0%} of the {stream} "
                f"stream, then unbalanced; consistent with a sequential payload of about "
                f"that size.")
    if kind == "whole_stream":
        return (f"Pairs of values are equalized over the whole {stream} stream (p > "
                f"{P_THRESHOLD} at every prefix, and the last {tail_pct} looks embedded on "
                f"its own); consistent with LSB embedding in every sample.")
    if ended_by == "end_of_stream":
        if tail["state"] == "unbalanced":
            why = (f"the last {tail_pct} of it has unbalanced pairs on its own, as in a clean "
                   f"image, so the payload probably ends earlier, but the test cannot locate "
                   f"where")
        elif tail["state"] == "smooth":
            why = (f"the last {tail_pct} of it has balanced control pairs as well, so a "
                   f"clean smooth tail cannot be told apart from an embedded one")
        else:
            why = f"the last {tail_pct} of it is too small to test on its own"
        return (f"Pairs of values look equalized at every prefix of the {stream} stream, "
                f"but {why}; a sequential payload is likely, its size is not estimated.")
    cause = ("the control pairs are balanced as well" if ended_by == "control_balanced"
             else "the test is undefined")
    text = (f"Pairs of values are equalized over the first {f_hat:.0%} of the {stream} "
            f"stream, but beyond that {cause}, so the attack is inconclusive beyond "
            f"{f_hat:.0%}: it does not locate the end of the payload, and {f_hat:.0%} is "
            f"not a size estimate.")
    if tail["state"] == "embedded":
        text += (f" The last {tail_pct} of the stream also looks embedded on its own, so "
                 f"the payload may fill the stream.")
    return text


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
        p-value curve per stream, the run length per stream with the reason
        each run ended, the tail test per stream and the stream that
        produced the score. ``estimated_sequential_fraction`` is the run
        length of that stream; ``estimate_kind`` says whether it is a size
        estimate (``"located"``, ``"whole_stream"``) or only the point past
        which the test is inconclusive (``"open"``).
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    streams: dict[str, np.ndarray] = {}
    if a.shape[2] > 1:
        streams["".join(labels)] = a.reshape(-1)
    for i, lab in enumerate(labels):
        streams[lab] = a[:, :, i].reshape(-1)

    curves = {}
    runs = {}
    tails = {}
    kinds = {}
    for name, stream in streams.items():
        curve = prefix_curve(stream)
        curves[name] = curve
        runs[name] = leading_run(curve)
        tails[name] = tail_test(stream)
        kinds[name] = estimate_kind(runs[name], tails[name])
    fractions = {k: r["fraction"] for k, r in runs.items()}

    # Streams whose run length is a size estimate (the end was located, or a
    # full-stream run is backed by its tail) outrank open runs, whose length
    # may only reflect where a smooth histogram hides the payload's end.
    primary = next(iter(streams))
    sized = [k for k in streams if kinds[k] in ("located", "whole_stream")]
    pool = sized or [k for k in streams if fractions[k] > 0] or [primary]
    best = max(pool, key=lambda k: fractions[k])
    f_hat = fractions[best]
    kind = kinds[best]
    score = interp_score(f_hat, SCORE_KNOTS)
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
    else:
        summary = _summary(f_hat, best, kind, runs[best]["ended_by"], tails[best])
    details = {
        "estimated_sequential_fraction": f_hat,
        "estimate_kind": kind,
        "run_ended_by": runs[best]["ended_by"],
        "stream": best,
        "per_stream_fraction": fractions,
        "per_stream_ended_by": {k: r["ended_by"] for k, r in runs.items()},
        "per_stream_estimate_kind": kinds,
        "tail_fraction": TAIL_FRACTION,
        "tail": tails,
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
