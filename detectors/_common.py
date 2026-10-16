"""Helpers shared by the detector modules (not a detector itself).

Every detector returns the same result shape::

    {
        "name":    "<registered detector name>",
        "score":   float in [0, 1],   # suspicion, 0 = nothing found
        "verdict": "clean" | "suspicious" | "likely_stego",
        "summary": "One short human-readable sentence.",
        "details": {...},             # JSON-serializable numbers and lists
    }

The verdict is always derived from the score with the same two cut points,
``SUSPICIOUS_AT`` and ``LIKELY_AT``. Each detector maps its own statistic
(an estimated embedding rate, a p-value curve, a signature hit, ...) onto the
score so that those two cut points land on thresholds that make sense for
that statistic. The thresholds are heuristics calibrated on synthetic covers
and a few test images, not error rates measured on a large image database;
treat a "likely_stego" verdict as a reason to look closer, not as proof.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

# Score cut points shared by every detector (heuristic, see module docstring).
SUSPICIOUS_AT = 0.35
LIKELY_AT = 0.65

VERDICTS = ("clean", "suspicious", "likely_stego")


def verdict_for(score: float) -> str:
    """Map a suspicion score in [0, 1] to a verdict string.

    Args:
        score: Suspicion score; values outside [0, 1] are clipped first.

    Returns:
        ``"clean"`` below ``SUSPICIOUS_AT``, ``"likely_stego"`` at or above
        ``LIKELY_AT``, otherwise ``"suspicious"``.
    """
    s = clip01(score)
    if s >= LIKELY_AT:
        return "likely_stego"
    if s >= SUSPICIOUS_AT:
        return "suspicious"
    return "clean"


def clip01(x: float) -> float:
    """Clip a number to [0, 1]; NaN becomes 0.0."""
    x = float(x)
    if math.isnan(x):
        return 0.0
    return min(1.0, max(0.0, x))


def interp_score(value: float, knots: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear map from a statistic to a score.

    Used to turn an estimate such as "embedding rate 0.12" into a score whose
    0.35 / 0.65 cut points sit on documented thresholds of that statistic.

    Args:
        value: The statistic. NaN maps to 0.0.
        knots: Increasing ``(statistic, score)`` points. Values below the first
            knot get the first score, above the last knot the last score.

    Returns:
        Score clipped to [0, 1].
    """
    v = float(value)
    if math.isnan(v):
        return 0.0
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return clip01(float(np.interp(v, xs, ys)))


def make_result(name: str, score: float, summary: str,
                details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a detector result dict in the shared schema.

    Args:
        name: Registered detector name.
        score: Suspicion score; clipped to [0, 1].
        summary: Short human-readable sentence.
        details: Extra JSON-serializable values (numpy types are converted).

    Returns:
        The result dict with ``name``, ``score``, ``verdict``, ``summary`` and
        ``details`` keys.
    """
    s = clip01(score)
    return {
        "name": name,
        "score": s,
        "verdict": verdict_for(s),
        "summary": str(summary),
        "details": to_jsonable(details or {}),
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types and non-finite floats for ``json.dumps``.

    numpy integers and floats become Python numbers, arrays become lists,
    tuples become lists, and NaN or infinite floats become ``None`` so the
    result also passes ``json.dumps(..., allow_nan=False)``.

    Args:
        obj: Any nesting of dicts, lists, tuples, numpy values and scalars.

    Returns:
        An equivalent structure built only from dict, list, str, int, float,
        bool and None.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def as_channels(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as an ``(H, W, C)`` uint8 array.

    A 2-D grayscale array gets a channel axis of length 1. Detectors call this
    so they work on any number of channels (``run_all`` may pass a subset such
    as only blue).

    Args:
        arr: ``(H, W)`` or ``(H, W, C)`` array of 8-bit sample values.

    Returns:
        ``(H, W, C)`` uint8 array (a view when no conversion is needed).

    Raises:
        ValueError: If the array is not 2-D or 3-D or has a non-integer dtype.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"expected an (H, W) or (H, W, C) array, got shape {a.shape}")
    if a.dtype != np.uint8:
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"expected 8-bit integer samples, got dtype {a.dtype}")
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def channel_labels(n: int, names: Iterable[str] | None) -> list[str]:
    """Labels for ``n`` channels: ``names`` when it fits, else R/G/B or c0..cN."""
    if names is not None:
        names = [str(c) for c in names]
        if len(names) == n:
            return names
    if n == 3:
        return ["R", "G", "B"]
    if n == 1:
        return ["Y"]
    return [f"c{i}" for i in range(n)]


def solve_quadratic_small_root(a: float, b: float, c: float) -> tuple[float, str]:
    """Solve ``a x^2 + b x + c = 0`` and return the root of smaller magnitude.

    Both the sample pair and the RS estimators end in a quadratic whose
    wanted root is the one closer to zero. The degenerate cases are handled
    the way most implementations do:

    * ``a == 0``: the linear root ``-c / b`` (NaN if ``b`` is also 0).
    * Negative discriminant (sampling noise near full embedding pushes the two
      roots off the real line): the real part ``-b / (2a)`` (the vertex).

    Args:
        a, b, c: Coefficients.

    Returns:
        ``(root, case)`` where ``case`` is ``"two_roots"``, ``"linear"``,
        ``"no_real_root"`` or ``"degenerate"`` (root NaN).
    """
    a, b, c = float(a), float(b), float(c)
    scale = max(abs(a), abs(b), abs(c), 1e-300)
    if abs(a) <= 1e-12 * scale:
        if abs(b) <= 1e-12 * scale:
            return float("nan"), "degenerate"
        return -c / b, "linear"
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return -b / (2.0 * a), "no_real_root"
    sq = math.sqrt(disc)
    # Numerically stable pair of roots (avoids cancellation in -b + sq).
    q = -0.5 * (b + math.copysign(sq, b)) if b != 0 else -0.5 * sq
    r1 = q / a
    r2 = c / q if q != 0 else -r1
    return (r1 if abs(r1) <= abs(r2) else r2), "two_roots"
