"""Blind steganalysis detectors for LSB embedding in images.

Each detector is a plain function that takes an ``(H, W, C)`` uint8 image
array (normally RGB) and returns a result dict. Detectors register
themselves with the ``@register`` decorator when their module is imported,
so importing this package makes all of them available::

    import detectors

    results = detectors.run_all("suspect.png")          # path, array or PIL image
    for name, r in results.items():
        print(f"{name:22s} {r['verdict']:13s} {r['score']:.2f}  {r['summary']}")

    overall = detectors.combine_results(results)        # {"score", "verdict", ...}
    detectors.list_detectors()                          # [{"name", "description"}]

Result schema (every detector, see ``_common.py``)::

    {"name": str, "score": float in [0, 1], "verdict": "clean" |
     "suspicious" | "likely_stego", "summary": str, "details": dict}

``details`` is always JSON-serializable (numpy values converted, NaN -> None).

Built-in detectors (run in this order):

    signatures            Scans the first few KB of the LSB stream (RGB, BGR,
                          single channels, and 2 bits per sample) for this
                          project's ``STGA`` payload header, common file
                          magic numbers (PNG, JPEG, ZIP, PDF, ...) and
                          readable text. A valid STGA header is definitive.
    chi_square            Westfeld & Pfitzmann (1999) chi-square attack on
                          pairs of values, on growing prefixes of the sample
                          stream; finds sequential embedding.
    sample_pair_analysis  Dumitrescu, Wu & Wang (2003) sample pair analysis;
                          estimates the embedding rate per channel.
    rs_analysis           Fridrich, Goljan & Du (2001) RS analysis; estimates
                          the embedding rate per channel.
    histogram_pairs       Single-image even/odd histogram pair imbalance
                          (the statistic of ``pairwise_stats.py`` without a
                          cover image): flattened pairs and the "comb" that
                          a text payload leaves.
    lsb_randomness        Entropy, monobit, runs and byte-uniformity tests on
                          the LSB stream plus LSB spatial correlation: flags
                          random-looking (encrypted or compressed) payloads.

What the detectors can and cannot see:
    * The rate estimators (SPA, RS) measure *how much* of the image carries
      LSB replacement, wherever it is; they are the most reliable signal for
      payloads above roughly 10% of capacity and see nothing below a few
      percent.
    * The chi-square attack only finds payloads written sequentially from
      the top of the image; a key-scattered payload (``stego.hide`` with a
      password) defeats it unless nearly every sample is used.
    * Signatures only see payloads stored in natural order (no key scatter).
    * All thresholds are heuristics calibrated on synthetic covers and test
      images, not error rates measured on a large image database.

Adding a detector: write a module with a function decorated by
``@register("name", "description")`` that returns ``_common.make_result(...)``
and import it at the bottom of this file. It may accept an optional
``channel_names`` keyword (the labels of the channels it was given).
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Callable

import numpy as np

from ._common import LIKELY_AT, SUSPICIOUS_AT, VERDICTS, make_result, verdict_for

__all__ = [
    "LIKELY_AT",
    "SUSPICIOUS_AT",
    "VERDICTS",
    "combine_results",
    "get_detector",
    "list_detectors",
    "load_rgb",
    "register",
    "run",
    "run_all",
    "unregister",
    "verdict_for",
]

DetectorFn = Callable[..., dict]

# name -> (function, description), in registration order.
_REGISTRY: dict[str, tuple[DetectorFn, str]] = {}


def register(name: str, description: str) -> Callable[[DetectorFn], DetectorFn]:
    """Decorator that adds a detector function to the registry.

    Args:
        name: Unique detector name (the key in ``run_all`` output).
        description: One-sentence description for menus and reports.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If a different function is already registered under
            ``name``.
    """
    def deco(fn: DetectorFn) -> DetectorFn:
        existing = _REGISTRY.get(name)
        if existing is not None and existing[0] is not fn and (
                existing[0].__module__, existing[0].__qualname__) != (fn.__module__, fn.__qualname__):
            raise ValueError(f"detector {name!r} is already registered")
        _REGISTRY[name] = (fn, description)
        fn.detector_name = name  # type: ignore[attr-defined]
        fn.detector_description = description  # type: ignore[attr-defined]
        return fn
    return deco


def unregister(name: str) -> None:
    """Remove a detector from the registry (no error if absent)."""
    _REGISTRY.pop(name, None)


def list_detectors() -> list[dict[str, str]]:
    """Return ``[{"name": ..., "description": ...}]`` in registration order."""
    return [{"name": n, "description": d} for n, (_, d) in _REGISTRY.items()]


def get_detector(name: str) -> DetectorFn:
    """Return the detector function registered under ``name``.

    Raises:
        KeyError: If no detector has that name.
    """
    try:
        return _REGISTRY[name][0]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; known: {', '.join(_REGISTRY)}") from None


def load_rgb(image: Any) -> np.ndarray:
    """Load an image as an ``(H, W, 3)`` uint8 RGB array.

    Args:
        image: A file path (str or ``os.PathLike``), a PIL image, or a numpy
            array of shape ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``. Palette,
            grayscale and alpha images are converted to plain RGB (alpha is
            dropped; the embedders never touch it).

    Returns:
        ``(H, W, 3)`` uint8 array.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If an array has an unsupported shape or dtype.
    """
    if isinstance(image, (str, os.PathLike)):
        from PIL import Image
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    if not isinstance(image, np.ndarray) and hasattr(image, "convert") and hasattr(image, "size"):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    a = np.asarray(image)
    if a.dtype != np.uint8:
        if not np.issubdtype(a.dtype, np.integer) or a.size and (a.min() < 0 or a.max() > 255):
            raise ValueError(f"expected uint8 samples, got dtype {a.dtype}")
        a = a.astype(np.uint8)
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return np.ascontiguousarray(a[:, :, :3])
    raise ValueError(f"expected an (H, W), (H, W, 3) or (H, W, 4) array, got shape {a.shape}")


def _select_channels(arr: np.ndarray, channels: str) -> np.ndarray:
    """Pick and order channels by letters from ``"RGB"`` (e.g. ``"B"``, ``"BGR"``)."""
    idx = []
    for ch in str(channels).upper():
        if ch not in "RGB":
            raise ValueError(f"channels must use the letters R, G, B; got {channels!r}")
        idx.append("RGB".index(ch))
    if not idx:
        raise ValueError("channels must name at least one of R, G, B")
    return np.ascontiguousarray(arr[:, :, idx])


def _call(fn: DetectorFn, name: str, arr: np.ndarray, labels: tuple[str, ...]) -> dict:
    """Call one detector, normalizing its result and catching failures."""
    try:
        params = inspect.signature(fn).parameters
        if "channel_names" in params:
            res = fn(arr, channel_names=labels)
        else:
            res = fn(arr)
    except Exception as exc:  # one failing detector must not sink a report
        return make_result(name, 0.0, f"Detector failed: {type(exc).__name__}: {exc}",
                           {"error": f"{type(exc).__name__}: {exc}", "inconclusive": True})
    if not isinstance(res, dict):
        return make_result(name, 0.0, "Detector returned no result.",
                           {"error": "non-dict result", "inconclusive": True})
    # Re-normalize so third-party detectors also satisfy the schema.
    return make_result(name, res.get("score", 0.0), res.get("summary", ""), res.get("details", {}))


def run(name: str, image: Any, channels: str = "RGB") -> dict:
    """Run one registered detector on an image.

    Args:
        name: Detector name (see ``list_detectors``).
        image: Path, PIL image or array (see ``load_rgb``).
        channels: Which channels to analyse and in what order, as letters
            from ``"RGB"``.

    Returns:
        The detector's result dict.
    """
    fn = get_detector(name)
    arr = _select_channels(load_rgb(image), channels)
    return _call(fn, name, arr, tuple(channels.upper()))


def run_all(image: Any, channels: str = "RGB") -> dict[str, dict]:
    """Run every registered detector on an image.

    Args:
        image: Path, PIL image or array (see ``load_rgb``).
        channels: Which channels to analyse and in what order, as letters
            from ``"RGB"``. ``"RGB"`` matches the embedding order of this
            project (row-major pixels, R then G then B within a pixel).

    Returns:
        ``{detector name: result dict}`` in registration order. A detector
        that raises gets a result with score 0.0 and ``details["error"]``.
    """
    arr = _select_channels(load_rgb(image), channels)
    labels = tuple(channels.upper())
    return {name: _call(fn, name, arr, labels) for name, (fn, _) in _REGISTRY.items()}


# Weight of each detector in combine_results(). The rate estimators and the
# signature scan are the most specific signals; the histogram and randomness
# heuristics are noisier on their own and count for less. Heuristic.
COMBINE_WEIGHTS = {
    "signatures": 1.0,
    "chi_square": 1.0,
    "sample_pair_analysis": 1.0,
    "rs_analysis": 1.0,
    "histogram_pairs": 0.8,
    "lsb_randomness": 0.75,
}


def combine_results(results: dict[str, dict]) -> dict[str, Any]:
    """Combine ``run_all`` output into one overall score and verdict.

    Heuristic: the overall score is the largest weighted detector score
    (see ``COMBINE_WEIGHTS``; unknown detectors weigh 0.75), raised by 0.1
    when SPA and RS both reach "suspicious" on their own, since two
    independent estimators agreeing is stronger evidence than either alone.

    Args:
        results: ``{name: result}`` as returned by ``run_all``.

    Returns:
        ``{"score": float, "verdict": str, "driver": name of the detector
        that set the score, "flagged": [names with a non-clean verdict]}``.
    """
    best, driver = 0.0, None
    for name, r in results.items():
        w = COMBINE_WEIGHTS.get(name, 0.75)
        s = float(r.get("score", 0.0)) * w
        if s > best:
            best, driver = s, name
    spa = results.get("sample_pair_analysis", {}).get("score", 0.0)
    rs = results.get("rs_analysis", {}).get("score", 0.0)
    if min(spa, rs) >= SUSPICIOUS_AT:
        best = min(1.0, best + 0.1)
    flagged = [n for n, r in results.items() if r.get("verdict") != "clean"]
    return {"score": best, "verdict": verdict_for(best), "driver": driver, "flagged": flagged}


# Import the built-in detectors last so they can import register() from this
# module; importing them registers them. The order here is the run order.
from . import signatures, chi_square, spa, rs, histogram_pairs, randomness  # noqa: E402,F401
