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

Input handling (``run`` and ``run_all``):
    * Grayscale images (mode L, a 2-D array, or an RGB file whose R, G and B
      are identical in every pixel) are analysed as one channel labelled
      ``"Y"``. Interleaving three identical channels would repeat every LSB
      three times in a row, which wrecks the runs test (a fully random LSB
      plane would read as "structured") and triples the chi-square counts.
      Such results carry ``details["input"]`` saying so.
    * Images with more than 8 bits per sample are rejected with
      ``ValueError``. Grayscale 16-bit and 32-bit integer modes (``I;16``,
      ``I``) are accepted only when every value fits in 0..255; float
      images (``F``) and 48/64-bit RGB(A) PNG or TIFF files are always
      rejected (Pillow has already discarded the low byte of the latter).
      Reducing such images to 8 bits keeps only the high byte (or clips),
      which removes the low-order bits an LSB payload lives in, so every
      detector would report "clean" whatever the file holds.

Adding a detector: write a module with a function decorated by
``@register("name", "description")`` that returns ``_common.make_result(...)``
and import it at the bottom of this file. It may accept an optional
``channel_names`` keyword (the labels of the channels it was given).
"""
from __future__ import annotations

import inspect
import os
import re
from typing import Any, Callable

import numpy as np

from ._common import LIKELY_AT, SUSPICIOUS_AT, VERDICTS, make_result, verdict_for

__all__ = [
    "GRAY_LABEL",
    "LIKELY_AT",
    "SUSPICIOUS_AT",
    "VERDICTS",
    "combine_results",
    "get_detector",
    "is_grayscale",
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


# Pillow raw modes that decode 16 bits per sample into an 8-bit mode by
# keeping only the high byte (48/64-bit RGB(A) PNG and TIFF, 16-bit LA PNG).
# Packed-pixel modes such as "BGR;16" (RGB565 BMP) do not match.
_SIXTEEN_BIT_RAWMODE = re.compile(r";16[BLN]")

# Label of the one channel analysed for a grayscale image; the same label the
# detectors give a 2-D array (see ``_common.channel_labels``).
GRAY_LABEL = "Y"


def _high_depth_error(what: str) -> ValueError:
    """The error raised for images with more than 8 bits per sample."""
    return ValueError(
        f"{what}: the detectors analyse 8-bit samples, and reducing this image to 8 bits "
        "would keep only the high byte (or clip), removing the low-order bits an LSB "
        "payload is stored in, so every detector would report clean whatever the file "
        "holds. Convert the image to 8 bits yourself to analyse its visible content, or "
        "pass the low byte as an array ((arr & 0xFF).astype(numpy.uint8)) to test its "
        "LSB plane: the signature scan and the LSB randomness tests apply there, the "
        "pair-based detectors (chi-square, SPA, RS, histogram pairs) model 8-bit image "
        "content and do not.")


def _sixteen_bit_rawmode(im: Any) -> str | None:
    """Return Pillow's raw mode when it decoded 16-bit samples into an 8-bit mode.

    Pillow opens 48-bit RGB and 64-bit RGBA PNG/TIFF files as plain ``RGB`` /
    ``RGBA``, silently dropping the low byte; only the tile's raw mode (for
    example ``"RGB;16B"``) shows it. ``load()`` clears the tile list, so for
    an already loaded image opened from a file the file header is read again.

    Args:
        im: A PIL image.

    Returns:
        The raw mode string, or None when the samples are 8 bits (or unknown).
    """
    tiles = list(getattr(im, "tile", None) or [])
    filename = getattr(im, "filename", None)
    if not tiles and filename and os.path.isfile(filename):
        from PIL import Image
        try:
            with Image.open(filename) as src:
                if src.mode == im.mode and src.size == im.size:
                    tiles = list(src.tile)
        except (OSError, ValueError, SyntaxError):  # unreadable now: nothing to report
            tiles = []
    for tile in tiles:
        args = tile[3] if len(tile) > 3 else None
        if isinstance(args, tuple):
            args = args[0] if args else None
        if isinstance(args, str) and _SIXTEEN_BIT_RAWMODE.search(args):
            return args
    return None


def _pil_to_rgb(im: Any) -> np.ndarray:
    """Convert a PIL image to ``(H, W, 3)`` uint8, refusing lossy bit-depth cuts.

    Raises:
        ValueError: For a float image, an integer image with values outside
            0..255, or a file decoded from 16 bits per sample (see
            ``_high_depth_error``).
    """
    mode = im.mode
    if mode == "F":
        raise _high_depth_error("floating-point image (mode F)")
    if mode == "I" or mode.startswith("I;"):
        # 16- or 32-bit integer samples. convert("RGB") would clip every value
        # above 255; accept the image only when that conversion is lossless.
        a = np.asarray(im)
        if a.size and (int(a.min()) < 0 or int(a.max()) > 255):
            raise _high_depth_error(
                f"{'16' if mode.startswith('I;16') else '32'}-bit integer image (mode {mode}, "
                f"values {int(a.min())}..{int(a.max())})")
        return np.repeat(a.astype(np.uint8)[:, :, None], 3, axis=2)
    raw = _sixteen_bit_rawmode(im)
    if raw is not None:
        raise _high_depth_error(
            f"16-bit-per-sample image (Pillow raw mode {raw}, opened as {mode})")
    return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def load_rgb(image: Any) -> np.ndarray:
    """Load an image as an ``(H, W, 3)`` uint8 RGB array.

    Args:
        image: A file path (str or ``os.PathLike``), a PIL image, or a numpy
            array of shape ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``. Palette,
            grayscale and alpha images are converted to plain RGB (alpha is
            dropped; the embedders never touch it), so a grayscale image comes
            back with R, G and B equal (``run`` and ``run_all`` then analyse
            it as one channel, see ``is_grayscale``). 16-bit and 32-bit
            integer images are accepted only when every value fits in 0..255.

    Returns:
        ``(H, W, 3)`` uint8 array.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If an array has an unsupported shape or dtype, or the
            image has more than 8 bits per sample (mode ``I;16``, ``I`` or
            ``F``, or a 48/64-bit PNG or TIFF that Pillow would cut to its
            high byte) and cannot be reduced to 8 bits without losing the
            low-order bits.
    """
    if isinstance(image, (str, os.PathLike)):
        from PIL import Image
        with Image.open(image) as im:
            return _pil_to_rgb(im)
    if not isinstance(image, np.ndarray) and hasattr(image, "convert") and hasattr(image, "size"):
        return _pil_to_rgb(image)
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


def is_grayscale(arr: np.ndarray) -> bool:
    """True when an ``(H, W, 3)`` array has R, G and B equal in every pixel.

    Such an image carries one channel of information: its interleaved LSB
    stream repeats every bit three times, and the stream-based tests (runs,
    chi-square counts) would misread those repeats as structure. A payload
    embedded over R, G and B breaks the equality, so such a stego image is
    still analysed as three channels.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return True
    if a.ndim != 3 or a.shape[2] < 3:
        return False
    return bool(np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 1], a[:, :, 2]))


def _prepare(image: Any, channels: str) -> tuple[np.ndarray, tuple[str, ...], dict | None]:
    """Load an image and pick the channels the detectors will see.

    A grayscale image (``is_grayscale``) analysed with more than one channel
    is reduced to a single channel labelled ``GRAY_LABEL``: the extra
    channels are exact copies, so nothing is lost, and the interleaved
    stream no longer repeats each bit. A single requested channel keeps its
    own label.

    Returns:
        ``(arr, labels, input_note)``; ``input_note`` is None unless the
        channels were reduced, else a dict for ``details["input"]``.
    """
    rgb = load_rgb(image)
    arr = _select_channels(rgb, channels)
    labels = tuple(str(channels).upper())
    if arr.shape[2] > 1 and is_grayscale(rgb):
        note = {
            "grayscale": True,
            "requested_channels": "".join(labels),
            "analysed_channels": GRAY_LABEL,
            "note": ("R, G and B are identical in every pixel (grayscale image), so one "
                     "channel was analysed; interleaving the copies would repeat every "
                     "LSB three times."),
        }
        return np.ascontiguousarray(rgb[:, :, :1]), (GRAY_LABEL,), note
    return arr, labels, None


def _call(fn: DetectorFn, name: str, arr: np.ndarray, labels: tuple[str, ...],
          input_note: dict | None = None) -> dict:
    """Call one detector, normalizing its result and catching failures.

    ``input_note`` (from ``_prepare``) is added to the result as
    ``details["input"]``.
    """
    try:
        params = inspect.signature(fn).parameters
        if "channel_names" in params:
            res = fn(arr, channel_names=labels)
        else:
            res = fn(arr)
    except Exception as exc:  # one failing detector must not sink a report
        res = {"score": 0.0, "summary": f"Detector failed: {type(exc).__name__}: {exc}",
               "details": {"error": f"{type(exc).__name__}: {exc}", "inconclusive": True}}
    if not isinstance(res, dict):
        res = {"score": 0.0, "summary": "Detector returned no result.",
               "details": {"error": "non-dict result", "inconclusive": True}}
    details = res.get("details", {})
    if input_note is not None:
        details = {**(details if isinstance(details, dict) else {}), "input": input_note}
    # Re-normalize so third-party detectors also satisfy the schema.
    return make_result(name, res.get("score", 0.0), res.get("summary", ""), details)


def run(name: str, image: Any, channels: str = "RGB") -> dict:
    """Run one registered detector on an image.

    Args:
        name: Detector name (see ``list_detectors``).
        image: Path, PIL image or array (see ``load_rgb``).
        channels: Which channels to analyse and in what order, as letters
            from ``"RGB"``. A grayscale image is analysed as one channel
            (see ``_prepare``).

    Returns:
        The detector's result dict.

    Raises:
        KeyError: If no detector has that name.
        ValueError: For bad ``channels`` or an image ``load_rgb`` rejects.
    """
    fn = get_detector(name)
    arr, labels, note = _prepare(image, channels)
    return _call(fn, name, arr, labels, note)


def run_all(image: Any, channels: str = "RGB") -> dict[str, dict]:
    """Run every registered detector on an image.

    Args:
        image: Path, PIL image or array (see ``load_rgb``).
        channels: Which channels to analyse and in what order, as letters
            from ``"RGB"``. ``"RGB"`` matches the embedding order of this
            project (row-major pixels, R then G then B within a pixel). A
            grayscale image is analysed as one channel labelled ``"Y"`` and
            every result then has ``details["input"]`` (see ``_prepare``).

    Returns:
        ``{detector name: result dict}`` in registration order. A detector
        that raises gets a result with score 0.0 and ``details["error"]``.

    Raises:
        ValueError: For bad ``channels`` or an image ``load_rgb`` rejects
            (for example a 16-bit PNG with values above 255).
    """
    arr, labels, note = _prepare(image, channels)
    return {name: _call(fn, name, arr, labels, note) for name, (fn, _) in _REGISTRY.items()}


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
    (see ``COMBINE_WEIGHTS``; unknown detectors weigh 0.75). Agreement
    between detectors earns no bonus. In particular SPA and RS are not
    independent: both estimate the embedding rate from the same pixels and
    react to the same cover structure, and on clean synthetic covers their
    per-channel estimates correlate at about r = 0.85 (0.57 to 0.85 across
    sizes and cover types). Adding to the score when both are suspicious
    would count largely the same evidence twice and turned some clean
    covers into "likely_stego".

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
    flagged = [n for n, r in results.items() if r.get("verdict") != "clean"]
    return {"score": best, "verdict": verdict_for(best), "driver": driver, "flagged": flagged}


# Import the built-in detectors last so they can import register() from this
# module; importing them registers them. The order here is the run order.
from . import signatures, chi_square, spa, rs, histogram_pairs, randomness  # noqa: E402,F401
