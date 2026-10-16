"""Randomness tests on the LSB plane (encrypted or compressed payloads).

Encrypted and well-compressed data are indistinguishable from fair coin
flips. When such a payload replaces the LSBs, the LSB stream passes the
standard randomness tests. This module runs them on the LSB stream in
embedding order (row-major pixels, channels interleaved R, G, B):

* Shannon entropy of the stream packed into bytes (MSB-first), in bits per
  byte. A uniform byte source gives about ``8 - 255 / (2 N ln 2)`` for ``N``
  bytes (the Miller-Madow small-sample bias), reported as the reference.
* Monobit (frequency) test: ``z = (ones - zeros) / sqrt(n)``.
* Wald-Wolfowitz runs test: ``z`` of the number of runs of equal bits.
* Chi-square uniformity of the packed bytes (255 dof, ``scipy.stats.chi2``).
* Spatial correlation of each channel's LSB plane with its horizontal
  neighbour, compared with the same correlation of the bit-1 plane.

The catch, and why the result is a heuristic: noisy natural photos also
have random-looking LSBs, because sensor noise of even one grey level
randomizes bit 0. A random LSB plane is therefore only *evidence* of a
payload when the image gives a reason to expect structure there:

* Smooth content. Pixel noise randomizes bit 0 long before bit 1 (by a
  Fourier argument the bit-0 correlation falls off like the fourth power of
  the bit-1 correlation), but when the bit-1 plane is still strongly
  correlated with its neighbours (``>= 0.6``) a clean bit-0 plane keeps
  clearly measurable correlation. Random LSBs under a smooth bit-1 plane are
  flagged ``likely_stego``.
* A sequential payload. The stream is cut into 16 chunks; if the leading
  chunks pass the tests and the rest of the stream clearly fails them, the
  random region is a candidate payload of about that length. That profile
  alone is not evidence: a noisy photo banner above a flat page or text
  body has it too. The region must also give a reason to expect structure,
  checked on the region itself: either its bit-1 plane is smooth (the test
  above, on the leading chunks), or sample pair analysis (Dumitrescu, Wu &
  Wang 2003) of the region's full rows estimates a high LSB embedding rate
  (full support from 0.6, ``SPA_REGION_KNOTS``). SPA is a context test
  here, not a second verdict: a payload
  that fills the region shows up as a rate near 1 there (also when it uses
  two bits per sample), while clean noisy content stays near 0. Bit-1
  smoothness alone is not enough, because covers with a comb-shaped
  histogram (contrast-stretched or otherwise edited photos) have structured
  LSBs without a smooth bit-1 plane; comparing bit-plane correlations with
  the rest of the image does not work either, since a flat page with text
  has low bit-plane correlations from glyph edges, not from noise. Without
  that context the region scores 0.3 like the whole-plane case.

A random LSB plane or region with no such context scores 0.3 (verdict
``clean``) and sets ``details["random_lsb"]`` (whole plane) or
``details["sequential_random_region"]`` (leading region): on its own it
cannot separate photo noise from a payload. Whenever the random region is
attributed to a payload, the summary adds "encrypted or compressed payload
likely", since text and other uncompressed data would fail the tests.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2 as chi2_dist

from . import register
from ._common import as_channels, channel_labels, interp_score, make_result
from .spa import spa_estimate

NAME = "lsb_randomness"

Z_MAX = 4.0               # |z| above this fails the monobit / runs tests
BYTE_P_MIN = 1e-4         # byte chi-square p below this fails uniformity
MIN_BYTES_FOR_CHI2 = 1280  # >= 5 expected per byte value
N_CHUNKS = 16
# Fewer LSBs than this (about a 32x32 RGB image) are not tested at all.
MIN_BITS = 3072
# Bit-1 neighbour correlation -> score when the LSB plane looks random.
CONTEXT_KNOTS = ((0.3, 0.3), (0.6, 0.85))
# Leading random fraction (sequential payload) -> score.
LEADING_KNOTS = ((1 / 16, 0.45), (2 / 16, 0.6), (4 / 16, 0.75), (1.0, 0.85))
# SPA embedding-rate estimate on the random leading region -> how far it
# supports a sequential payload there. Heuristic: on the synthetic covers, the
# test images and noisy photo banners above flat pages, clean regions estimate
# at most about 0.1 and regions filled with random payload bits (1 or 2 bits
# per sample) at least about 0.85.
SPA_REGION_KNOTS = ((0.3, 0.3), (0.6, 0.85))
# Fewer horizontal pairs per channel than this in the region's full rows and
# the SPA context is not computed (its estimate is too noisy).
MIN_REGION_PAIRS = 1000


def bit_stream_stats(bits: np.ndarray) -> dict[str, Any]:
    """Randomness statistics of a 0/1 stream.

    Args:
        bits: 1-D array of 0/1 values.

    Returns:
        Dict with ``n_bits``, ``ones_fraction``, ``monobit_z``, ``runs``,
        ``runs_z``, ``n_bytes``, ``entropy`` (bits per byte),
        ``entropy_random`` (reference for uniform bytes), ``byte_chi2`` and
        ``byte_chi2_p`` (None when there are too few bytes).
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = int(b.size)
    out: dict[str, Any] = {"n_bits": n}
    if n < 2:
        return {**out, "ones_fraction": float("nan"), "monobit_z": float("nan"),
                "runs": 0, "runs_z": float("nan"), "n_bytes": 0, "entropy": float("nan"),
                "entropy_random": float("nan"), "byte_chi2": None, "byte_chi2_p": None}
    n1 = int(b.sum())
    n0 = n - n1
    out["ones_fraction"] = n1 / n
    out["monobit_z"] = (n1 - n0) / math.sqrt(n)
    runs = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    out["runs"] = runs
    mu = 2.0 * n0 * n1 / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    out["runs_z"] = (runs - mu) / math.sqrt(var) if var > 0 else float("nan")

    nbytes = n // 8
    out["n_bytes"] = nbytes
    if nbytes == 0:
        out.update(entropy=float("nan"), entropy_random=float("nan"), byte_chi2=None, byte_chi2_p=None)
        return out
    packed = np.packbits(b[: nbytes * 8])
    counts = np.bincount(packed, minlength=256).astype(np.float64)
    p = counts[counts > 0] / nbytes
    out["entropy"] = float(-(p * np.log2(p)).sum())
    out["entropy_random"] = 8.0 - 255.0 / (2.0 * nbytes * math.log(2))
    if nbytes >= MIN_BYTES_FOR_CHI2:
        exp = nbytes / 256.0
        stat = float(((counts - exp) ** 2 / exp).sum())
        out["byte_chi2"] = stat
        out["byte_chi2_p"] = float(chi2_dist.sf(stat, 255))
    else:
        out["byte_chi2"] = None
        out["byte_chi2_p"] = None
    return out


def lag_correlation(bits: np.ndarray, lag: int) -> float:
    """Correlation between ``bits[i]`` and ``bits[i + lag]`` (1.0 for a constant stream)."""
    b = np.asarray(bits, dtype=np.float64).ravel()
    if b.size <= lag:
        return float("nan")
    return _corr(b[:-lag], b[lag:])


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; a constant input counts as perfectly structured (1.0)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return 1.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def neighbour_correlation(plane: np.ndarray) -> float:
    """Correlation of a bit plane with its right-hand neighbour."""
    p = np.asarray(plane)
    if p.shape[1] < 2:
        return float("nan")
    return _corr(p[:, :-1], p[:, 1:])


def _corr_limit(n: int) -> float:
    """|correlation| below this is consistent with independent bits (about 4 sigma)."""
    return 4.0 / math.sqrt(max(n, 1)) + 0.01


def looks_random(stats: dict[str, Any], corr: float | None = None, n_corr: int = 0) -> bool:
    """True when a stream passes every test at the module's thresholds."""
    if not (abs(stats.get("monobit_z", float("nan"))) < Z_MAX):
        return False
    rz = stats.get("runs_z", float("nan"))
    if not (abs(rz) < Z_MAX):
        return False
    p = stats.get("byte_chi2_p")
    if p is not None and p < BYTE_P_MIN:
        return False
    if corr is not None and not (abs(corr) < _corr_limit(n_corr)):
        return False
    return True


@register(NAME, "Entropy, monobit, runs and byte-uniformity tests plus spatial correlation "
                "of the LSB plane; flags random-looking (encrypted or compressed) payloads.")
def lsb_randomness(arr: np.ndarray,
                   channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Test the LSB stream for randomness and decide whether that is suspicious.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details`` holds the
        stream statistics, per-channel LSB and bit-1 neighbour correlations,
        the chunk profile (with each chunk's bit-1 lag correlation), the
        ``leading_context`` of a sequential random region (None when there is
        none) and the ``random_lsb`` / ``sequential_random_region`` /
        ``sequential_context`` / ``payload_hint`` flags.
    """
    a = as_channels(arr)
    h, w, c = a.shape
    labels = channel_labels(c, channel_names)
    stream = (a.reshape(-1) & 1).astype(np.uint8)
    if stream.size < MIN_BITS or w < 2:
        return make_result(NAME, 0.0,
                           f"Inconclusive: {stream.size} LSBs are too few for randomness tests "
                           f"(needs {MIN_BITS}).",
                           {"inconclusive": True, "random_lsb": False, "payload_hint": None,
                            "n_bits": int(stream.size), "min_bits": MIN_BITS})
    overall = bit_stream_stats(stream)

    per_channel = {}
    n_pairs = h * max(w - 1, 0)
    for i, lab in enumerate(labels):
        lsb = a[:, :, i] & 1
        bit1 = (a[:, :, i] >> 1) & 1
        per_channel[lab] = {
            "ones_fraction": float(lsb.mean()) if lsb.size else float("nan"),
            "lsb_neighbour_corr": neighbour_correlation(lsb),
            "bit1_neighbour_corr": neighbour_correlation(bit1),
        }
    corr_limit = _corr_limit(n_pairs)
    lsb_corrs = [v["lsb_neighbour_corr"] for v in per_channel.values()]
    bit1_corrs = [v["bit1_neighbour_corr"] for v in per_channel.values()]
    spatially_random = all(np.isfinite(x) and abs(x) < corr_limit for x in lsb_corrs)
    random_lsb = bool(spatially_random and looks_random(overall))
    finite_bit1 = [x for x in bit1_corrs if np.isfinite(x)]
    bit1_context = float(np.mean(finite_bit1)) if finite_bit1 else float("nan")

    # Chunk profile in embedding order; lag = channels -> same channel, next pixel.
    # The bit-1 stream in the same order tells how smooth each chunk's content is.
    bit1_stream = ((a.reshape(-1) >> 1) & 1).astype(np.uint8)
    chunks = []
    bounds = np.linspace(0, stream.size, N_CHUNKS + 1).astype(int)
    for k in range(N_CHUNKS):
        lo, hi = bounds[k], bounds[k + 1]
        seg = stream[lo:hi]
        st = bit_stream_stats(seg)
        corr = lag_correlation(seg, c) if seg.size > c else float("nan")
        chunks.append({"start": int(lo), "end": int(hi),
                       "entropy": st["entropy"], "monobit_z": st["monobit_z"],
                       "runs_z": st["runs_z"], "byte_chi2_p": st["byte_chi2_p"],
                       "lag_corr": corr,
                       "bit1_lag_corr": lag_correlation(bit1_stream[lo:hi], c),
                       "random": bool(seg.size >= 64 and looks_random(st, corr, max(seg.size - c, 1)))})
    leading = 0
    for ch in chunks:
        if not ch["random"]:
            break
        leading += 1
    rest = chunks[leading:]
    rest_fail = (sum(1 for ch in rest if not ch["random"]) / len(rest)) if rest else 0.0
    sequential = bool(0 < leading < N_CHUNKS and rest_fail >= 0.75)
    leading_fraction = leading / N_CHUNKS
    context = _leading_context(a, chunks[:leading], int(bounds[leading])) if sequential else None

    hint = None
    if random_lsb:
        score = interp_score(bit1_context, CONTEXT_KNOTS)
        if score >= 0.65:
            hint = "encrypted or compressed payload likely"
            summary = (f"LSB plane is statistically random although the bit-1 plane is smooth "
                       f"(neighbour correlation {bit1_context:.2f}); a natural LSB plane would keep "
                       f"some of that structure. Encrypted or compressed payload likely.")
        else:
            hint = "encrypted or compressed payload likely, if a payload is present"
            summary = ("LSB plane is statistically random: typical of a noisy photo, and also what "
                       "an encrypted or compressed payload looks like; not evidence on its own.")
    elif sequential:
        support = context["support"]
        score = min(interp_score(leading_fraction, LEADING_KNOTS), support)
        if support >= 0.65:
            hint = "encrypted or compressed payload likely"
            summary = (f"The first {leading_fraction:.0%} of the LSB stream is statistically random "
                       f"while the rest is structured: consistent with a sequentially embedded "
                       f"payload. Encrypted or compressed payload likely.")
        else:
            hint = "encrypted or compressed payload likely, if a payload is present"
            rate = context["spa_rate"]
            if rate is None:
                why = ("that region is too small for sample pair analysis and its bit-1 plane is "
                       "not smooth")
            else:
                why = (f"sample pair analysis of that region estimates an LSB embedding rate of "
                       f"only {rate:.0%} (a payload filling it would show close to 100%)")
            summary = (f"The first {leading_fraction:.0%} of the LSB stream is statistically random "
                       f"while the rest is structured, but {why}: noisier content there, such as a "
                       f"photo above flat graphics, explains the random LSBs as well as a payload "
                       f"would; " + ("not evidence on its own." if score < 0.35 else "weak evidence."))
    else:
        score = 0.0
        summary = (f"LSB plane is structured (entropy {overall['entropy']:.3f} bits/byte, monobit "
                   f"z {overall['monobit_z']:.1f}, runs z {overall['runs_z']:.1f}); no random-looking "
                   f"payload region.")
    details = {
        "random_lsb": random_lsb,
        "payload_hint": hint,
        "stream": overall,
        "channels": per_channel,
        "lsb_corr_limit": corr_limit,
        "bit1_neighbour_corr_mean": bit1_context,
        "sequential_random_region": sequential,
        "leading_random_fraction": leading_fraction,
        "sequential_context": bool(context is not None and context["support"] >= 0.65),
        "leading_context": context,
        "chunks": chunks,
        "thresholds": {"z_max": Z_MAX, "byte_p_min": BYTE_P_MIN,
                       "context_knots": CONTEXT_KNOTS, "leading_knots": LEADING_KNOTS,
                       "spa_region_knots": SPA_REGION_KNOTS,
                       "min_region_pairs": MIN_REGION_PAIRS},
    }
    return make_result(NAME, score, summary, details)


def _leading_context(a: np.ndarray, leading: list[dict[str, Any]],
                     n_samples: int) -> dict[str, Any]:
    """Whether a random leading region gives a reason to expect structure.

    Heuristic, see the module docstring. Two scores are computed on the
    region alone and the larger one is the support for a payload there:

    * ``smooth_score``: ``CONTEXT_KNOTS`` applied to the mean bit-1 lag
      correlation of the leading chunks (the whole-plane test, restricted to
      the region); 0.3 to 0.85.
    * ``spa_score``: ``SPA_REGION_KNOTS`` applied to the sample pair analysis
      (Dumitrescu, Wu & Wang 2003) embedding-rate estimate of the region's
      full rows, averaged over channels; 0.3 to 0.85, or 0.0 when those rows
      hold fewer than ``MIN_REGION_PAIRS`` horizontal pairs per channel.

    Args:
        a: ``(H, W, C)`` uint8 image array.
        leading: Chunk dicts of the random leading region (with
            ``bit1_lag_corr``).
        n_samples: Length of the region in samples (row-major pixels,
            channels interleaved).

    Returns:
        Dict with ``leading_bit1_corr``, ``region_rows``, ``region_pairs``
        (per channel), ``spa_rates`` (per channel, empty when not computed),
        ``spa_rate`` (their mean, None when not computed), ``smooth_score``,
        ``spa_score`` and ``support``.
    """
    h, w, c = a.shape
    b1 = [ch["bit1_lag_corr"] for ch in leading if np.isfinite(ch["bit1_lag_corr"])]
    lead1 = float(np.mean(b1)) if b1 else float("nan")
    smooth_score = interp_score(lead1, CONTEXT_KNOTS)

    # Only full rows: SPA counts horizontal pairs, and the region may end mid-row.
    rows = min(h, n_samples // (w * c))
    pairs = rows * (w - 1)
    rates: list[float] = []
    if pairs >= MIN_REGION_PAIRS:
        rates = [spa_estimate(a[:rows, :, i])["rate"] for i in range(c)]
    finite = [r for r in rates if np.isfinite(r)]
    spa_rate = float(np.mean(finite)) if finite else None
    spa_score = interp_score(spa_rate, SPA_REGION_KNOTS) if spa_rate is not None else 0.0
    return {"leading_bit1_corr": lead1, "region_rows": int(rows), "region_pairs": int(pairs),
            "spa_rates": rates, "spa_rate": spa_rate, "smooth_score": smooth_score,
            "spa_score": spa_score, "support": max(smooth_score, spa_score)}
