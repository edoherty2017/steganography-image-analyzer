#!/usr/bin/env python3
"""make_all.py: experiment runner for the steganalysis project.

Two experiments, both run in-process (every module is imported directly; no
subprocesses, so the script works from any working directory and with any
interpreter that has the dependencies):

``classic``  The original three-case histogram experiment on one cover image:

               A) full-low      Original size, short message (low rate).
               B) resized-high  Cover resized to --resize x --resize, then
                                filled to capacity with a repeated message.
               C) full-high     Original size, filled to capacity by a very
                                long repeated message.

             Each case embeds UTF-8 text plus the legacy 64-zero-bit
             delimiter in the R, G, B LSBs (row-major), then writes the
             steganalysis.analyze() histogram figures and the
             pairwise_stats.py even/odd statistics (JSON + LaTeX table).

             Differences from the original script (commit a5a3774):

             * Cases B and C cut the repeated message at a whole UTF-8
               character (fit_text_bits) instead of at a bit, so the
               delimiter stays byte-aligned and decoder.decode_legacy() can
               read it back. Case A is never cut. The stego image is
               pixel-identical to the original's when the message fits
               without a cut, or when it is ASCII and the case's cover has a
               W*H that is a multiple of 8 (as at the default 500x500 for
               case B). Otherwise the payload ends up to 7 bits earlier
               (more when a multi-byte character is split), the delimiter
               moves up with it, the last few samples keep their cover
               values, and that case's pairwise statistics differ slightly
               (a handful of samples at the end of the stream).
             * The pairwise captions name the analyzed channel; the original
               always said "blue channel".
             * The default --out is results/ in the repository, not
               results/ under the current working directory.

``detection`` A detection-curve experiment for the blind detectors in
             ``detectors/``. For each cover and each embedding rate it hides
             a payload and runs every registered detector, then writes a CSV
             of every result, a JSON summary, a LaTeX table and one figure
             per detector plus a combined figure. Payload variants:

               random     Uniform random bits written directly with the
                          image carrier (an idealized encrypted payload).
               encrypted  A real AES-256-GCM payload made by stego.hide()
                          (STGA header + ciphertext), sized to the rate.
               text       Repeated English text (optional; the biased-bit
                          payload of the classic experiment).

             Each in two modes: ``sequential`` (natural order from the top
             row, the legacy layout) and ``scattered`` (positions permuted
             by a key, as stego.hide() does with a password).

The embedding rate is the fraction of R, G, B samples whose LSB carries a
payload bit (1 bit per sample), the quantity that sample pair analysis and RS
analysis estimate. About half of the carrying samples actually change value,
because a random bit already matches the LSB half of the time.

Usage (any working directory)::

    python make_all.py detection                      # default covers, tag = timestamp
    python make_all.py detection --tag detection-demo
    python make_all.py detection --quick --out /tmp/exp
    python make_all.py detection photo.png synthetic:natural --rates 0,0.1,1

    python make_all.py classic COVER [--out results] [--resize 500]
                       [--channel b] [--short "message"] [--long-multiplier 5000]
    python make_all.py COVER [options]                # same as classic (old form)
    python make_all.py [options] COVER                # old form, options first

Detection outputs, under ``--out/--tag`` (default results/experiments/<tag>)::

    detection.csv              One row per cover x variant x rate x trial x
                               detector (plus an "overall" row from
                               detectors.combine_results).
    detection_summary.json     Configuration, curves (mean/min/max per rate)
                               and headline numbers per detector and variant.
    detection_table.tex        Mean suspicion score per detector and rate
                               (booktabs).
    detection_<detector>.png   Score vs true rate, and the detector's own rate
                               estimate vs true rate where it reports one.
    detection_combined.png     Every detector's score curve, one panel per
                               payload variant.

Classic outputs (RID = run timestamp, BASE = cover stem, CASE = full-low |
resized-high | full-high, CH = --channel), same names as before::

    {BASE}_{RID}_{CASE}_enc.png              Stego image.
    {BASE}_{RID}_resized-high_resized.png    Resized cover (case B only).
    {RID}_{CASE}_{CH}_{std,pair}_{cover,stego}.png, {RID}_{CASE}_{CH}_comp.png
                                             Figures from steganalysis.analyze.
    {RID}_{CASE}_pairwise.json, {RID}_{CASE}_pairwise.tex
                                             Statistics from pairwise_stats.

All detector thresholds are heuristics calibrated on synthetic covers and a
few test images; the curves here show how each detector responds on these
particular covers, not error rates on an image database.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

import detectors
import pairwise_stats
import stego
from carriers.image_lsb import ImageLSBCarrier
from payload import HEADER_LEN

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Length of the legacy end-of-message marker: 64 zero bits (eight bytes)
# appended after the classic payload. decoder.decode_legacy() finds it at a
# byte boundary.
DELIM_BITS = 64

DEFAULT_SHORT = "I am a very sneaky dinosaur... Nobody knows I am here."

DEFAULT_RATES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
QUICK_RATES = (0.0, 0.1, 0.5, 1.0)
PAYLOADS = ("random", "encrypted", "text")
DEFAULT_PAYLOADS = ("random", "encrypted")
MODES = ("sequential", "scattered")
SYNTHETIC_KINDS = ("natural", "stretched", "gradient")
DEFAULT_SIZE = 512
QUICK_SIZE = 128
DEFAULT_SEED = 5800
SMOOTH_BLUE = REPO_ROOT / "test-images" / "smoothBlue.png"

# Scatter key for the "scattered" mode. For the encrypted variant it is also
# the password, exactly as stego.hide() uses a password as the scatter key.
SCATTER_KEY = "make_all-detection"

# Plaintext for the encrypted and text variants (repeated to the needed
# length). Its bits are biased (about 46% ones), which matters for the text
# variant; encryption makes the encrypted variant's bits uniform.
EXPERIMENT_TEXT = (
    "The quick brown fox jumps over the lazy dog while the analyst watches "
    "the histogram. "
)

# Detectors that report their own estimate of the embedding rate, and the
# details key that holds it. For SPA and RS this is the mean over the three
# channels, which is the quantity the true rate measures (their headline
# "estimated_rate" is the largest channel, a conservative detection value
# that is biased upward as an estimate). chi_square estimates the length of
# a sequential payload, so it only tracks the true rate in sequential mode.
ESTIMATE_KEYS = {
    "sample_pair_analysis": "mean_rate",
    "rs_analysis": "mean_rate",
    "chi_square": "estimated_sequential_fraction",
}

# Pseudo-detector name for detectors.combine_results() in the CSV and plots.
OVERALL = "overall"

DISPLAY_NAMES = {
    "signatures": "Signatures",
    "chi_square": "Chi-square attack (Westfeld & Pfitzmann 1999)",
    "sample_pair_analysis": "Sample pair analysis (Dumitrescu, Wu & Wang 2003)",
    "rs_analysis": "RS analysis (Fridrich, Goljan & Du 2001)",
    "histogram_pairs": "Histogram pairs (heuristic)",
    "lsb_randomness": "LSB randomness (heuristic)",
    OVERALL: "Overall (combine_results)",
}

SHORT_NAMES = {
    "signatures": "Signatures",
    "chi_square": "Chi-square",
    "sample_pair_analysis": "SPA",
    "rs_analysis": "RS",
    "histogram_pairs": "Histogram pairs",
    "lsb_randomness": "LSB randomness",
    OVERALL: "Overall",
}

PAYLOAD_LABELS = {
    "random": "Random bits",
    "encrypted": "Encrypted (stego.hide)",
    "text": "ASCII text",
}
MODE_LABELS = {"sequential": "sequential", "scattered": "key-scattered"}

CSV_FIELDS = (
    "cover", "width", "height", "payload", "mode", "variant", "nominal_rate",
    "true_rate", "payload_bits", "changed_fraction", "trial", "detector",
    "score", "verdict", "estimate", "inconclusive", "summary",
)

Log = Callable[[str], None]


def ts_tag() -> str:
    """Return a run ID timestamp such as ``20250811-171038``.

    Returns:
        str: Local time formatted as ``%Y%m%d-%H%M%S``. It has one-second
        resolution, so two runs started in the same second that write to the
        same output directory share a tag and overwrite each other's files.
    """
    return time.strftime("%Y%m%d-%H%M%S")


# ===========================================================================
# Classic experiment (the original three runs)
# ===========================================================================

def to_bits_utf8(s: str) -> np.ndarray:
    """Convert a string to a flat array of bits (MSB first) from its UTF-8 bytes.

    Args:
        s: Text to embed.

    Returns:
        np.ndarray: uint8 array of 0/1 values with length ``8 * len(s.encode("utf-8"))``.
    """
    b = s.encode("utf-8")
    # unpackbits expands each byte into 8 bits, most significant bit first.
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)


def fit_text_bits(text: str, max_bits: int) -> np.ndarray:
    """Return the bits of the longest prefix of ``text`` that fits ``max_bits``.

    The cut is made at a whole UTF-8 character, so the embedded message never
    ends in a partial byte or a partial multi-byte character and decodes
    cleanly with decoder.decode_legacy().

    Args:
        text: Message to embed (usually a long repeated string).
        max_bits: Number of payload bits available.

    Returns:
        np.ndarray: uint8 0/1 array whose length is a multiple of 8 and at
        most ``max_bits``.
    """
    data = text.encode("utf-8")
    max_bytes = max(0, max_bits) // 8
    if len(data) > max_bytes:
        # errors="ignore" drops a multi-byte character cut in half.
        data = data[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def embed_bits_rgb(in_path: str, out_path: str, payload_bits: np.ndarray):
    """LSB-embed a bit stream, followed by the 64-zero-bit delimiter, into an image.

    This is the legacy delimiter format the classic experiment has always
    used (decoder.decode_legacy() reads it). Bits go into the least
    significant bit of R, then G, then B of each pixel, pixels in row-major
    order. Samples after the end of the stream are left unchanged. The
    result is pixel-for-pixel what the original per-pixel loop produced; the
    loop is now a single vectorized NumPy assignment.

    Args:
        in_path: Cover image path. It is converted to 8-bit RGB, which drops any
            alpha channel.
        out_path: Where to save the stego image. The format comes from the file
            extension and must be lossless, or the LSBs are destroyed (the
            classic experiment always passes a .png path).
        payload_bits: 1-D uint8 array of 0/1 values (see to_bits_utf8).

    Returns:
        tuple[int, int, int]: ``(W, H, n)``, where ``n`` is the number of bits
        written (payload plus delimiter).

    Raises:
        ValueError: If the payload plus delimiter is larger than ``3 * W * H`` bits.

    Side effects:
        Writes ``out_path``.
    """
    with Image.open(in_path) as img:
        rgb = img.convert("RGB")
    W, H = rgb.size
    arr = np.array(rgb, dtype=np.uint8)  # shape (H, W, 3), writable copy
    capacity = 3 * W * H  # one bit per color channel
    bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    total = bits.size + DELIM_BITS
    if total > capacity:
        raise ValueError(f"Payload too large ({total} bits) for capacity {capacity} bits")
    stream = np.concatenate([bits, np.zeros(DELIM_BITS, dtype=np.uint8)])
    # reshape(-1) of a C-contiguous array is a view in R, G, B, row-major
    # order. 0xFE clears the LSB; "| bit" writes the payload bit.
    flat = arr.reshape(-1)
    flat[:total] = (flat[:total] & 0xFE) | stream
    Image.fromarray(arr).save(out_path)
    return W, H, total


def resized_copy(in_path: str, out_path: str, size: int = 500):
    """Save a square ``size`` x ``size`` RGB copy of an image (Lanczos resampling).

    The aspect ratio is not preserved, and a cover smaller than ``size`` is
    upscaled. Case B embeds into this copy, and at the default 500x500 the
    default repeated message fills its whole capacity.

    Args:
        in_path: Source image path.
        out_path: Destination path. The format comes from the extension.
        size: Width and height of the output, in pixels.

    Raises:
        ValueError: If ``size`` is not positive.

    Side effects:
        Writes ``out_path``.
    """
    if size <= 0:
        raise ValueError(f"resize must be positive, got {size}")
    with Image.open(in_path) as img:
        out = img.convert("RGB").resize((size, size), resample=Image.LANCZOS)
    out.save(out_path)


def capacity_bits(path: str) -> int:
    """Return the raw LSB capacity of an image, ``3 * W * H`` bits.

    This includes the space that the 64-bit delimiter needs, so callers subtract
    it to get the usable payload size.

    Args:
        path: Image path.

    Returns:
        int: Number of embeddable bits (3 per RGB pixel).
    """
    with Image.open(path) as img:
        W, H = img.size
    return 3 * W * H


def _check_channel(channel: str) -> str:
    """Validate a classic-mode channel letter and return it lowercased.

    steganalysis.analyze() silently falls back to blue for an unknown
    channel, so the check happens here instead.
    """
    ch = str(channel).strip().lower()
    if ch not in ("r", "g", "b"):
        raise ValueError(f"channel must be 'r', 'g' or 'b', got {channel!r}")
    return ch


def run_steganalysis(cover, stego_path, outdir, tag, channel="b"):
    """Write the five histogram figures for a cover/stego pair.

    Calls steganalysis.analyze() in-process. The figures are
    ``{tag}_{channel}_std_cover.png``, ``_std_stego.png``,
    ``_pair_cover.png``, ``_pair_stego.png`` and ``_comp.png`` (a 2x2
    composite) in ``outdir``.

    Args:
        cover: Cover image path.
        stego_path: Stego image path (same size as the cover).
        outdir: Output directory.
        tag: Run tag used as the filename prefix.
        channel: "r", "g" or "b".

    Returns:
        dict: steganalysis.analyze()'s mapping of figure name to path.

    Side effects:
        Writes the PNG files and prints their paths (steganalysis.analyze
        prints them).
    """
    # Imported here because steganalysis imports matplotlib.pyplot at module
    # level; the CLI selects the non-interactive Agg backend before this runs.
    import steganalysis

    return steganalysis.analyze(cover, stego_path, out_dir=outdir, tag=tag,
                                channel=_check_channel(channel))


def run_pairwise_stats(cover, stego_path, outdir, tag, caption=None, channel="b"):
    """Compute even/odd pair statistics for a cover/stego pair and save them.

    Uses pairwise_stats.chan_hist(), pair_stats() and latex_table() directly
    and writes the same JSON object the pairwise_stats.py CLI prints.

    For each pair of values (2k, 2k+1) in the channel histogram, the
    statistic is ``d_k = h[2k] - h[2k+1]``, summarized with L1, L2, median
    and max of ``|d_k|``. Flipping an LSB only moves a pixel between 2k and
    2k+1. With uniformly random payload bits, heavy embedding makes the two
    counts equal in expectation, so these numbers drop compared with the
    cover. The classic payload is repeated ASCII text (about 46% ones), so
    the numbers can rise instead.

    Args:
        cover: Cover image path.
        stego_path: Stego image path (must be the same size as the cover).
        outdir: Directory for the JSON and LaTeX outputs.
        tag: Run tag used as the filename prefix and stored in the JSON.
        caption: Caption for the LaTeX table. None gives
            pairwise_stats.default_caption(), which names the channel.
        channel: "r", "g" or "b".

    Returns:
        dict: Keys "tag", "channel" ("R"/"G"/"B"), "cover" and "stego" (each
        with file, L1, L2, median_abs, max_abs, max_k), and "delta"
        (L1_drop, L2_drop and median_abs_drop, each equal to cover minus
        stego, so a negative value means the statistic grew).

    Raises:
        ValueError: If the two images differ in size.

    Side effects:
        Writes ``{outdir}/{tag}_pairwise.json`` and ``{outdir}/{tag}_pairwise.tex``.
    """
    ch = pairwise_stats.parse_channel(_check_channel(channel))
    Hc, size_c = pairwise_stats.chan_hist(cover, ch)
    Hs, size_s = pairwise_stats.chan_hist(stego_path, ch)
    if size_c != size_s:
        raise ValueError(f"size mismatch: {cover}={size_c}, {stego_path}={size_s}")
    Sc, Ss = pairwise_stats.pair_stats(Hc), pairwise_stats.pair_stats(Hs)
    out = {
        "tag": tag,
        "channel": "RGB"[ch],
        "cover": {"file": os.path.basename(cover), **Sc},
        "stego": {"file": os.path.basename(stego_path), **Ss},
        "delta": {
            "L1_drop": Sc["L1"] - Ss["L1"],
            "L2_drop": Sc["L2"] - Ss["L2"],
            "median_abs_drop": Sc["median_abs"] - Ss["median_abs"],
        },
    }
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"{tag}_pairwise.json")
    tex_path = os.path.join(outdir, f"{tag}_pairwise.tex")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(tex_path, "w") as f:
        f.write(pairwise_stats.latex_table(Sc, Ss, caption or pairwise_stats.default_caption(ch)))
    return out


def run_classic(cover: str, out_dir: str, *, resize: int = 500, channel: str = "b",
                short: str = DEFAULT_SHORT, long_multiplier: int = 5000,
                rid: str | None = None, log: Log = print) -> dict[str, Any]:
    """Run the three classic cases (A full-low, B resized-high, C full-high).

    Args:
        cover: Path to the cover image.
        out_dir: Output directory, created if it does not exist.
        resize: Side length in pixels of the square cover for case B.
        channel: Channel analyzed by both analyses: "r", "g" or "b".
        short: Message for case A. For B and C it is repeated, with a space
            after each copy.
        long_multiplier: Number of repeats for case B. Case C uses 10 times
            this. Both are cut to capacity at a whole character.
        rid: Run ID used in every file name; defaults to ts_tag().
        log: Function used for progress messages.

    Returns:
        dict: ``{"rid", "out_dir", "tags", "cases"}``; ``cases`` maps each
        case name to its cover, stego, figures, pairwise result and the
        paths of the JSON and LaTeX files.

    Raises:
        ValueError: If ``channel`` is not r/g/b, if ``resize`` is not
            positive, or if the short message plus delimiter does not fit.
        FileNotFoundError: If ``cover`` does not exist.
    """
    channel = _check_channel(channel)
    if not os.path.exists(cover):
        raise FileNotFoundError(f"cover not found: {cover}")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(cover))[0]
    rid = rid or ts_tag()
    name = pairwise_stats.CHANNEL_NAMES["rgb".index(channel)]
    cases: dict[str, dict[str, Any]] = {}

    def _case(case: str, cov: str, bits: np.ndarray, caption: str) -> None:
        tag = f"{rid}_{case}"
        enc = os.path.join(out_dir, f"{base}_{tag}_enc.png")
        _, _, n = embed_bits_rgb(cov, enc, bits)
        figs = run_steganalysis(cov, enc, outdir=out_dir, tag=tag, channel=channel)
        stats = run_pairwise_stats(cov, enc, outdir=out_dir, tag=tag,
                                   caption=caption, channel=channel)
        cases[case] = {
            "cover": cov, "stego": enc, "bits_written": n, "figures": figs,
            "pairwise": stats,
            "json": os.path.join(out_dir, f"{tag}_pairwise.json"),
            "tex": os.path.join(out_dir, f"{tag}_pairwise.tex"),
        }
        log(f"{case}: {n} bits written -> {enc}")

    # A) full-size low rate: only 8 * len(short) + 64 LSBs are written, a
    # tiny fraction of capacity.
    _case("full-low", cover, to_bits_utf8(short),
          f"Pairwise even/odd statistics ({name} channel), full-size low-rate case.")

    # B) resized high rate: shrink the cover so the repeated message fills
    # every LSB (delimiter included).
    covB = os.path.join(out_dir, f"{base}_{rid}_resized-high_resized.png")
    resized_copy(cover, covB, size=resize)
    bitsB = fit_text_bits((short + " ") * long_multiplier, capacity_bits(covB) - DELIM_BITS)
    _case("resized-high", covB, bitsB,
          f"Pairwise even/odd statistics ({name} channel), resized high-rate case.")

    # C) full-size high rate: ten times the repeats of case B, cut to
    # capacity. Covers larger than about 7.3 MP are not completely filled.
    bitsC = fit_text_bits((short + " ") * (long_multiplier * 10),
                          capacity_bits(cover) - DELIM_BITS)
    _case("full-high", cover, bitsC,
          f"Pairwise even/odd statistics ({name} channel), full-size high-rate case via long message.")

    tags = [f"{rid}_{c}" for c in cases]
    return {"rid": rid, "out_dir": os.path.abspath(out_dir), "tags": tags, "cases": cases}


# ===========================================================================
# Detection experiment
# ===========================================================================

@dataclass
class Cover:
    """A cover image for the detection experiment.

    Attributes:
        name: Short unique name used in the CSV and plots.
        source: The file path, or ``synthetic:<kind>``.
        pixels: ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array (alpha kept, as
            the image carrier keeps it; detectors see only R, G, B).
    """

    name: str
    source: str
    pixels: np.ndarray

    @property
    def rgb(self) -> np.ndarray:
        """The R, G, B channels as an ``(H, W, 3)`` array."""
        return np.ascontiguousarray(self.pixels[..., :3])


def _smooth_field(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Gaussian-blurred white noise with unit variance (via FFT, no SciPy).

    The blur wraps around the edges, which is harmless for a texture.
    """
    noise = rng.normal(size=(size, size))
    f = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    # Fourier transform of a Gaussian kernel with standard deviation sigma.
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    return field / (field.std() or 1.0)


def synthetic_cover(kind: str = "natural", seed: int = DEFAULT_SEED,
                    size: int = DEFAULT_SIZE) -> np.ndarray:
    """Generate a deterministic synthetic RGB cover.

    Kinds:
        natural    A gentle gradient, large smooth blobs and a low-frequency
                   sine texture per channel, plus Gaussian sensor noise
                   (sigma 1). Neighbouring pixels are correlated as in a
                   photo, and the histogram is smooth.
        stretched  The same content rendered into about 180 grey levels and
                   then linearly stretched to 0-255, as a levels or contrast
                   adjustment does. Some values are never used, so the
                   histogram has structure at the scale of single values,
                   which the chi-square attack and the histogram-pair
                   heuristic need.
        gradient   Wide linear gradients plus Gaussian noise (sigma 2); the
                   same recipe as the test suite's synthetic_rgb fixture.

    Args:
        kind: One of SYNTHETIC_KINDS.
        seed: Seed for every random draw; the same seed and size always give
            the same image.
        size: Width and height in pixels.

    Returns:
        np.ndarray: ``(size, size, 3)`` uint8 array.

    Raises:
        ValueError: If ``kind`` is unknown or ``size`` is below 16.
    """
    if kind not in SYNTHETIC_KINDS:
        raise ValueError(f"unknown synthetic cover kind {kind!r}; choose from {SYNTHETIC_KINDS}")
    if size < 16:
        raise ValueError(f"synthetic covers need size >= 16, got {size}")
    rng = np.random.default_rng(seed)
    if kind == "gradient":
        y, x = np.mgrid[0:size, 0:size].astype(float) * (255.0 / (size - 1))
        base = np.stack([x, y, (x + y) / 2.0], axis=-1)
        return np.clip(np.rint(base + rng.normal(0.0, 2.0, base.shape)), 0, 255).astype(np.uint8)

    y, x = np.mgrid[0:size, 0:size].astype(float) / (size - 1)
    chans = []
    for _ in range(3):
        wx, wy = rng.uniform(0.3, 1.0, 2)
        fx, fy = rng.uniform(1.0, 3.0), rng.uniform(0.5, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        chans.append(30.0 + 140.0 * (wx * x + wy * y) / (wx + wy)
                     + 22.0 * _smooth_field(rng, size, size / 12.0)
                     + 12.0 * np.sin(2.0 * np.pi * (fx * x + fy * y) + phase))
    img = np.stack(chans, axis=-1) + rng.normal(0.0, 1.0, (size, size, 3))
    if kind == "stretched":
        levels = 180
        img = np.clip(np.rint(img * levels / 255.0), 0, levels) * 255.0 / levels
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def default_cover_specs(quick: bool = False) -> list[str]:
    """Default covers: test-images/smoothBlue.png (if present, not in quick
    mode) plus the ``natural`` and ``stretched`` synthetic covers."""
    specs = [] if quick or not SMOOTH_BLUE.exists() else [str(SMOOTH_BLUE)]
    return specs + ["synthetic:natural", "synthetic:stretched"]


def load_covers(specs: Iterable[str | os.PathLike | Cover], *, size: int = DEFAULT_SIZE,
                seed: int = DEFAULT_SEED) -> list[Cover]:
    """Turn cover specs into Cover objects.

    Args:
        specs: Image paths, ``synthetic:<kind>`` strings, or ready Cover
            objects (passed through).
        size: Side length of synthetic covers.
        seed: Base seed; the i-th spec in the list uses ``seed + i`` so two
            synthetic covers of the same kind still differ.

    Returns:
        list[Cover]: Covers with unique names (a numeric suffix is added
        to repeated names).

    Raises:
        ValueError: For an unknown synthetic kind or an empty list.
        FileNotFoundError: If an image path does not exist.
    """
    covers: list[Cover] = []
    seen: dict[str, int] = {}
    for i, spec in enumerate(specs):
        if isinstance(spec, Cover):
            cov = spec
        else:
            s = os.fspath(spec)
            if s.startswith("synthetic:"):
                kind = s.split(":", 1)[1]
                cov = Cover(f"synthetic-{kind}", s, synthetic_cover(kind, seed + i, size))
            else:
                if not os.path.exists(s):
                    raise FileNotFoundError(f"cover not found: {s}")
                pixels = ImageLSBCarrier.load(s).pixels
                cov = Cover(Path(s).stem, s, pixels)
        n = seen.get(cov.name, 0)
        seen[cov.name] = n + 1
        if n:
            cov = Cover(f"{cov.name}-{n + 1}", cov.source, cov.pixels)
        covers.append(cov)
    if not covers:
        raise ValueError("no covers given")
    return covers


def _variant(payload: str, mode: str) -> str:
    return f"{payload}-{mode}"


def variant_label(variant: str) -> str:
    """Human label for a variant id such as ``"random-scattered"``."""
    payload, mode = variant.split("-", 1)
    return f"{PAYLOAD_LABELS.get(payload, payload)}, {MODE_LABELS.get(mode, mode)}"


def _text_bits(n: int) -> np.ndarray:
    """``n`` bits of EXPERIMENT_TEXT repeated (MSB-first ASCII)."""
    one = np.unpackbits(np.frombuffer(EXPERIMENT_TEXT.encode("ascii"), dtype=np.uint8))
    return np.resize(one, n).astype(np.uint8)


def _embed(cover: Cover, cover_png: Path, payload: str, mode: str, rate: float,
           rng: np.random.Generator, img_dir: Path, stem: str) -> tuple[np.ndarray, dict] | None:
    """Embed one payload variant at one rate.

    Returns:
        ``(stego_rgb, info)`` where info has ``payload_bits`` and
        ``true_rate``, or None when the encrypted variant cannot be made
        that small (its fixed 58-byte header and crypto overhead exceed the
        requested rate).
    """
    n_samples = cover.rgb.size
    key = SCATTER_KEY if mode == "scattered" else None
    if payload in ("random", "text"):
        n_bits = int(round(rate * n_samples))
        bits = (rng.integers(0, 2, n_bits, dtype=np.uint8) if payload == "random"
                else _text_bits(n_bits))
        carrier = ImageLSBCarrier(cover.pixels)
        carrier.embed(bits, key=key)
        if img_dir is not None:
            carrier.save(img_dir / f"{stem}.png")
        return carrier.rgb.copy(), {"payload_bits": n_bits, "true_rate": n_bits / n_samples}

    # Encrypted: a real stego.hide() frame sized so header + crypto overhead
    # + plaintext fill ``rate`` of the samples (rounded down to whole bytes).
    data_len = int(rate * n_samples) // 8 - HEADER_LEN - stego.ENCRYPTION_OVERHEAD
    if data_len < 0:
        return None
    plain = (EXPERIMENT_TEXT * (data_len // len(EXPERIMENT_TEXT) + 1)).encode("ascii")[:data_len]
    out = img_dir / f"{stem}.png"
    summary = stego.hide(cover_png, out, plain, password=SCATTER_KEY,
                         key_scatter=(mode == "scattered"))
    stego_rgb = ImageLSBCarrier.load(out).rgb.copy()
    return stego_rgb, {"payload_bits": summary["bits_used"],
                       "true_rate": summary["bits_used"] / n_samples}


def _result_rows(base: dict, results: dict[str, dict]) -> list[dict]:
    """One CSV row per detector result plus the combine_results() row."""
    rows = []
    for name, r in results.items():
        details = r.get("details") or {}
        est = details.get(ESTIMATE_KEYS[name]) if name in ESTIMATE_KEYS else None
        rows.append({
            **base, "detector": name, "score": float(r["score"]),
            "verdict": r["verdict"],
            "estimate": None if est is None else float(est),
            "inconclusive": bool(details.get("inconclusive")),
            "summary": r.get("summary", ""),
        })
    overall = detectors.combine_results(results)
    rows.append({
        **base, "detector": OVERALL, "score": float(overall["score"]),
        "verdict": overall["verdict"], "estimate": None, "inconclusive": False,
        "summary": f"driver={overall['driver']}; flagged={','.join(overall['flagged']) or 'none'}",
    })
    return rows


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def aggregate(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Summarize result rows into curves.

    Args:
        rows: Rows as produced by run_detection() (the CSV rows).

    Returns:
        ``{variant: {detector: [point, ...]}}`` with points sorted by rate.
        Each point has ``rate`` (nominal), ``true_rate`` (mean), ``n`` (runs),
        ``score_mean/min/max``, ``flagged`` (fraction of runs not "clean"),
        ``likely`` (fraction "likely_stego"), ``inconclusive`` (fraction),
        and ``estimate_mean/min/max`` (None when the detector reports no
        estimate).
    """
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["variant"], r["detector"], r["nominal_rate"]), []).append(r)
    curves: dict[str, dict[str, list[dict]]] = {}
    for (variant, det, rate), rs in groups.items():
        scores = [r["score"] for r in rs]
        ests = [r["estimate"] for r in rs if r["estimate"] is not None]
        curves.setdefault(variant, {}).setdefault(det, []).append({
            "rate": rate,
            "true_rate": _mean([r["true_rate"] for r in rs]),
            "n": len(rs),
            "score_mean": _mean(scores),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "flagged": sum(r["verdict"] != "clean" for r in rs) / len(rs),
            "likely": sum(r["verdict"] == "likely_stego" for r in rs) / len(rs),
            "inconclusive": sum(bool(r["inconclusive"]) for r in rs) / len(rs),
            "estimate_mean": _mean(ests),
            "estimate_min": float(min(ests)) if ests else None,
            "estimate_max": float(max(ests)) if ests else None,
        })
    for dets in curves.values():
        for pts in dets.values():
            pts.sort(key=lambda p: p["rate"])
    return curves


def headline(curves: dict, rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Headline numbers per variant and detector.

    Returns:
        ``{variant: {detector: {...}}}`` with ``score_at_zero`` (mean score
        on the clean covers), ``flagged_at_zero`` (fraction of clean covers
        not called clean: the false-alarm rate on these covers),
        ``reliable_from_rate`` (smallest nonzero rate from which every run
        at that rate and every higher rate is not "clean", or None), and,
        for detectors with an estimate, ``estimate_mae`` and
        ``estimate_bias`` (mean absolute and signed error of the estimate
        against the true rate, over all rates).
    """
    out: dict[str, dict[str, dict]] = {}
    for variant, dets in curves.items():
        for det, pts in dets.items():
            zero = [p for p in pts if p["rate"] == 0]
            reliable = None
            for p in reversed([p for p in pts if p["rate"] > 0]):
                if p["flagged"] < 1.0:
                    break
                reliable = p["rate"]
            entry: dict[str, Any] = {
                "score_at_zero": zero[0]["score_mean"] if zero else None,
                "flagged_at_zero": zero[0]["flagged"] if zero else None,
                "reliable_from_rate": reliable,
            }
            errs = [r["estimate"] - r["true_rate"] for r in rows
                    if r["variant"] == variant and r["detector"] == det
                    and r["estimate"] is not None]
            if errs:
                entry["estimate_mae"] = float(np.mean(np.abs(errs)))
                entry["estimate_bias"] = float(np.mean(errs))
            out.setdefault(variant, {})[det] = entry
    return out


def write_csv(rows: list[dict], path: Path) -> Path:
    """Write the result rows as CSV (columns CSV_FIELDS; None -> empty)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})
    return path


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#")):
        s = s.replace(a, b)
    return s


def latex_table(curves: dict, heads: dict, variants: Sequence[str],
                det_names: Sequence[str], rates: Sequence[float], n_covers: int) -> str:
    """LaTeX table of mean suspicion score per detector and rate.

    One block per payload variant; cells of 0.65 or more (the
    "likely_stego" band) are bold. The last column is the mean absolute
    error of the detector's own rate estimate, where it reports one. Needs
    the ``booktabs`` package.
    """
    cols = len(rates)
    head = " & ".join(f"{100 * r:g}\\%" for r in rates)
    lines = [
        "% Generated by make_all.py detection. Requires \\usepackage{booktabs}.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Mean detector suspicion score (0--1) by embedding rate, over "
        f"{n_covers} cover{'s' if n_covers != 1 else ''}. "
        r"Bold: $\geq 0.65$ (likely stego). Est.\ MAE: mean absolute error of the "
        r"detector's own rate estimate. All thresholds are heuristics.}",
        r"\label{tab:detection}",
        f"\\begin{{tabular}}{{l{'r' * cols}r}}",
        r"\toprule",
        f"Detector & \\multicolumn{{{cols}}}{{c}}{{Embedding rate}} & Est. \\\\",
        f"\\cmidrule(lr){{2-{cols + 1}}}",
        f" & {head} & MAE \\\\",
    ]
    for variant in variants:
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{cols + 2}}}{{l}}{{\\emph{{{_tex_escape(variant_label(variant))}}}}} \\\\")
        for det in list(det_names) + [OVERALL]:
            pts = {p["rate"]: p for p in curves.get(variant, {}).get(det, [])}
            cells = []
            for r in rates:
                p = pts.get(r)
                if p is None:
                    cells.append("--")
                    continue
                v = f"{p['score_mean']:.2f}"
                cells.append(f"\\textbf{{{v}}}" if p["score_mean"] >= detectors.LIKELY_AT else v)
            mae = heads.get(variant, {}).get(det, {}).get("estimate_mae")
            mae_s = "--" if mae is None else f"{mae:.3f}"
            name = _tex_escape(SHORT_NAMES.get(det, det))
            lines.append(f"{name} & {' & '.join(cells)} & {mae_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots (matplotlib object API on the Agg canvas: never touches pyplot, so it
# cannot change the GUI's backend or leave figures open)
# ---------------------------------------------------------------------------

# Light-mode chart tokens and the categorical palette in fixed slot order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def _variant_style(variant: str) -> tuple[str, str]:
    """Fixed colour and marker per variant, so a variant keeps its colour
    whichever subset of variants a run includes."""
    order = [_variant(p, m) for p in PAYLOADS for m in MODES]
    i = order.index(variant) if variant in order else len(SERIES) - 1
    return SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]


def _detector_style(det: str, det_names: Sequence[str]) -> tuple[str, str]:
    """Colour and marker by registry position; detectors past the palette
    (plug-ins) share the muted grey rather than reusing a hue."""
    if det == OVERALL:
        return INK, "*"
    i = list(det_names).index(det)
    if i >= len(SERIES):
        return MUTED, MARKERS[i % len(MARKERS)]
    return SERIES[i], MARKERS[i]


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, length=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def _thresholds(ax) -> None:
    """Hairlines at the verdict thresholds, labelled in muted ink."""
    for y, label in ((detectors.SUSPICIOUS_AT, "suspicious"), (detectors.LIKELY_AT, "likely stego")):
        ax.axhline(y, color=BASELINE, linewidth=0.8, zorder=1)
        ax.text(1.02, y + 0.012, label, color=MUTED, fontsize=7, ha="right", va="bottom")


# (line width, marker size, z-order) per payload in the per-detector
# figures. An encrypted body is statistically random, so its curves often
# coincide with the random-bit ones; drawing random wider and underneath
# keeps both visible.
_PAYLOAD_WEIGHT = {"random": (3.6, 8.5, 3), "text": (2.6, 7.0, 4), "encrypted": (1.6, 5.5, 5)}


def _line(ax, pts: list[dict], key: str, color: str, marker: str, label: str,
          band: bool = True, lw: float = 1.8, ms: float = 6.0, z: int = 3) -> None:
    xs = [p["true_rate"] for p in pts]
    ys = [p[f"{key}_mean"] for p in pts]
    keep = [i for i, y in enumerate(ys) if y is not None]
    if not keep:
        return
    xs = [xs[i] for i in keep]
    if band:
        lo = [pts[i][f"{key}_min"] for i in keep]
        hi = [pts[i][f"{key}_max"] for i in keep]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(xs, [ys[i] for i in keep], color=color, linewidth=lw, marker=marker,
            markersize=ms, markerfacecolor=color, markeredgecolor=SURFACE,
            markeredgewidth=1.2, solid_joinstyle="round", solid_capstyle="round",
            label=label, zorder=z)


def _new_figure(w: float, h: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def _legend(fig, handles_from, ncol: int, extra: Sequence = ()) -> None:
    handles, labels = handles_from.get_legend_handles_labels()
    if len(handles) < 2:
        return
    for h in extra:
        handles.append(h)
        labels.append(h.get_label())
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                     fontsize=8, handlelength=2.2, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK_2)


def plot_detector(curves: dict, det: str, variants: Sequence[str], out_path: Path,
                  n_covers: int) -> Path:
    """Detection-curve figure for one detector.

    Left panel: suspicion score vs true embedding rate, one line per variant
    (mean over covers and trials, with a min-max band). Right panel (only
    for detectors that estimate the rate): the estimate vs the true rate,
    with the identity line for a perfect estimator.
    """
    has_est = det in ESTIMATE_KEYS
    fig = _new_figure(10.0 if has_est else 5.6, 4.6)
    axes = fig.subplots(1, 2 if has_est else 1, squeeze=False)[0]
    xlabel = "True embedding rate (share of R, G, B samples)"
    _style_axes(axes[0], xlabel, "Suspicion score", "Score vs true rate")
    _thresholds(axes[0])
    for v in variants:
        pts = curves.get(v, {}).get(det, [])
        color, marker = _variant_style(v)
        lw, ms, z = _PAYLOAD_WEIGHT.get(v.split("-", 1)[0], (1.8, 6.0, 3))
        _line(axes[0], pts, "score", color, marker, variant_label(v), lw=lw, ms=ms, z=z)
    extra = []
    if has_est:
        ax = axes[1]
        est_name = ESTIMATE_KEYS[det].replace("_", " ")
        _style_axes(ax, xlabel, f"Detector estimate ({est_name})", "Estimate vs true rate")
        (ident,) = ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, zorder=1,
                           label="Perfect estimate (y = x)")
        extra.append(ident)
        for v in variants:
            pts = curves.get(v, {}).get(det, [])
            color, marker = _variant_style(v)
            lw, ms, z = _PAYLOAD_WEIGHT.get(v.split("-", 1)[0], (1.8, 6.0, 3))
            _line(ax, pts, "estimate", color, marker, variant_label(v), lw=lw, ms=ms, z=z)
    fig.suptitle(DISPLAY_NAMES.get(det, det), color=INK, fontsize=12, x=0.01, ha="left")
    sub = (f"Mean over {n_covers} cover{'s' if n_covers != 1 else ''}; "
           "shaded band = min to max across covers and trials."
           + (" " if has_est else "\n")
           + "Random-bit lines are drawn wider, underneath, so coinciding curves stay visible.")
    fig.text(0.01, 0.905, sub, color=INK_2, fontsize=7.5, ha="left", va="top")
    n_items = len(variants) + len(extra)
    ncol = min(n_items, 3 if has_est else 2)
    _legend(fig, axes[0], ncol=ncol, extra=extra)
    rows_legend = -(-n_items // ncol)
    fig.subplots_adjust(left=0.07 if has_est else 0.12, right=0.98,
                        top=0.80 if has_est else 0.77,
                        bottom=0.17 + 0.05 * (rows_legend - 1), wspace=0.22)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    return out_path


def plot_combined(curves: dict, det_names: Sequence[str], variants: Sequence[str],
                  out_path: Path, n_covers: int) -> Path:
    """One panel per variant; one score line per detector plus the overall
    combine_results() score (ink, no band)."""
    n = len(variants)
    ncols = 1 if n == 1 else 2
    nrows = -(-n // ncols)
    fig = _new_figure(5.2 * ncols, 3.6 * nrows + 1.2)
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax, v in zip(axes, variants):
        _style_axes(ax, "True embedding rate", "Suspicion score", variant_label(v))
        _thresholds(ax)
        for det in det_names:
            color, marker = _detector_style(det, det_names)
            _line(ax, curves.get(v, {}).get(det, []), "score", color, marker,
                  SHORT_NAMES.get(det, det), band=False, lw=1.5)
        _line(ax, curves.get(v, {}).get(OVERALL, []), "score", INK, "*",
              SHORT_NAMES[OVERALL], band=False, lw=2.2)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Detection curves: mean suspicion score vs embedding rate",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 1 - 0.55 / (3.6 * nrows + 1.2), f"Mean over {n_covers} cover"
             f"{'s' if n_covers != 1 else ''}. Thresholds: 0.35 suspicious, 0.65 likely stego "
             "(heuristics).", color=INK_2, fontsize=8, ha="left", va="top")
    _legend(fig, axes[0], ncol=min(len(det_names) + 1, 7))
    h = 3.6 * nrows + 1.2
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 0.95 / h, bottom=0.75 / h,
                        hspace=0.45, wspace=0.22)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    return out_path


# ---------------------------------------------------------------------------
# Detection driver
# ---------------------------------------------------------------------------

def _portable(source: str) -> str:
    """A cover source as stored in the summary: paths inside the repository
    become repository-relative (POSIX separators); anything else is kept."""
    if source.startswith("synthetic:"):
        return source
    try:
        return Path(source).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return source


def _relative_files(files: dict, base: Path) -> dict:
    """Copy of the ``files`` mapping with paths relative to ``base``."""
    def rel(p: str) -> str:
        try:
            return Path(p).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return p
    return {k: ({n: rel(v2) for n, v2 in v.items()} if isinstance(v, dict) else rel(v))
            for k, v in files.items()}


def _parse_list(value: str | Sequence, cast=str) -> list:
    if isinstance(value, str):
        return [cast(v.strip()) for v in value.split(",") if v.strip()]
    return [cast(v) for v in value]


def run_detection(out_dir: str | os.PathLike, *, covers: Sequence | None = None,
                  rates: Sequence[float] | None = None,
                  payloads: Sequence[str] = DEFAULT_PAYLOADS,
                  modes: Sequence[str] = MODES, size: int | None = None,
                  trials: int = 1, seed: int = DEFAULT_SEED, quick: bool = False,
                  keep_images: bool = False, plots: bool = True,
                  tag: str | None = None, log: Log = print) -> dict[str, Any]:
    """Run the detection experiment and write its outputs into ``out_dir``.

    For every cover, rate, payload variant and trial: embed, run every
    registered detector (detectors.run_all) and combine_results(). Rate 0 is
    the untouched cover; it is analysed once per cover and shared by all
    variants so every curve starts from the same clean baseline.

    Args:
        out_dir: Directory for the outputs (created). Nothing is written
            anywhere else; temporary stego files go to a temp folder unless
            ``keep_images`` is set.
        covers: Paths, ``synthetic:<kind>`` specs or Cover objects. None uses
            default_cover_specs(quick).
        rates: Nominal embedding rates in [0, 1]. None uses DEFAULT_RATES, or
            QUICK_RATES with ``quick``.
        payloads: Any of PAYLOADS.
        modes: Any of MODES.
        size: Side length of synthetic covers (default 512, or 128 with
            ``quick``).
        trials: Embeddings per nonzero rate, each with fresh random payload
            bits.
        seed: Seed for synthetic covers and payload bits. The encrypted
            variant also draws a random salt and nonce inside
            payload.pack(), so its rows differ slightly between runs.
        quick: Small, fast defaults (for tests and a first look).
        keep_images: Save every stego image under ``out_dir/stego/``.
        plots: Write the figures.
        tag: Label stored in the summary (defaults to the folder name).
        log: Function used for progress messages.

    Returns:
        dict: The JSON summary. In the returned dict ``"files"`` holds
        absolute paths; in detection_summary.json it holds names relative
        to ``out_dir`` (and cover sources inside the repository are stored
        relative to it), so the file can be committed and moved.

    Raises:
        ValueError: For an unknown payload or mode, a rate outside [0, 1],
            or trials below 1.
    """
    t0 = time.time()
    out = Path(out_dir)
    rates = [float(r) for r in (rates if rates is not None else (QUICK_RATES if quick else DEFAULT_RATES))]
    if any(not 0.0 <= r <= 1.0 for r in rates):
        raise ValueError(f"rates must be in [0, 1], got {rates}")
    rates = sorted(set(rates))
    payloads = list(dict.fromkeys(payloads))
    modes = list(dict.fromkeys(modes))
    for p in payloads:
        if p not in PAYLOADS:
            raise ValueError(f"unknown payload {p!r}; choose from {PAYLOADS}")
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown mode {m!r}; choose from {MODES}")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    size = size or (QUICK_SIZE if quick else DEFAULT_SIZE)
    cover_list = load_covers(covers if covers is not None else default_cover_specs(quick),
                             size=size, seed=seed)
    variants = [_variant(p, m) for p in payloads for m in modes]
    det_info = detectors.list_detectors()
    det_names = [d["name"] for d in det_info]
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="make_all_") as tmp:
        work = Path(tmp)
        img_dir = out / "stego" if keep_images else work
        img_dir.mkdir(parents=True, exist_ok=True)
        for ci, cov in enumerate(cover_list):
            tc = time.time()
            cover_png = work / f"cover_{ci}.png"
            Image.fromarray(cov.pixels).save(cover_png)
            h, w = cov.pixels.shape[:2]
            n_samples = cov.rgb.size
            clean = detectors.run_all(cov.rgb)
            for ri, rate in enumerate(rates):
                for pi, payload in enumerate(payloads):
                    for mi, mode in enumerate(modes):
                        for trial in range(trials if rate > 0 else 1):
                            base = {"cover": cov.name, "width": w, "height": h,
                                    "payload": payload, "mode": mode,
                                    "variant": _variant(payload, mode),
                                    "nominal_rate": rate, "trial": trial}
                            if rate == 0:
                                results = clean
                                info = {"payload_bits": 0, "true_rate": 0.0}
                                changed = 0.0
                            else:
                                rng = np.random.default_rng(
                                    np.random.SeedSequence([seed, ci, ri, pi, mi, trial]))
                                stem = f"{cov.name}_{payload}-{mode}_r{rate:g}_t{trial}"
                                made = _embed(cov, cover_png, payload, mode, rate, rng,
                                              img_dir if keep_images or payload == "encrypted" else None,
                                              stem)
                                if made is None:
                                    skipped.append({**base, "reason": "payload overhead exceeds rate"})
                                    continue
                                stego_rgb, info = made
                                changed = float(np.count_nonzero(stego_rgb != cov.rgb)) / n_samples
                                results = detectors.run_all(stego_rgb)
                            base.update(true_rate=info["true_rate"],
                                        payload_bits=info["payload_bits"],
                                        changed_fraction=changed)
                            rows.extend(_result_rows(base, results))
            log(f"[detection] {cov.name} ({w}x{h}): {time.time() - tc:.1f}s")

    curves = aggregate(rows)
    heads = headline(curves, rows)

    files: dict[str, str] = {}
    files["csv"] = str(write_csv(rows, out / "detection.csv"))
    tex = latex_table(curves, heads, variants, det_names, rates, len(cover_list))
    (out / "detection_table.tex").write_text(tex, encoding="utf-8")
    files["latex"] = str(out / "detection_table.tex")
    figures: dict[str, str] = {}
    if plots:
        for det in det_names + [OVERALL]:
            p = plot_detector(curves, det, variants, out / f"detection_{det}.png",
                              len(cover_list))
            figures[det] = str(p)
        figures["combined"] = str(plot_combined(curves, det_names, variants,
                                                out / "detection_combined.png",
                                                len(cover_list)))
    files["figures"] = figures  # type: ignore[assignment]
    files["json"] = str(out / "detection_summary.json")

    summary = {
        "experiment": "detection",
        "tag": tag or out.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_s": round(time.time() - t0, 2),
        "config": {
            "covers": [{"name": c.name, "source": _portable(c.source),
                        "width": int(c.pixels.shape[1]), "height": int(c.pixels.shape[0])}
                       for c in cover_list],
            "rates": rates, "payloads": payloads, "modes": modes,
            "variants": variants, "trials": trials, "seed": seed, "quick": quick,
            "synthetic_size": size, "bits_per_sample": 1,
            "rate_definition": "share of R, G, B samples whose LSB carries a payload bit",
            "scatter": "carriers.base.scatter_order with a fixed key (the password, "
                       "for the encrypted variant)",
            "encrypted_note": "stego.hide() with a password: AES-256-GCM body behind a "
                              "14-byte STGA header; salt and nonce are random per run",
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "thresholds": {"suspicious": detectors.SUSPICIOUS_AT, "likely_stego": detectors.LIKELY_AT},
        "detectors": det_info,
        "estimate_keys": ESTIMATE_KEYS,
        "headline": heads,
        "curves": curves,
        "skipped": skipped,
        "files": files,
    }
    on_disk = {**summary, "files": _relative_files(files, out)}
    with open(out / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(on_disk, f, indent=2, allow_nan=False)
    log(f"[detection] {len(rows)} rows, {len(skipped)} skipped, "
        f"{summary['runtime_s']}s -> {out}")
    return summary


# ===========================================================================
# Command line
# ===========================================================================

SUBCOMMANDS = ("classic", "detection")


def _add_classic_arguments(c: argparse.ArgumentParser) -> list[argparse.Action]:
    """Add the ``classic`` subcommand's arguments to a parser.

    Shared by build_parser() and _normalize_argv(), so the old-form detection
    always knows exactly which classic options take a value.

    Args:
        c: Parser to add the arguments to.

    Returns:
        list[argparse.Action]: The actions that were added.
    """
    return [
        c.add_argument("cover", help="Path to the cover image"),
        c.add_argument("--out", default=str(REPO_ROOT / "results"),
                       help="Output directory (default: results/ in the repository)"),
        c.add_argument("--resize", type=int, default=500,
                       help="Side length of the square cover for case B (default 500)"),
        c.add_argument("--channel", default="b", choices=["r", "g", "b"],
                       help="Channel analysed (default b)"),
        c.add_argument("--short", default=DEFAULT_SHORT, help="Message for case A"),
        c.add_argument("--long-multiplier", type=int, default=5000,
                       help="Repeats of the message for case B (case C uses 10x)"),
    ]


def _classic_options() -> tuple[set[str], set[str]]:
    """Return ``(all_options, value_options)`` for the classic subcommand.

    ``all_options`` holds every option string (``-h`` and ``--help``
    included); ``value_options`` the ones that consume the next argument.
    """
    probe = argparse.ArgumentParser(add_help=False)
    actions = _add_classic_arguments(probe)
    every = {"-h", "--help"}
    takes_value: set[str] = set()
    for action in actions:
        every.update(action.option_strings)
        if action.option_strings and action.nargs != 0:
            takes_value.update(action.option_strings)
    return every, takes_value


def _first_positional(argv: Sequence[str]) -> int | None:
    """Index of the first positional argument in a classic-style command line.

    Options are skipped together with their value (``--out DIR``; the
    ``--out=DIR`` form has none to skip). Unique prefixes of long options
    count, as argparse accepts them (``--res 64`` is ``--resize 64``).
    Everything after a bare ``--`` is positional.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        int | None: Index into ``argv``, or None if there is no positional
        argument.
    """
    every, takes_value = _classic_options()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            return i + 1 if i + 1 < len(argv) else None
        if not tok.startswith("-") or tok == "-":
            return i
        i += 1
        if "=" in tok:
            continue  # --opt=value carries its own value
        if tok.startswith("--"):
            matches = [o for o in every if o.startswith(tok)]
            name = tok if tok in every else (matches[0] if len(matches) == 1 else None)
        else:
            name = tok
        if name in takes_value:
            i += 1  # skip the option's value
    return None


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Keep the old flat ``make_all.py [options] COVER [options]`` form working.

    The original script had no subcommands: one positional cover path plus
    options, in any order. If the first positional argument (skipping
    options and their values) is not a subcommand name, the command line is
    in that old form and ``classic`` is inserted in front of it. This also
    covers options before the cover (``--out X --resize 64 cover.png``) and
    option values that happen to equal a subcommand name
    (``--short classic cover.png``).

    An empty command line and a bare ``-h`` / ``--help`` are left alone so
    argparse shows the top-level usage with both subcommands. Options with
    no cover at all (``--out X``) become ``classic --out X``, so the error
    names the missing cover, as the original script's did.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        list[str]: Arguments for build_parser().parse_args().
    """
    argv = list(argv)
    if not argv or argv in (["-h"], ["--help"]):
        return argv
    i = _first_positional(argv)
    if i is not None and argv[i] in SUBCOMMANDS:
        return argv
    return ["classic", *argv]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the ``classic`` and ``detection`` subcommands."""
    ap = argparse.ArgumentParser(
        prog="make_all.py",
        description="Experiment runner: classic histogram cases and detector detection curves.")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("classic", help="Original three-case histogram experiment on one cover.")
    _add_classic_arguments(c)

    d = sub.add_parser("detection", help="Detection curves for every registered detector.")
    d.add_argument("covers", nargs="*",
                   help="Cover image paths or synthetic:{natural,stretched,gradient} "
                        "(default: test-images/smoothBlue.png if present, "
                        "synthetic:natural, synthetic:stretched)")
    d.add_argument("--out", default=str(REPO_ROOT / "results" / "experiments"),
                   help="Parent folder; outputs go to OUT/TAG (default: results/experiments)")
    d.add_argument("--tag", default=None, help="Run name (default: detection-<timestamp>)")
    d.add_argument("--rates", default=None,
                   help="Comma-separated rates in [0, 1] (default: "
                        + ",".join(f"{r:g}" for r in DEFAULT_RATES) + ")")
    d.add_argument("--payloads", default=",".join(DEFAULT_PAYLOADS),
                   help=f"Comma-separated subset of {','.join(PAYLOADS)} (default: random,encrypted)")
    d.add_argument("--modes", default=",".join(MODES),
                   help="Comma-separated subset of sequential,scattered (default: both)")
    d.add_argument("--size", type=int, default=None,
                   help=f"Synthetic cover size in pixels (default {DEFAULT_SIZE}, quick {QUICK_SIZE})")
    d.add_argument("--trials", type=int, default=1, help="Embeddings per nonzero rate (default 1)")
    d.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default {DEFAULT_SEED})")
    d.add_argument("--quick", action="store_true",
                   help=f"Fast run: synthetic covers only, {QUICK_SIZE} px, rates "
                        + ",".join(f"{r:g}" for r in QUICK_RATES))
    d.add_argument("--keep-images", action="store_true", help="Save every stego image under OUT/TAG/stego/")
    d.add_argument("--no-plots", action="store_true", help="Skip the figures")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. See the module docstring for usage.

    Returns:
        int: Process exit status (0 on success, 1 on a handled error).
    """
    # The runner is headless: pick the non-interactive backend before
    # steganalysis imports pyplot. Library callers keep their own backend.
    import matplotlib
    matplotlib.use("Agg")

    args = build_parser().parse_args(_normalize_argv(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "classic":
            res = run_classic(args.cover, args.out, resize=args.resize, channel=args.channel,
                              short=args.short, long_multiplier=args.long_multiplier)
            print("\nAll artifacts written under:", res["out_dir"])
            print("Run tags:\n-", "\n- ".join(res["tags"]))
        else:
            tag = args.tag or f"detection-{ts_tag()}"
            out_dir = Path(args.out) / tag
            summary = run_detection(
                out_dir, covers=args.covers or None,
                rates=_parse_list(args.rates, float) if args.rates else None,
                payloads=_parse_list(args.payloads), modes=_parse_list(args.modes),
                size=args.size, trials=args.trials, seed=args.seed, quick=args.quick,
                keep_images=args.keep_images, plots=not args.no_plots, tag=tag)
            print("Outputs:")
            for k, v in summary["files"].items():
                if isinstance(v, dict):
                    for name, p in v.items():
                        print(f"  figure {name}: {p}")
                else:
                    print(f"  {k}: {v}")
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
