#!/usr/bin/env python3
# pairwise_stats.py — compute even/odd pair statistics used to detect LSB embedding.
"""Even/odd pair statistics for detecting LSB steganography in one color channel.

For a cover image and its stego (encoded) version, this module builds the
256-bin histogram h of one channel (blue by default) and computes the pair
differences

    Delta_k = h[2k] - h[2k+1],   k = 0..127

then summarizes them with the L1 norm, the L2 norm, the median |Delta_k|, the
largest |Delta_k|, and the pair index k where that maximum occurs.

Why even/odd pairs: the legacy embedder (make_all.py's classic mode,
embed_bits_rgb, which produced the images in results/) writes the message bits
followed by a 64-zero-bit delimiter, one bit into the least significant bit
of each R, G, B byte in raster order from the top-left pixel (3 bits per
pixel), and stops as soon as that stream ends. Pixels past the end are left
unchanged. (encoder.py now writes an STGA frame instead of the delimiter
format, in the same R, G, B LSB order, so the statistics behave the same
way.) Replacing the LSB can only move a value between 2k and 2k+1, never
into a different pair. If the payload bits are close to uniform, each pair's
two counts drift toward each other over the changed pixels, so |Delta_k| (and
its norms) tends to shrink at high embedding rates. A short message changes
too few pixels to matter (the full-low run in results/ moves L1 by only +6).
Biased payloads also break the effect: plain ASCII text such as make_all.py's
default message is about 46% one-bits, so a changed pixel lands on the even
value of its pair about 54% of the time instead of 50%, and |Delta_k| need not
shrink (both high-rate runs in results/ show L1 rising).

How it fits with the other modules:
    encoder.py / decoder.py  LSB embed and extract.
    main.py                  CLI front end for encode/decode.
    steganalysis.py          Plots of the standard and pairwise histograms.
    pairwise_stats.py        This module: the numbers (JSON + LaTeX table).
    make_all.py              Experiment runner; its classic mode imports
                             chan_hist, pair_stats, latex_table and
                             default_caption for each cover/stego pair.
    gui.py                   Tkinter app (does not call this module).

Usage (from the repo root):
    python3 pairwise_stats.py COVER STEGO [-c b] [--json OUT.json]
        [--latex OUT.tex] [--caption TEXT] [--tag TAG]

    e.g. python3 pairwise_stats.py cover.png stego.png -c b \\
             --json results/run_pairwise.json --latex results/run_pairwise.tex

On success the JSON result is always printed to stdout. --json also writes that
JSON object to a file, and --latex writes a LaTeX table of the same numbers;
both create missing parent directories.
"""

import argparse, os, sys, json
import numpy as np
from PIL import Image

def parse_channel(s: str) -> int:
    """Convert a channel name or index to an RGB channel index.

    Used as the argparse ``type`` for ``-c/--channel``; argparse also applies
    it to the string default ``"b"``.

    Args:
        s: One of r/red/g/green/b/blue or 0/1/2 (case and surrounding
            whitespace ignored).

    Returns:
        0 for red, 1 for green, 2 for blue.

    Raises:
        argparse.ArgumentTypeError: If ``s`` is not a recognized channel.
    """
    table = {"r":0,"red":0,"g":1,"green":1,"b":2,"blue":2,"0":0,"1":1,"2":2}
    s = str(s).strip().lower()
    if s not in table:
        raise argparse.ArgumentTypeError("channel must be r/g/b or 0/1/2")
    return table[s]

def chan_hist(path: str, ch: int = 2) -> tuple[np.ndarray, tuple[int,int]]:
    """Load an image and return the 256-bin histogram of one channel.

    The image is converted to RGB first, so 8-bit palette, grayscale, and
    RGBA inputs are all handled (alpha is dropped, not composited). 16-bit
    images are clipped to 0..255 by PIL, not rescaled.

    Args:
        path: Path to the image file.
        ch: Channel index (0=R, 1=G, 2=B). Defaults to blue. Any other
            value raises IndexError.

    Returns:
        A tuple ``(h, size)`` where ``h`` is an int64 array of length 256
        (``h[v]`` = number of pixels whose channel value is ``v``) and
        ``size`` is ``(width, height)`` from PIL.

    Raises:
        SystemExit: If ``path`` does not exist (raised through ``sys.exit``
            even when called as a library function; if nothing catches it,
            Python prints the message to stderr and exits with status 1).
        PIL.UnidentifiedImageError, OSError: If the file cannot be read as
            an image (not caught here).
    """
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    img = Image.open(path).convert("RGB")
    # Take one channel of the H x W x 3 array and flatten it to a 1-D array of
    # pixel values.
    arr = np.asarray(img, dtype=np.uint8)[:, :, ch].ravel()
    # minlength=256 guarantees a bin for every value 0..255 even if unused.
    # bincount already returns a signed integer array (platform intp, int64 on
    # 64-bit builds); the cast only pins the dtype to int64 everywhere.
    h = np.bincount(arr, minlength=256).astype(np.int64)
    return h, img.size  # (W,H)

def pair_stats(h: np.ndarray) -> dict:
    """Summarize the even/odd pair differences of a 256-bin histogram.

    Each pair (2k, 2k+1) holds two values that differ only in the LSB, so LSB
    replacement moves pixels within a pair but never between pairs.

    Args:
        h: Signed integer histogram of length 256 (as returned by
            ``chan_hist``). An unsigned array would wrap on subtraction.

    Returns:
        A dict with:
            L1: Sum of |Delta_k| (int).
            L2: Euclidean norm of Delta (float).
            median_abs: Median of |Delta_k| over the 128 pairs (float; can
                end in .5 because the count of pairs is even).
            max_abs: Largest |Delta_k| (int).
            max_k: Pair index k of that maximum, i.e. the pair of values
                (2k, 2k+1), not a pixel value. Ties go to the smallest k.
    """
    # Even bins minus odd bins: d[k] = h[2k] - h[2k+1], 128 entries.
    d = h[0::2] - h[1::2]
    a = np.abs(d)
    return {
        "L1": int(a.sum()),
        # Square in float64 so large counts cannot overflow.
        "L2": float(np.sqrt((d.astype(np.float64)**2).sum())),
        "median_abs": float(np.median(a)),
        "max_abs": int(a.max()),
        "max_k": int(a.argmax())
    }

CHANNEL_NAMES = ("red", "green", "blue")

def default_caption(ch: int) -> str:
    """Return the default LaTeX caption for a channel index.

    Args:
        ch: Channel index (0=R, 1=G, 2=B), as returned by ``parse_channel``.

    Returns:
        ``"Pairwise even/odd statistics (<name> channel)."`` with the name of
        the channel that was actually analyzed.
    """
    return f"Pairwise even/odd statistics ({CHANNEL_NAMES[ch]} channel)."

def latex_table(Sc: dict, Ss: dict, caption: str) -> str:
    """Format cover and stego pair statistics as a LaTeX table.

    Pure function: returns the text and writes nothing. The table uses
    ``\\toprule``/``\\midrule``/``\\bottomrule`` and the ``[H]`` float
    placement, so the document needs the ``booktabs`` and ``float`` packages.

    Args:
        Sc: ``pair_stats`` result for the cover (original) image.
        Ss: ``pair_stats`` result for the stego (encoded) image.
        caption: Table caption, inserted verbatim (LaTeX special characters
            such as ``_``, ``%`` or ``&`` are not escaped).

    Returns:
        The LaTeX source of the table, with rows for Original, Encoded, and
        the change (Encoded minus Original) in L1, L2 and median |Delta_k|.
    """
    # The "Change (E-O)" row is Encoded minus Original, so a positive value
    # means that statistic grew in the stego image (e.g. Encoded L1 36924 vs
    # Original 19786 prints +17138). This is the opposite sign of the JSON
    # "*_drop" fields, which are Original minus Encoded. The signs can
    # disagree across statistics (full-high in results/: L1 rises while L2
    # falls).
    L1_change = Ss["L1"] - Sc["L1"]
    L2_change = Ss["L2"] - Sc["L2"]
    Med_change = Ss["median_abs"] - Sc["median_abs"]
    return r"""
\begin{table}[H]\centering
\caption{%s}
\begin{tabular}{lrrrrr}
\toprule
Image & $\|\Delta\|_{1}$ & $\|\Delta\|_{2}$ & Median $|\Delta_k|$ & $\max |\Delta_k|$ & $k_{\max}$ \\
\midrule
Original & %d & %.3f & %.3f & %d & %d \\
Encoded  & %d & %.3f & %.3f & %d & %d \\
\midrule
Change (E$-$O) & %+d & %+0.3f & %+0.3f & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
""" % (caption,
       Sc["L1"], Sc["L2"], Sc["median_abs"], Sc["max_abs"], Sc["max_k"],
       Ss["L1"], Ss["L2"], Ss["median_abs"], Ss["max_abs"], Ss["max_k"],
       L1_change, L2_change, Med_change)

if __name__ == "__main__":
    # CLI entry point. Side effects: prints the JSON result to stdout, and
    # optionally writes it to --json and a LaTeX table to --latex. Exits with
    # an error message if an input file is missing or the image sizes differ.
    p = argparse.ArgumentParser(description="Compute pairwise even/odd stats (Δ_k) for LSB steganalysis.")
    p.add_argument("cover", help="Path to cover/original image (PNG/BMP/TIF)")
    p.add_argument("stego", help="Path to stego/encoded image (same size)")
    p.add_argument("-c","--channel", type=parse_channel, default="b",
                   help="Channel r/g/b or 0/1/2 (default: b)")
    p.add_argument("--json", help="Write results to JSON")
    p.add_argument("--latex", help="Write LaTeX table to this path")
    # The default caption is filled in after parsing, so it names the channel
    # that -c actually selected.
    p.add_argument("--caption", default=None,
                   help="Caption for LaTeX table (default: "
                        "'Pairwise even/odd statistics (<channel> channel).')")
    p.add_argument("--tag", default=None, help="Optional tag echoed into JSON object")
    args = p.parse_args()
    if args.caption is None:
        args.caption = default_caption(args.channel)

    Hc, size_c = chan_hist(args.cover, args.channel)
    Hs, size_s = chan_hist(args.stego, args.channel)
    # Only the dimensions are compared; the pixels need not correspond.
    if size_c != size_s:
        sys.exit(f"ERROR: size mismatch: {args.cover}={size_c}, {args.stego}={size_s}")

    Sc, Ss = pair_stats(Hc), pair_stats(Hs)
    out = {
        "tag": args.tag,
        "channel": ["R","G","B"][args.channel],
        "cover": {"file": os.path.basename(args.cover), **Sc},
        "stego": {"file": os.path.basename(args.stego), **Ss},
        # Each "*_drop" is cover minus stego: positive means that statistic is
        # smaller for the stego image (L1, L2 and median can disagree).
        "delta": {
            "L1_drop": Sc["L1"] - Ss["L1"],
            "L2_drop": Sc["L2"] - Ss["L2"],
            "median_abs_drop": Sc["median_abs"] - Ss["median_abs"]
        }
    }
    print(json.dumps(out, indent=2))

    if args.json:
        # dirname is "" for a bare filename; fall back to the current dir.
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f: json.dump(out, f, indent=2)
    if args.latex:
        os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
        with open(args.latex, "w") as f: f.write(latex_table(Sc, Ss, args.caption))
