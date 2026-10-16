#!/usr/bin/env python3
# steganalysis.py — unified API for GUI + CLI + make_all.py
"""Histogram-based steganalysis plots for LSB steganography.

Compares a cover (original) image with a stego (encoded) image on one color
channel and saves matplotlib figures of two kinds:

* Standard histogram: the pixel count h(v) for each channel value v = 0..255.
* Pairwise-difference histogram: for each even/odd value pair (2k, 2k+1),
  k = 0..127, the difference Delta_k = h(2k) - h(2k+1), shown as a histogram
  of those 128 differences.

Why the even/odd pairs matter: LSB embedding sets a channel value to
``(x & 0b11111110) | bit``, which clears the lowest bit and then writes the
message bit. That can only move a value between 2k and 2k+1, never into
another pair. If a fraction p of a channel's values are overwritten and a
fraction q of the message bits are 1, each overwritten value lands on 2k+1
with probability q and on 2k otherwise, so in expectation

    Delta_k -> (1 - p) * Delta_k + p * (1 - 2q) * (h(2k) + h(2k+1)).

Only with equiprobable bits (q = 1/2, e.g. an encrypted or compressed
payload) does this reduce to (1 - p) * Delta_k, pulling each pair's counts
together. That is the pairs-of-values assumption behind the
Westfeld-Pfitzmann chi-square attack. Even then a fully embedded channel's
Delta_k do not sit at 0: they scatter with a standard deviation of about
sqrt(h(2k) + h(2k+1)).

The payloads in this project are ASCII/UTF-8 text, whose bits are not 50/50
(the top bit of every ASCII byte is 0). For make_all.py's default message q
is about 0.46, so a saturated embed pushes every Delta_k toward
(1 - 2q) * (h(2k) + h(2k+1)) > 0. The stego pairwise histogram therefore
shifts to the positive side instead of collapsing onto 0. In the repo's own
blue-channel results (results/*_pairwise.json), median |Delta_k| rose from 28
to 121 (resized high-rate) and from 60 to 190.5 (full-size high-rate). The
largest imbalance, at k = 0 (values 0 and 1), was pulled in: max |Delta_k|
fell from 8605 to 1203 in the full-size high-rate case. At low embedding
rates few values change, so cover and stego plots look nearly identical.

How it fits with the other modules:

* encoder.py / stego.py: the LSB embedder. With 1 bit per sample and no
  password, each RGB pixel carries 3 payload bits (one per channel, in
  raster order), so an image has 3 * width * height LSB slots. The payload
  is an STGA frame (payload.py): a 14-byte header (magic, version, flags,
  body length, CRC-32) followed by the body, so the header takes 112 of
  those slots and stego.capacity_bytes() is ``slots // 8 - 14``. Images
  written by the original encoder (text followed by a 64-zero-bit
  delimiter) are still read by decoder.py's legacy fallback.
* main.py: command-line front end. ``main.py compare`` calls
  ``analyze_comparison`` with its ``--channel`` and then prints the
  pairwise_stats.py numbers for the same channel.
* steganalysis.py (this file): the plots. ``analyze`` serves make_all.py,
  whose ``run_steganalysis`` imports this module and calls it in-process.
  ``analyze_comparison`` serves gui.py's Compare tab and ``main.py
  compare``. Its own CLI (below) is a convenience that neither caller uses.
* pairwise_stats.py: the numeric counterpart for the same Delta_k values. It
  reports the L1 and L2 norms of Delta, the median and max of |Delta_k| and
  the k of that max. It always prints JSON to stdout and writes a JSON file
  or LaTeX table only with ``--json`` or ``--latex``.
* make_all.py: experiment runner. Its ``classic`` experiment has three cases
  (full-size low-rate, resized high-rate, full-size high-rate). It embeds
  UTF-8 text plus the legacy 64-zero-bit delimiter with its own NumPy
  embedder (it does not import encoder.py), then calls ``analyze`` and
  pairwise_stats.py for each case. Everything runs in-process, so it works
  from any working directory.
* gui.py: Tkinter app. Its Compare tab calls ``analyze_comparison`` in a
  background thread with the channel chosen in its Red/Green/Blue box.

Usage (from the repository root)::

    # Five per-run figures named {tag}_{channel}_*.png (make_all.py style).
    python steganalysis.py cover.png stego.png --out results --tag run1 --channel b

    # One composite at a chosen path plus four fixed-name plots beside it
    # (GUI style).
    python steganalysis.py cover.png stego.png -o results/comparison_analysis.png

``--out`` and ``-o`` can be combined in one call. With neither, nothing is
written.
"""

import argparse, os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Channel letter -> index along the last axis of an (H, W, 3) RGB array.
CHANNELS = {"r":0, "g":1, "b":2}
# Channel index -> name used in plot titles.
CHANNEL_NAMES = ("Red", "Green", "Blue")

def _load_rgb(path):
    """Load an image as an RGB uint8 array.

    Args:
        path: Path to any image file Pillow can open.

    Returns:
        numpy.ndarray of shape (height, width, 3) and dtype uint8. Alpha,
        grayscale and palette images are converted to plain RGB first.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def _std_hist(ax, chan_data, title):
    """Draw a 256-bin histogram of one channel's values onto ``ax``.

    Args:
        ax: Matplotlib Axes to draw on.
        chan_data: 1-D array of channel values in 0..255.
        title: Axes title.

    Returns:
        None. Only modifies ``ax``.
    """
    # bins=256 over [0, 256) gives one unit-wide bin [v, v+1) per integer value v.
    ax.hist(chan_data, bins=256, range=(0,256))
    ax.set_title(title)
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Frequency (Number of Pixels)")
    # NOTE: the bar for value 255 spans [255, 256] and starts exactly at this
    # right limit, so it is clipped to zero width and the count of saturated
    # (255) values is never drawn.
    ax.set_xlim(0,255)

def _pairwise_hist(ax, chan_data, title, zoom=True, annotate_max=True):
    """Draw the even/odd pair-difference histogram for one channel onto ``ax``.

    Computes Delta_k = h(2k) - h(2k+1) for k = 0..127, where h is the 256-bin
    value histogram, then plots a 50-bin histogram of those 128 differences.
    Embedding with equiprobable message bits pulls each pair's two counts
    together. The text payloads used here are biased toward 0 bits, so a
    heavily embedded image's differences shift to the positive side instead
    (see the module docstring).

    Args:
        ax: Matplotlib Axes to draw on.
        chan_data: 1-D array of channel values in 0..255.
        title: Axes title.
        zoom: If True, crop the x-axis to +/- the 98th percentile of
            |Delta_k|. Bars outside that window are cut off. The 50 bins are
            still spread over the full min..max range, so one large outlier
            leaves only a few wide bins inside the window. Each call picks its
            own window, so cover and stego panels usually have different x
            scales.
        annotate_max: If True, write max|Delta_k| in the top-right corner. This
            is the true maximum, even when ``zoom`` hides that bar.

    Returns:
        numpy.ndarray of the 128 differences Delta_k (dtype float).
    """
    hist, _ = np.histogram(chan_data, bins=256, range=(0,256))
    # Delta_k = h(2k) - h(2k+1) for pairs (0,1), (2,3), ..., (254,255): 128 values.
    # Rewriting an LSB moves a value only within its pair, so embedding changes
    # only these within-pair balances (toward 0 for every pair only if message
    # bits are 50/50).
    diffs = np.array([float(hist[i])-float(hist[i+1]) for i in range(0,255,2)], dtype=float)
    if zoom and len(diffs)>0:
        # auto-zoom to central mass for readability
        lo, hi = np.percentile(np.abs(diffs), [2,98])
        # NOTE: lo is never used after this line, so the max(1.0, hi) floor has
        # no effect. When hi is 0 (at least 126 of the 128 pairs equal, e.g. a
        # single-color channel), set_xlim(-0.0, 0.0) runs; matplotlib warns
        # about identical limits and widens the axis to only +/-0.05.
        lo = -max(1.0, hi)
        # Symmetric window [-hi, hi]. set_xlim also turns off x autoscaling, so
        # the hist() call below keeps this range.
        ax.set_xlim(-abs(hi), abs(hi))
    # NOTE: the 50 bins span the full min..max of diffs, not the zoom window, so
    # zoom only crops. With one large outlier (k = 0 in every run in results/)
    # the window holds a few wide bins, e.g. 5 bins of width ~180 in the
    # full-size low-rate run.
    ax.hist(diffs, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Difference between Adjacent Color Pair Counts")
    ax.set_ylabel("Frequency of that Difference")
    if annotate_max and diffs.size>0:
        # Axes-fraction coordinates: (0.98, 0.90) is near the top-right corner.
        ax.text(0.98, 0.90, f"max|Δ|={int(np.max(np.abs(diffs)))}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    return diffs

def analyze(cover_path:str, stego_path:str, out_dir:str=".", tag:str="", channel:str="b"):
    """Save cover-vs-stego histogram figures for one channel (make_all.py path).

    make_all.run_steganalysis imports this module and calls this function
    in-process, after rejecting channels other than r, g or b. The two images
    are not checked to be the same size (pairwise_stats.py does check).

    Args:
        cover_path: Path to the original (cover) image.
        stego_path: Path to the encoded (stego) image.
        out_dir: Directory for the figures. Created if missing.
        tag: Run tag used as the filename prefix. The default "" gives names
            that start with "_".
        channel: "r", "g" or "b". The channel lookup ignores case, but the
            filenames use the string as given. Unknown values fall back to
            blue. The CLI accepts only lowercase r, g or b.

    Returns:
        dict mapping "std_cover", "std_stego", "pair_cover", "pair_stego" and
        "composite" to the paths written.

    Side effects:
        Writes five 150-dpi PNGs, overwriting files with the same name:
          {out_dir}/{tag}_{channel}_std_cover.png
          {out_dir}/{tag}_{channel}_std_stego.png
          {out_dir}/{tag}_{channel}_pair_cover.png
          {out_dir}/{tag}_{channel}_pair_stego.png
          {out_dir}/{tag}_{channel}_comp.png  (same four plots redrawn as 2x2)
        Prints each path to stdout.

    Raises:
        FileNotFoundError: If either image path does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    # NOTE: an unknown channel silently analyzes blue (index 2), while the
    # filenames and the composite title still use that string. Only direct
    # callers can hit this; the CLI's choices= rejects unknown channels.
    ch = CHANNELS.get(channel.lower(), 2)

    cov = _load_rgb(cover_path); ste = _load_rgb(stego_path)
    # Keep one channel and flatten (H, W) to a 1-D array of values.
    cov_c = cov[:,:,ch].flatten(); ste_c = ste[:,:,ch].flatten()

    std_cov = os.path.join(out_dir, f"{tag}_{channel}_std_cover.png")
    std_ste = os.path.join(out_dir, f"{tag}_{channel}_std_stego.png")
    pr_cov  = os.path.join(out_dir, f"{tag}_{channel}_pair_cover.png")
    pr_ste  = os.path.join(out_dir, f"{tag}_{channel}_pair_stego.png")
    comp    = os.path.join(out_dir, f"{tag}_{channel}_comp.png")

    # individual plots
    fig, ax = plt.subplots(figsize=(6,4))
    _std_hist(ax, cov_c, "Original: Standard Histogram")
    fig.tight_layout(); fig.savefig(std_cov, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    _std_hist(ax, ste_c, "Encoded: Standard Histogram")
    fig.tight_layout(); fig.savefig(std_ste, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    _pairwise_hist(ax, cov_c, "Original: Pairwise Differences")
    fig.tight_layout(); fig.savefig(pr_cov, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    _pairwise_hist(ax, ste_c, "Encoded: Pairwise Differences")
    fig.tight_layout(); fig.savefig(pr_ste, dpi=150); plt.close(fig)

    # composite 2x2
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    _std_hist(axs[0,0], cov_c, "1. Original: Standard")
    _std_hist(axs[0,1], ste_c, "2. Encoded: Standard")
    _pairwise_hist(axs[1,0], cov_c, "3. Original: Pairwise")
    _pairwise_hist(axs[1,1], ste_c, "4. Encoded: Pairwise")
    fig.suptitle("Comprehensive Steganalysis (channel {})".format(channel.upper()))
    # rect leaves the top 4% of the figure free for the suptitle.
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(comp, dpi=150); plt.close(fig)

    print("std_cover:", std_cov)
    print("std_stego:", std_ste)
    print("pair_cover:", pr_cov)
    print("pair_stego:", pr_ste)
    print("composite:", comp)

    return {"std_cover":std_cov,"std_stego":std_ste,"pair_cover":pr_cov,"pair_stego":pr_ste,"composite":comp}

def analyze_comparison(original_image_path, encoded_image_path, output_path=None, channel="b"):
    """Save a 2x2 cover-vs-stego composite plus four single plots (GUI path).

    Called by gui.py's Compare tab and by ``main.py compare``, both with the
    channel the user chose. Also reachable through this file's CLI with
    ``-o``. The two images are not checked to be the same size.

    Args:
        original_image_path: Path to the original (cover) image.
        encoded_image_path: Path to the encoded (stego) image.
        output_path: Path for the composite image. If None, uses
            "comparison_analysis.png" in the encoded image's directory.
        channel: "r", "g" or "b" (case-insensitive). Unknown values fall back
            to blue. The composite's suptitle and all four single-plot titles
            name the channel actually analyzed, e.g. "(Red Channel)", so a
            fallback is labeled "(Blue Channel)".

    Returns:
        dict mapping "composite", "p1", "p2", "p3" and "p4" to the paths
        written.

    Side effects:
        Creates the output directory if needed and writes five 150-dpi
        figures: the composite at ``output_path`` (matplotlib picks the format
        from its extension, so a .png path gives a PNG) and, in the same
        directory, these four PNGs:
          1_original_standard_hist.png
          2_encoded_standard_hist.png
          3_original_difference_analysis.png
          4_encoded_difference_analysis.png
        The four fixed names are overwritten on every call. Prints nothing.

    Raises:
        FileNotFoundError: If either image path does not exist.
    """
    ch = CHANNELS.get(channel.lower(), 2)
    # Titles are derived from the resolved index, not the string passed in,
    # so they always name the channel whose data is plotted.
    label = f"({CHANNEL_NAMES[ch]} Channel)"
    cov = _load_rgb(original_image_path)
    ste = _load_rgb(encoded_image_path)
    # Keep one channel and flatten (H, W) to a 1-D array of values.
    cov_c = cov[:,:,ch].flatten(); ste_c = ste[:,:,ch].flatten()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(encoded_image_path) or ".", "comparison_analysis.png")
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # composite
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    _std_hist(axs[0,0], cov_c, "1. Original Image: Standard Histogram")
    _std_hist(axs[0,1], ste_c, "2. Encoded Image: Standard Histogram")
    _pairwise_hist(axs[1,0], cov_c, "3. Original Image: Difference Analysis")
    _pairwise_hist(axs[1,1], ste_c, "4. Encoded Image: Difference Analysis")
    fig.suptitle(f"Comprehensive Steganalysis: Before vs. After {label}")
    # rect leaves the top 5% of the figure free for the suptitle.
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(output_path, dpi=150); plt.close(fig)

    # four individual plots with fixed names
    p1 = os.path.join(out_dir, "1_original_standard_hist.png")
    p2 = os.path.join(out_dir, "2_encoded_standard_hist.png")
    p3 = os.path.join(out_dir, "3_original_difference_analysis.png")
    p4 = os.path.join(out_dir, "4_encoded_difference_analysis.png")

    fig, ax = plt.subplots(figsize=(12, 8)); _std_hist(ax, cov_c, f"Original Image: Standard Histogram {label}")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _std_hist(ax, ste_c, f"Encoded Image: Standard Histogram {label}")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _pairwise_hist(ax, cov_c, f"Original Image: Difference Analysis {label}")
    fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _pairwise_hist(ax, ste_c, f"Encoded Image: Difference Analysis {label}")
    fig.tight_layout(); fig.savefig(p4, dpi=150); plt.close(fig)

    return {"composite":output_path, "p1":p1, "p2":p2, "p3":p3, "p4":p4}

def _main():
    """Command-line entry point. See the module docstring for usage.

    Parses ``cover``, ``stego``, ``--out``, ``--tag``, ``--channel`` and
    ``-o/--output``, then runs ``analyze`` if ``--out`` is set and
    ``analyze_comparison`` if ``-o`` is set (both run if both are set).
    ``--tag`` defaults to "run" and ``--channel`` to "b". ``--channel``
    accepts only lowercase r, g or b.

    Side effects:
        Writes the figures described in those two functions. ``analyze`` also
        prints the paths it wrote. argparse prints usage and exits with status
        2 on invalid arguments.
    """
    ap = argparse.ArgumentParser(description="Steganalysis plots (standard + pairwise).")
    ap.add_argument("cover", help="Path to cover/original image")
    ap.add_argument("stego", help="Path to stego/encoded image")
    ap.add_argument("--out", default=None, help="Output directory for 5 files (for make_all.py).")
    ap.add_argument("--tag", default="run", help="Run tag used in filenames (for make_all.py).")
    ap.add_argument("--channel", default="b", choices=list(CHANNELS.keys()))
    ap.add_argument("-o","--output", default=None, help="Optional single composite path (for GUI-like use).")
    args = ap.parse_args()

    # NOTE: with neither --out nor -o, nothing runs and the script exits 0
    # without writing or printing anything.
    if args.out:
        analyze(args.cover, args.stego, out_dir=args.out, tag=args.tag, channel=args.channel)
    if args.output:
        # Writes the composite at this path plus four fixed-name plots beside it.
        analyze_comparison(args.cover, args.stego, output_path=args.output, channel=args.channel)

if __name__ == "__main__":
    _main()
