#!/usr/bin/env python3
# steganalysis.py — unified API for GUI + CLI + make_all.py

import argparse, os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CHANNELS = {"r":0, "g":1, "b":2}

def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def _std_hist(ax, chan_data, title):
    ax.hist(chan_data, bins=256, range=(0,256))
    ax.set_title(title)
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Frequency (Number of Pixels)")
    ax.set_xlim(0,255)

def _pairwise_hist(ax, chan_data, title, zoom=True, annotate_max=True):
    hist, _ = np.histogram(chan_data, bins=256, range=(0,256))
    diffs = np.array([float(hist[i])-float(hist[i+1]) for i in range(0,255,2)], dtype=float)
    if zoom and len(diffs)>0:
        # auto-zoom to central mass for readability
        lo, hi = np.percentile(np.abs(diffs), [2,98])
        lo = -max(1.0, hi)
        ax.set_xlim(-abs(hi), abs(hi))
    ax.hist(diffs, bins=50)
    ax.set_title(title)
    ax.set_xlabel("Difference between Adjacent Color Pair Counts")
    ax.set_ylabel("Frequency of that Difference")
    if annotate_max and diffs.size>0:
        ax.text(0.98, 0.90, f"max|Δ|={int(np.max(np.abs(diffs)))}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    return diffs

def analyze(cover_path:str, stego_path:str, out_dir:str=".", tag:str="", channel:str="b"):
    """
    Entry for make_all.py. Saves five figures:
      {out}/{tag}_{channel}_std_cover.png
      {out}/{tag}_{channel}_std_stego.png
      {out}/{tag}_{channel}_pair_cover.png
      {out}/{tag}_{channel}_pair_stego.png
      {out}/{tag}_{channel}_comp.png
    """
    os.makedirs(out_dir, exist_ok=True)
    ch = CHANNELS.get(channel.lower(), 2)

    cov = _load_rgb(cover_path); ste = _load_rgb(stego_path)
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
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(comp, dpi=150); plt.close(fig)

    print("std_cover:", std_cov)
    print("std_stego:", std_ste)
    print("pair_cover:", pr_cov)
    print("pair_stego:", pr_ste)
    print("composite:", comp)

    return {"std_cover":std_cov,"std_stego":std_ste,"pair_cover":pr_cov,"pair_stego":pr_ste,"composite":comp}

def analyze_comparison(original_image_path, encoded_image_path, output_path=None, channel="b"):
    """
    Back-compat for the GUI. Saves a composite at output_path and also writes four
    fixed-named plots in the same directory:
      1_original_standard_hist.png
      2_encoded_standard_hist.png
      3_original_difference_analysis.png
      4_encoded_difference_analysis.png
    """
    ch = CHANNELS.get(channel.lower(), 2)
    cov = _load_rgb(original_image_path)
    ste = _load_rgb(encoded_image_path)
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
    fig.suptitle("Comprehensive Steganalysis: Before vs. After")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(output_path, dpi=150); plt.close(fig)

    # four individual plots with fixed names
    p1 = os.path.join(out_dir, "1_original_standard_hist.png")
    p2 = os.path.join(out_dir, "2_encoded_standard_hist.png")
    p3 = os.path.join(out_dir, "3_original_difference_analysis.png")
    p4 = os.path.join(out_dir, "4_encoded_difference_analysis.png")

    fig, ax = plt.subplots(figsize=(12, 8)); _std_hist(ax, cov_c, "Original Image: Standard Histogram (Blue Channel)")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _std_hist(ax, ste_c, "Encoded Image: Standard Histogram (Blue Channel)")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _pairwise_hist(ax, cov_c, "Original Image: Difference Analysis (Blue Channel)")
    fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8)); _pairwise_hist(ax, ste_c, "Encoded Image: Difference Analysis (Blue Channel)")
    fig.tight_layout(); fig.savefig(p4, dpi=150); plt.close(fig)

    return {"composite":output_path, "p1":p1, "p2":p2, "p3":p3, "p4":p4}

def _main():
    ap = argparse.ArgumentParser(description="Steganalysis plots (standard + pairwise).")
    ap.add_argument("cover", help="Path to cover/original image")
    ap.add_argument("stego", help="Path to stego/encoded image")
    ap.add_argument("--out", default=None, help="Output directory for 5 files (for make_all.py).")
    ap.add_argument("--tag", default="run", help="Run tag used in filenames (for make_all.py).")
    ap.add_argument("--channel", default="b", choices=list(CHANNELS.keys()))
    ap.add_argument("-o","--output", default=None, help="Optional single composite path (for GUI-like use).")
    args = ap.parse_args()

    if args.out:
        analyze(args.cover, args.stego, out_dir=args.out, tag=args.tag, channel=args.channel)
    if args.output:
        analyze_comparison(args.cover, args.stego, output_path=args.output, channel=args.channel)

if __name__ == "__main__":
    _main()
