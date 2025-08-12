#!/usr/bin/env python3
# pairwise_stats.py — compute even/odd pair statistics used to detect LSB embedding.

import argparse, os, sys, json
import numpy as np
from PIL import Image

def parse_channel(s: str) -> int:
    table = {"r":0,"red":0,"g":1,"green":1,"b":2,"blue":2,"0":0,"1":1,"2":2}
    s = str(s).strip().lower()
    if s not in table:
        raise argparse.ArgumentTypeError("channel must be r/g/b or 0/1/2")
    return table[s]

def chan_hist(path: str, ch: int = 2) -> tuple[np.ndarray, tuple[int,int]]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)[:, :, ch].ravel()
    h = np.bincount(arr, minlength=256).astype(np.int64)
    return h, img.size  # (W,H)

def pair_stats(h: np.ndarray) -> dict:
    d = h[0::2] - h[1::2]
    a = np.abs(d)
    return {
        "L1": int(a.sum()),
        "L2": float(np.sqrt((d.astype(np.float64)**2).sum())),
        "median_abs": float(np.median(a)),
        "max_abs": int(a.max()),
        "max_k": int(a.argmax())
    }

def latex_table(Sc: dict, Ss: dict, caption: str) -> str:
    L1_drop = Sc["L1"] - Ss["L1"]
    L2_drop = Sc["L2"] - Ss["L2"]
    Med_drop = Sc["median_abs"] - Ss["median_abs"]
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
       L1_drop, L2_drop, Med_drop)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute pairwise even/odd stats (Δ_k) for LSB steganalysis.")
    p.add_argument("cover", help="Path to cover/original image (PNG/BMP/TIF)")
    p.add_argument("stego", help="Path to stego/encoded image (same size)")
    p.add_argument("-c","--channel", type=parse_channel, default="b",
                   help="Channel r/g/b or 0/1/2 (default: b)")
    p.add_argument("--json", help="Write results to JSON")
    p.add_argument("--latex", help="Write LaTeX table to this path")
    p.add_argument("--caption", default="Pairwise even/odd statistics (blue channel).",
                   help="Caption for LaTeX table")
    p.add_argument("--tag", default=None, help="Optional tag echoed into JSON object")
    args = p.parse_args()

    Hc, size_c = chan_hist(args.cover, args.channel)
    Hs, size_s = chan_hist(args.stego, args.channel)
    if size_c != size_s:
        sys.exit(f"ERROR: size mismatch: {args.cover}={size_c}, {args.stego}={size_s}")

    Sc, Ss = pair_stats(Hc), pair_stats(Hs)
    out = {
        "tag": args.tag,
        "channel": ["R","G","B"][args.channel],
        "cover": {"file": os.path.basename(args.cover), **Sc},
        "stego": {"file": os.path.basename(args.stego), **Ss},
        "delta": {
            "L1_drop": Sc["L1"] - Ss["L1"],
            "L2_drop": Sc["L2"] - Ss["L2"],
            "median_abs_drop": Sc["median_abs"] - Ss["median_abs"]
        }
    }
    print(json.dumps(out, indent=2))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f: json.dump(out, f, indent=2)
    if args.latex:
        os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
        with open(args.latex, "w") as f: f.write(latex_table(Sc, Ss, args.caption))
