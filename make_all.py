#!/usr/bin/env python3
# make_all.py — build all artifacts from one cover image with short filenames

import argparse, os, time, json, subprocess
import numpy as np
from PIL import Image

DELIM_BITS = 64

def ts_tag(): return time.strftime("%Y%m%d-%H%M%S")

def to_bits_utf8(s: str) -> np.ndarray:
    b = s.encode("utf-8")
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def embed_bits_rgb(in_path: str, out_path: str, payload_bits: np.ndarray):
    img = Image.open(in_path).convert("RGB")
    W, H = img.size
    arr = np.array(img, dtype=np.uint8)
    capacity = 3 * W * H
    total = payload_bits.size + DELIM_BITS
    if total > capacity:
        raise ValueError(f"Payload too large ({total} bits) for capacity {capacity} bits")
    flat = arr.reshape(-1, 3)
    stream = np.concatenate([payload_bits, np.zeros(DELIM_BITS, dtype=np.uint8)], axis=0)
    idx = 0
    for i in range(flat.shape[0]):
        for c in range(3):
            if idx >= stream.size: break
            flat[i, c] = (flat[i, c] & 0xFE) | stream[idx]; idx += 1
        if idx >= stream.size: break
    Image.fromarray(flat.reshape(H, W, 3)).save(out_path)  # no 'mode' arg (silences deprecation)
    return W, H, stream.size

def resized_copy(in_path: str, out_path: str, size: int = 500):
    img = Image.open(in_path).convert("RGB")
    img = img.resize((size, size), resample=Image.LANCZOS)
    img.save(out_path)

def capacity_bits(path: str) -> int:
    W, H = Image.open(path).convert("RGB").size
    return 3 * W * H

# --- wrappers use subprocess so imports never get stale ---
def run_steganalysis(cover, stego, outdir, tag, channel="b"):
    cmd = [
        "python3", "steganalysis.py", cover, stego,
        "--out", outdir, "--tag", tag, "--channel", channel
    ]
    subprocess.check_call(cmd)

def run_pairwise_stats(cover, stego, outdir, tag, caption, channel="b"):
    json_path = os.path.join(outdir, f"{tag}_pairwise.json")
    tex_path  = os.path.join(outdir, f"{tag}_pairwise.tex")
    cmd = [
        "python3", "pairwise_stats.py", cover, stego,
        "--json", json_path, "--latex", tex_path,
        "--caption", caption, "--tag", tag, "-c", channel
    ]
    subprocess.check_call(cmd)
    with open(json_path) as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Build all LSB artifacts from a single cover image.")
    ap.add_argument("cover", help="Path to the cover image (e.g., test-images/stego.png)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resize", type=int, default=500)
    ap.add_argument("--channel", default="b")
    ap.add_argument("--short", default="I am a very sneaky dinosaur... Nobody knows I am here.")
    ap.add_argument("--long-multiplier", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.cover))[0]
    rid  = ts_tag()

    # A) full-size LOW-rate (short message)
    tagA = f"{rid}_full-low"
    encA = os.path.join(args.out, f"{base}_{tagA}_enc.png")
    bitsA = to_bits_utf8(args.short)
    embed_bits_rgb(args.cover, encA, bitsA)
    run_steganalysis(args.cover, encA, outdir=args.out, tag=tagA, channel=args.channel)
    run_pairwise_stats(args.cover, encA, outdir=args.out, tag=tagA,
                       caption="Pairwise even/odd statistics (blue channel), full-size low-rate case.",
                       channel=args.channel)

    # B) RESIZED HIGH-rate
    tagB = f"{rid}_resized-high"
    covB = os.path.join(args.out, f"{base}_{tagB}_resized.png")
    encB = os.path.join(args.out, f"{base}_{tagB}_enc.png")
    resized_copy(args.cover, covB, size=args.resize)
    capB = capacity_bits(covB) - 64
    bitsB = to_bits_utf8((args.short + " ") * args.long_multiplier)
    if bitsB.size > capB: bitsB = bitsB[:capB]
    embed_bits_rgb(covB, encB, bitsB)
    run_steganalysis(covB, encB, outdir=args.out, tag=tagB, channel=args.channel)
    run_pairwise_stats(covB, encB, outdir=args.out, tag=tagB,
                       caption="Pairwise even/odd statistics (blue channel), resized high-rate case.",
                       channel=args.channel)

    # C) full-size HIGH-rate (very long message)
    tagC = f"{rid}_full-high"
    encC = os.path.join(args.out, f"{base}_{tagC}_enc.png")
    capC = capacity_bits(args.cover) - 64
    bitsC = to_bits_utf8((args.short + " ") * (args.long_multiplier * 10))
    if bitsC.size > capC: bitsC = bitsC[:capC]
    embed_bits_rgb(args.cover, encC, bitsC)
    run_steganalysis(args.cover, encC, outdir=args.out, tag=tagC, channel=args.channel)
    run_pairwise_stats(args.cover, encC, outdir=args.out, tag=tagC,
                       caption="Pairwise even/odd statistics (blue channel), full-size high-rate case via long message.",
                       channel=args.channel)

    print("\nAll artifacts written under:", os.path.abspath(args.out))
    print("Run tags:\n-", tagA, "\n-", tagB, "\n-", tagC)

if __name__ == "__main__":
    main()
