#!/usr/bin/env python3
"""make_all.py: build every LSB experiment artifact from one cover image.

This script is the batch experiment runner for the project. Given one cover
image, it runs three embedding scenarios and writes the stego images, histogram
plots and statistics tables into one output directory. Filenames are short and
carry a timestamp:

  A) full-low      Original size, short message (low embedding rate).
  B) resized-high  Cover resized to --resize x --resize, then filled to capacity
                   with a repeated message (high embedding rate).
  C) full-high     Original size, filled to capacity by a very long repeated
                   message (the default message length fills covers up to about
                   7.3 megapixels).

For each scenario it writes the stego PNG. It then runs steganalysis.py
(histogram and pairwise-difference plots) and pairwise_stats.py (even/odd pair
statistics as JSON plus a LaTeX table) on the cover/stego pair.

How it fits with the other modules:
  encoder.py         encode(): hides a text message in an image's RGB LSBs.
  decoder.py         decode(): recovers the message by searching for the
                     64-zero-bit delimiter.
  main.py            Command-line ``encode`` / ``decode`` front end.
  steganalysis.py    Standard and pairwise-difference histogram plots.
  pairwise_stats.py  Numeric even/odd pair statistics (JSON + LaTeX table).
  make_all.py        This file. It has its own NumPy embedder (embed_bits_rgb)
                     rather than importing encoder.py, but it uses the same
                     layout: one payload bit in the LSB of each R, G, B byte,
                     pixels in row-major order, then 64 zero bits. Text becomes
                     UTF-8 bytes here, not ord() values (see to_bits_utf8), so
                     only ASCII messages give pixels identical to encoder.py.
  gui.py             Tkinter app that wraps encoder, decoder and steganalysis.

Usage (run from the repository root, because the helper scripts are invoked
by relative path):
    python3 make_all.py COVER [--out results] [--resize 500] [--channel b]
                        [--short "message"] [--long-multiplier 5000]

Outputs (RID = run timestamp YYYYmmdd-HHMMSS, BASE = cover filename stem,
CASE = full-low | resized-high | full-high, CH = --channel):
    {BASE}_{RID}_{CASE}_enc.png               Stego image.
    {BASE}_{RID}_resized-high_resized.png     Resized cover (case B only).
    {RID}_{CASE}_{CH}_{std,pair}_{cover,stego}.png, {RID}_{CASE}_{CH}_comp.png
                                              Plots from steganalysis.py.
    {RID}_{CASE}_pairwise.json, {RID}_{CASE}_pairwise.tex
                                              Statistics from pairwise_stats.py.
"""

import argparse, os, time, json, subprocess
import numpy as np
from PIL import Image

# Length of the end-of-message marker: 64 zero bits (eight bytes' worth) appended
# after the payload. This matches DELIMITER in encoder.py / decoder.py.
DELIM_BITS = 64

def ts_tag():
    """Return a run ID timestamp such as ``20250811-171038``.

    Returns:
        str: Local time formatted as ``%Y%m%d-%H%M%S``. It has one-second
        resolution, so two runs started in the same second that write to the
        same --out directory share a tag and overwrite each other's files.
    """
    return time.strftime("%Y%m%d-%H%M%S")

def to_bits_utf8(s: str) -> np.ndarray:
    """Convert a string to a flat array of bits (MSB first) from its UTF-8 bytes.

    Args:
        s: Text to embed.

    Returns:
        np.ndarray: uint8 array of 0/1 values with length ``8 * len(s.encode("utf-8"))``.
    """
    # NOTE: encoder.py writes format(ord(ch), '08b') per character and decoder.py
    # reads chr(byte) per byte, so non-ASCII text embedded here comes back as
    # mojibake through decoder.py (e.g. 'é' -> 'Ã©'). ASCII is the same both ways.
    b = s.encode("utf-8")
    # unpackbits expands each byte into 8 bits, most significant bit first.
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def embed_bits_rgb(in_path: str, out_path: str, payload_bits: np.ndarray):
    """LSB-embed a bit stream, followed by the 64-zero-bit delimiter, into an image.

    Bits go into the least significant bit of R, then G, then B of each pixel,
    pixels in row-major order (the same order encoder.py uses). Pixels after the
    end of the stream are left unchanged.

    Args:
        in_path: Cover image path. It is converted to 8-bit RGB, which drops any
            alpha channel.
        out_path: Where to save the stego image. The format comes from the file
            extension and must be lossless, or the LSBs are destroyed (make_all
            always passes a .png path).
        payload_bits: 1-D uint8 array of 0/1 values (see to_bits_utf8).

    Returns:
        tuple[int, int, int]: ``(W, H, n)``, where ``n`` is the number of bits
        written (payload plus delimiter).

    Raises:
        ValueError: If the payload plus delimiter is larger than ``3 * W * H`` bits.

    Side effects:
        Writes ``out_path``.
    """
    img = Image.open(in_path).convert("RGB")
    W, H = img.size
    arr = np.array(img, dtype=np.uint8)  # shape (H, W, 3)
    # Capacity: one bit per color channel, so 3 bits per RGB pixel.
    capacity = 3 * W * H
    total = payload_bits.size + DELIM_BITS
    if total > capacity:
        raise ValueError(f"Payload too large ({total} bits) for capacity {capacity} bits")
    # View the image as a list of (R, G, B) pixels. reshape returns a view, so
    # writing to flat also changes arr.
    flat = arr.reshape(-1, 3)
    # Payload followed by 64 zero bits. The decoder treats the first run of 64
    # zeros as end-of-message.
    # NOTE: decoder.py matches the first 64-zero run, so if the payload's last
    # byte ends in 0 bits (any even byte, e.g. '.' = 0x2E) the match starts
    # inside that byte and decoder.py drops the last character. With the default
    # --short message, the trailing '.' is lost.
    stream = np.concatenate([payload_bits, np.zeros(DELIM_BITS, dtype=np.uint8)], axis=0)
    idx = 0
    # Performance: this Python loop runs once per channel value written (up to
    # 3 * W * H times), about 1 s per megapixel at full capacity. A vectorized
    # NumPy assignment would be much faster.
    for i in range(flat.shape[0]):
        for c in range(3):
            # The inner break only leaves the channel loop. The outer check below
            # stops the pixel loop.
            if idx >= stream.size: break
            # 0xFE = 0b11111110: "& 0xFE" clears the LSB, "| bit" sets it to the
            # payload bit. The other 7 bits are unchanged, so each value moves by
            # at most 1.
            flat[i, c] = (flat[i, c] & 0xFE) | stream[idx]; idx += 1
        if idx >= stream.size: break
    Image.fromarray(flat.reshape(H, W, 3)).save(out_path)  # no 'mode' arg (silences deprecation)
    return W, H, stream.size

def resized_copy(in_path: str, out_path: str, size: int = 500):
    """Save a square ``size`` x ``size`` RGB copy of an image (Lanczos resampling).

    The aspect ratio is not preserved, and a cover smaller than ``size`` is
    upscaled. Case B embeds into this copy, and at the default 500x500 the
    default repeated message fills its whole capacity.

    Args:
        in_path: Source image path.
        out_path: Destination path. The format comes from the extension.
        size: Width and height of the output, in pixels.

    Side effects:
        Writes ``out_path``.
    """
    img = Image.open(in_path).convert("RGB")
    img = img.resize((size, size), resample=Image.LANCZOS)
    img.save(out_path)

def capacity_bits(path: str) -> int:
    """Return the raw LSB capacity of an image, ``3 * W * H`` bits.

    This includes the space that the 64-bit delimiter needs, so callers subtract
    it to get the usable payload size.

    Args:
        path: Image path.

    Returns:
        int: Number of embeddable bits (3 per RGB pixel).
    """
    W, H = Image.open(path).convert("RGB").size
    return 3 * W * H

# --- wrappers use subprocess so imports never get stale ---
# NOTE: Both wrappers run "python3" from PATH (not sys.executable) and give the
# scripts by relative path. If the venv is not active, the child process can be a
# different interpreter without numpy/matplotlib. If the working directory is not
# the repo root, the scripts are not found.
def run_steganalysis(cover, stego, outdir, tag, channel="b"):
    """Run steganalysis.py on a cover/stego pair to produce the histogram plots.

    The script writes five PNGs into ``outdir``:
    ``{tag}_{channel}_std_cover.png``, ``_std_stego.png``, ``_pair_cover.png``,
    ``_pair_stego.png`` and ``_comp.png`` (a 2x2 composite).

    Args:
        cover: Cover image path.
        stego: Stego image path (same size as the cover).
        outdir: Output directory.
        tag: Run tag used as the filename prefix.
        channel: Channel to analyze. steganalysis.py accepts only "r", "g" or "b".

    Raises:
        subprocess.CalledProcessError: If the script exits with a non-zero status,
            for example when the channel is invalid.

    Side effects:
        Writes the PNG files. The child process prints their paths to stdout.
    """
    cmd = [
        "python3", "steganalysis.py", cover, stego,
        "--out", outdir, "--tag", tag, "--channel", channel
    ]
    subprocess.check_call(cmd)

def run_pairwise_stats(cover, stego, outdir, tag, caption, channel="b"):
    """Run pairwise_stats.py on a cover/stego pair and return its results.

    For each pair of values (2k, 2k+1) in the channel histogram, the script
    measures the count difference ``d_k = h[2k] - h[2k+1]`` and summarizes it
    with L1, L2, median and max of ``|d_k|``. Flipping an LSB only moves a pixel
    between 2k and 2k+1. With uniformly random payload bits, heavy embedding
    makes the two counts equal in expectation, so these numbers drop compared
    with the cover (the classic pairs-of-values signal). make_all's payload is
    repeated text, not random bits, so the numbers can rise instead (see the
    NOTE in case B of main).

    Args:
        cover: Cover image path.
        stego: Stego image path (must be the same size as the cover).
        outdir: Directory for the JSON and LaTeX outputs.
        tag: Run tag used as the filename prefix and stored in the JSON.
        caption: Caption for the LaTeX table.
        channel: Channel to analyze. pairwise_stats.py accepts "r"/"g"/"b",
            "red"/"green"/"blue" or "0"/"1"/"2".

    Returns:
        dict: The parsed JSON, with keys "tag", "channel" ("R"/"G"/"B"),
        "cover" and "stego" (each with file, L1, L2, median_abs, max_abs,
        max_k), and "delta" (L1_drop, L2_drop and median_abs_drop, each equal to
        cover minus stego, so a negative value means the statistic grew).

    Raises:
        subprocess.CalledProcessError: If the script exits with a non-zero status,
            for example when a file is missing or the image sizes differ.

    Side effects:
        Writes ``{outdir}/{tag}_pairwise.json`` and ``{outdir}/{tag}_pairwise.tex``.
        The child process also prints the JSON to stdout.
    """
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
    """Parse the CLI arguments and run the three scenarios (A, B, C).

    Arguments:
        cover: Path to the cover image.
        --out: Output directory, created if it does not exist (default "results").
        --resize: Side length in pixels of the square cover for case B (default 500).
        --channel: Channel passed to both analysis scripts (default "b"). It
            must be "r", "g" or "b", because steganalysis.py runs first and
            rejects anything else.
        --short: Message for case A. For B and C it is repeated, with a space
            added after each copy.
        --long-multiplier: Number of repeats for case B. Case C uses 10 times this.

    Raises:
        ValueError: If the --short message (plus delimiter) does not fit the
            cover, or if --resize is 4 or less (a 1x1 to 4x4 cover cannot hold
            even the 64-bit delimiter, and PIL rejects sizes of 0 or below).
        subprocess.CalledProcessError: If either analysis script fails.

    Side effects:
        Writes all artifacts under ``--out`` (see the module docstring) and
        prints the output directory and the three run tags.
    """
    ap = argparse.ArgumentParser(description="Build all LSB artifacts from a single cover image.")
    ap.add_argument("cover", help="Path to the cover image (e.g., test-images/stego.png)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resize", type=int, default=500)
    ap.add_argument("--channel", default="b")
    ap.add_argument("--short", default="I am a very sneaky dinosaur... Nobody knows I am here.")
    ap.add_argument("--long-multiplier", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.cover))[0]  # cover filename without extension
    rid  = ts_tag()  # one timestamp shared by all three cases in this run

    # A) full-size LOW-rate (short message)
    # Only 8 * (UTF-8 byte length of --short) + 64 LSBs are written (496 with the
    # default), and only about half of those actually flip. That is a tiny
    # fraction of capacity.
    tagA = f"{rid}_full-low"
    encA = os.path.join(args.out, f"{base}_{tagA}_enc.png")
    bitsA = to_bits_utf8(args.short)
    embed_bits_rgb(args.cover, encA, bitsA)
    run_steganalysis(args.cover, encA, outdir=args.out, tag=tagA, channel=args.channel)
    # NOTE: The three captions always say "blue channel", even when --channel is
    # r or g.
    run_pairwise_stats(args.cover, encA, outdir=args.out, tag=tagA,
                       caption="Pairwise even/odd statistics (blue channel), full-size low-rate case.",
                       channel=args.channel)

    # B) RESIZED HIGH-rate
    # Resize the cover (usually shrinking it) so the repeated message fills every
    # LSB (100% of capacity, delimiter included). With the defaults, 2.2M payload
    # bits are available for 749,936 usable bits at 500x500.
    tagB = f"{rid}_resized-high"
    covB = os.path.join(args.out, f"{base}_{tagB}_resized.png")
    encB = os.path.join(args.out, f"{base}_{tagB}_enc.png")
    resized_copy(args.cover, covB, size=args.resize)
    # Usable payload bits = raw capacity minus the 64-bit delimiter.
    # The literal 64 duplicates DELIM_BITS, so the two must be kept equal by hand.
    capB = capacity_bits(covB) - 64
    # NOTE: Repeated ASCII text is not uniform random bits. The default --short
    # is about 45.7% ones, so full embedding pushes each (2k, 2k+1) pair toward
    # a 54:46 split rather than 50:50, and |d_k| ends near 8.6% of the pair's
    # total. In results/20250811-171038_*_pairwise.json, L1 and median |d_k|
    # rose (negative "drop") in both high-rate cases.
    bitsB = to_bits_utf8((args.short + " ") * args.long_multiplier)
    # Truncate to fit.
    # NOTE: The cut is at a bit, not a byte or UTF-8 character boundary. If capB
    # is not a multiple of 8 (at the default 500x500 it is), the last character
    # is cut off, and decoder.py drops the partial byte. The statistics are not
    # affected.
    if bitsB.size > capB: bitsB = bitsB[:capB]
    embed_bits_rgb(covB, encB, bitsB)
    # Analyze against the resized cover, not the original, so the sizes match.
    run_steganalysis(covB, encB, outdir=args.out, tag=tagB, channel=args.channel)
    run_pairwise_stats(covB, encB, outdir=args.out, tag=tagB,
                       caption="Pairwise even/odd statistics (blue channel), resized high-rate case.",
                       channel=args.channel)

    # C) full-size HIGH-rate (very long message)
    # Ten times the repeats of case B (22M bits by default), truncated to
    # capacity. Covers larger than about 7.3 MP are not completely filled.
    tagC = f"{rid}_full-high"
    encC = os.path.join(args.out, f"{base}_{tagC}_enc.png")
    capC = capacity_bits(args.cover) - 64  # same delimiter reservation as case B
    bitsC = to_bits_utf8((args.short + " ") * (args.long_multiplier * 10))
    if bitsC.size > capC: bitsC = bitsC[:capC]  # same bit-level truncation as case B
    embed_bits_rgb(args.cover, encC, bitsC)
    run_steganalysis(args.cover, encC, outdir=args.out, tag=tagC, channel=args.channel)
    run_pairwise_stats(args.cover, encC, outdir=args.out, tag=tagC,
                       caption="Pairwise even/odd statistics (blue channel), full-size high-rate case via long message.",
                       channel=args.channel)

    print("\nAll artifacts written under:", os.path.abspath(args.out))
    print("Run tags:\n-", tagA, "\n-", tagB, "\n-", tagC)

if __name__ == "__main__":
    main()
