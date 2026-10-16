#!/usr/bin/env python3
# main.py
"""Command-line interface for the Steganography Image Analyzer.

One entry point for hiding, revealing and analysing data hidden in the least
significant bits (LSBs) of images and WAV audio. Each subcommand is a thin
layer over a library module, so everything here can also be done from
Python:

    encode      Hide a message or a file (stego.hide). Optional password
                (AES-256-GCM with a scrypt key; the password also scatters
                the bits over the cover), zlib compression, 1 or 2 bits per
                sample.
    decode      Recover it (stego.reveal), falling back to the original
                delimiter format of this project for old images
                (decoder.decode_legacy) when that reading looks like text.
    capacity    How many bytes a cover can hold.
    analyze     Full steganalysis of one image (report.analyze_image): the
                statistical detectors (chi-square, sample pair analysis, RS
                analysis, histogram pairs, LSB randomness), the signature
                scan, payload header detection and brute force, with an
                optional Markdown, JSON or HTML report (report.write_report).
    bitplanes   Bit-plane grid and cover-vs-stego LSB difference mask
                (bitplanes.py).
    bruteforce  Try ~100 bit layouts and rank what comes out (bruteforce.py),
                optionally with a password word list for scattered payloads
                (the word list also works on WAV files).
    compare     Cover-vs-stego histogram comparison plots
                (steganalysis.analyze_comparison) plus even/odd pair
                statistics (pairwise_stats.pair_stats).
    plugins     List the registered carriers (cover formats) and detectors.

Examples (run from the repository root)::

    # The original syntax still works.
    python main.py encode -i cover.png -m "secret text" -o stego.png
    python main.py decode -i stego.png

    # Encrypted, compressed file payload at 2 bits per sample.
    python main.py encode -i cover.jpg -f notes.pdf -o stego.png \\
        --password-prompt --compress --bits 2
    python main.py decode -i stego.png --password-prompt --out recovered/

    # Audio carrier.
    python main.py encode -i song.wav -m "hi" -o song_stego.wav

    python main.py capacity -i cover.png
    python main.py analyze -i suspect.png --report reports/ --format html
    python main.py analyze -i suspect.png --json > suspect.json
    python main.py bitplanes -i stego.png -o planes.png --diff cover.png
    python main.py bruteforce -i suspect.png --top 5 --save recovered/
    python main.py compare cover.png stego.png -o plots/comparison.png
    python main.py plugins

Passwords: ``--password PW`` is visible to other users in the process list
and is saved in shell history; ``--password-prompt`` asks for it instead
(twice when encoding). When standard input is not a terminal,
``--password-prompt`` reads one line from it, so scripts can pipe the
password in. An empty password is refused either way, so an unset shell
variable (``--password "$PW"``) cannot silently write an unencrypted,
unscattered payload.

Untrusted output: recovered text comes from the stego file, so ``decode``
replaces control characters (escape sequences, carriage returns, C1
controls, bidirectional overrides) with U+FFFD before printing, the same
rule ``analyze`` and ``bruteforce`` use for their previews. ``decode --out``
writes the exact bytes.

Legacy fallback: the original delimiter format has no header, and any
image whose LSB stream contains 64 zero bits (a flat black or even-valued
region is enough) "decodes" to whatever comes before them. So ``decode``
accepts a legacy reading only when ``bruteforce.score_candidate`` reads it
as text ending at the delimiter with a heuristic score of at least
LEGACY_MIN_SCORE, and says on stderr when the score is low. ``--legacy``
keeps the "is it text" check but drops the score floor.

Carriers: ``encode``, ``decode`` and ``capacity`` work on every registered
carrier (images and WAV). ``analyze``, ``bitplanes``, ``compare`` and the
layout part of ``bruteforce`` read images only; for a WAV file
``bruteforce`` runs only the ``--wordlist`` attack.

Exit status: 0 on success; 1 on an error or when nothing hidden is found
(the message goes to standard error): ``decode`` with no frame and no
plausible legacy message, and ``bruteforce --wordlist`` when no password
matched and no layout scored BRUTEFORCE_FOUND_SCORE or more (without
``--wordlist`` the brute force only ranks layouts and exits 0); 2 for
invalid arguments (from argparse or the option checks in
_check_combinations, e.g. an empty ``--password`` or a ``--channels``
letter outside RGBA); 130 when interrupted with Ctrl-C.

Heavy modules (NumPy, Pillow, Matplotlib, the detectors) are imported inside
each subcommand, so ``--help`` and the light subcommands start quickly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

PROG = "main.py"

# Lowest heuristic score (bruteforce.score_candidate, a ranking confidence in
# [0, 1], not a probability) at which decode accepts a legacy delimiter
# reading when the image holds no STGA frame. bruteforce scores a legacy
# reading 0.15 to 0.95 when the bytes before the 64 zero bits are printable
# text: 0.15 for 1 to 3 ASCII characters, more for longer word-like text
# ("hello" 0.28, "attack at dawn" 0.55, a 40-character sentence 0.8). Text
# that is repetitive ("UUUU", which smooth gradients produce) gets 0.05, and
# binary noise is not text at all. 0.15 is bruteforce's own floor for the
# legacy band, so short legacy messages still decode.
LEGACY_MIN_SCORE = 0.15

# Below this score a legacy reading is weak evidence, and decode says so on
# stderr. It is the cut point report.py uses before it reports a legacy
# message (report._LEGACY_MIN_SCORE), so analyze stays the conservative one.
LEGACY_CONFIDENT_SCORE = 0.5

# With --wordlist, bruteforce exits 1 when no password matched and no layout
# candidate scored at least this much (heuristic). bruteforce reads a layout
# further than --max-bytes from the same score (bruteforce._EXPAND_MIN_SCORE),
# and on the test covers every layout of a clean image stayed below 0.3.
BRUTEFORCE_FOUND_SCORE = 0.5

# Format characters kept although str.isprintable() rejects them: the zero
# width non-joiner and joiner, which emoji sequences and some scripts need and
# which do nothing in a terminal.
_SAFE_FORMAT_CHARS = "‌‍"


class CLIError(Exception):
    """An expected failure: its message is printed to stderr and main() returns 1."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_password(args: argparse.Namespace, *, confirm: bool) -> str | None:
    """Return the password from ``--password`` or ``--password-prompt``.

    With ``--password-prompt`` on a terminal, getpass asks without echoing
    (twice when ``confirm`` is True). When stdin is not a terminal, one line
    is read from it instead, so the prompt never blocks a script.

    Raises:
        CLIError: If the two entries differ or the password is empty.
    """
    if not getattr(args, "password_prompt", False):
        return getattr(args, "password", None)
    if sys.stdin is not None and sys.stdin.isatty():
        import getpass

        password = getpass.getpass("Password: ")
        if confirm and getpass.getpass("Repeat password: ") != password:
            raise CLIError("the passwords do not match")
    else:
        password = (sys.stdin.readline() if sys.stdin is not None else "").rstrip("\r\n")
    if not password:
        raise CLIError("empty password")
    return password


def _is_dir_target(out: str) -> bool:
    """True when ``--out`` names a folder (existing, or ending in a separator)."""
    return os.path.isdir(out) or out.endswith(("/", os.sep))


def _write_output(data: bytes, out: str, found: Any = None) -> Path:
    """Write recovered bytes for ``decode --out``.

    A folder target gets a new file named after the stored file name (or
    payload.txt / payload.bin) and never overwrites anything
    (stego.save_payload). Any other target is written exactly there,
    replacing an existing file, with parent folders created.
    """
    import stego

    if _is_dir_target(out):
        if found is None:
            import payload

            found = payload.Payload(data=data)
        return stego.save_payload(found, out)
    path = Path(out)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _terminal_safe(text: str, *, one_line: bool = False) -> str:
    """Make untrusted text safe to print to a terminal.

    Recovered payload text, stored file names and paths built from them come
    from the stego file, and terminals act on control characters: ESC starts
    sequences that retitle the window, clear the screen, recolour text or
    move the cursor to fake earlier output (and, on some terminals, write
    the clipboard with OSC 52); CR overwrites the current line. Every
    character ``str.isprintable()`` rejects (C0 and C1 controls, DEL,
    bidirectional overrides and other format characters, line and paragraph
    separators) becomes U+FFFD, except newline and tab and the zero-width
    joiners. CRLF line ends become plain newlines first. This is the rule
    report._printable applies to analyze's previews.

    Args:
        text: Untrusted text.
        one_line: Also replace newlines and tabs (for names and paths that
            are printed inside a sentence).

    Returns:
        The text with unsafe characters replaced.
    """
    text = text.replace("\r\n", "\n")
    keep = _SAFE_FORMAT_CHARS if one_line else "\n\t" + _SAFE_FORMAT_CHARS
    return "".join(ch if ch.isprintable() or ch in keep else "�" for ch in text)


def _print_text_message(text: str) -> None:
    """Print a recovered message in the format the original CLI used.

    The text is untrusted, so control characters are replaced first
    (_terminal_safe); a note on stderr says how many, and that ``--out``
    gives the exact bytes.
    """
    safe = _terminal_safe(text)
    print("Decoded Message Found:")
    print(safe)
    replaced = safe.count("�") - text.count("�")
    if replaced:
        print(f"note: {_plural(replaced, 'control character')} replaced with U+FFFD for "
              "display; use --out PATH to save the exact bytes", file=sys.stderr)


def _read_legacy(path: str) -> dict[str, Any]:
    """Read the legacy delimiter format and judge whether it looks like text.

    ``decoder.decode_legacy`` returns the bytes before the first 64 zero
    bits of the RGB LSB stream, which is *something* for any image with a
    flat black or even-valued region. So the same bytes, cut exactly at the
    delimiter, are also scored with ``bruteforce.score_candidate`` under
    the legacy layout (R, G, B LSBs, row-major, most significant bit
    first). That is a heuristic text score, not a statistical test.

    Args:
        path: The image.

    Returns:
        A dict with ``text`` (decoder.legacy_text_from_pixels: None when the
        stream has no delimiter, "" when it starts with one), ``raw_bytes``
        (bytes before the delimiter), ``is_text`` (True when the bytes read
        as printable text that ends at the delimiter) and ``score`` (the
        heuristic score of that text, 0.0 when it is not text).

    Raises:
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    import numpy as np
    from PIL import Image

    import bruteforce as bf
    import decoder

    with Image.open(path) as img:
        # Same conversion as decoder.decode_legacy (alpha dropped).
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    text = decoder.legacy_text_from_pixels(rgb)
    result: dict[str, Any] = {"text": text, "raw_bytes": 0, "is_text": False, "score": 0.0}
    if not text:
        return result

    delim = decoder.LEGACY_DELIMITER_BYTES
    # decode_legacy decodes the bytes as UTF-8, or as Latin-1 (one byte per
    # character) when they are not UTF-8; the UTF-8 length is never the
    # smaller, so this window holds the message bytes and the delimiter.
    window = bf.extract_bytes(rgb, bf.Strategy("RGB", 0, "msb", "row"),
                              len(text.encode("utf-8")) + len(delim))
    end = window.find(delim)
    if end <= 0:
        return result
    raw = window[:end]
    cand = bf.score_candidate(raw + delim)
    fmt = (cand.get("info") or {}).get("format")
    if fmt is not None:
        is_text = cand["kind"] == "text" and fmt == "legacy-delimiter"
    else:
        # Older scoring without the format field: the text candidate must
        # cover every byte before the delimiter.
        is_text = cand["kind"] == "text" and cand.get("data") == raw
    result.update(raw_bytes=len(raw), is_text=is_text,
                  score=float(cand["score"]) if is_text else 0.0)
    return result


def _plural(n: int, word: str) -> str:
    """Format a count with its noun, e.g. "1 byte" or "1,024 bytes"."""
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def _pct(part: int, whole: int) -> str:
    """Format part/whole as a percentage string."""
    return f"{(part / whole if whole else 0.0):.2%}"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_encode(args: argparse.Namespace) -> int:
    """Hide a message or a file in a cover (stego.hide)."""
    import stego

    password = _get_password(args, confirm=True)
    if args.file is not None:
        try:
            data: bytes | str = Path(args.file).read_bytes()
        except IsADirectoryError:
            raise CLIError(f"--file must be a file, not a folder: {args.file}") from None
        filename = os.path.basename(args.file)
    else:
        data, filename = args.message, None

    summary = stego.hide(
        args.image, args.output, data,
        password=password, filename=filename, compress=args.compress,
        bits=args.bits, key_scatter=not args.no_scatter,
    )
    if filename:
        print(f"Successfully hid file '{filename}' in '{args.output}'")
    else:
        print(f"Successfully encoded message in '{args.output}'")
    traits = [
        summary["carrier"],
        f"{summary['bits_per_sample']} bit(s)/sample",
        "encrypted" if summary["encrypted"] else "not encrypted",
        "scattered" if summary["scattered"] else "sequential",
    ]
    if summary["compressed"]:
        traits.append("compressed")
    print(f"  {_plural(summary['data_bytes'], 'byte')} of data in a {summary['frame_bytes']:,}-byte "
          f"frame, {summary['utilization']:.2%} of capacity ({', '.join(traits)})")
    return EXIT_OK


def cmd_decode(args: argparse.Namespace) -> int:
    """Recover a hidden payload (stego.reveal, with the legacy fallback).

    The legacy reading is accepted only when it reads as text ending at the
    64 zero bits and, unless ``--legacy`` was given, scores at least
    LEGACY_MIN_SCORE (see _read_legacy). Otherwise nothing was found and
    the command fails with exit status 1.
    """
    import carriers
    import stego
    from carriers.image_lsb import ImageLSBCarrier

    password = _get_password(args, confirm=False)
    if not os.path.exists(args.image):
        raise FileNotFoundError(2, "No such file", args.image)

    reason = "no hidden message found"
    if not args.legacy:
        try:
            carrier_cls = carriers.carrier_for_path(args.image)
        except ValueError:
            carrier_cls = None  # unknown extension: only the legacy reader can try
        if carrier_cls is not None:
            try:
                found = stego.reveal(args.image, password=password, bits=args.bits)
            except stego.NoPayloadError as exc:
                # Same rule as decoder.decode(): only "no frame at all" falls
                # back; damaged frames and wrong passwords are reported.
                if not issubclass(carrier_cls, ImageLSBCarrier):
                    raise CLIError(str(exc)) from None
                reason = str(exc)
            else:
                return _emit_payload(found, args.out)

    legacy = _read_legacy(args.image)
    text, score = legacy["text"], legacy["score"]
    if not text:
        if args.legacy:
            raise CLIError("no legacy-format message found (no text followed by "
                           "64 zero bits at the start of the RGB LSB stream)")
        raise CLIError(f"{reason}; the legacy delimiter format was not found either")
    if not legacy["is_text"]:
        detail = (f"the {_plural(legacy['raw_bytes'], 'byte')} before the first 64 zero "
                  "bits of the RGB LSB stream are not text")
    elif not args.legacy and score < LEGACY_MIN_SCORE:
        detail = (f"the text before the first 64 zero bits does not look like a message "
                  f"(heuristic score {score:.2f}; --legacy prints it anyway)")
    else:
        detail = None
    if detail is not None:
        if args.legacy:
            raise CLIError(f"no legacy-format message found: {detail}")
        raise CLIError(f"{reason}; the legacy delimiter format was not found either ({detail})")

    if not args.legacy:
        print("note: no STGA frame; read the legacy delimiter format "
              f"(text followed by 64 zero bits), heuristic text score {score:.2f}",
              file=sys.stderr)
    if score < LEGACY_CONFIDENT_SCORE:
        print("note: a score this low is weak evidence: short or repetitive text before "
              "a run of zero bits also occurs by chance in clean images", file=sys.stderr)
    if args.out:
        data = text.encode("utf-8")
        path = _write_output(data, args.out)
        print(f"Saved {len(data):,} bytes to '{_terminal_safe(str(path), one_line=True)}'")
        return EXIT_OK
    _print_text_message(text)
    return EXIT_OK


def _emit_payload(found: Any, out: str | None) -> int:
    """Print or save a payload returned by stego.reveal().

    The stored file name and the text are untrusted and are printed through
    _terminal_safe; the saved file gets the exact bytes.
    """
    if out:
        path = _write_output(found.data, out, found)
        print(f"Saved {len(found.data):,} bytes to '{_terminal_safe(str(path), one_line=True)}'")
        return EXIT_OK
    if found.filename:
        print(f"Hidden file found: '{_terminal_safe(found.filename, one_line=True)}' "
              f"({len(found.data):,} bytes). Use --out DIR to save it.")
        return EXIT_OK
    try:
        text = found.data.decode("utf-8")
    except UnicodeDecodeError:
        print(f"Hidden binary payload found ({len(found.data):,} bytes, not UTF-8 text). "
              "Use --out PATH to save it.")
        return EXIT_OK
    _print_text_message(text)
    return EXIT_OK


def cmd_capacity(args: argparse.Namespace) -> int:
    """Print how many bytes of data a cover holds."""
    import stego

    carrier = stego.load_carrier(args.image, 1)
    samples = carrier.capacity_bits()  # at 1 bit per sample: one bit per sample
    overhead = stego.ENCRYPTION_OVERHEAD if args.encrypted else 0
    print(f"{args.image} ({carrier.name}): {samples:,} samples")
    for bits in ([args.bits] if args.bits else [1, 2]):
        n = max(0, stego.capacity_bytes(args.image, bits) - overhead)
        label = "bit/sample: " if bits == 1 else "bits/sample:"
        print(f"  {bits} {label} {n:,} bytes of data"
              + (" (with encryption)" if args.encrypted else ""))
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run the full single-image analysis and optionally write a report."""
    import report

    password = _get_password(args, confirm=False)
    result = report.analyze_image(args.image, run_bruteforce=not args.no_bruteforce,
                                  top_k=args.top, password=password)
    path = None
    if args.report:
        path = report.write_report(result, args.report, fmt=args.format or "md",
                                   figures=not args.no_figures)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(report.format_text(result))
    if path:
        # stdout stays pure JSON with --json.
        print(f"Report written to '{path}'", file=sys.stderr if args.json else sys.stdout)
    return EXIT_OK


def cmd_bitplanes(args: argparse.Namespace) -> int:
    """Write a bit-plane grid, and optionally an LSB diff against a cover."""
    import bitplanes
    import numpy as np
    from PIL import Image

    grid = bitplanes.save_bitplane_grid(args.image, args.output, channels=args.channels)
    print(f"Bit-plane grid written to '{grid}'")
    if args.amplify:
        Path(args.amplify).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(bitplanes.lsb_amplify(args.image)).save(args.amplify)
        print(f"Amplified LSB image written to '{args.amplify}'")
    if args.diff:
        out = args.diff_out
        if not out:
            grid_path = Path(args.output)
            out = str(grid_path.with_name(f"{grid_path.stem}_lsb_diff.png"))
        mask = bitplanes.lsb_diff(args.diff, args.image)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(out)
        changed, total = int(np.count_nonzero(mask)), mask.size
        print(f"LSB diff written to '{out}': {changed:,} of {total:,} pixels "
              f"({_pct(changed, total)}) have a changed LSB")
    return EXIT_OK


def cmd_bruteforce(args: argparse.Namespace) -> int:
    """Rank many bit layouts by how much the extracted data looks like a payload.

    The layouts are pixel layouts, so they are tried on images only. The
    ``--wordlist`` attack goes through the carrier registry and also works
    on WAV files, where it is the only thing run. With ``--wordlist`` the
    command fails (exit 1) when no password matched and no layout scored
    BRUTEFORCE_FOUND_SCORE or more; without it, it only ranks and exits 0.
    """
    import bruteforce as bf
    import carriers
    from carriers.image_lsb import ImageLSBCarrier

    if not os.path.exists(args.image):
        raise FileNotFoundError(2, "No such file", args.image)
    try:
        carrier_cls = carriers.carrier_for_path(args.image)
    except ValueError:
        carrier_cls = None  # unknown extension: Pillow may still read it
    is_image = carrier_cls is None or issubclass(carrier_cls, ImageLSBCarrier)
    if not is_image and not args.wordlist:
        raise CLIError(f"the layout brute force reads images only, and {args.image!r} is a "
                       f"{carrier_cls.name} file; use --wordlist FILE to try passwords "
                       "against a payload scattered with one, or decode for an unscattered one")

    results: list[dict[str, Any]] = []
    if is_image:
        results = bf.bruteforce(args.image, top_k=args.top, max_bytes=args.max_bytes,
                                max_bit=args.max_bit)
    else:
        print(f"note: '{args.image}' is not an image ({carrier_cls.name}); "
              "only the word list is tried", file=sys.stderr)
    hits = None
    if args.wordlist:
        with open(args.wordlist, encoding="utf-8", errors="replace") as fh:
            hits = bf.dictionary_attack(args.image, fh)

    if args.json:
        doc: dict[str, Any] = {"candidates": [
            {"label": r["label"], "strategy": r["strategy"].to_dict(), "score": r["score"],
             "kind": r["kind"], "preview": r["preview"], "info": r["info"]} for r in results]}
        if hits is not None:
            doc["wordlist_hits"] = [
                {k: v for k, v in h.items() if k != "data"} for h in hits]
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        if is_image:
            print(f"{'score':>5}  {'layout':<26} {'kind':<10} preview")
        for r in results:
            print(f"{r['score']:5.2f}  {r['label']:<26} {r['kind']:<10} {r['preview']}")
        if hits is not None:
            if not hits:
                print("wordlist: no password matched")
            for h in hits:
                print(f"wordlist: password {h['password']!r} "
                      f"({h['bits_per_sample']} bit/sample): {h['preview']}")

    note = sys.stderr if args.json else sys.stdout
    if args.save:
        best = hits[0] if hits else (results[0] if results else None)
        if best is None:
            raise CLIError("nothing to save: no candidates")
        if best["score"] < 0.3:
            print(f"warning: the best candidate only scored {best['score']:.2f}; "
                  "it is probably noise", file=sys.stderr)
        print(f"Saved best candidate to '{bf.save_candidate(best, args.save)}'", file=note)

    if hits is not None and not hits and not any(
            r["score"] >= BRUTEFORCE_FOUND_SCORE for r in results):
        found_by_layouts = (f", and no layout scored {BRUTEFORCE_FOUND_SCORE:.2f} or more"
                            if is_image else "")
        raise CLIError(f"no password in the word list matched{found_by_layouts}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    """Cover-vs-stego histogram plots, pair statistics and LSB change count."""
    import matplotlib

    matplotlib.use("Agg")  # files only; never open a window from the CLI
    import numpy as np
    from PIL import Image

    import pairwise_stats
    import steganalysis

    for path in (args.cover, args.stego):
        if not os.path.exists(path):
            raise FileNotFoundError(2, "No such file", path)
    paths = steganalysis.analyze_comparison(args.cover, args.stego,
                                            output_path=args.output, channel=args.channel)
    print(f"Comparison figure written to '{paths['composite']}'")
    for key in ("p1", "p2", "p3", "p4"):
        print(f"  also '{paths[key]}'")

    ch = "rgb".index(args.channel)
    with Image.open(args.cover) as a, Image.open(args.stego) as b:
        cover = np.asarray(a.convert("RGB"), dtype=np.uint8)
        stego_arr = np.asarray(b.convert("RGB"), dtype=np.uint8)
    name = {"r": "red", "g": "green", "b": "blue"}[args.channel]
    print(f"Even/odd pair statistics, {name} channel (Delta_k = h(2k) - h(2k+1)):")
    for label, arr in (("cover", cover), ("stego", stego_arr)):
        h = np.bincount(arr[..., ch].ravel(), minlength=256).astype(np.int64)
        s = pairwise_stats.pair_stats(h)
        print(f"  {label}: L1 {s['L1']:,}  L2 {s['L2']:,.1f}  median |Delta| "
              f"{s['median_abs']:g}  max |Delta| {s['max_abs']:,} at k={s['max_k']}")
    if cover.shape != stego_arr.shape:
        print("note: the images differ in size, so LSBs were not compared pixel by pixel",
              file=sys.stderr)
    else:
        changed_px = int(np.count_nonzero(((cover ^ stego_arr) & 1).any(axis=2)))
        total = cover.shape[0] * cover.shape[1]
        print(f"{changed_px:,} of {total:,} pixels ({_pct(changed_px, total)}) "
              "have a changed R, G or B LSB")
    return EXIT_OK


def cmd_plugins(args: argparse.Namespace) -> int:
    """List the registered carriers and detectors."""
    import bruteforce as bf
    import carriers
    import detectors
    import report

    carrier_rows = [{
        "name": cls.name,
        "description": cls.description,
        "output_extensions": list(cls.extensions),
        "cover_extensions": list(cls.cover_extensions),
    } for cls in carriers.list_carriers()]
    detector_rows = detectors.list_detectors()
    n_strategies = len(bf.enumerate_strategies())
    if args.json:
        print(json.dumps({"carriers": carrier_rows, "detectors": detector_rows,
                          "bruteforce_strategies": n_strategies,
                          "report_formats": list(report.REPORT_FORMATS)}, indent=2))
        return EXIT_OK
    print("Carriers (cover formats, carriers/ package):")
    for row in carrier_rows:
        print(f"  {row['name']:<12} {row['description']}")
        print(f"  {'':<12} writes: {' '.join(row['output_extensions'])}")
        if row["cover_extensions"]:
            print(f"  {'':<12} also reads covers: {' '.join(row['cover_extensions'])}")
    print("Detectors (detectors/ package):")
    for row in detector_rows:
        print(f"  {row['name']:<22} {row['description']}")
    print(f"Brute-force layouts: {n_strategies} by default (bruteforce.enumerate_strategies), "
          "plus a password word-list attack")
    print(f"Report formats: {', '.join(report.REPORT_FORMATS)}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _add_password_options(p: argparse.ArgumentParser, *, purpose: str) -> None:
    """Add the mutually exclusive --password / --password-prompt pair."""
    group = p.add_mutually_exclusive_group()
    group.add_argument("--password", metavar="PW", help=f"password {purpose} "
                       "(visible in the process list; prefer --password-prompt)")
    group.add_argument("--password-prompt", action="store_true",
                       help="ask for the password without echoing it (reads one "
                            "line from stdin when it is not a terminal)")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Hide, reveal and detect data hidden in the LSBs of images and WAV audio.",
        epilog="Run 'python main.py <command> -h' for the options of one command.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("encode", help="hide a message or file in a cover",
                       description="Hide a message or a file in a cover image or WAV file.")
    p.add_argument("-i", "--image", required=True,
                   help="cover file: any image Pillow reads (JPEG is fine as a cover) or PCM WAV")
    p.add_argument("-o", "--output", required=True,
                   help="stego file to write: .png, .bmp, .tif/.tiff or .wav (lossless only)")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("-m", "--message", help="text to hide (stored as UTF-8)")
    what.add_argument("-f", "--file", help="file to hide; its name is stored with it")
    _add_password_options(p, purpose="to encrypt with (AES-256-GCM) and scatter the bits")
    p.add_argument("--bits", type=int, choices=(1, 2), default=1,
                   help="low bits used per sample (default 1; 2 doubles capacity)")
    p.add_argument("--compress", action="store_true", help="zlib-compress before hiding")
    p.add_argument("--no-scatter", action="store_true",
                   help="with a password, still write the bits in natural order")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("decode", help="recover a hidden message or file",
                       description="Recover data hidden by encode (or by the original "
                                   "delimiter-format encoder).")
    p.add_argument("-i", "--image", required=True, help="stego file")
    _add_password_options(p, purpose="used when hiding")
    p.add_argument("--bits", type=int, choices=(1, 2), default=None,
                   help="bits per sample used when hiding (default: try 1, then 2)")
    p.add_argument("--out", metavar="PATH",
                   help="save the payload: into a folder (existing, or PATH ending in /) "
                        "under its stored name without overwriting, or exactly to PATH")
    p.add_argument("--legacy", action="store_true",
                   help="read only the original format (text + 64 zero bits)")
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("capacity", help="show how many bytes a cover holds",
                       description="Show how many bytes of data fit in a cover.")
    p.add_argument("-i", "--image", required=True, help="cover file")
    p.add_argument("--bits", type=int, choices=(1, 2), default=None,
                   help="bits per sample (default: show 1 and 2)")
    p.add_argument("--encrypted", action="store_true",
                   help="subtract the encryption overhead (44 bytes)")
    p.set_defaults(func=cmd_capacity)

    p = sub.add_parser("analyze", help="full steganalysis of one image",
                       description="Run every detector, the signature scan, payload "
                                   "detection and brute force on one image.")
    p.add_argument("-i", "--image", required=True, help="suspect image")
    p.add_argument("--report", metavar="DIR", help="write a report into DIR")
    p.add_argument("--format", choices=("md", "json", "html"), default=None,
                   help="report format (default md; needs --report)")
    p.add_argument("--no-bruteforce", action="store_true", help="skip the brute-force layouts")
    p.add_argument("--no-figures", action="store_true",
                   help="do not draw figures in md/html reports")
    p.add_argument("--top", type=int, default=5, help="brute-force candidates kept (default 5)")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    _add_password_options(p, purpose="to find and decrypt a scattered payload")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bitplanes", help="bit-plane grid and LSB difference mask",
                       description="Draw the bit planes of an image (visual attack) and, "
                                   "with --diff, mark the pixels whose LSBs differ from a cover.")
    p.add_argument("-i", "--image", required=True, help="image to inspect")
    p.add_argument("-o", "--output", required=True, help="PNG file for the bit-plane grid")
    p.add_argument("--channels", default="RGB", help="channel letters from RGBA (default RGB)")
    p.add_argument("--diff", metavar="COVER", help="cover image to compare LSBs against")
    p.add_argument("--diff-out", metavar="PNG",
                   help="where to write the diff mask (default: <output>_lsb_diff.png)")
    p.add_argument("--amplify", metavar="PNG", help="also write the amplified LSB image")
    p.set_defaults(func=cmd_bitplanes)

    p = sub.add_parser("bruteforce", help="try many bit layouts and rank the results",
                       description="Extract bytes under many bit layouts and rank them by "
                                   "how much they look like a payload (heuristic scores).")
    p.add_argument("-i", "--image", required=True,
                   help="suspect image (or a WAV file, with --wordlist)")
    p.add_argument("--top", type=int, default=10, help="candidates to show (default 10)")
    p.add_argument("--max-bit", type=int, default=1,
                   help="highest bit plane to try, 0-7 (default 1)")
    p.add_argument("--max-bytes", type=int, default=4096,
                   help="bytes read per layout before expanding good ones (default 4096)")
    p.add_argument("--save", metavar="DIR", help="save the best candidate's data into DIR")
    p.add_argument("--wordlist", metavar="FILE",
                   help="also try each line of FILE as the password of a scattered payload "
                        "(the only attack run on a WAV file)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.set_defaults(func=cmd_bruteforce)

    p = sub.add_parser("compare", help="cover-vs-stego histogram comparison",
                       description="Plot standard and even/odd pair histograms of a cover "
                                   "and its stego version and print pair statistics.")
    p.add_argument("cover", help="original image")
    p.add_argument("stego", help="stego image")
    p.add_argument("-o", "--output", default=None,
                   help="composite PNG (default: comparison_analysis.png next to STEGO); "
                        "four single plots are written beside it")
    p.add_argument("--channel", choices=("r", "g", "b"), default="b",
                   help="channel to plot (default b)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("plugins", help="list carriers and detectors",
                       description="List the registered carrier plug-ins and detectors.")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.set_defaults(func=cmd_plugins)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _error(message: str) -> int:
    """Print an error in argparse's style and return EXIT_ERROR."""
    print(f"{PROG}: error: {message}", file=sys.stderr)
    return EXIT_ERROR


def _check_combinations(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations argparse cannot express (exits with status 2)."""
    if args.command == "analyze" and args.format and not args.report:
        parser.error("analyze: --format needs --report DIR")
    if args.command == "bitplanes" and args.diff_out and not args.diff:
        parser.error("bitplanes: --diff-out needs --diff COVER")
    if args.command == "bruteforce" and args.max_bytes <= 0:
        parser.error("bruteforce: --max-bytes must be positive")
    if args.command == "bruteforce" and not 0 <= args.max_bit <= 7:
        parser.error("bruteforce: --max-bit must be 0..7")
    if args.command in ("analyze", "bruteforce") and args.top < 1:
        parser.error(f"{args.command}: --top must be at least 1")
    if args.command == "bitplanes":
        # Same letters as bitplanes._grid_rows (case-insensitive, repeats
        # allowed). Whether the image has an alpha channel is checked later.
        if not args.channels or any(c not in "RGBA" for c in args.channels.upper()):
            parser.error(f"bitplanes: --channels takes letters from RGBA, "
                         f"got {args.channels!r}")
    if getattr(args, "password", None) == "":
        # payload.pack() treats an empty password as none, which would
        # silently write a plaintext, unscattered frame (fail open).
        parser.error(f"{args.command}: --password is empty (an unset shell variable?); "
                     "give a password or use --password-prompt")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, run one subcommand and return the exit status.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        EXIT_OK (0) on success, EXIT_ERROR (1) on an expected failure (the
        message is printed to stderr), EXIT_INTERRUPTED (130) on Ctrl-C.

    Raises:
        SystemExit: From argparse, with status 2 for invalid arguments and
            0 for ``-h/--help``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_combinations(parser, args)
    try:
        return args.func(args)
    except CLIError as exc:
        return _error(str(exc))
    except FileNotFoundError as exc:
        return _error(f"file not found: {exc.filename}" if exc.filename else str(exc))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return EXIT_INTERRUPTED
    except ValueError as exc:
        # CapacityError, LossyFormatError, PayloadError, NoPayloadError and
        # WrongPasswordError are all ValueErrors with useful messages.
        import payload

        message = str(exc)
        if isinstance(exc, payload.WrongPasswordError):
            message += " (pass the right --password, or use --password-prompt)"
        return _error(message)
    except OSError as exc:
        # Includes PIL.UnidentifiedImageError (not an image) and permissions.
        return _error(str(exc))
    except wave.Error as exc:
        # The WAV carrier lets the stdlib's error through for a corrupt file.
        return _error(f"{getattr(args, 'image', 'input')}: not a readable PCM WAV file ({exc})")


# Run the CLI only when executed as a script, not when imported.
if __name__ == "__main__":
    sys.exit(main())
