"""Tkinter desktop app for LSB steganography and steganalysis.

The Steganography Image Analyzer hides data in images (and PCM WAV audio),
reads it back, and looks for data that something else has hidden. The window
has a toolbar, six tabs and a status bar:

    Encode       Pick a cover (PNG, BMP or TIFF; JPEG, GIF and other Pillow
                 formats as a cover only; 8/16-bit PCM WAV), type a message
                 or choose a file to hide, and set an optional password
                 (AES-256-GCM; by default it also scatters the bits over the
                 whole cover), zlib compression and 1 or 2 bits per sample.
                 A live meter shows payload bytes against the cover's
                 capacity. The stego file is saved as PNG, BMP, TIFF or WAV
                 only (never over the cover itself), and the tab previews
                 the cover, the stego image and a mask of the pixels whose
                 low bits changed.
    Decode       Recover a hidden payload (stego.reveal). Text is shown in
                 the tab; a hidden file (or any payload) can be saved. Images
                 in the original delimiter format (text followed by 64 zero
                 bits) are read by the legacy fallback when no password is
                 given, and its result is shown only when it reads as text
                 (a heuristic: the format has no header, so dark images
                 "decode" to noise). A wrong or missing password, no
                 payload, and a damaged payload each get their own message.
    Analyze      Runs report.analyze_image (every detector, STGA header
                 detection and brute force) in the background, then shows an
                 overall verdict banner, a table of detector results and the
                 text summary. "Export report" writes Markdown, HTML or JSON.
    Bitplanes    Any bit plane of any channel, the amplified LSB image and
                 the full bit-plane grid. Looking at the LSB plane is the
                 "visual attack" of Westfeld & Pfitzmann (1999); it is a
                 heuristic aid, not a test.
    Brute force  Tries about a hundred bit layouts (bruteforce.bruteforce)
                 in the background and lists the candidates ranked by a
                 heuristic score; a word list can be tried as passwords
                 against payloads scattered with a password key
                 (bruteforce.dictionary_attack). Any candidate can be saved.
    Compare      The original cover-vs-stego histogram comparison
                 (steganalysis.analyze_comparison), the even/odd pair
                 statistics of pairwise_stats.py, and a mask of the pixels
                 whose LSBs differ.

Toolbar: Open... (into the current tab), Resize for Test (a 400x400 copy of
the cover), Fill Max Space (Lorem ipsum that uses the cover's whole capacity,
a demo of full-rate embedding) and Reset All. Menu: File (Open, Exit) and
Help (About, which lists the registered carrier and detector plug-ins).

Threading. Slow work (hiding, decoding, analysis, brute force, figures) runs
on a worker thread started by SteganoApp.run_in_background(). The worker
function only computes: it receives plain values (paths, strings, numbers),
never a widget, and returns plain values or PIL images. Its result or
exception goes into a queue.Queue, and a root.after() poll on the Tk thread
picks it up and updates the widgets. Tk is not thread-safe, so this is the
only way results reach the UI. A tab's input can change while its job runs
(File > Open, typing a path, Reset All); the job's callbacks compare the
input with the path the job started with and drop a result that belongs to
another file (ToolTab._is_stale). matplotlib is switched to the Agg backend
before steganalysis (which uses pyplot) is imported, so figures can be drawn
off the Tk thread; the GUI only shows them as saved PNGs.

How it fits with the other modules:
    stego.py           hide(), reveal(), probe(), capacity_bytes() and
                       save_payload(): the Encode and Decode tabs.
    payload.py         The STGA frame and its errors (WrongPasswordError,
                       PayloadError); pack() sizes the capacity meter.
    carriers/          The plug-in registry: which file types can be covers
                       and outputs (it builds the file dialog filters).
    decoder.py         The legacy delimiter format (LEGACY_DELIMITER_BYTES),
                       read by legacy_bytes() here for the Decode fallback.
    report.py          analyze_image(), format_text(), write_report(): the
                       Analyze tab.
    detectors/         The statistical detectors behind report.py; listed in
                       Help > About.
    bitplanes.py       bitplane(), lsb_amplify(), save_bitplane_grid(),
                       lsb_diff(): the Bitplanes tab and the change masks.
    bruteforce.py      bruteforce(), dictionary_attack(), save_candidate():
                       the Brute force tab.
    steganalysis.py    analyze_comparison(): the Compare tab's histograms.
    pairwise_stats.py  pair_stats(): the Compare tab's numbers.
    main.py            The command-line front end (not used here).

Run it from the repository root::

    python gui.py

Side effects: generated files (the comparison figures, bit-plane grids and
the Resize for Test copy) go to ``results/`` in the current working
directory, created when first needed; a SteganoApp built with
``output_dir=...`` uses that folder instead. Stego files, recovered payloads
and reports go where the user chooses. The GUI prints nothing; results and
errors go to the tabs, the status bar and message boxes. The one exception is an
internal error (a bug) in a Tk callback, whose traceback is printed to stderr
as Tk would print it, besides being shown in a dialog.
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import textwrap
import threading
import traceback
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any, Callable

import matplotlib

# steganalysis.py imports matplotlib.pyplot. With an interactive backend
# (TkAgg or macosx) pyplot would tie figures to a GUI event loop and break
# when called from the worker thread. The GUI only displays saved PNGs, so the
# file-only Agg backend loses nothing. This must run before the import below.
matplotlib.use("Agg")

import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError

import bitplanes
import bruteforce
import carriers
import decoder
import detectors
import pairwise_stats
import payload
import report
import steganalysis
import stego
from carriers.base import CapacityError, LossyFormatError
from carriers.image_lsb import ImageLSBCarrier


APP_TITLE = "Steganography Image Analyzer"

# Short description used in Help > About.
PROJECT_DESCRIPTION = (
    "A cross-platform desktop application in Python and Tkinter for "
    "steganography and steganalysis on images. It hides and recovers least "
    "significant bit (LSB) payloads in lossless formats such as PNG, BMP and "
    "TIFF (and PCM WAV audio), visualizes bit planes, and reconstructs hidden "
    "messages and files. To find payloads hidden by others it combines "
    "histogram comparison, the chi-square attack, sample pair analysis, RS "
    "analysis, entropy heuristics and randomness tests, plus brute-force "
    "decoding over many bit layouts. Cover formats and detectors are plug-ins "
    "(the carriers/ and detectors/ packages), so audio, video or new "
    "decoding strategies can be added without changing this window."
)

# Filler text that "Fill Max Space" repeats and truncates to the cover's capacity.
LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris"
    " nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit"
    " esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in"
    " culpa qui officia deserunt mollit anim id est laborum. "
)

# Longest line of the Fill Max Space filler. A tk.Text lays out a line as a
# whole, so a capacity-sized filler on one line (9 MB for a 12-megapixel
# cover at 2 bits) froze the window for seconds per keystroke; in short lines
# typing stays instant.
FILLER_LINE_WIDTH = 76

# LOREM_IPSUM broken into lines: textwrap swaps spaces for newlines one for
# one, so the block is still ASCII and exactly as long as LOREM_IPSUM.
FILLER_BLOCK = textwrap.fill(LOREM_IPSUM.strip(), FILLER_LINE_WIDTH) + "\n"

# Legacy-format plausibility (see legacy_text()). Valid UTF-8 counts as text
# when at least this share of its characters is printable, newline, CR or
# tab: the rule report.py uses to call a payload text.
LEGACY_MIN_PRINTABLE = 0.9

# Bytes that are not valid UTF-8 are read as Latin-1 (the original encoder
# stored code points 128-255 as single bytes) only when at least this share
# is ASCII and every character is printable: bruteforce.py's legacy rule.
LEGACY_MIN_ASCII_LATIN1 = 0.6

# A Latin-1 character that is not printable text (C0/C1 controls, DEL, NBSP,
# soft hyphen); tab, LF and CR are allowed.
_LATIN1_NOT_TEXT = re.compile(r"[^\t\n\r\x20-\x7e\xa1-\xac\xae-\xff]")

# How often (ms) the Tk thread checks whether a background job has finished.
POLL_MS = 50

# Largest side, in pixels, of an image kept for on-screen display. Bigger
# images are shrunk once (in the worker thread where possible) so that
# re-fitting on every window resize stays fast. Display only: hiding,
# decoding and analysis always use the full-resolution file.
PREVIEW_MAX_PX = 2048

# Bytes shown in the hex view of a binary payload.
HEX_PREVIEW_BYTES = 512

# Characters of a payload or candidate kept in the text views.
TEXT_PREVIEW_CHARS = 200_000

# Verdict -> (banner text, foreground, background). The cut points (0.35 and
# 0.65) are the detectors' heuristic thresholds, not calibrated probabilities.
VERDICT_STYLE = {
    "clean": ("Clean", "#14532d", "#dcfce7"),
    "suspicious": ("Suspicious", "#713f12", "#fef3c7"),
    "likely_stego": ("Likely stego", "#7f1d1d", "#fee2e2"),
}
NEUTRAL_BANNER = ("#1f2937", "#e5e7eb")

# Colors for tk.Text widgets. ttk follows the light "clam" theme, but a plain
# tk.Text takes the system colors, which in macOS dark mode means white text;
# setting both explicitly keeps text readable on every platform.
TEXT_COLORS = {
    "background": "#ffffff",
    "foreground": "#1f2328",
    "insertbackground": "#1f2328",
    "selectbackground": "#b6d3f5",
    "selectforeground": "#1f2328",
}
TEXT_DISABLED_BACKGROUND = "#eeeeee"

# Readable names for file dialog filters, by extension.
_FORMAT_NAMES = {
    ".png": "PNG image",
    ".bmp": "BMP image",
    ".tif": "TIFF image",
    ".tiff": "TIFF image",
    ".wav": "WAV audio",
    ".jpg": "JPEG image (cover only)",
    ".jpeg": "JPEG image (cover only)",
    ".jpe": "JPEG image (cover only)",
    ".jfif": "JPEG image (cover only)",
    ".gif": "GIF image (cover only)",
    ".webp": "WebP image (cover only)",
}


# ---------------------------------------------------------------------------
# File types (built from the carrier registry, so plug-ins show up)
# ---------------------------------------------------------------------------

def _unique(items) -> list:
    """Return the items in order with duplicates removed."""
    return list(dict.fromkeys(items))


def cover_extensions() -> list[str]:
    """Extensions any registered carrier can read as a cover."""
    return _unique(ext for cls in carriers.list_carriers() for ext in cls.all_extensions())


def stego_extensions() -> list[str]:
    """Lossless extensions a registered carrier can write (and so decode)."""
    return _unique(ext for cls in carriers.list_carriers() for ext in cls.extensions)


def image_extensions() -> list[str]:
    """Extensions the image tools (analysis, bit planes, brute force) accept."""
    return list(ImageLSBCarrier.all_extensions())


def filetypes_for(extensions: list[str], label: str = "Supported files") -> list[tuple[str, str]]:
    """Build a Tk file dialog filter list.

    Args:
        extensions: Lowercase extensions with the dot, e.g. [".png", ".wav"].
        label: Name of the first entry, which matches every extension.

    Returns:
        [(label, "*.png *.wav"), ("PNG image", "*.png"), ..., ("All files", "*")],
        with extensions that share a name (".tif", ".tiff") grouped.
    """
    groups: dict[str, list[str]] = {}
    for ext in extensions:
        name = _FORMAT_NAMES.get(ext, f"{ext.lstrip('.').upper()} file")
        groups.setdefault(name, []).append(ext)
    types = [(label, " ".join(f"*{e}" for e in extensions))]
    types += [(name, " ".join(f"*{e}" for e in exts)) for name, exts in groups.items()]
    types.append(("All files", "*"))
    return types


def is_image_path(path: str) -> bool:
    """True if the image carrier (not audio or another plug-in) handles ``path``.

    Unregistered extensions count as images, since only Pillow can try them.
    """
    try:
        return issubclass(carriers.carrier_for_path(path), ImageLSBCarrier)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Worker-side helpers: plain values in, plain values or PIL images out.
# None of these may touch a Tk widget; they run on the background thread.
# ---------------------------------------------------------------------------

def display_copy(img: Image.Image, max_px: int = PREVIEW_MAX_PX,
                 resample=Image.Resampling.LANCZOS) -> Image.Image:
    """Return a copy of ``img`` no larger than ``max_px`` on either side.

    Modes other than RGB/RGBA are converted to RGB (palette, grayscale,
    CMYK, ...) so Tk can show the result.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    else:
        img = img.copy()
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), resample)
    return img


def load_preview(path: str, max_px: int = PREVIEW_MAX_PX) -> Image.Image:
    """Open an image file and return a display-sized copy.

    Raises:
        FileNotFoundError, PIL.UnidentifiedImageError: As Image.open does.
    """
    with Image.open(path) as img:
        img.load()
        return display_copy(img, max_px)


def describe_cover(path: str) -> tuple[str, Image.Image | None]:
    """Return a one-line description and a preview image (None for audio).

    Args:
        path: A cover file of any registered carrier.
    """
    if is_image_path(path):
        with Image.open(path) as img:
            img.load()
            text = f"{img.width} x {img.height} px, {img.format or 'image'} ({img.mode})"
            return text, display_copy(img)
    if Path(path).suffix.lower() == ".wav":
        with wave.open(path, "rb") as w:
            seconds = w.getnframes() / float(w.getframerate() or 1)
            text = (f"WAV audio, {w.getnchannels()} ch, {8 * w.getsampwidth()}-bit, "
                    f"{w.getframerate():,} Hz, {seconds:.1f} s")
        return text, None
    return f"{carriers.carrier_for_path(path).name} cover", None


def inspect_cover(path: str) -> dict[str, Any]:
    """Measure a cover for the Encode tab.

    Returns:
        {"capacity": {1: bytes, 2: bytes}, "description": str,
        "preview": PIL image or None, "carrier": registry name}, where the
        capacity is the largest frame body stego.hide() can store
        (stego.capacity_bytes) at 1 and 2 bits per sample.
    """
    capacity = {bits: stego.capacity_bytes(path, bits) for bits in (1, 2)}
    description, preview = describe_cover(path)
    return {
        "capacity": capacity,
        "description": description,
        "preview": preview,
        "carrier": carriers.carrier_for_path(path).name,
    }


def frame_body_len(data: bytes, *, filename: str | None = None,
                   compress: bool = False, encrypted: bool = False) -> int:
    """Exact size of the frame body stego.hide() will store for ``data``.

    Uses payload.pack() itself (without a password, so no scrypt work) and
    adds the fixed AES-GCM overhead (salt + nonce + tag) when ``encrypted``;
    GCM ciphertext is exactly as long as its plaintext, so the result is
    exact. The 14-byte header is not counted: the capacity it is compared to
    (stego.capacity_bytes) already leaves room for it.
    """
    body = len(payload.pack(data, filename=filename, compress=compress)) - payload.HEADER_LEN
    return body + (stego.ENCRYPTION_OVERHEAD if encrypted else 0)


def legacy_bytes(path: str) -> bytes | None:
    """Read the raw bytes of a legacy delimiter-format message.

    The same reading as decoder.decode_legacy(): the image converted to RGB,
    the LSB of every R, G and B value in row-major order packed MSB-first
    into bytes, cut at the first 8 zero bytes on a byte boundary. Unlike
    decode_legacy() the bytes are returned before any text decoding, so
    their count is exact and the caller can judge whether they are text.

    Args:
        path: The image.

    Returns:
        The bytes before the delimiter (b"" when the stream starts with
        it), or None when there is no delimiter.

    Raises:
        FileNotFoundError, PIL.UnidentifiedImageError: As Image.open does.
    """
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    lsb = rgb.reshape(-1) & 1
    # Whole bytes only: packbits would pad a partial last byte with zeros.
    whole = lsb.size - lsb.size % 8
    stream = np.packbits(lsb[:whole]).tobytes()
    end = stream.find(decoder.LEGACY_DELIMITER_BYTES)
    return None if end == -1 else stream[:end]


def legacy_text(raw: bytes) -> tuple[str, str] | None:
    """Decide whether legacy-format bytes read as a text message (heuristic).

    The legacy format has no header and no checksum, and any 8 zero bytes in
    the LSB stream look like its delimiter. Flat black or dark regions are
    full of them, so on a clean image (or a payload read with the wrong
    password or bit depth) the bytes in front of the first zero run are
    LSB noise. They are reported as a message only when they look like
    text:

    * valid UTF-8 (strict) with at least LEGACY_MIN_PRINTABLE (90%) of the
      characters printable, newline, CR or tab, the rule report.py uses to
      call a payload text; or
    * not valid UTF-8 but every byte printable Latin-1 and at least
      LEGACY_MIN_ASCII_LATIN1 (60%) ASCII, bruteforce.py's rule for the
      original encoder, which stored code points 128-255 as single bytes;
    * and in both cases at least one letter or digit, so a few stray
      printable bytes in front of a dark region are not called a message.

    This is a heuristic filter, not a proof either way: noise from a natural
    image fails it after a few bytes (only about 37% of random bytes are
    printable ASCII, and invalid UTF-8 appears almost at once), but a very
    short false hit can still pass.

    Args:
        raw: Bytes from legacy_bytes().

    Returns:
        (text, encoding) with encoding "utf-8" or "latin-1", or None when
        the bytes are empty or do not read as text.
    """
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        codes = np.frombuffer(raw, dtype=np.uint8)
        if np.count_nonzero(codes < 0x80) < LEGACY_MIN_ASCII_LATIN1 * codes.size:
            return None
        text = raw.decode("latin-1")
        if _LATIN1_NOT_TEXT.search(text):
            return None
        encoding = "latin-1"
    else:
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
        if printable < LEGACY_MIN_PRINTABLE * len(text):
            return None
        encoding = "utf-8"
    if not any(ch.isalnum() for ch in text):
        return None
    return text, encoding


def decode_file(path: str, password: str | None = None, bits: int | None = None) -> dict[str, Any]:
    """Recover a hidden payload, falling back to the legacy text format.

    Tries stego.reveal() first. A wrong password or a damaged frame is
    reported, never "repaired" by the legacy reader. Only when no STGA
    header is found at all (NoPayloadError) in an image is the legacy
    reader tried, and only when it can apply: the legacy format has no
    password and uses 1 bit per sample, so it is skipped when a password is
    given or 2 bits per sample are forced. Its result is accepted only when
    it reads as text (legacy_text(), a heuristic), because the format has
    no header and dark images "decode" to noise.

    Args:
        path: The stego file.
        password: Password, or None.
        bits: 1, 2, or None to try both.

    Returns:
        {"format": "stga" or "legacy", "payload": payload.Payload,
        "header": stego.probe() dict or None, "preview": PIL image or None};
        a legacy result also has "legacy_encoding" ("utf-8" or "latin-1")
        and "legacy_bytes" (the stored byte count). The payload data is the
        text as UTF-8, which is what the tab shows, counts and saves.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        payload.WrongPasswordError: Encrypted frame, password missing/wrong.
        stego.NoPayloadError: Nothing found (also a wrong password when the
            payload was scattered with it). The message says whether the
            legacy format was tried and, when ``bits`` was forced, whether
            a frame exists at the other bit depth.
        payload.PayloadError: Damaged frame (CRC mismatch, cropped file).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(2, "No such file", path)
    try:
        carrier_cls = carriers.carrier_for_path(path)
    except ValueError:
        carrier_cls = None  # unregistered extension: only the legacy reader can try
    image = carrier_cls is None or issubclass(carrier_cls, ImageLSBCarrier)
    preview = None
    if image:
        try:
            preview = load_preview(path)
        except (OSError, UnidentifiedImageError):
            preview = None

    reasons: list[str] = []
    if carrier_cls is not None:
        try:
            found = stego.reveal(path, password=password, bits=bits)
        except stego.NoPayloadError as exc:
            if not image:
                raise
            reasons.append(str(exc))
        else:
            header = stego.probe(path, password=password, bits=bits)
            return {"format": "stga", "payload": found, "header": header, "preview": preview}
        if bits is not None:
            # A frame hidden at the other bit depth reads as nothing here.
            other = stego.probe(path, password=password, bits=None)
            if other is not None and other["bits_per_sample"] != bits:
                reasons.append(f"but an STGA frame was found at {other['bits_per_sample']} "
                               "bit(s) per sample: set Bits per sample to Auto")
    else:
        reasons.append(f"no hidden payload found in {path!r}")

    if password:
        reasons.append("the legacy format (text followed by 64 zero bits) was not tried "
                       "because it has no password: clear the password to try it")
    elif bits == 2:
        reasons.append("the legacy format (text followed by 64 zero bits) was not tried "
                       "because it uses 1 bit per sample")
    else:
        raw = legacy_bytes(path)
        hit = legacy_text(raw) if raw else None
        if hit is not None:
            text, encoding = hit
            return {"format": "legacy", "payload": payload.Payload(text.encode("utf-8")),
                    "header": None, "preview": preview,
                    "legacy_encoding": encoding, "legacy_bytes": len(raw)}
        if raw:
            reasons.append("the bytes in front of the first 64 zero bits (the legacy "
                           "format's delimiter) do not read as text")
        else:
            reasons.append("no legacy-format text (text followed by 64 zero bits) was found")
    raise stego.NoPayloadError("; ".join(reasons))


def hex_dump(data: bytes, limit: int = HEX_PREVIEW_BYTES) -> str:
    """Classic 16-bytes-per-line hex view of the first ``limit`` bytes."""
    lines = []
    for offset in range(0, min(len(data), limit), 16):
        chunk = data[offset:offset + 16]
        hexes = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset:08x}  {hexes:<47}  {text}")
    if len(data) > limit:
        lines.append(f"... {len(data) - limit:,} more bytes")
    return "\n".join(lines)


def one_line(text: str, limit: int = 120) -> str:
    """Collapse whitespace and control characters for a single table cell."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def safe_stem(path: str) -> str:
    """File-name-safe stem of ``path`` for generated output names."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(path).stem).strip("._")
    return stem or "image"


def same_file(a: str | os.PathLike, b: str | os.PathLike) -> bool:
    """True if ``a`` and ``b`` name the same existing file.

    The operating system decides (os.path.samefile), so a different
    spelling, letter case on a case-insensitive disk, ".." or a link all
    count. If either file does not exist they cannot be the same file.
    """
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def describe_error(exc: BaseException) -> tuple[str, str]:
    """Turn an exception into a (title, message) pair a user can act on.

    The order matters: WrongPasswordError and NoPayloadError are both
    PayloadErrors, and CapacityError / LossyFormatError are ValueErrors.
    """
    if isinstance(exc, payload.WrongPasswordError):
        return ("Wrong or missing password",
                "This file holds an encrypted payload. Enter the password it was "
                f"hidden with and try again.\n\nDetails: {exc}")
    if isinstance(exc, stego.NoPayloadError):
        return ("No hidden payload found",
                "No payload header was found, and no text in the legacy format (text "
                "followed by 64 zero bits) was recognized. If the data was hidden with "
                "a password and key scatter, the same password is needed to find "
                "it: a wrong password looks exactly like no payload. The details say "
                f"whether the legacy format was tried.\n\nDetails: {exc}")
    if isinstance(exc, payload.PayloadError):
        return ("Damaged payload",
                "A payload header was found but the data behind it is damaged (CRC "
                "mismatch, or the file was cropped, resized or re-saved).\n\n"
                f"Details: {exc}")
    if isinstance(exc, LossyFormatError):
        return ("Lossless output needed",
                "Stego files must be saved in a lossless format (PNG, BMP, TIFF, or "
                "WAV for audio). Lossy formats such as JPEG destroy the hidden bits."
                f"\n\nDetails: {exc}")
    if isinstance(exc, CapacityError):
        return ("Payload too large",
                "The payload does not fit in this cover. Use a larger cover, 2 bits "
                f"per sample, compression, or less data.\n\nDetails: {exc}")
    if isinstance(exc, FileNotFoundError):
        return ("File not found", f"{exc.filename or exc}")
    if isinstance(exc, UnidentifiedImageError):
        return ("Not an image", f"The file could not be read as an image.\n\nDetails: {exc}")
    if isinstance(exc, wave.Error):
        return ("Unreadable WAV file", f"Only PCM WAV audio is supported.\n\nDetails: {exc}")
    if isinstance(exc, (ValueError, TypeError, OSError)):
        return ("Error", str(exc) or type(exc).__name__)
    return ("Unexpected error", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ImageTab(ttk.Frame):
    """Notebook tab that shows one image scaled to fit the tab.

    The image is rescaled from the stored original whenever the frame is
    resized, preserving aspect ratio. There are no scrollbars or zoom
    controls. Shows placeholder text (default "No image") when empty.

    Attributes:
        label: ttk.Label that displays the scaled image (or placeholder text).
    """

    def __init__(self, parent, placeholder: str = "No image",
                 resample=Image.Resampling.LANCZOS,
                 on_error: Callable[[str, str], None] | None = None):
        """Create an empty tab.

        Args:
            parent: Containing widget, normally a ttk.Notebook.
            placeholder: Text shown while there is no image.
            resample: Pillow filter used when scaling. LANCZOS suits photos;
                NEAREST keeps black/white bit planes and masks crisp
                instead of blurring them to gray.
            on_error: Called as on_error(title, message) when a file cannot
                be opened. Defaults to an error message box.
        """
        super().__init__(parent)
        self._placeholder = placeholder
        self._resample = resample
        self._on_error = on_error or (lambda title, msg: messagebox.showerror(title, msg))
        # Display-sized source image; every resize is computed from this so
        # repeated resizing does not degrade quality.
        self._pil_original: Image.Image | None = None
        # Current scaled PhotoImage. Tk does not hold a Python reference, so
        # it must be kept alive here or the image disappears when GC'd.
        self._photo: ImageTk.PhotoImage | None = None

        self.label = ttk.Label(self, text=placeholder, anchor="center", justify="center")
        self.label.pack(fill="both", expand=True)

        # Re-render when the tab area resizes
        self.bind("<Configure>", self._on_resize)

    @property
    def image(self) -> Image.Image | None:
        """The image being shown (display-sized), or None."""
        return self._pil_original

    def set_image_path(self, path: str):
        """Load an image file and display it.

        Args:
            path: Path to an image file Pillow can open.

        Side effects:
            If the file cannot be opened or read, reports it through
            ``on_error`` and leaves the current image unchanged; the error
            is not re-raised.
        """
        try:
            img = load_preview(path)
        except Exception as e:
            self._on_error("Error", f"Failed to open image: {e}")
            return
        self.set_pil_image(img)

    def set_pil_image(self, img: Image.Image):
        """Display an already-loaded PIL image.

        Modes other than RGB/RGBA (e.g. palette, grayscale or 1-bit) are
        converted first, and images larger than PREVIEW_MAX_PX are shrunk
        once for display. Rendering is scheduled 10 ms later with after()
        rather than done immediately, and is redone by _on_resize whenever
        the tab later changes size.

        Args:
            img: Image to show. A display copy is stored as the new original.
        """
        self._pil_original = display_copy(img, PREVIEW_MAX_PX, self._resample)
        self.after(10, self._render_fit)

    def set_message(self, text: str):
        """Remove the image and show ``text`` instead (e.g. for audio covers)."""
        self._pil_original = None
        self._photo = None
        self.label.configure(image="", text=text)
        self.label.image = None

    def clear(self):
        """Remove the image and show the placeholder text."""
        self.set_message(self._placeholder)

    # ---- internals ----
    def _on_resize(self, _event):
        """Handle <Configure> events by re-fitting the image, if any.

        Args:
            _event: Tk event object (unused).
        """
        if self._pil_original is not None:
            self._render_fit()

    def _render_fit(self):
        """Scale the original image to fit the frame and show it.

        Uses the largest uniform scale that fits inside the frame minus a
        16 px margin (floored at 5%), resampling with the tab's filter. Does
        nothing if there is no image, the image has a zero dimension, or the
        widget no longer exists.
        """
        if self._pil_original is None or not self.winfo_exists():
            return
        # winfo_* report 1 before the widget is first laid out. The max(1, ...)
        # clamps keep w and h at least 1 after the 16 px margin is
        # subtracted, so they never go negative. (The divisors below are
        # img_w and img_h, which the == 0 check guards, and scale is also
        # floored at 0.05.)
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        # Pad a bit so the image doesn't smack the edges
        w = max(1, w - 16)
        h = max(1, h - 16)
        img_w, img_h = self._pil_original.size
        if img_w == 0 or img_h == 0:
            return
        # Fit-inside scale: the tighter of the two axes wins, so the whole
        # image is visible and the aspect ratio is preserved. Small images
        # are scaled up as well as large ones down.
        scale = min(w / img_w, h / img_h)
        scale = max(scale, 0.05)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = self._pil_original.resize((new_w, new_h), self._resample)
        # master= ties the PhotoImage to this window's Tk interpreter rather
        # than to whichever Tk root happens to be the default.
        self._photo = ImageTk.PhotoImage(resized, master=self.label)
        self.label.configure(image=self._photo, text="")
        # Extra reference on the widget itself (common Tkinter idiom) so the
        # PhotoImage is not garbage-collected while displayed.
        self.label.image = self._photo


def _scrolled_text(parent, **kwargs) -> tuple[ttk.Frame, tk.Text]:
    """Return (frame, text): a Text widget with a vertical scrollbar."""
    frame = ttk.Frame(parent, padding=0)
    kwargs.setdefault("wrap", "word")
    for option, value in TEXT_COLORS.items():
        kwargs.setdefault(option, value)
    text = tk.Text(frame, undo=False, **kwargs)
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.grid(row=0, column=0, sticky="nsew")
    bar.grid(row=0, column=1, sticky="ns")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    return frame, text


def _set_readonly_text(widget: tk.Text, content: str):
    """Replace the contents of a read-only (disabled) Text widget."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


def _scrolled_tree(parent, columns: tuple[str, ...], **kwargs) -> tuple[ttk.Frame, ttk.Treeview]:
    """Return (frame, tree): a headings-only Treeview with a vertical scrollbar."""
    frame = ttk.Frame(parent, padding=0)
    tree = ttk.Treeview(frame, columns=columns, show="headings", **kwargs)
    bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.grid(row=0, column=0, sticky="nsew")
    bar.grid(row=0, column=1, sticky="ns")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    return frame, tree


class _Job:
    """One background computation and the callbacks for its outcome.

    Attributes:
        results: Queue that receives exactly one (ok, value) pair from the
            worker thread: (True, return value) or (False, exception).
    """

    def __init__(self, func: Callable[[], Any], on_done: Callable[[Any], None],
                 on_error: Callable[[BaseException], None]):
        self.func = func
        self.on_done = on_done
        self.on_error = on_error
        self.results: queue.Queue = queue.Queue(maxsize=1)

    def run(self):
        """Worker-thread body: compute and post the outcome. Never touches Tk."""
        try:
            self.results.put((True, self.func()))
        except BaseException as exc:  # noqa: BLE001 - every outcome must reach the UI
            # Catch everything (even SystemExit) so the job always reports
            # back; otherwise its tab would stay "busy" forever.
            self.results.put((False, exc))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

class ToolTab(ttk.Frame):
    """Base class for the notebook tabs.

    Gives each tab a reference to the app, a "busy" flag that blocks a
    second run while one is in progress, and helpers for the common
    "path + Browse..." rows.

    Attributes:
        app: The SteganoApp that owns this tab.
        title: Text on the notebook tab.
    """

    title = "Tab"

    def __init__(self, notebook: ttk.Notebook, app: "SteganoApp"):
        super().__init__(notebook, padding=8)
        self.app = app
        self._busy = False
        # Buttons disabled while this tab's job runs.
        self._action_buttons: list[ttk.Button] = []

    @property
    def busy(self) -> bool:
        """True while this tab's background job is running."""
        return self._busy

    def start_job(self, func: Callable[[], Any], on_done: Callable[[Any], None],
                  on_error: Callable[[BaseException], None] | None = None,
                  status: str | None = None) -> bool:
        """Run ``func`` in the background, one job per tab at a time.

        Args:
            func: Worker function. It must not touch any widget.
            on_done: Called on the Tk thread with func's return value.
            on_error: Called on the Tk thread with func's exception.
                Defaults to an error message box built by describe_error().
            status: Status bar text while the job runs.

        Returns:
            True if the job started, False if this tab was already busy.
        """
        if self._busy:
            self.app.set_status(f"{self.title}: still working, please wait...")
            return False
        self._busy = True
        for button in self._action_buttons:
            button.state(["disabled"])

        def finish():
            self._busy = False
            for button in self._action_buttons:
                button.state(["!disabled"])

        def done(value):
            finish()
            on_done(value)

        def failed(exc):
            finish()
            (on_error or self.app.show_exception)(exc)

        self.app.run_in_background(func, done, failed, status=status)
        return True

    def _is_stale(self, var: tk.StringVar, path: str) -> bool:
        """Check, when a job finishes, whether its input is still the chosen one.

        A job keeps running when File > Open, Browse..., typing in the entry
        or Reset All changes the tab's input, and its result belongs to the
        old file. The done and error callbacks call this first and drop the
        outcome when it returns True, so a result is never shown (or saved)
        under another file's name.

        Args:
            var: The input variable the job read its path from.
            path: The (stripped) path the job was started with.

        Returns:
            True if ``var`` no longer holds ``path``. Then the tab's
            _discard_stale() has run and the status bar says so.
        """
        if var.get().strip() == path:
            return False
        self._discard_stale()
        self.app.set_status(f"{self.title}: discarded the result for "
                            f"{os.path.basename(path) or 'the previous file'}, because "
                            "another file was chosen while it was computed")
        return True

    def _discard_stale(self):
        """Hook: tidy up after a stale result was dropped (see _is_stale)."""

    def _path_row(self, parent, row: int, label: str, var: tk.StringVar,
                  browse: Callable[[], None], apply: Callable[[], None] | None = None) -> ttk.Entry:
        """Add "label [entry] Browse..." on one grid row of ``parent``.

        Args:
            apply: Called when Return is pressed in the entry (to load a
                typed path).
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(parent, text="Browse…", command=browse).grid(
            row=row, column=2, sticky="e", padx=(6, 0), pady=2)
        if apply is not None:
            entry.bind("<Return>", lambda _e: apply())
        parent.columnconfigure(1, weight=1)
        return entry

    def _password_entry(self, parent, var: tk.StringVar, show_var: tk.BooleanVar) -> ttk.Frame:
        """A masked password entry with a "Show" check box, in its own frame."""
        frame = ttk.Frame(parent, padding=0)
        entry = ttk.Entry(frame, textvariable=var, show="*", width=24)
        entry.pack(side="left", fill="x", expand=True)

        def toggle():
            entry.configure(show="" if show_var.get() else "*")

        ttk.Checkbutton(frame, text="Show", variable=show_var, command=toggle).pack(
            side="left", padx=(6, 0))
        return frame

    # Hooks for File > Open and Reset All; overridden by each tab.
    def open_filetypes(self) -> list[tuple[str, str]]:
        """File dialog filters for File > Open in this tab."""
        return filetypes_for(image_extensions(), "Images")

    def open_path(self, path: str) -> bool | None:
        """Load ``path`` as this tab's main input (File > Open).

        Returns:
            False if the tab refused the file (it is busy); otherwise None.
        """
        raise NotImplementedError

    def reset(self):
        """Clear this tab's inputs and results."""
        raise NotImplementedError


class EncodeTab(ToolTab):
    """Hide a message or a file in a cover and save the stego file.

    State:
        capacity: {1: bytes, 2: bytes} payload room of the current cover
            (stego.capacity_bytes), or None while unknown or measuring.
        stego_path: The last stego file written, or None.
        last_summary: stego.hide()'s summary dict for that file, or None.
    """

    title = "Encode"

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.cover_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="text")
        self.file_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.show_password_var = tk.BooleanVar(value=False)
        self.compress_var = tk.BooleanVar(value=False)
        self.bits_var = tk.IntVar(value=1)
        self.scatter_var = tk.BooleanVar(value=True)

        self.capacity: dict[int, int] | None = None
        self.stego_path: str | None = None
        self.last_summary: dict[str, Any] | None = None
        # Compressed size of a file payload, keyed by (path, mtime, size),
        # measured in the background because the file may be large.
        self._file_body: dict[tuple, int] = {}
        self._measuring: set[tuple] = set()
        self._meter_after: str | None = None
        self._build()

    # ---- layout ----
    def _build(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=1)

        cover = ttk.Labelframe(left, text="Cover")
        cover.grid(row=0, column=0, sticky="ew")
        self._path_row(cover, 0, "File", self.cover_var, self.select_cover,
                       apply=lambda: self.set_cover(self.cover_var.get().strip()))
        self.cover_info = ttk.Label(cover, text="PNG, BMP, TIFF, JPEG (read only) or PCM WAV",
                                    style="Muted.TLabel")
        self.cover_info.grid(row=1, column=0, columnspan=3, sticky="w")

        data = ttk.Labelframe(left, text="Payload")
        data.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        modes = ttk.Frame(data, padding=0)
        modes.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(modes, text="Text message", value="text", variable=self.mode_var,
                        command=self._mode_changed).pack(side="left")
        ttk.Radiobutton(modes, text="File", value="file", variable=self.mode_var,
                        command=self._mode_changed).pack(side="left", padx=(12, 0))

        text_frame, self.message_text = _scrolled_text(data, height=8)
        text_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(6, 6))
        # <<Modified>> also fires for paste and programmatic inserts, unlike
        # <KeyRelease>, so the meter always matches the text.
        self.message_text.bind("<<Modified>>", self._on_text_modified)
        # Tk's Text class binding for Control-o inserts a newline before the
        # application-wide shortcut runs; this widget binding opens instead.
        self.message_text.bind("<Control-o>", self._open_shortcut)
        self.file_entry = self._path_row(data, 2, "File", self.file_var, self.select_payload_file,
                                         apply=lambda: self.set_payload_file(self.file_var.get().strip()))
        data.columnconfigure(1, weight=1)
        data.rowconfigure(1, weight=1)

        options = ttk.Labelframe(left, text="Options")
        options.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(options, text="Password").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._password_entry(options, self.password_var, self.show_password_var).grid(
            row=0, column=1, columnspan=3, sticky="ew")
        ttk.Checkbutton(options, text="Scatter bits using the password", variable=self.scatter_var).grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(options, text="Compress (zlib)", variable=self.compress_var,
                        command=self.update_meter).grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Label(options, text="Bits per sample").grid(row=3, column=0, sticky="w", pady=(4, 0))
        bits = ttk.Frame(options, padding=0)
        bits.grid(row=3, column=1, columnspan=3, sticky="w", pady=(4, 0))
        for value, text in ((1, "1 (least visible)"), (2, "2 (twice the room)")):
            ttk.Radiobutton(bits, text=text, value=value, variable=self.bits_var,
                            command=self.update_meter).pack(side="left", padx=(0, 12))
        options.columnconfigure(1, weight=1)
        self.password_var.trace_add("write", lambda *_: self._schedule_meter())

        meter = ttk.Labelframe(left, text="Capacity")
        meter.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.meter_bar = ttk.Progressbar(meter, orient="horizontal", mode="determinate")
        self.meter_bar.grid(row=0, column=0, sticky="ew")
        self.meter_label = ttk.Label(meter, text="Choose a cover to see its capacity")
        self.meter_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.remaining_label = ttk.Label(meter, text="", style="Muted.TLabel")
        self.remaining_label.grid(row=2, column=0, sticky="w")
        meter.columnconfigure(0, weight=1)

        actions = ttk.Frame(left, padding=(0, 8, 0, 0))
        actions.grid(row=4, column=0, sticky="ew")
        hide_button = ttk.Button(actions, text="Hide and Save…", style="Accent.TButton",
                                 command=self.hide)
        hide_button.pack(side="left")
        self._action_buttons.append(hide_button)
        self.result_label = ttk.Label(actions, text="", style="Muted.TLabel", wraplength=360)
        self.result_label.pack(side="left", padx=(10, 0), fill="x", expand=True)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        right = ttk.Frame(paned, padding=0)
        paned.add(right, weight=2)
        self.previews = ttk.Notebook(right)
        self.previews.pack(fill="both", expand=True)
        self.preview_cover = ImageTab(self.previews, "No cover selected", on_error=self.app.show_error)
        self.preview_stego = ImageTab(self.previews, "Hide something to see the stego image",
                                      on_error=self.app.show_error)
        self.preview_diff = ImageTab(self.previews, "Changed pixels appear here after hiding",
                                     resample=Image.Resampling.NEAREST, on_error=self.app.show_error)
        self.previews.add(self.preview_cover, text="Cover")
        self.previews.add(self.preview_stego, text="Stego")
        self.previews.add(self.preview_diff, text="Changed LSBs")
        self._mode_changed()

    # ---- inputs ----
    def open_filetypes(self):
        return filetypes_for(cover_extensions(), "Covers (images and audio)")

    def open_path(self, path: str):
        self.set_cover(path)

    def select_cover(self):
        """Ask for a cover file and make it the active cover."""
        path = self.app.ask_open_path("Choose a cover", self.open_filetypes())
        if path:
            self.set_cover(path)

    def set_cover(self, path: str | os.PathLike | None):
        """Make ``path`` the cover; measure its capacity in the background.

        The previous stego preview is cleared. Measuring is not exclusive:
        choosing another cover while one is measured simply makes the older
        result stale, and it is ignored when it arrives.
        """
        path = os.fspath(path) if path else ""
        self.cover_var.set(path)
        self.capacity = None
        self.stego_path = None
        self.last_summary = None
        self.preview_stego.clear()
        self.preview_diff.clear()
        self.result_label.configure(text="")
        if not path:
            self.preview_cover.clear()
            self.cover_info.configure(text="PNG, BMP, TIFF, JPEG (read only) or PCM WAV")
            self.update_meter()
            return
        self.cover_info.configure(text="Measuring capacity…")
        self.update_meter()

        def done(info):
            if self.cover_var.get() != path:
                return  # a newer cover was chosen meanwhile
            self.capacity = info["capacity"]
            self.cover_info.configure(text=f"{info['description']} · {info['carrier']} carrier")
            if info["preview"] is not None:
                self.preview_cover.set_pil_image(info["preview"])
            else:
                self.preview_cover.set_message(f"{info['description']}\n\n(no preview for audio)")
            self.previews.select(self.preview_cover)
            self.update_meter()
            self.app.set_status(f"Cover: {os.path.basename(path)} "
                                f"({self.capacity[1]:,} bytes at 1 bit/sample)")

        def failed(exc):
            if self.cover_var.get() != path:
                return
            title, message = describe_error(exc)
            self.cover_info.configure(text=f"{title}: {one_line(str(exc), 90)}")
            self.preview_cover.set_message("Cannot use this file as a cover")
            self.update_meter()
            self.app.show_error(title, message)

        self.app.run_in_background(lambda: inspect_cover(path), done, failed,
                                   status=f"Measuring {os.path.basename(path)}…")

    def select_payload_file(self):
        """Ask for any file to hide."""
        path = self.app.ask_open_path("Choose a file to hide", [("All files", "*")])
        if path:
            self.set_payload_file(path)

    def set_payload_file(self, path: str | os.PathLike | None):
        """Hide the file at ``path`` (switches the payload to File mode)."""
        self.file_var.set(os.fspath(path) if path else "")
        self.mode_var.set("file")
        self._mode_changed()

    def set_message(self, text: str):
        """Replace the message text (switches the payload to Text mode)."""
        self.mode_var.set("text")
        self._mode_changed()
        self.message_text.delete("1.0", "end")
        self.message_text.insert("1.0", text)
        self.update_meter()

    def get_message(self) -> str:
        """The message exactly as typed ("end-1c" drops Tk's own final newline)."""
        return self.message_text.get("1.0", "end-1c")

    def _mode_changed(self):
        """Enable the text box or the file row to match the payload mode."""
        text_mode = self.mode_var.get() == "text"
        self.message_text.configure(state="normal" if text_mode else "disabled",
                                    background=TEXT_COLORS["background"] if text_mode
                                    else TEXT_DISABLED_BACKGROUND)
        self.file_entry.state(["disabled"] if text_mode else ["!disabled"])
        self.update_meter()

    def _open_shortcut(self, _event):
        """Control-o in the message box: File > Open, without the newline."""
        self.app.open_file()
        return "break"

    def _on_text_modified(self, _event):
        """<<Modified>> handler: reset Tk's flag and refresh the meter soon."""
        self.message_text.edit_modified(False)
        self._schedule_meter()

    def _schedule_meter(self):
        """Refresh the meter 120 ms after the last change (debounced)."""
        if self._meter_after is not None:
            self.after_cancel(self._meter_after)
        self._meter_after = self.after(120, self.update_meter)

    # ---- capacity meter ----
    def payload_body_len(self) -> int | None:
        """Bytes the frame body will take, or None if not known yet.

        Text is measured directly (fast). A file is measured by its size, or,
        when compressed, by a background job whose result is cached; None is
        returned until it arrives.
        """
        encrypted = bool(self.password_var.get())
        compress = self.compress_var.get()
        if self.mode_var.get() == "text":
            return frame_body_len(self.get_message().encode("utf-8"),
                                  compress=compress, encrypted=encrypted)
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            return None
        name = os.path.basename(path)
        stat = os.stat(path)
        if not compress:
            # Uncompressed FILE record: 2-byte name length + UTF-8 name + data.
            body = stat.st_size + 2 + len(name.encode("utf-8"))
            return body + (stego.ENCRYPTION_OVERHEAD if encrypted else 0)
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key in self._file_body:
            return self._file_body[key] + (stego.ENCRYPTION_OVERHEAD if encrypted else 0)
        if key in self._measuring:
            return None  # already being measured
        self._measuring.add(key)

        def measure():
            return frame_body_len(Path(path).read_bytes(), filename=name, compress=True)

        def done(size):
            self._measuring.discard(key)
            self._file_body[key] = size
            self.update_meter()

        def failed(_exc):
            self._measuring.discard(key)

        self.app.run_in_background(measure, done, failed,
                                   status="Measuring compressed size…")
        return None

    def update_meter(self) -> tuple[int | None, int | None]:
        """Redraw the capacity meter.

        Returns:
            (used, available): frame body bytes and the cover's room at the
            selected bits per sample; either is None while unknown.
        """
        self._meter_after = None
        cover = self.cover_var.get()
        bits = self.bits_var.get()
        room = self.capacity.get(bits) if self.capacity else None
        try:
            used = self.payload_body_len()
        except (OSError, ValueError):
            used = None
        self.meter_bar.configure(style="Horizontal.TProgressbar")
        self.remaining_label.configure(style="Muted.TLabel")
        if not cover:
            self.meter_bar.configure(maximum=1, value=0)
            self.meter_label.configure(text="Choose a cover to see its capacity")
            self.remaining_label.configure(text="")
        elif room is None:
            self.meter_bar.configure(maximum=1, value=0)
            self.meter_label.configure(text="Measuring capacity…")
            self.remaining_label.configure(text="")
        elif used is None:
            self.meter_bar.configure(maximum=max(room, 1), value=0)
            self.meter_label.configure(text=f"? / {room:,} bytes (measuring payload…)")
            self.remaining_label.configure(text="")
        else:
            pct = used / room if room else float("inf")
            self.meter_bar.configure(maximum=max(room, 1), value=min(used, room))
            self.meter_label.configure(
                text=f"{used:,} / {room:,} bytes used ({pct:.1%}) at {bits} bit(s) per sample")
            if used <= room:
                self.remaining_label.configure(text=f"Remaining: {room - used:,} bytes")
            else:
                self.meter_bar.configure(style="Over.Horizontal.TProgressbar")
                self.remaining_label.configure(
                    text=f"Over capacity by {used - room:,} bytes: use 2 bits, compression "
                         "or a larger cover", style="Error.TLabel")
        return used, room

    # ---- toolbar actions ----
    def fill_max(self):
        """Fill the message box with Lorem ipsum that uses the whole capacity.

        The filler is ASCII in lines of at most FILLER_LINE_WIDTH
        characters, so characters equal bytes. The fill size is the
        room at the selected bits per sample, minus the encryption overhead
        when a password is set; with compression on, the text then compresses
        to a fraction of that.
        """
        if not self.cover_var.get():
            self.app.show_warning("No cover", "Choose a cover image first.")
            return
        if self.capacity is None:
            self.app.show_warning("Measuring", "The cover's capacity is still being measured.")
            return
        room = self.capacity[self.bits_var.get()]
        if self.password_var.get():
            room -= stego.ENCRYPTION_OVERHEAD
        room = max(0, room)
        # Repeat the filler enough times to cover room (one extra copy handles
        # the remainder), then cut to exactly room characters. FILLER_BLOCK
        # is broken into short lines: one multi-megabyte line would make the
        # text box re-lay it out on every keystroke.
        self.set_message((FILLER_BLOCK * (room // len(FILLER_BLOCK) + 1))[:room])
        self.app.set_status(f"Filled {room:,} bytes: 100% of the cover's capacity "
                            f"at {self.bits_var.get()} bit(s) per sample")

    def resize_cover(self):
        """Save a copy of the image cover that fits within 400x400 and use it.

        The copy is shrunk in memory (aspect ratio kept; never enlarged) and
        saved as ``<output_dir>/<name>_resized.png``, overwriting any file of
        that name; the original cover is not modified.
        """
        cover = self.cover_var.get().strip()
        if not cover:
            self.app.show_warning("No cover", "Please select an image first.")
            return
        if not is_image_path(cover):
            self.app.show_warning("Not an image", "Resize for Test works on image covers only.")
            return
        out = os.path.join(self.app.ensure_output_dir(), f"{safe_stem(cover)}_resized.png")

        def work():
            with Image.open(cover) as img:
                img.load()
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                # thumbnail() resizes in place to fit the box and only shrinks.
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                img.save(out, "PNG")
            return out

        def done(path):
            if self.cover_var.get().strip() != cover:
                # Another cover was chosen meanwhile; do not replace it.
                self.app.set_status(f"Resized copy saved as {os.path.basename(path)}; not used, "
                                    "because the cover was changed meanwhile.")
                return
            self.set_cover(path)
            self.app.set_status(f"Resized to {os.path.basename(path)}; it is now the cover.")

        self.start_job(work, done, status="Resizing…")

    # ---- hide ----
    def hide(self, output_path: str | os.PathLike | None = None) -> bool:
        """Hide the payload in the cover and save the stego file.

        Args:
            output_path: Where to save. If None, a save dialog asks; its
                filters allow only the lossless formats of the cover's
                carrier (PNG/BMP/TIFF for images, WAV for audio).

        Returns:
            True if the background job started.
        """
        if self.busy:
            self.app.set_status("Encode: still working, please wait...")
            return False
        cover = self.cover_var.get().strip()
        if not cover:
            self.app.show_warning("No cover", "Choose a cover first.")
            return False
        text_mode = self.mode_var.get() == "text"
        message = self.get_message() if text_mode else None
        file_path = None if text_mode else self.file_var.get().strip()
        if text_mode and not message:
            self.app.show_warning("Nothing to hide", "Type a message, or switch to File.")
            return False
        if not text_mode and not file_path:
            self.app.show_warning("Nothing to hide", "Choose a file to hide.")
            return False
        try:
            out_exts = list(carriers.carrier_for_path(cover).extensions)
        except ValueError as exc:
            self.app.show_exception(exc)
            return False
        if output_path is None:
            default_ext = out_exts[0]
            output_path = self.app.ask_save_path(
                "Save stego file as", filetypes_for(out_exts, "Lossless formats"),
                default_ext, f"{safe_stem(cover)}_encoded{default_ext}")
            if not output_path:
                return False
        output_path = os.fspath(output_path)
        if same_file(cover, output_path):
            # The native save dialog only asks "Replace?". Overwriting the
            # cover would destroy the original, and the change mask would
            # then compare the stego file with itself ("0 pixels changed").
            self.result_label.configure(text="Not saved: the output would replace the cover.",
                                        style="Error.TLabel")
            self.app.set_status("Hiding cancelled: the output file is the cover")
            self.app.show_error("Choose another file",
                                "The stego file would replace the cover, destroying the original "
                                "image. Save it under a different name.\n\n"
                                f"Cover: {cover}")
            return False
        password = self.password_var.get() or None
        compress = self.compress_var.get()
        bits = self.bits_var.get()
        scatter = self.scatter_var.get()
        filename = os.path.basename(file_path) if file_path else None
        image = is_image_path(cover)

        def work():
            # Runs on the worker thread: plain values only, no widgets.
            data = message if text_mode else Path(file_path).read_bytes()
            summary = stego.hide(cover, output_path, data, password=password, filename=filename,
                                 compress=compress, bits=bits, key_scatter=scatter)
            result = {"summary": summary, "preview": None, "diff": None, "changed": None}
            if image:
                result["preview"] = load_preview(output_path)
                mask = bitplanes.lsb_diff(cover, output_path, bits=bits)
                result["changed"] = (int(np.count_nonzero(mask)), int(mask.size))
                result["diff"] = display_copy(Image.fromarray(mask), PREVIEW_MAX_PX,
                                              Image.Resampling.NEAREST)
            return result

        def done(result):
            if self.cover_var.get().strip() == cover:
                self._hidden(result)
                return
            # The cover was changed (or Reset All) while hiding. The file is
            # written, so say so, but do not show it under the new cover.
            out = result["summary"]["output"]
            self.app.set_status(f"Saved {os.path.basename(out)}; not shown, because the cover "
                                "was changed meanwhile")
            self.app.show_info("Saved", f"Stego file saved as\n{out}\n\nThe cover was changed "
                                        "while hiding, so the result is not shown.")

        def failed(exc):
            if self.cover_var.get().strip() == cover:
                self._hide_failed(exc)
                return
            title, message = describe_error(exc)
            self.app.set_status(f"Hiding in {os.path.basename(cover)} failed: {title}")
            self.app.show_error(title, message)

        return self.start_job(work, done, failed, status=f"Hiding in {os.path.basename(cover)}…")

    def _hidden(self, result: dict[str, Any]):
        """Show the outcome of hide() and hand the files to Decode/Compare."""
        s = result["summary"]
        self.last_summary = s
        self.stego_path = s["output"]
        extras = [f"{s['bits_per_sample']} bit(s)/sample"]
        if s["encrypted"]:
            extras.append("encrypted")
        if s["scattered"]:
            extras.append("scattered")
        if s["compressed"]:
            extras.append("compressed")
        text = (f"Hid {s['data_bytes']:,} bytes ({s['frame_bytes']:,} with the frame) in "
                f"{os.path.basename(s['output'])}: {s['utilization']:.1%} of capacity, "
                + ", ".join(extras) + ".")
        if result["changed"] is not None:
            changed, total = result["changed"]
            text += f" {changed:,} of {total:,} pixels changed."
        self.result_label.configure(text=text, style="Ok.TLabel")
        if result["preview"] is not None:
            self.preview_stego.set_pil_image(result["preview"])
            self.preview_diff.set_pil_image(result["diff"])
            self.previews.select(self.preview_stego)
        else:
            self.preview_stego.set_message(f"Saved {os.path.basename(s['output'])}\n\n(no preview for audio)")
            self.preview_diff.set_message("(no pixel mask for audio)")
        self.app.stego_written(s["cover"], s["output"])
        self.app.set_status(text)
        self.app.show_info("Saved", f"Stego file saved as\n{s['output']}\n\n{text}")

    def _hide_failed(self, exc: BaseException):
        title, message = describe_error(exc)
        self.result_label.configure(text=f"{title}.", style="Error.TLabel")
        self.app.set_status(f"Hiding failed: {title}")
        self.app.show_error(title, message)

    def reset(self):
        self.set_cover(None)
        self.file_var.set("")
        self.password_var.set("")
        self.compress_var.set(False)
        self.bits_var.set(1)
        self.scatter_var.set(True)
        self.set_message("")


class DecodeTab(ToolTab):
    """Recover a hidden payload, with the legacy-format fallback.

    State:
        found: The recovered payload.Payload, or None.
        found_format: "stga" or "legacy", or None.
        found_path: The file ``found`` came from, or None. Saved payloads
            are named after it, not after whatever the entry holds now.
    """

    title = "Decode"

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.path_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.show_password_var = tk.BooleanVar(value=False)
        self.bits_var = tk.StringVar(value="Auto")
        self.found: payload.Payload | None = None
        self.found_format: str | None = None
        self.found_path: str | None = None
        self._build()

    def _build(self):
        form = ttk.Frame(self, padding=0)
        form.pack(side="top", fill="x")
        self._path_row(form, 0, "Stego file", self.path_var, self.select_file,
                       apply=lambda: self.decode())
        ttk.Label(form, text="Password").grid(row=1, column=0, sticky="w", pady=2)
        opts = ttk.Frame(form, padding=0)
        opts.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        self._password_entry(opts, self.password_var, self.show_password_var).pack(side="left")
        ttk.Label(opts, text="Bits per sample").pack(side="left", padx=(16, 4))
        ttk.Combobox(opts, textvariable=self.bits_var, values=("Auto", "1", "2"),
                     state="readonly", width=6).pack(side="left")
        decode_button = ttk.Button(opts, text="Decode", style="Accent.TButton", command=self.decode)
        decode_button.pack(side="right")
        self._action_buttons.append(decode_button)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left, weight=2)
        self.headline = ttk.Label(left, text="Choose a stego file and press Decode.", style="Title.TLabel")
        self.headline.grid(row=0, column=0, sticky="w")
        self.detail = ttk.Label(left, text="", style="Muted.TLabel", wraplength=520, justify="left")
        self.detail.grid(row=1, column=0, sticky="w")
        frame, self.output_text = _scrolled_text(left, height=12, state="disabled")
        frame.grid(row=2, column=0, sticky="nsew", pady=(6, 6))
        buttons = ttk.Frame(left, padding=0)
        buttons.grid(row=3, column=0, sticky="w")
        self.save_button = ttk.Button(buttons, text="Save recovered data…",
                                      command=self.save_recovered)
        self.save_button.pack(side="left")
        self.copy_button = ttk.Button(buttons, text="Copy text", command=self.copy_text)
        self.copy_button.pack(side="left", padx=(6, 0))
        self.save_button.state(["disabled"])
        self.copy_button.state(["disabled"])
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        right = ttk.Labelframe(paned, text="Stego file")
        paned.add(right, weight=1)
        self.preview = ImageTab(right, "No file", on_error=self.app.show_error)
        self.preview.pack(fill="both", expand=True)

    def open_filetypes(self):
        return filetypes_for(stego_extensions(), "Stego files (lossless)")

    def open_path(self, path: str):
        self.set_path(path)

    def select_file(self):
        """Ask for a stego file (lossless formats only)."""
        path = self.app.ask_open_path("Choose a stego file", self.open_filetypes())
        if path:
            self.set_path(path)

    def set_path(self, path: str | os.PathLike | None):
        """Set the file to decode and clear the previous result."""
        self.path_var.set(os.fspath(path) if path else "")
        self._clear_result("Press Decode to look for a hidden payload." if path else
                           "Choose a stego file and press Decode.")
        self.preview.clear()

    def _clear_result(self, headline: str):
        self.found = None
        self.found_format = None
        self.found_path = None
        self.headline.configure(text=headline, style="Title.TLabel")
        self.detail.configure(text="")
        _set_readonly_text(self.output_text, "")
        self.save_button.state(["disabled"])
        self.copy_button.state(["disabled"])

    def decode(self, path: str | os.PathLike | None = None) -> bool:
        """Look for a payload in the chosen file, in the background.

        Args:
            path: Optional file to decode instead of the one in the entry.

        Returns:
            True if the background job started.
        """
        if self.busy:
            self.app.set_status("Decode: still working, please wait...")
            return False
        if path is not None:
            self.set_path(path)
        path = self.path_var.get().strip()
        if not path:
            self.app.show_warning("No file", "Choose a stego file first.")
            return False
        password = self.password_var.get() or None
        bits = {"Auto": None, "1": 1, "2": 2}[self.bits_var.get()]
        self._clear_result("Decoding…")

        def done(result):
            if not self._is_stale(self.path_var, path):
                self._decoded(result, path)

        def failed(exc):
            if not self._is_stale(self.path_var, path):
                self._decode_failed(exc)

        return self.start_job(lambda: decode_file(path, password, bits), done, failed,
                              status=f"Decoding {os.path.basename(path)}…")

    def _discard_stale(self):
        """A decode finished for a file that is no longer chosen: show nothing."""
        self._clear_result("Press Decode to look for a hidden payload." if self.path_var.get()
                           else "Choose a stego file and press Decode.")
        self.preview.clear()

    def _decoded(self, result: dict[str, Any], path: str | None = None):
        """Show a recovered payload: text inline, files and binary as hex.

        Args:
            result: decode_file()'s dict.
            path: The file it came from; defaults to the entry.
        """
        found: payload.Payload = result["payload"]
        self.found = found
        self.found_format = result["format"]
        self.found_path = path or self.path_var.get().strip()
        if result["preview"] is not None:
            self.preview.set_pil_image(result["preview"])
        else:
            self.preview.set_message("(no preview for audio)")

        if result["format"] == "legacy":
            details = ("Legacy format: text followed by 64 zero bits, read at byte boundaries. "
                       "It has no header or checksum; these bytes were accepted because they "
                       "read as text (a heuristic check), which a stray hit can still pass.")
            if result.get("legacy_encoding") == "latin-1":
                details += (f" Stored as {result['legacy_bytes']:,} single-byte Latin-1 "
                            "characters; shown and saved as UTF-8.")
        else:
            h = result["header"] or {}
            parts = ["STGA frame", f"{h.get('bits_per_sample', '?')} bit(s) per sample",
                     "scattered with the password" if h.get("scattered") else "sequential"]
            parts += [flag for flag, on in (("encrypted", found.encrypted),
                                            ("compressed", found.compressed)) if on]
            details = ", ".join(parts) + ". CRC-32 verified."
        try:
            text = found.data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        size = len(found.data)
        if found.filename:
            headline = f"Hidden file: {found.filename} ({size:,} bytes)"
            body = text if text is not None else hex_dump(found.data)
        elif text is None:
            headline = f"Hidden binary data ({size:,} bytes, not UTF-8 text)"
            body = hex_dump(found.data)
        else:
            headline = f"Hidden message ({size:,} bytes)"
            body = text
        if len(body) > TEXT_PREVIEW_CHARS:
            body = body[:TEXT_PREVIEW_CHARS] + f"\n\n[... shortened; save it to see all {size:,} bytes]"
        self.headline.configure(text=headline, style="OkTitle.TLabel")
        self.detail.configure(text=details)
        _set_readonly_text(self.output_text, body)
        self.save_button.state(["!disabled"])
        self.copy_button.state(["!disabled"] if text is not None else ["disabled"])
        self.app.set_status(f"{headline} found in {os.path.basename(self.found_path)}")

    def _decode_failed(self, exc: BaseException):
        title, message = describe_error(exc)
        self.headline.configure(text=title, style="ErrorTitle.TLabel")
        self.detail.configure(text="")
        _set_readonly_text(self.output_text, message)
        self.app.set_status(f"Decode: {title}")
        self.app.show_error(title, message)

    def recovered_text(self) -> str | None:
        """The recovered payload as text, or None if absent or not UTF-8."""
        if self.found is None:
            return None
        try:
            return self.found.data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def save_recovered(self, out_dir: str | os.PathLike | None = None) -> str | None:
        """Save the recovered payload into a folder, never overwriting.

        A hidden file keeps its stored (sanitized) name; text becomes
        ``<stego name>_payload.txt`` and other data ``.bin`` (see
        stego.save_payload), where the stego name is that of the file the
        payload was decoded from (found_path).

        Args:
            out_dir: Folder to save into. If None, a folder dialog asks.

        Returns:
            The path written, or None if nothing was saved.
        """
        if self.found is None:
            self.app.show_warning("Nothing to save", "Decode a file first.")
            return None
        if out_dir is None:
            out_dir = self.app.ask_directory("Save the recovered data in…")
            if not out_dir:
                return None
        try:
            source = self.found_path or self.path_var.get()
            path = stego.save_payload(self.found, out_dir,
                                      default_stem=f"{safe_stem(source)}_payload")
        except OSError as exc:
            self.app.show_exception(exc)
            return None
        self.app.set_status(f"Saved {len(self.found.data):,} bytes to {path}")
        self.app.show_info("Saved", f"Recovered data saved as\n{path}")
        return str(path)

    def copy_text(self):
        """Copy the recovered text to the clipboard."""
        text = self.recovered_text()
        if text is not None:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.app.set_status("Recovered text copied to the clipboard")

    def reset(self):
        self.password_var.set("")
        self.bits_var.set("Auto")
        self.set_path(None)


class AnalyzeTab(ToolTab):
    """Full steganalysis of one image (report.analyze_image) and export.

    State:
        result: The last analyze_image() dict, or None.
        last_report: Path of the last exported report, or None.
    """

    title = "Analyze"

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.path_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.show_password_var = tk.BooleanVar(value=False)
        self.bruteforce_var = tk.BooleanVar(value=True)
        self.format_var = tk.StringVar(value="html")
        self.result: dict[str, Any] | None = None
        self.last_report: str | None = None
        self._build()

    def _build(self):
        form = ttk.Frame(self, padding=0)
        form.pack(side="top", fill="x")
        self._path_row(form, 0, "Image", self.path_var, self.select_file, apply=lambda: self.analyze())
        ttk.Label(form, text="Password").grid(row=1, column=0, sticky="w", pady=2)
        opts = ttk.Frame(form, padding=0)
        opts.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        self._password_entry(opts, self.password_var, self.show_password_var).pack(side="left")
        ttk.Checkbutton(opts, text="Include brute force", variable=self.bruteforce_var).pack(
            side="left", padx=(16, 0))
        run = ttk.Button(opts, text="Analyze", style="Accent.TButton", command=self.analyze)
        run.pack(side="right")
        self._action_buttons.append(run)

        self.banner = tk.Label(self, text="No analysis yet", anchor="w", justify="left",
                               font=self.app.fonts["banner"], padx=12, pady=8,
                               fg=NEUTRAL_BANNER[0], bg=NEUTRAL_BANNER[1], wraplength=1000)
        self.banner.pack(side="top", fill="x", pady=(8, 8))
        # Keep the banner text wrapped to the window width.
        self.banner.bind("<Configure>", lambda e: self.banner.configure(wraplength=max(200, e.width - 24)))

        export = ttk.Frame(self, padding=(0, 6, 0, 0))
        export.pack(side="bottom", fill="x")
        ttk.Label(export, text="Report format").pack(side="left")
        ttk.Combobox(export, textvariable=self.format_var, values=report.REPORT_FORMATS,
                     state="readonly", width=6).pack(side="left", padx=(6, 6))
        self.export_button = ttk.Button(export, text="Export report…", command=self.export_report)
        self.export_button.pack(side="left")
        self.export_button.state(["disabled"])
        self._action_buttons.append(self.export_button)
        ttk.Label(export, text="Scores are heuristics, not probabilities.",
                  style="Muted.TLabel").pack(side="right")

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)
        columns = ("detector", "score", "verdict", "summary")
        tree_frame, self.tree = _scrolled_tree(paned, columns, height=8, selectmode="browse")
        for col, text, width, stretch in (("detector", "Detector", 170, False),
                                          ("score", "Score", 70, False),
                                          ("verdict", "Verdict", 110, False),
                                          ("summary", "Summary", 520, True)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, stretch=stretch,
                             anchor="e" if col == "score" else "w")
        for verdict, (_text, fg, _bg) in VERDICT_STYLE.items():
            self.tree.tag_configure(verdict, foreground=fg)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(tree_frame, weight=1)

        self.details = ttk.Notebook(paned)
        summary_frame, self.summary_text = _scrolled_text(self.details, height=10, state="disabled",
                                                          font=self.app.fonts["mono"], wrap="none")
        detail_frame, self.detail_text = _scrolled_text(self.details, height=10, state="disabled",
                                                        font=self.app.fonts["mono"], wrap="none")
        self.details.add(summary_frame, text="Summary")
        self.details.add(detail_frame, text="Selected detector")
        paned.add(self.details, weight=1)

    def open_path(self, path: str):
        self.set_path(path)

    def select_file(self):
        """Ask for an image to analyze."""
        path = self.app.ask_open_path("Choose an image to analyze", self.open_filetypes())
        if path:
            self.set_path(path)

    def set_path(self, path: str | os.PathLike | None):
        """Set the image to analyze and clear the previous result."""
        self.path_var.set(os.fspath(path) if path else "")
        self._clear()

    def _clear(self):
        self.result = None
        self.tree.delete(*self.tree.get_children())
        self.banner.configure(text="No analysis yet", fg=NEUTRAL_BANNER[0], bg=NEUTRAL_BANNER[1])
        _set_readonly_text(self.summary_text, "")
        _set_readonly_text(self.detail_text, "")
        self.export_button.state(["disabled"])

    def analyze(self, path: str | os.PathLike | None = None) -> bool:
        """Run report.analyze_image on the image in a background thread.

        Returns:
            True if the job started (False if one is already running or no
            image is chosen).
        """
        # Check busy before touching the input, so a refused run does not
        # show the new path over the running job's file.
        if self.busy:
            self.app.set_status("Analyze: still working, please wait...")
            return False
        if path is not None:
            self.set_path(path)
        path = self.path_var.get().strip()
        if not path:
            self.app.show_warning("No image", "Choose an image to analyze.")
            return False
        password = self.password_var.get() or None
        run_bf = self.bruteforce_var.get()
        self._clear()
        self.banner.configure(text=f"Analyzing {os.path.basename(path)}…")

        def done(result):
            if not self._is_stale(self.path_var, path):
                self._analyzed(result)

        def failed(exc):
            if not self._is_stale(self.path_var, path):
                self._analyze_failed(exc)

        return self.start_job(
            lambda: report.analyze_image(path, run_bruteforce=run_bf, password=password),
            done, failed,
            status=f"Analyzing {os.path.basename(path)} (detectors"
                   + (", brute force" if run_bf else "") + ")…")

    def _discard_stale(self):
        """An analysis finished for a file that is no longer chosen: show nothing."""
        self._clear()  # also disables Export, which finish() re-enabled

    def _analyzed(self, result: dict[str, Any]):
        """Fill the banner, the detector table and the summary."""
        self.result = result
        overall = result["overall"]
        text, fg, bg = VERDICT_STYLE.get(overall["verdict"], ("Unknown", *NEUTRAL_BANNER))
        self.banner.configure(text=f"{text}  ·  score {overall['score']:.2f}  ·  "
                                   f"{overall.get('summary', '')}", fg=fg, bg=bg)
        for name, det in result["detectors"].items():
            self.tree.insert("", "end", iid=name, tags=(det["verdict"],), values=(
                name, f"{det['score']:.2f}", det["verdict"].replace("_", " "),
                one_line(det.get("summary", ""), 300)))
        _set_readonly_text(self.summary_text, report.format_text(result))
        self.details.select(0)
        self.export_button.state(["!disabled"])
        self.app.set_status(f"Analysis of {result['file_name']}: {text.lower()} "
                            f"(score {overall['score']:.2f})")

    def _analyze_failed(self, exc: BaseException):
        title, message = describe_error(exc)
        self.export_button.state(["disabled"])  # finish() re-enabled it; nothing to export
        self.banner.configure(text=f"{title}: {one_line(str(exc), 200)}",
                              fg=NEUTRAL_BANNER[0], bg=NEUTRAL_BANNER[1])
        self.app.set_status(f"Analysis failed: {title}")
        self.app.show_error(title, message)

    def _on_select(self, _event=None):
        """Show the selected detector's full result as JSON."""
        if not self.result:
            return
        selection = self.tree.selection()
        if not selection:
            return
        det = self.result["detectors"].get(selection[0])
        if det is not None:
            _set_readonly_text(self.detail_text, json.dumps(det, indent=2, ensure_ascii=False))
            self.details.select(1)

    def export_report(self, out_dir: str | os.PathLike | None = None,
                      fmt: str | None = None) -> bool:
        """Write the last result as a report (report.write_report) in the background.

        Args:
            out_dir: Folder for ``<image>_report.<fmt>`` (and the figure PNGs
                for Markdown). If None, a folder dialog asks.
            fmt: "md", "html" or "json"; defaults to the combo box.

        Returns:
            True if the export job started.
        """
        if self.result is None:
            self.app.show_warning("Nothing to export", "Run an analysis first.")
            return False
        fmt = fmt or self.format_var.get()
        if out_dir is None:
            out_dir = self.app.ask_directory("Save the report in…")
            if not out_dir:
                return False
        result, out_dir = self.result, os.fspath(out_dir)

        def done(path):
            self.last_report = path
            self.app.set_status(f"Report written to {path}")
            self.app.show_info("Report saved", f"Report written to\n{path}")

        return self.start_job(lambda: report.write_report(result, out_dir, fmt=fmt), done,
                              status=f"Writing {fmt} report…")

    def reset(self):
        self.password_var.set("")
        self.set_path(None)
        self.last_report = None


class BitplanesTab(ToolTab):
    """Bit-plane views: one plane, the amplified LSBs and the full grid.

    State:
        array: The loaded image as a uint8 array (H, W, 3 or 4), or None.
        grid_path: The last bit-plane grid PNG written, or None.
    """

    title = "Bitplanes"

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.path_var = tk.StringVar()
        self.channel_var = tk.StringVar(value="R")
        self.bit_var = tk.IntVar(value=0)
        self.array: np.ndarray | None = None
        self.grid_path: str | None = None
        self._build()

    def _build(self):
        form = ttk.Frame(self, padding=0)
        form.pack(side="top", fill="x")
        self._path_row(form, 0, "Image", self.path_var, self.select_file,
                       apply=lambda: self.load(self.path_var.get().strip()))
        selectors = ttk.Frame(form, padding=0)
        selectors.grid(row=1, column=1, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(form, text="Plane").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.channel_buttons: dict[str, ttk.Radiobutton] = {}
        for letter in "RGBA":
            button = ttk.Radiobutton(selectors, text=letter, value=letter, variable=self.channel_var,
                                     command=self.show_plane, style="Toolbutton")
            button.pack(side="left")
            self.channel_buttons[letter] = button
        self.channel_buttons["A"].state(["disabled"])
        ttk.Label(selectors, text="bit").pack(side="left", padx=(14, 4))
        for bit in range(7, -1, -1):
            text = f"{bit}" + (" MSB" if bit == 7 else " LSB" if bit == 0 else "")
            ttk.Radiobutton(selectors, text=text, value=bit, variable=self.bit_var,
                            command=self.show_plane, style="Toolbutton").pack(side="left")
        self.plane_info = ttk.Label(form, text="", style="Muted.TLabel")
        self.plane_info.grid(row=2, column=1, columnspan=2, sticky="w")

        # Packed before the expanding notebook so it keeps its space when the
        # window shrinks.
        ttk.Label(self, text="A noise-like LSB plane is a hint, not proof: noisy photos look the "
                             "same, and scattered or short payloads leave no visible trace "
                             "(visual attack, Westfeld & Pfitzmann 1999).",
                  style="Muted.TLabel", wraplength=900).pack(side="bottom", fill="x", pady=(6, 0))
        self.views = ttk.Notebook(self)
        self.views.pack(fill="both", expand=True, pady=(8, 0))
        nearest = Image.Resampling.NEAREST
        self.view_plane = ImageTab(self.views, "Choose an image", resample=nearest,
                                   on_error=self.app.show_error)
        self.view_amplified = ImageTab(self.views, "Amplified LSBs appear here", resample=nearest,
                                       on_error=self.app.show_error)
        self.view_grid = ImageTab(self.views, "The full bit-plane grid appears here",
                                  on_error=self.app.show_error)
        self.views.add(self.view_plane, text="Selected plane")
        self.views.add(self.view_amplified, text="LSB amplified")
        self.views.add(self.view_grid, text="All planes")

    def open_path(self, path: str) -> bool | None:
        # Loading starts the job at once, so a busy tab refuses the file
        # (load() then leaves the input alone and says "still working").
        return None if self.load(path) else False

    def select_file(self):
        """Ask for an image to slice."""
        path = self.app.ask_open_path("Choose an image", self.open_filetypes())
        if path:
            self.load(path)

    def load(self, path: str | os.PathLike | None = None) -> bool:
        """Load an image and draw its amplified LSBs and bit-plane grid.

        The array, the amplified image and the grid PNG (written to
        ``<output_dir>/<name>_bitplanes.png``) are built in the background.
        Switching planes afterwards is a single NumPy operation on the
        cached array, done on the Tk thread.

        Returns:
            True if the background job started.
        """
        # Check busy before touching the input, so a refused load does not
        # show the new path over the planes of the file being sliced.
        if self.busy:
            self.app.set_status("Bitplanes: still working, please wait...")
            return False
        if path is not None:
            self.path_var.set(os.fspath(path))
        path = self.path_var.get().strip()
        if not path:
            self.app.show_warning("No image", "Choose an image first.")
            return False
        grid_out = os.path.join(self.app.ensure_output_dir(), f"{safe_stem(path)}_bitplanes.png")

        def work():
            with Image.open(path) as img:
                has_alpha = img.mode in ("RGBA", "LA", "PA") or (
                    img.mode == "P" and "transparency" in img.info)
                arr = np.asarray(img.convert("RGBA" if has_alpha else "RGB"))
            amplified = display_copy(Image.fromarray(bitplanes.lsb_amplify(arr)), PREVIEW_MAX_PX,
                                     Image.Resampling.NEAREST)
            grid = bitplanes.save_bitplane_grid(arr, grid_out, channels="RGBA" if has_alpha else "RGB")
            return {"array": arr, "amplified": amplified, "grid": grid, "grid_image": load_preview(grid)}

        def done(result):
            if not self._is_stale(self.path_var, path):
                self._loaded(result)

        def failed(exc):
            if not self._is_stale(self.path_var, path):
                self.app.show_exception(exc)

        return self.start_job(work, done, failed, status=f"Slicing {os.path.basename(path)}…")

    def _discard_stale(self):
        """Slicing finished for a file that is no longer chosen: show no planes."""
        self._clear_views()

    def _clear_views(self):
        """Forget the loaded image and empty the three views."""
        self.array = None
        self.grid_path = None
        self.plane_info.configure(text="")
        for view in (self.view_plane, self.view_amplified, self.view_grid):
            view.clear()

    def _loaded(self, result: dict[str, Any]):
        self.array = result["array"]
        self.grid_path = result["grid"]
        has_alpha = self.array.shape[2] == 4
        self.channel_buttons["A"].state(["!disabled"] if has_alpha else ["disabled"])
        if not has_alpha and self.channel_var.get() == "A":
            self.channel_var.set("R")
        self.view_amplified.set_pil_image(result["amplified"])
        self.view_grid.set_pil_image(result["grid_image"])
        self.show_plane()
        h, w = self.array.shape[:2]
        self.app.set_status(f"Bit planes of {os.path.basename(self.path_var.get())} ({w} x {h}); "
                            f"grid saved to {self.grid_path}")

    def show_plane(self) -> np.ndarray | None:
        """Show the selected channel/bit plane (white = 1).

        Returns:
            The (H, W) 0/255 plane, or None if no image is loaded.
        """
        if self.array is None:
            return None
        letter, bit = self.channel_var.get(), self.bit_var.get()
        plane = bitplanes.bitplane(self.array, bitplanes.CHANNEL_INDEX[letter], bit)
        ones = float(np.count_nonzero(plane)) / plane.size if plane.size else 0.0
        self.view_plane.set_pil_image(Image.fromarray(plane))
        self.plane_info.configure(text=f"Channel {letter}, bit {bit}: {ones:.1%} of pixels are 1")
        self.views.select(self.view_plane)
        return plane

    def reset(self):
        self.path_var.set("")
        self._clear_views()


class BruteforceTab(ToolTab):
    """Brute-force extraction over many bit layouts, plus a word-list attack.

    State:
        candidates: Candidate dicts in display order (bruteforce() results,
            with word-list hits first), including their raw ``data``.
    """

    title = "Brute force"

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.path_var = tk.StringVar()
        self.top_var = tk.IntVar(value=10)
        self.max_bit_var = tk.IntVar(value=1)
        self.wordlist_var = tk.StringVar()
        self.candidates: list[dict[str, Any]] = []
        self._build()

    def _build(self):
        form = ttk.Frame(self, padding=0)
        form.pack(side="top", fill="x")
        self._path_row(form, 0, "Image", self.path_var, self.select_file, apply=lambda: self.run())
        opts = ttk.Frame(form, padding=0)
        opts.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(form, text="Search").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(opts, text="Top").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=200, textvariable=self.top_var, width=5).pack(side="left", padx=(4, 12))
        ttk.Label(opts, text="Highest bit plane").pack(side="left")
        ttk.Spinbox(opts, from_=0, to=7, textvariable=self.max_bit_var, width=3).pack(side="left", padx=(4, 12))
        self.count_label = ttk.Label(opts, text="", style="Muted.TLabel")
        self.count_label.pack(side="left")
        run = ttk.Button(opts, text="Run brute force", style="Accent.TButton", command=self.run)
        run.pack(side="right")
        self._action_buttons.append(run)
        self._path_row(form, 2, "Password list", self.wordlist_var, self.select_wordlist)
        attack = ttk.Button(form, text="Try passwords", command=self.run_wordlist)
        attack.grid(row=3, column=2, sticky="e", pady=2)
        self._action_buttons.append(attack)
        ttk.Label(form, text="Finds payloads scattered with a password key, if the password is on the list.",
                  style="Muted.TLabel").grid(row=3, column=1, sticky="w")
        self._update_count()
        self.max_bit_var.trace_add("write", lambda *_: self._update_count())

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, pady=(8, 0))
        columns = ("rank", "strategy", "kind", "score", "preview")
        tree_frame, self.tree = _scrolled_tree(paned, columns, height=10, selectmode="browse")
        for col, text, width, stretch in (("rank", "#", 40, False), ("strategy", "Strategy", 210, False),
                                          ("kind", "Kind", 90, False), ("score", "Score", 60, False),
                                          ("preview", "Preview", 520, True)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, stretch=stretch,
                             anchor="e" if col in ("rank", "score") else "w")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(tree_frame, weight=1)

        bottom = ttk.Frame(paned, padding=0)
        text_frame, self.preview_text = _scrolled_text(bottom, height=8, state="disabled",
                                                       font=self.app.fonts["mono"])
        text_frame.pack(fill="both", expand=True)
        buttons = ttk.Frame(bottom, padding=(0, 6, 0, 0))
        buttons.pack(fill="x")
        self.save_button = ttk.Button(buttons, text="Save candidate…", command=self.save_candidate)
        self.save_button.pack(side="left")
        self.save_button.state(["disabled"])
        ttk.Label(buttons, text="Scores rank candidates; they are heuristics, not probabilities.",
                  style="Muted.TLabel").pack(side="right")
        paned.add(bottom, weight=1)

    def _update_count(self):
        try:
            n = len(bruteforce.enumerate_strategies(max(0, min(7, int(self.max_bit_var.get())))))
        except (tk.TclError, ValueError):
            return
        self.count_label.configure(text=f"{n} layouts")

    def open_path(self, path: str):
        self.set_path(path)

    def select_file(self):
        """Ask for an image to search."""
        path = self.app.ask_open_path("Choose an image", self.open_filetypes())
        if path:
            self.set_path(path)

    def select_wordlist(self):
        """Ask for a password list (one password per line)."""
        path = self.app.ask_open_path("Choose a password list",
                                      [("Text files", "*.txt *.lst"), ("All files", "*")])
        if path:
            self.wordlist_var.set(path)

    def set_path(self, path: str | os.PathLike | None):
        """Set the image to search and clear the previous candidates."""
        self.path_var.set(os.fspath(path) if path else "")
        self._show_candidates([])

    def _show_candidates(self, candidates: list[dict[str, Any]]):
        self.candidates = list(candidates)
        self.tree.delete(*self.tree.get_children())
        for i, cand in enumerate(self.candidates):
            self.tree.insert("", "end", iid=str(i), values=(
                i + 1, cand.get("label", "?"), cand.get("kind", "?"), f"{cand.get('score', 0):.2f}",
                one_line(cand.get("preview", ""))))
        _set_readonly_text(self.preview_text, "")
        self.save_button.state(["disabled"])
        if self.candidates:
            self.tree.selection_set("0")
            self._on_select()

    def run(self, path: str | os.PathLike | None = None) -> bool:
        """Run bruteforce.bruteforce on the image in a background thread.

        Returns:
            True if the job started.
        """
        # Check busy before touching the input, so a refused run does not
        # clear the table and show the new path over the running search.
        if self.busy:
            self.app.set_status("Brute force: still working, please wait...")
            return False
        if path is not None:
            self.set_path(path)
        path = self.path_var.get().strip()
        if not path:
            self.app.show_warning("No image", "Choose an image first.")
            return False
        try:
            top_k = max(1, int(self.top_var.get()))
            max_bit = max(0, min(7, int(self.max_bit_var.get())))
        except (tk.TclError, ValueError):
            self.app.show_warning("Invalid setting", "Top and highest bit plane must be whole numbers.")
            return False

        def done(found):
            if self._is_stale(self.path_var, path):
                return
            self._show_candidates(found)
            best = found[0] if found else None
            self.app.set_status(
                f"Brute force: {len(found)} candidates; best: {best['label']} ({best['kind']}, "
                f"score {best['score']:.2f})" if best else "Brute force: no candidates")

        def failed(exc):
            if not self._is_stale(self.path_var, path):
                self.app.show_exception(exc)

        return self.start_job(lambda: bruteforce.bruteforce(path, top_k=top_k, max_bit=max_bit),
                              done, failed, status=f"Brute-forcing {os.path.basename(path)}…")

    def _discard_stale(self):
        """A search finished for a file that is no longer chosen: list nothing."""
        self._show_candidates([])

    def run_wordlist(self, wordlist: str | os.PathLike | None = None) -> bool:
        """Try each line of a password list as a scatter key (dictionary_attack).

        Hits are added above the brute-force candidates. Runs in the
        background; the cost is one permutation of the samples per password.

        Returns:
            True if the job started.
        """
        if self.busy:
            self.app.set_status("Brute force: still working, please wait...")
            return False
        if wordlist is not None:
            self.wordlist_var.set(os.fspath(wordlist))
        path = self.path_var.get().strip()
        wordlist = self.wordlist_var.get().strip()
        if not path or not wordlist:
            self.app.show_warning("Missing input", "Choose an image and a password list first.")
            return False

        def work():
            with open(wordlist, encoding="utf-8", errors="replace") as fh:
                return bruteforce.dictionary_attack(path, fh.read().splitlines())

        def failed(exc):
            if not self._is_stale(self.path_var, path):
                self.app.show_exception(exc)

        def done(hits):
            if self._is_stale(self.path_var, path):
                return
            for hit in hits:
                hit.setdefault("label", f"password {hit['password']!r}, "
                                        f"{hit['bits_per_sample']} bit(s)/sample")
            existing = [c for c in self.candidates if "password" not in c]
            self._show_candidates(hits + existing)
            if hits:
                self.app.set_status(f"Password list: {len(hits)} hit(s); "
                                    f"first: {hits[0]['password']!r}")
            else:
                self.app.set_status("Password list: no password matched")
                self.app.show_info("No match", "No password in the list unlocked a payload.")

        return self.start_job(work, done, failed, status="Trying passwords…")

    def selected(self) -> dict[str, Any] | None:
        """The selected candidate, or None."""
        selection = self.tree.selection()
        if not selection:
            return None
        return self.candidates[int(selection[0])]

    def _on_select(self, _event=None):
        cand = self.selected()
        if cand is None:
            return
        info = json.dumps(cand.get("info", {}), indent=2, ensure_ascii=False, default=str)
        text = (f"{cand.get('label', '')}\nkind {cand.get('kind')}, score {cand.get('score', 0):.2f}, "
                f"{len(cand.get('data', b'')):,} bytes\n\n{cand.get('preview', '')}\n\n{info}")
        _set_readonly_text(self.preview_text, text)
        self.save_button.state(["!disabled"])

    def save_candidate(self, out_dir: str | os.PathLike | None = None,
                       index: int | None = None) -> str | None:
        """Save a candidate's recovered bytes (bruteforce.save_candidate).

        Args:
            out_dir: Folder to save into. If None, a folder dialog asks.
            index: Candidate to save; defaults to the selected one.

        Returns:
            The path written, or None.
        """
        cand = self.candidates[index] if index is not None else self.selected()
        if cand is None:
            self.app.show_warning("Nothing selected", "Select a candidate first.")
            return None
        if out_dir is None:
            out_dir = self.app.ask_directory("Save the candidate in…")
            if not out_dir:
                return None
        try:
            path = bruteforce.save_candidate(cand, out_dir)
        except OSError as exc:
            self.app.show_exception(exc)
            return None
        self.app.set_status(f"Candidate saved to {path}")
        self.app.show_info("Saved", f"Candidate saved as\n{path}")
        return path

    def reset(self):
        self.wordlist_var.set("")
        self.set_path(None)


class CompareTab(ToolTab):
    """Cover vs stego: histogram comparison, pair statistics and LSB changes.

    State:
        figure_path: The last composite figure written, or None.
        changed: (changed pixels, total pixels) of the last comparison, or
            None when the images differ in size.
    """

    title = "Compare"

    CHANNELS = {"Red": "r", "Green": "g", "Blue": "b"}

    def __init__(self, notebook, app):
        super().__init__(notebook, app)
        self.cover_var = tk.StringVar()
        self.stego_var = tk.StringVar()
        self.channel_var = tk.StringVar(value="Blue")
        self.figure_path: str | None = None
        self.changed: tuple[int, int] | None = None
        self._build()

    def _build(self):
        form = ttk.Frame(self, padding=0)
        form.pack(side="top", fill="x")
        self._path_row(form, 0, "Cover", self.cover_var, self.select_cover)
        self._path_row(form, 1, "Stego", self.stego_var, self.select_stego, apply=lambda: self.compare())
        opts = ttk.Frame(form, padding=0)
        opts.grid(row=2, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(form, text="Channel").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Combobox(opts, textvariable=self.channel_var, values=tuple(self.CHANNELS),
                     state="readonly", width=8).pack(side="left")
        run = ttk.Button(opts, text="Compare", style="Accent.TButton", command=self.compare)
        run.pack(side="right")
        self._action_buttons.append(run)
        self.stats_label = ttk.Label(self, text="", font=self.app.fonts["mono"], justify="left")
        self.stats_label.pack(side="top", fill="x", pady=(6, 0))

        self.views = ttk.Notebook(self)
        self.views.pack(fill="both", expand=True, pady=(6, 0))
        self.view_hist = ImageTab(self.views, "Histograms appear here", on_error=self.app.show_error)
        self.view_diff = ImageTab(self.views, "Pixels whose R, G or B LSB changed appear white",
                                  resample=Image.Resampling.NEAREST, on_error=self.app.show_error)
        self.views.add(self.view_hist, text="Histograms")
        self.views.add(self.view_diff, text="LSB differences")

    def open_path(self, path: str):
        if not self.cover_var.get():
            self.cover_var.set(path)
        else:
            self.stego_var.set(path)

    def select_cover(self):
        path = self.app.ask_open_path("Choose the cover (original)", self.open_filetypes())
        if path:
            self.cover_var.set(path)

    def select_stego(self):
        path = self.app.ask_open_path("Choose the stego image",
                                      filetypes_for(list(ImageLSBCarrier.extensions),
                                                    "Stego images (lossless)"))
        if path:
            self.stego_var.set(path)

    def set_paths(self, cover: str | os.PathLike | None, stego_path: str | os.PathLike | None):
        """Set both images (used after hiding in the Encode tab)."""
        self.cover_var.set(os.fspath(cover) if cover else "")
        self.stego_var.set(os.fspath(stego_path) if stego_path else "")

    def compare(self, cover: str | os.PathLike | None = None,
                stego_path: str | os.PathLike | None = None) -> bool:
        """Draw the comparison figures and the LSB change mask in the background.

        The composite and the four single plots of
        steganalysis.analyze_comparison go to ``<output_dir>``
        (comparison_analysis.png, 1_original_standard_hist.png, ...),
        overwriting the previous comparison.

        Returns:
            True if the job started.
        """
        if self.busy:
            self.app.set_status("Compare: still working, please wait...")
            return False
        if cover is not None or stego_path is not None:
            self.set_paths(cover or self.cover_var.get(), stego_path or self.stego_var.get())
        cover, stego_file = self.cover_var.get().strip(), self.stego_var.get().strip()
        if not cover or not stego_file:
            self.app.show_warning("Two images needed", "Choose a cover and a stego image.")
            return False
        channel = self.CHANNELS[self.channel_var.get()]
        out = os.path.join(self.app.ensure_output_dir(), "comparison_analysis.png")

        def work():
            for p in (cover, stego_file):
                if not os.path.isfile(p):
                    raise FileNotFoundError(2, "No such file", p)
            paths = steganalysis.analyze_comparison(cover, stego_file, output_path=out, channel=channel)
            with Image.open(cover) as a, Image.open(stego_file) as b:
                cov = np.asarray(a.convert("RGB"), dtype=np.uint8)
                ste = np.asarray(b.convert("RGB"), dtype=np.uint8)
            ch = "rgb".index(channel)
            stats = {label: pairwise_stats.pair_stats(
                         np.bincount(arr[..., ch].ravel(), minlength=256).astype(np.int64))
                     for label, arr in (("cover", cov), ("stego", ste))}
            result = {"figure": paths["composite"], "figure_image": load_preview(paths["composite"]),
                      "stats": stats, "diff": None, "changed": None}
            if cov.shape == ste.shape:
                mask = bitplanes.lsb_diff(cov, ste)
                result["changed"] = (int(np.count_nonzero(mask)), int(mask.size))
                result["diff"] = display_copy(Image.fromarray(mask), PREVIEW_MAX_PX,
                                              Image.Resampling.NEAREST)
            return result

        def stale() -> bool:
            return (self._is_stale(self.cover_var, cover)
                    or self._is_stale(self.stego_var, stego_file))

        def done(result):
            if not stale():
                self._compared(result)

        def failed(exc):
            if not stale():
                self.app.show_exception(exc)

        return self.start_job(work, done, failed, status="Comparing…")

    def _compared(self, result: dict[str, Any]):
        self.figure_path = result["figure"]
        self.changed = result["changed"]
        self.view_hist.set_pil_image(result["figure_image"])
        lines = [f"Even/odd pairs, {self.channel_var.get().lower()} channel "
                 "(Delta_k = h(2k) - h(2k+1)):"]
        for label in ("cover", "stego"):
            s = result["stats"][label]
            lines.append(f"  {label:<5}  L1 {s['L1']:>10,}   L2 {s['L2']:>12,.1f}   "
                         f"median |Delta| {s['median_abs']:>8g}   max |Delta| {s['max_abs']:>8,} at k={s['max_k']}")
        if result["changed"] is not None:
            changed, total = result["changed"]
            lines.append(f"  {changed:,} of {total:,} pixels ({changed / max(total, 1):.2%}) "
                         "have a changed R, G or B LSB")
            self.view_diff.set_pil_image(result["diff"])
        else:
            lines.append("  The images differ in size, so LSBs cannot be compared pixel by pixel.")
            self.view_diff.set_message("The images differ in size")
        self.stats_label.configure(text="\n".join(lines))
        self.views.select(self.view_hist)
        self.app.set_status(f"Comparison saved to {self.figure_path}")

    def reset(self):
        self.set_paths(None, None)
        self.figure_path = None
        self.changed = None
        self.stats_label.configure(text="")
        self.view_hist.clear()
        self.view_diff.clear()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SteganoApp:
    """Main application window: menu, toolbar, tabs, status bar and jobs.

    Attributes:
        root: The Tk root window.
        output_dir: Folder for generated figures and the resized cover.
        interactive: When False, message boxes and file dialogs are not
            shown (dialogs return "cancelled"); messages are still recorded
            in ``messages``. Used by the tests and for scripting.
        messages: Every (kind, title, text) shown or suppressed, in order.
        callback_errors: Exceptions raised inside Tk callbacks.
        notebook: The ttk.Notebook holding the six tabs.
        encode_tab, decode_tab, analyze_tab, bitplanes_tab, bruteforce_tab,
        compare_tab: The tabs.
    """

    def __init__(self, root: tk.Tk, *, output_dir: str | os.PathLike = "results",
                 interactive: bool = True):
        """Configure the root window and build the UI.

        Args:
            root: The Tk root window to populate.
            output_dir: Folder for generated files; created when first needed.
            interactive: Show dialogs (True) or only record them (False).
        """
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x820")
        self.root.minsize(960, 640)
        self.output_dir = os.fspath(output_dir)
        self.interactive = interactive
        self.messages: list[tuple[str, str, str]] = []
        self.callback_errors: list[BaseException] = []
        self._jobs: list[_Job] = []
        self._poll_id: str | None = None
        self._progress_on = False
        self._closed = False

        self._setup_style()
        # Errors inside button handlers and after() callbacks are shown in a
        # dialog (and recorded) instead of only being printed to stderr.
        self.root.report_callback_exception = self._report_callback_exception

        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._build_tabs()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.set_status("Ready")

    # ------------- UI BUILDERS -------------
    def _setup_style(self):
        """Pick a ttk theme and define the named styles and fonts used by the tabs."""
        style = ttk.Style(self.root)
        # Use the first theme this Tk build accepts; theme_use raises
        # TclError for unknown themes. "clam" ships with every Tk, so in
        # practice it is always chosen and the app looks the same everywhere.
        for candidate in ("clam", "vista", "default", "aqua"):
            try:
                style.theme_use(candidate)
                break
            except tk.TclError:
                continue
        base = tkfont.nametofont("TkDefaultFont")
        size = base.actual("size")

        def bigger(points: int) -> int:
            # Tk font sizes are points when positive and pixels when negative
            # (common on X11), so "bigger" means further from zero.
            return size + points if size >= 0 else size - points

        self.fonts = {
            "bold": base.copy(),
            "title": base.copy(),
            "banner": base.copy(),
            "mono": tkfont.nametofont("TkFixedFont"),
        }
        self.fonts["bold"].configure(weight="bold")
        self.fonts["title"].configure(weight="bold", size=bigger(2))
        self.fonts["banner"].configure(weight="bold", size=bigger(3))
        self.colors = {"muted": "#555b66",
                       "error": "#b42318", "ok": "#1a7f37", "accent": "#2457a6"}
        style.configure("TFrame", padding=6)
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(2, 2))
        style.configure("TLabelframe", padding=8)
        style.configure("Toolbutton", padding=(8, 3))
        style.configure("Accent.TButton", font=self.fonts["bold"])
        style.configure("Muted.TLabel", foreground=self.colors["muted"])
        style.configure("Error.TLabel", foreground=self.colors["error"])
        style.configure("Ok.TLabel", foreground=self.colors["ok"])
        style.configure("Title.TLabel", font=self.fonts["title"])
        style.configure("OkTitle.TLabel", font=self.fonts["title"], foreground=self.colors["ok"])
        style.configure("ErrorTitle.TLabel", font=self.fonts["title"], foreground=self.colors["error"])
        style.configure("Treeview", rowheight=max(22, int(abs(size) * 2.2)))
        style.configure("Over.Horizontal.TProgressbar", background=self.colors["error"])

    def _build_menu(self):
        """Create the File and Help menus and bind the Open shortcut."""
        aqua = self.root.tk.call("tk", "windowingsystem") == "aqua"
        # tearoff=False here too: on X11 a menu's entry 0 is otherwise a
        # dashed tear-off line, which has no label.
        menubar = tk.Menu(self.root, tearoff=False)
        filemenu = tk.Menu(menubar, tearoff=False)
        filemenu.add_command(label="Open…", command=self.open_file,
                             accelerator="Command-O" if aqua else "Ctrl+O")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=False)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)
        self.menubar = menubar

        # The accelerator text above is only a label; these bindings make the
        # shortcut actually work, application-wide.
        self.root.bind_all("<Control-o>", lambda e: self.open_file())
        if aqua:
            self.root.bind_all("<Command-o>", lambda e: self.open_file())

    def _build_toolbar(self):
        """Create the toolbar (Open, Resize for Test, Fill Max Space, Reset All)."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Open…", command=self.open_file).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Resize for Test", command=self.resize_image).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Fill Max Space", command=self.fill_max_message).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Reset All", command=self.reset_all).pack(side="left")
        ttk.Label(toolbar, text=APP_TITLE, style="Muted.TLabel").pack(side="right")

    def _build_statusbar(self):
        """Create the status line (with a progress indicator) at the bottom."""
        bar = ttk.Frame(self.root, padding=(6, 2))
        bar.pack(side="bottom", fill="x")
        self.status = ttk.Label(bar, text="", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        # Packed only while background work is running.

    def _build_tabs(self):
        """Create the notebook and the six tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self.encode_tab = EncodeTab(self.notebook, self)
        self.decode_tab = DecodeTab(self.notebook, self)
        self.analyze_tab = AnalyzeTab(self.notebook, self)
        self.bitplanes_tab = BitplanesTab(self.notebook, self)
        self.bruteforce_tab = BruteforceTab(self.notebook, self)
        self.compare_tab = CompareTab(self.notebook, self)
        for tab in self.tabs:
            self.notebook.add(tab, text=tab.title)

    @property
    def tabs(self) -> list[ToolTab]:
        """The tabs in notebook order."""
        return [self.encode_tab, self.decode_tab, self.analyze_tab,
                self.bitplanes_tab, self.bruteforce_tab, self.compare_tab]

    def current_tab(self) -> ToolTab:
        """The tab currently shown."""
        return self.root.nametowidget(self.notebook.select())

    # ------------- Status, dialogs, errors -------------
    def set_status(self, text: str):
        """Show text in the status bar.

        Args:
            text: Message to display.
        """
        self.status.configure(text=text)

    def _dialog(self, kind: str, title: str, text: str):
        """Record a message and, when interactive, show it in a message box."""
        self.messages.append((kind, title, text))
        if self.interactive and not self._closed:
            show = {"info": messagebox.showinfo, "warning": messagebox.showwarning,
                    "error": messagebox.showerror}[kind]
            show(title, text, parent=self.root)

    def show_info(self, title: str, text: str):
        """Information message box."""
        self._dialog("info", title, text)

    def show_warning(self, title: str, text: str):
        """Warning message box (missing input and the like)."""
        self._dialog("warning", title, text)

    def show_error(self, title: str, text: str):
        """Error message box."""
        self._dialog("error", title, text)

    def show_exception(self, exc: BaseException):
        """Error message box for an exception, worded by describe_error()."""
        self.show_error(*describe_error(exc))

    def _report_callback_exception(self, exc_type, exc, tb):
        """Tk hook for exceptions in callbacks: record and show them.

        The traceback is also printed to stderr (as Tk itself would), since
        an exception here is a bug in the GUI, not a user error.
        """
        self.callback_errors.append(exc)
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        self.set_status(f"Internal error: {exc_type.__name__}: {exc}")
        self.show_error("Unexpected error", f"{exc_type.__name__}: {exc}")

    def ask_open_path(self, title: str, filetypes: list[tuple[str, str]]) -> str:
        """File-open dialog; returns "" when cancelled or not interactive."""
        if not self.interactive:
            return ""
        return filedialog.askopenfilename(parent=self.root, title=title, filetypes=filetypes) or ""

    def ask_save_path(self, title: str, filetypes: list[tuple[str, str]],
                      defaultextension: str, initialfile: str) -> str:
        """File-save dialog starting in output_dir; "" when cancelled."""
        if not self.interactive:
            return ""
        return filedialog.asksaveasfilename(
            parent=self.root, title=title, filetypes=filetypes, defaultextension=defaultextension,
            initialfile=initialfile, initialdir=self.ensure_output_dir()) or ""

    def ask_directory(self, title: str) -> str:
        """Folder dialog starting in output_dir; "" when cancelled."""
        if not self.interactive:
            return ""
        return filedialog.askdirectory(parent=self.root, title=title,
                                       initialdir=self.ensure_output_dir()) or ""

    def ensure_output_dir(self) -> str:
        """Create output_dir if needed and return it."""
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    # ------------- Background jobs -------------
    @property
    def busy(self) -> bool:
        """True while any background job is running."""
        return bool(self._jobs)

    def run_in_background(self, func: Callable[[], Any], on_done: Callable[[Any], None],
                          on_error: Callable[[BaseException], None] | None = None,
                          *, status: str | None = None):
        """Run ``func`` on a worker thread; deliver the outcome on the Tk thread.

        ``func`` must not touch Tk. Its return value is passed to ``on_done``
        (or its exception to ``on_error``, by default an error dialog) from a
        root.after() poll on the Tk thread, so the callbacks may update any
        widget.

        Args:
            func: Zero-argument worker function.
            on_done: Called with func's return value.
            on_error: Called with func's exception.
            status: Status bar text while it runs.
        """
        job = _Job(func, on_done, on_error or self.show_exception)
        self._jobs.append(job)
        if status:
            self.set_status(status)
        if not self._progress_on:
            self._progress_on = True
            self.progress.pack(side="right")
            self.progress.start(12)
        threading.Thread(target=job.run, name="stegano-worker", daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.root.after(POLL_MS, self._poll_jobs)

    def _poll_jobs(self):
        """Tk-thread poll: hand finished jobs' results to their callbacks."""
        self._poll_id = None
        if self._closed:
            return
        for job in list(self._jobs):
            try:
                ok, value = job.results.get_nowait()
            except queue.Empty:
                continue
            # Remove first, so a callback that starts a new job (or checks
            # `busy`) sees the right state.
            self._jobs.remove(job)
            try:
                (job.on_done if ok else job.on_error)(value)
            except Exception as exc:  # a bug in a callback must not stop the poll
                self._report_callback_exception(type(exc), exc, exc.__traceback__)
        if not self._jobs:
            self._progress_on = False
            self.progress.stop()
            self.progress.pack_forget()
        elif self._poll_id is None and not self._closed:
            self._poll_id = self.root.after(POLL_MS, self._poll_jobs)

    # ------------- Actions -------------
    def open_file(self, path: str | os.PathLike | None = None):
        """File > Open: load a file into the current tab's main input.

        Args:
            path: File to open; if None, a dialog asks (filtered for the tab).
        """
        tab = self.current_tab()
        if path is None:
            path = self.ask_open_path(f"Open in {tab.title}", tab.open_filetypes())
            if not path:
                return
        if tab.open_path(os.fspath(path)) is False:
            return  # refused while busy; the tab's "still working" status stays
        self.set_status(f"Opened {os.path.basename(os.fspath(path))} in {tab.title}")

    def resize_image(self):
        """Toolbar: shrink the Encode tab's image cover to fit 400x400."""
        self.notebook.select(self.encode_tab)
        self.encode_tab.resize_cover()

    def fill_max_message(self):
        """Toolbar: fill the Encode message to the cover's full capacity."""
        self.notebook.select(self.encode_tab)
        self.encode_tab.fill_max()

    def stego_written(self, cover: str, stego_path: str):
        """After hiding: offer the new file to the Decode and Compare tabs."""
        self.decode_tab.set_path(stego_path)
        if is_image_path(stego_path):
            self.compare_tab.set_paths(cover, stego_path)

    def reset_all(self):
        """Clear every tab. Files already written are left in place."""
        for tab in self.tabs:
            tab.reset()
        self.notebook.select(self.encode_tab)
        self.set_status("Ready")

    def about_text(self) -> str:
        """The Help > About text: description plus the registered plug-ins."""
        lines = [APP_TITLE, "CS 5800 final project, Northeastern University, 2025. Ethan Doherty.", "",
                 PROJECT_DESCRIPTION, "",
                 "Payload format: a self-describing STGA frame (magic, length, CRC-32) with optional "
                 "zlib compression and AES-256-GCM encryption (scrypt key). A password also scatters "
                 "the bits in a password-derived order.", "",
                 "Carrier plug-ins (carriers/ package):"]
        for cls in carriers.list_carriers():
            lines.append(f"  {cls.name}: {cls.description}")
            lines.append(f"      writes {' '.join(cls.extensions)}")
            if cls.cover_extensions:
                lines.append(f"      also reads covers {' '.join(cls.cover_extensions)}")
        lines += ["", "Detector plug-ins (detectors/ package):"]
        for row in detectors.list_detectors():
            lines.append(f"  {row['name']}: {row['description']}")
        lines += ["", f"Brute force: {len(bruteforce.enumerate_strategies())} bit layouts by default, "
                      "plus a password word-list attack on scattered payloads.",
                  "", "All scores and verdicts are heuristics, not calibrated probabilities."]
        return "\n".join(lines)

    def show_about(self) -> tk.Toplevel:
        """Help > About: a small window with the project and plug-in lists."""
        win = tk.Toplevel(self.root)
        win.title(f"About {APP_TITLE}")
        win.geometry("640x520")
        win.transient(self.root)
        ttk.Label(win, text=APP_TITLE, style="Title.TLabel").pack(side="top", anchor="w", padx=12, pady=(10, 0))
        frame, text = _scrolled_text(win, height=20)
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        _set_readonly_text(text, self.about_text())
        ttk.Button(win, text="Close", command=win.destroy).pack(side="bottom", pady=(0, 10))
        self.about_window = win
        return win

    def close(self):
        """Stop polling and destroy the window (File > Exit, window close)."""
        if self._closed:
            return
        self._closed = True
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        # Worker threads are daemons; a result that arrives after this is
        # simply never read.
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    """Launch the GUI: python gui.py"""
    root = tk.Tk()
    SteganoApp(root)
    root.mainloop()


# Launch the GUI when run as a script: python gui.py
if __name__ == "__main__":
    main()
