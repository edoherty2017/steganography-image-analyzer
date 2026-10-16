# bruteforce.py
"""Brute-force LSB extraction: try many bit layouts and rank what comes out.

A hidden message is only readable if you read the bits back in exactly the
order they were written. Tools disagree on that order: which colour channels
are used and in what sequence, which bit plane carries the data, whether the
eight bits of a byte are packed most- or least-significant first, and whether
pixels are visited row by row or column by column. This module enumerates
those layouts (``Strategy`` objects), pulls the first few kilobytes out of the
image under each one, and scores every result for "this looks like a real
payload". The best-scoring strategies come first.

A strategy is four choices (plus an optional fifth):

    channels    "RGB", "RBG", "GRB", "GBR", "BRG", "BGR", "R", "G", "B", and
                "RGBA" / "BGRA" when the image has an alpha channel. The
                order is the order samples are read inside each pixel.
    bit         The bit plane, 0 = least significant bit (LSB).
    bit_order   "msb": the first extracted bit becomes the most significant
                bit of the byte (what encoder.py and payload frames use).
                "lsb": the first extracted bit becomes the least significant.
    traversal   "row": pixels left to right, then top to bottom (row-major,
                like encoder.py). "col": pixels top to bottom, then left to
                right (column-major). Channels are read inside each pixel
                in both cases.
    depth       Bits read from each sample (default 1). With depth 2 and
                bit 0, each sample gives bit 1 then bit 0, the layout of
                the carriers package with ``bits_per_sample=2``.

So ``Strategy("RGB", 0, "msb", "row")`` is the layout of this project's
legacy encoder.py and of the unscattered ImageLSBCarrier, and
``Strategy("RGB", 0, "msb", "row", depth=2)`` is that carrier's 2-bit mode.

Scoring (``score_candidate``) is a set of heuristics, not a statistical test,
and the score is a ranking confidence in [0, 1], not a probability:

    1.00        This project's STGA frame (see payload.py) with a matching
                CRC32. The header is parsed locally so this module does not
                depend on payload.py.
    0.97        A known file signature (PNG, JPEG, GIF, ZIP, PDF, GZIP, BMP,
                RIFF/WAV) whose internal structure was walked to its end.
    0.90        STGA header that is valid but whose body is not fully
                extracted yet (bruteforce() then reads further and re-checks).
    0.88        A file signature whose file continues past the extracted
                bytes (bruteforce() reads further here too).
    0.80        A signature-only format (7z, RAR, ELF, FLAC, Ogg, MP3/ID3,
                TIFF) whose end cannot be located without a full parser.
    0.55, 0.50  An STGA header whose CRC does not match, or a file signature
                whose structure does not parse (damaged, or a false match).
    0.15-0.95   The legacy format: printable text followed by 8 zero bytes
                (encoder.py's 64-bit delimiter), searched at byte-aligned
                offsets only, so a final character with trailing zero bits
                is not cut off.
    0.00-0.85   A run of printable UTF-8 / ASCII text. Long, varied,
                word-like runs score higher, and a run that ends in a NUL
                byte needs fewer characters than one that just stops.
                Repetitive runs such as "UUUUUUUU" (which smooth gradients
                can produce in an LSB plane) score near 0.
    0.00-0.10   Anything else ("binary"). High-entropy noise, which is what
                the LSB plane of an ordinary photo looks like, scores 0.

The text heuristics were tuned by hand on the synthetic covers in the test
suite and on this repository's own result images: there, every layout of a
clean cover scored below 0.3, and hidden natural-language messages of about
25 characters or more scored above 0.5. Shorter messages still tend to rank
first, but with a lower score, because a few printable bytes are also what
noise produces by chance.

Limitations. bruteforce() tries sequential layouts only. Payloads scattered
with a key (the carriers package does this when a password is given) look
like noise to every strategy; dictionary_attack() covers that case by trying
a list of candidate passwords as scatter keys. Data encrypted without any
recognisable header cannot be told apart from noise by content at all; the
detectors package is the tool for spotting such embeddings statistically.

Typical use::

    from bruteforce import bruteforce, save_candidate
    for cand in bruteforce("suspect.png", top_k=5):
        print(f"{cand['score']:.2f}  {cand['strategy']}  {cand['kind']}  "
              f"{cand['preview']}")
    save_candidate(bruteforce("suspect.png")[0], "recovered/")

    from bruteforce import dictionary_attack
    for hit in dictionary_attack("suspect.png", open("words.txt")):
        print(hit["password"], hit["preview"])

or from the command line::

    python bruteforce.py suspect.png --top 5 --save recovered/
    python bruteforce.py suspect.png --wordlist words.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Strategy space
# ---------------------------------------------------------------------------

# Index of each channel letter in an (H, W, C) RGB or RGBA array.
_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2, "A": 3}

# Channel orders tried on every image, and the extra ones when alpha exists.
CHANNEL_ORDERS = ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR", "R", "G", "B")
ALPHA_CHANNEL_ORDERS = ("RGBA", "BGRA")

BIT_ORDERS = ("msb", "lsb")
TRAVERSALS = ("row", "col")


@dataclass(frozen=True)
class Strategy:
    """One way of reading bits out of an image.

    Attributes:
        channels: Channel letters in the order they are read inside each
            pixel, e.g. "RGB", "BGR", "G" or "RGBA". Letters must be unique
            and drawn from R, G, B, A.
        bit: Bit plane to read, 0 (the LSB) to 7.
        bit_order: "msb" packs each group of 8 extracted bits most
            significant bit first; "lsb" packs them least significant first.
        traversal: "row" visits pixels row-major (x inner); "col" visits
            them column-major (y inner).
        depth: Bits taken from each sample, highest first: planes
            ``bit + depth - 1`` down to ``bit``. 1 reads a single plane.

    Raises:
        ValueError: If any field is out of range.
    """

    channels: str = "RGB"
    bit: int = 0
    bit_order: str = "msb"
    traversal: str = "row"
    depth: int = 1

    def __post_init__(self) -> None:
        if (
            not self.channels
            or any(c not in _CHANNEL_INDEX for c in self.channels)
            or len(set(self.channels)) != len(self.channels)
        ):
            raise ValueError(
                f"channels must be unique letters from 'RGBA', got {self.channels!r}"
            )
        if not isinstance(self.bit, (int, np.integer)) or not 0 <= self.bit <= 7:
            raise ValueError(f"bit must be an integer 0..7, got {self.bit!r}")
        if self.bit_order not in BIT_ORDERS:
            raise ValueError(f"bit_order must be 'msb' or 'lsb', got {self.bit_order!r}")
        if self.traversal not in TRAVERSALS:
            raise ValueError(f"traversal must be 'row' or 'col', got {self.traversal!r}")
        if not isinstance(self.depth, (int, np.integer)) or not 1 <= self.depth <= 8 - self.bit:
            raise ValueError(f"depth must be 1..{8 - self.bit} for bit {self.bit}, got {self.depth!r}")

    @property
    def planes(self) -> str:
        """The bit planes read, e.g. ``"bit0"`` or ``"bits1-0"``."""
        if self.depth == 1:
            return f"bit{self.bit}"
        return f"bits{self.bit + self.depth - 1}-{self.bit}"

    @property
    def label(self) -> str:
        """Short human-readable name, e.g. ``"BGR bit0 lsb-first col"``."""
        return f"{self.channels} {self.planes} {self.bit_order}-first {self.traversal}"

    @property
    def slug(self) -> str:
        """File-name-safe name, e.g. ``"BGR_bit0_lsb_col"``."""
        return f"{self.channels}_{self.planes}_{self.bit_order}_{self.traversal}"

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a JSON-serializable dict."""
        return asdict(self)

    def __str__(self) -> str:
        return self.label


def enumerate_strategies(max_bit: int = 1, *, has_alpha: bool = False,
                         two_bit: bool = True) -> list[Strategy]:
    """List every strategy to try, most common layouts first.

    The order is bit plane (outermost), then channel order, then bit order,
    then traversal, so ``Strategy("RGB", 0, "msb", "row")`` (the legacy
    encoder's layout) is always first. bruteforce() sorts stably by score,
    so this order also breaks ties.

    Args:
        max_bit: Highest bit plane to include (0 = LSB only). 0..7.
        has_alpha: Also include "RGBA" and "BGRA" channel orders.
        two_bit: When max_bit >= 1, also include the two-bits-per-sample
            layouts (planes 1 and 0 together, depth=2) used by the carriers
            package's ``bits_per_sample=2`` mode. They come last.

    Returns:
        One strategy per channel order x bit plane x bit order x traversal:
        ``9 (+2 with alpha) * (max_bit + 1) * 2 * 2``, which is 72 for the
        defaults without alpha, plus ``9 (+2) * 2 * 2`` = 36 two-bit ones
        when ``two_bit`` applies (108 in total by default).

    Raises:
        ValueError: If max_bit is outside 0..7.
    """
    if not 0 <= int(max_bit) <= 7:
        raise ValueError(f"max_bit must be 0..7, got {max_bit!r}")
    orders = CHANNEL_ORDERS + (ALPHA_CHANNEL_ORDERS if has_alpha else ())
    strategies = [
        Strategy(channels, bit, bit_order, traversal)
        for bit in range(int(max_bit) + 1)
        for channels in orders
        for bit_order in BIT_ORDERS
        for traversal in TRAVERSALS
    ]
    if two_bit and max_bit >= 1:
        strategies += [
            Strategy(channels, 0, bit_order, traversal, depth=2)
            for channels in orders
            for bit_order in BIT_ORDERS
            for traversal in TRAVERSALS
        ]
    return strategies


# ---------------------------------------------------------------------------
# Image loading and extraction
# ---------------------------------------------------------------------------

def _load_array(image: Any) -> np.ndarray:
    """Return the image as an (H, W, 3) or (H, W, 4) uint8 array.

    Args:
        image: A path (str or PathLike), a PIL image, or an array. Arrays may
            be (H, W) grayscale, (H, W, 1), (H, W, 3) or (H, W, 4); grayscale
            is repeated into three channels, the same as Pillow's
            ``convert("RGB")``. Images with transparency become RGBA, all
            others RGB.

    Returns:
        A C-contiguous uint8 array.

    Raises:
        ValueError: If the array has an unsupported shape or values outside
            0..255.
    """
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as img:
            return _load_array(img)
    if isinstance(image, Image.Image):
        has_alpha = image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
            image.mode == "P" and "transparency" in image.info
        )
        return np.asarray(image.convert("RGBA" if has_alpha else "RGB"), dtype=np.uint8)

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if not np.issubdtype(arr.dtype, np.integer) or (
            arr.size and (arr.min() < 0 or arr.max() > 255)
        ):
            raise ValueError("image array must hold integers in 0..255")
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an (H, W), (H, W, 3) or (H, W, 4) array, got {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def capacity_bytes(arr: np.ndarray, strategy: Strategy) -> int:
    """Number of whole bytes a strategy can read from an image.

    Args:
        arr: (H, W, C) uint8 array.
        strategy: The layout.

    Returns:
        ``H * W * len(strategy.channels) * strategy.depth // 8``.
    """
    return arr.shape[0] * arr.shape[1] * len(strategy.channels) * strategy.depth // 8


def extract_bytes(arr: np.ndarray, strategy: Strategy, n_bytes: int) -> bytes:
    """Read bytes out of an image under one strategy.

    Fully vectorized: only the pixels needed for ``n_bytes`` are touched
    (whole rows for "row", whole columns for "col"), then the bit plane(s)
    are isolated with shifts and a mask and packed with ``np.packbits``.

    Args:
        arr: (H, W, 3) or (H, W, 4) uint8 array (see _load_array()).
        strategy: Which channels, bit plane, bit order and traversal to use.
        n_bytes: How many bytes to return at most.

    Returns:
        ``min(n_bytes, capacity_bytes(arr, strategy))`` bytes.

    Raises:
        ValueError: If n_bytes is negative or the strategy reads an alpha
            channel the array does not have.
    """
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    channel_idx = [_CHANNEL_INDEX[c] for c in strategy.channels]
    if max(channel_idx) >= arr.shape[2]:
        raise ValueError(f"strategy {strategy} needs an alpha channel")
    height, width = arr.shape[:2]
    n_ch = len(channel_idx)
    n_bits = min(int(n_bytes), capacity_bytes(arr, strategy)) * 8
    if n_bits == 0:
        return b""
    n_samples = -(-n_bits // strategy.depth)  # ceiling division
    n_pixels = -(-n_samples // n_ch)

    if strategy.traversal == "col":
        # Column-major: x outer, y inner. Take just the columns needed, then
        # transpose so that a row-major flatten walks down each column.
        n_cols = min(width, -(-n_pixels // height))
        region = arr[:, :n_cols, :].transpose(1, 0, 2)
    else:
        n_rows = min(height, -(-n_pixels // width))
        region = arr[:n_rows]

    # Fancy indexing on the last axis puts the channels in strategy order;
    # the row-major flatten then interleaves them inside each pixel.
    samples = region[..., channel_idx].reshape(-1)[:n_samples]
    if strategy.depth == 1:
        bits = (samples >> strategy.bit) & 1
    else:
        # One row of `depth` bits per sample, highest plane first.
        shifts = np.arange(strategy.bit + strategy.depth - 1, strategy.bit - 1, -1,
                           dtype=np.uint8)
        bits = ((samples[:, None] >> shifts) & 1).reshape(-1)[:n_bits]
    bitorder = "big" if strategy.bit_order == "msb" else "little"
    return np.packbits(bits, bitorder=bitorder).tobytes()


# ---------------------------------------------------------------------------
# Small helpers for scoring
# ---------------------------------------------------------------------------

def _entropy(data: bytes) -> float:
    """Shannon entropy of the byte histogram, in bits per byte (0..8)."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# Characters that end a text run: C0 controls other than tab/LF/CR, DEL, the
# C1 controls, U+FFFD (what invalid UTF-8 decodes to) and the non-characters.
_NON_TEXT = re.compile(r"[^\t\n\r\x20-\x7e -￼\U00010000-\U0010ffff]")

# Lookup table: True for printable ASCII bytes plus tab, LF and CR.
_PRINTABLE_ASCII = np.zeros(256, dtype=bool)
_PRINTABLE_ASCII[0x20:0x7F] = True
_PRINTABLE_ASCII[[0x09, 0x0A, 0x0D]] = True

# Latin-1 printable bytes, for legacy messages written by encoder.py, which
# stores code points 128..255 as single bytes rather than UTF-8.
_PRINTABLE_LATIN1 = _PRINTABLE_ASCII.copy()
_PRINTABLE_LATIN1[0xA0:] = True

# Legacy end-of-message marker: 64 zero bits (encoder.DELIMITER).
_LEGACY_DELIM = b"\x00" * 8

# Text runs shorter than this are ignored; runs found away from the start of
# the extracted bytes must be longer, because random bytes are printable
# ASCII with probability 98/256 and so produce runs of about 8-10 characters
# somewhere in every few kilobytes of noise.
_MIN_LEADING_TEXT = 4
_MIN_INNER_TEXT = 16

# How far into the extracted bytes a file signature or STGA header may start.
# Some tools write a length prefix before the data.
_MAX_SIG_OFFSET = 32

# Preview length in characters.
_PREVIEW_CHARS = 80


def _safe_preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Make a short single-line, printable preview of some text."""
    out = []
    for ch in text[:limit]:
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append("?")
    preview = "".join(out)
    return preview + ("..." if len(text) > limit else "")


def _hex_preview(data: bytes, n: int = 16) -> str:
    return " ".join(f"{b:02x}" for b in data[:n]) + (" ..." if len(data) > n else "")


def _text_quality(text: str, min_len: int = 3) -> float:
    """Heuristic "looks like a real message" factor for a text run, 0..1.

    Product of five terms:

    * length: 0 at ``min_len`` characters or fewer, rising on a log scale
      to 1 at 64. Callers pass a larger ``min_len`` when the run has less
      supporting evidence: random bytes are printable with probability
      about 0.4, so across ~100 strategies an unterminated run of 6 or 7
      printable characters at the start is common in plain noise, while a
      run that ends in a NUL byte or the legacy delimiter is much rarer.
    * diversity: character entropy relative to its maximum for the run
      length (capped at 16 symbols). Runs such as "UUUU" or "abab" get 0.
    * aperiodicity: the LSB planes of smooth gradients and flat regions
      repeat with short periods (e.g. 3 bytes when three channels cycle),
      which can decode to printable but repetitive strings such as
      "~@xxHz~@xxHz". For lags 1..12 the share of positions where the text
      equals itself shifted by that lag is measured; natural language stays
      near 0.1, so a best lag above 0.3 is penalized and 0.6 or more gives 0.
    * word-likeness: share of letters, digits and spaces. Natural language,
      hex and base64 are near 1; printable noise is about 0.66.
    * script consistency: random bytes that happen to be valid UTF-8 mostly
      form two-byte characters between U+0080 and U+07FF (IPA, Greek,
      Arabic, ...), sprinkled among ASCII. Real text is either almost all
      ASCII/Latin or mostly another script, so a share of such characters
      between about 10% and 65% gives 0.
    """
    n = len(text)
    if n <= max(3, min_len):
        return 0.0
    length = _clamp(math.log2(n / min_len) / math.log2(64 / min_len))
    codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / n
    h = float(-(p * np.log2(p)).sum())
    h_max = math.log2(min(n, 16))
    diversity = _clamp((h / h_max - 0.3) / 0.4)
    max_match = max(
        (float(np.mean(codes[lag:] == codes[:-lag])) for lag in range(1, min(12, n // 2) + 1)),
        default=0.0,
    )
    aperiodic = _clamp((0.6 - max_match) / 0.3)
    wordy = sum(1 for ch in text if ch.isalnum() or ch == " ") / n
    word_like = _clamp((wordy - 0.4) / 0.4)
    # Latin-1 symbols (U+0080-U+00BF) and the non-Latin two-byte range
    # (U+0250-U+07FF); accented Latin letters do not count.
    exotic = float(np.mean(((codes >= 0x80) & (codes < 0xC0))
                           | ((codes >= 0x250) & (codes < 0x800))))
    consistent = 1.0 - _clamp(min(exotic, 0.75 - exotic) / 0.1)
    return length * diversity * aperiodic * word_like * consistent


def _text_prefix(data: bytes) -> str | None:
    """Return data decoded as text if it is printable UTF-8, else None.

    A multi-byte character cut off by the end of ``data`` is dropped rather
    than counted as an error, so a fixed-size slice of text still passes.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data":
            return None
        text = data[:exc.start].decode("utf-8")
    return text if text and _NON_TEXT.search(text) is None else None


def _ascii_share(text: str) -> float:
    """Fraction of characters that are ASCII."""
    return sum(1 for ch in text if ord(ch) < 128) / len(text) if text else 0.0


def _result(score: float, kind: str, preview: str, data: bytes,
            info: dict[str, Any]) -> dict[str, Any]:
    return {"score": float(round(score, 4)), "kind": kind, "preview": preview,
            "data": data, "info": info}


# ---------------------------------------------------------------------------
# STGA frames (this project's payload format, parsed locally)
# ---------------------------------------------------------------------------

_STGA_MAGIC = b"STGA"
_STGA_VERSION = 1
_STGA_HEADER = struct.Struct(">4sBBII")  # magic, version, flags, body_len, crc32
_STGA_HEADER_LEN = _STGA_HEADER.size     # 14
_STGA_FLAGS = {0x01: "encrypted", 0x02: "file", 0x04: "compressed"}
_STGA_FLAGS_ALL = 0x07


def _check_stga(data: bytes) -> dict[str, Any] | None:
    """Look for a valid STGA frame header and check its CRC if possible.

    Args:
        data: Extracted bytes.

    Returns:
        A candidate result, or None if no valid header is present.
    """
    start = data.find(_STGA_MAGIC)
    while start != -1:
        header = data[start:start + _STGA_HEADER_LEN]
        if len(header) == _STGA_HEADER_LEN:
            _, version, flags, body_len, crc = _STGA_HEADER.unpack(header)
            if version == _STGA_VERSION and not flags & ~_STGA_FLAGS_ALL:
                break
        start = data.find(_STGA_MAGIC, start + 1)
    else:
        return None

    flag_names = [name for bit, name in _STGA_FLAGS.items() if flags & bit]
    frame_end = start + _STGA_HEADER_LEN + body_len
    complete = frame_end <= len(data)
    info: dict[str, Any] = {
        "offset": start,
        "version": version,
        "flags": flags,
        "flag_names": flag_names,
        "body_len": body_len,
        "frame_len": _STGA_HEADER_LEN + body_len,
        "crc32": f"0x{crc:08x}",
        "encrypted": bool(flags & 0x01),
        "compressed": bool(flags & 0x04),
        "has_filename": bool(flags & 0x02),
        "complete": complete,
    }
    desc = (f"STGA v{version} frame, {body_len}-byte body, "
            f"flags: {', '.join(flag_names) or 'none'}")
    if not complete:
        info["crc_ok"] = None
        info["needs_bytes"] = frame_end
        info["needs_exact"] = True
        score = 0.90
        preview = f"{desc} (body not yet extracted)"
        return _result(score - (0.02 if start else 0.0), "stga", preview,
                       data[start:], info)

    body = data[start + _STGA_HEADER_LEN:frame_end]
    crc_ok = (zlib.crc32(body) & 0xFFFFFFFF) == crc
    info["crc_ok"] = crc_ok
    if crc_ok:
        score = 1.0
        preview = f"{desc}, CRC ok"
        if flags == 0:
            # A plain frame's body is the hidden data itself.
            snippet = _text_prefix(body[:96])
            if snippet:
                preview += f': "{_safe_preview(snippet, 48)}"'
            else:
                preview += f": {_hex_preview(body, 12)}"
    else:
        score = 0.55
        preview = f"{desc}, CRC MISMATCH (damaged or false match)"
    return _result(score - (0.02 if start else 0.0), "stga", preview,
                   data[start:frame_end], info)


# ---------------------------------------------------------------------------
# File signatures and structure walks
# ---------------------------------------------------------------------------
#
# Each walker gets the bytes starting at the signature and returns
# (status, length, details):
#   "complete":  the file ends at ``length``.
#   "truncated": the file is longer than the bytes given; ``length`` is a
#                lower bound on its total size (exact if details["exact"]).
#   "invalid":   the signature matched but the structure does not parse.

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _walk_png(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    pos, first, details = 8, True, {}
    while True:
        if pos + 8 > len(buf):
            return "truncated", pos + 12, details
        length, ctype = struct.unpack(">I4s", buf[pos:pos + 8])
        if not ctype.isalpha() or length > 0x7FFFFFFF:
            return "invalid", None, details
        if first:
            if ctype != b"IHDR" or length != 13:
                return "invalid", None, details
        end = pos + 12 + length
        if end > len(buf):
            return "truncated", end, details
        (crc,) = struct.unpack(">I", buf[end - 4:end])
        if zlib.crc32(buf[pos + 4:end - 4]) & 0xFFFFFFFF != crc:
            return "invalid", None, details
        if first:
            w, h = struct.unpack(">II", buf[pos + 8:pos + 16])
            details.update(width=w, height=h)
            first = False
        if ctype == b"IEND":
            return "complete", end, details
        pos = end


# Marker bytes that may follow FF D8 FF at the start of a JPEG.
_JPEG_FIRST_MARKERS = set(range(0xE0, 0xF0)) | {0xDB, 0xC0, 0xC1, 0xC2, 0xC4, 0xFE, 0xDD}


def _walk_jpeg(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    pos, n, details = 2, len(buf), {}
    while True:
        if pos >= n:
            return "truncated", pos + 2, details
        if buf[pos] != 0xFF:
            return "invalid", None, details
        while pos < n and buf[pos] == 0xFF:  # optional fill bytes
            pos += 1
        if pos >= n:
            return "truncated", pos + 2, details
        marker = buf[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            return "complete", pos, details
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:  # standalone markers
            continue
        if marker == 0x00:
            return "invalid", None, details
        if pos + 2 > n:
            return "truncated", pos + 2, details
        (seg_len,) = struct.unpack(">H", buf[pos:pos + 2])
        if seg_len < 2:
            return "invalid", None, details
        if marker in (0xC0, 0xC1, 0xC2) and pos + 7 <= n:
            h, w = struct.unpack(">HH", buf[pos + 3:pos + 7])
            details.update(width=w, height=h)
        pos += seg_len
        if marker == 0xDA:  # start of scan: skip entropy-coded data
            while True:
                i = buf.find(b"\xff", pos)
                if i == -1 or i + 1 >= n:
                    return "truncated", max(n, pos) + 2, details
                nxt = buf[i + 1]
                if nxt == 0x00 or 0xD0 <= nxt <= 0xD7:  # stuffed byte / restart
                    pos = i + 2
                elif nxt == 0xFF:
                    pos = i + 1
                else:
                    pos = i
                    break


def _walk_gif(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    n = len(buf)
    if n < 13:
        return "truncated", 13, {}
    w, h, packed = struct.unpack("<HHB", buf[6:11])
    details = {"width": w, "height": h}
    pos = 13
    if packed & 0x80:
        pos += 3 * (2 ** ((packed & 0x07) + 1))

    def skip_sub_blocks(p: int) -> int | None:
        while p < n:
            size = buf[p]
            p += 1
            if size == 0:
                return p
            p += size
        return None

    while True:
        if pos >= n:
            return "truncated", pos + 1, details
        block = buf[pos]
        if block == 0x3B:  # trailer
            return "complete", pos + 1, details
        if block == 0x21:  # extension: introducer, label, sub-blocks
            nxt = skip_sub_blocks(pos + 2)
        elif block == 0x2C:  # image descriptor (10 bytes), local table, LZW data
            if pos + 10 > n:
                return "truncated", pos + 11, details
            local = buf[pos + 9]
            pos += 10
            if local & 0x80:
                pos += 3 * (2 ** ((local & 0x07) + 1))
            nxt = skip_sub_blocks(pos + 1)  # +1 skips the LZW minimum code size
        else:
            return "invalid", None, details
        if nxt is None:
            return "truncated", n + 1, details
        pos = nxt


def _walk_zip(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    # The end-of-central-directory record closes the archive.
    i = buf.rfind(b"PK\x05\x06")
    if i == -1 or i + 22 > len(buf):
        return "truncated", len(buf) + 1, {}
    (comment_len,) = struct.unpack("<H", buf[i + 20:i + 22])
    end = i + 22 + comment_len
    if end > len(buf):
        return "truncated", end, {"exact": True}
    return "complete", end, {}


def _walk_pdf(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    # Heuristic: the last "%%EOF" in the bytes seen so far. A PDF with
    # incremental updates has several, so a truncated window can stop early.
    i = buf.rfind(b"%%EOF")
    if i == -1:
        return "truncated", len(buf) + 1, {}
    end = i + 5
    if buf[end:end + 2] == b"\r\n":
        end += 2
    elif buf[end:end + 1] in (b"\n", b"\r"):
        end += 1
    return "complete", end, {}


def _walk_gzip(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    # Inflate (discarding the output) to find where the first member ends.
    # Output is capped so a crafted "zip bomb" cannot exhaust memory.
    inflater = zlib.decompressobj(wbits=31)
    chunk, produced = buf, 0
    try:
        while True:
            produced += len(inflater.decompress(chunk, 1 << 20))
            if inflater.eof:
                return "complete", len(buf) - len(inflater.unused_data), \
                    {"uncompressed_bytes": produced}
            chunk = inflater.unconsumed_tail
            if not chunk:
                return "truncated", len(buf) + 1, {}
            if produced > 64 * (1 << 20):
                return "truncated", len(buf) + 1, {"note": "inflates past 64 MiB"}
    except zlib.error:
        return "invalid", None, {}


_BMP_DIB_SIZES = {12, 40, 52, 56, 64, 108, 124}


def _walk_bmp(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    size, _, pixel_offset, dib = struct.unpack("<IIII", buf[2:18])
    if size < 14 + dib or pixel_offset > size:
        return "invalid", None, {}
    details: dict[str, Any] = {"exact": True}
    if dib >= 40 and len(buf) >= 26:
        w, h = struct.unpack("<ii", buf[18:26])
        details.update(width=w, height=abs(h))
    if size > len(buf):
        return "truncated", size, details
    return "complete", size, details


def _walk_riff(buf: bytes) -> tuple[str, int | None, dict[str, Any]]:
    (size,) = struct.unpack("<I", buf[4:8])
    end = 8 + size + (size & 1)
    details = {"exact": True, "form": buf[8:12].decode("latin-1")}
    if end > len(buf):
        return "truncated", end, details
    return "complete", end, details


def _bmp_ok(b: bytes) -> bool:
    if len(b) < 18 or b[6:10] != b"\x00\x00\x00\x00":
        return False
    return struct.unpack("<I", b[14:18])[0] in _BMP_DIB_SIZES


def _riff_ext(b: bytes) -> str | None:
    if len(b) < 12:
        return None
    form = b[8:12]
    known = {b"WAVE": "wav", b"AVI ": "avi", b"WEBP": "webp"}
    if form in known:
        return known[form]
    return "riff" if all(48 <= c <= 122 or c == 32 for c in form) else None


# (magic, ext, description, extra check on bytes from the signature on,
#  structure walker or None for signature-only formats)
_SIGNATURES: list[tuple[bytes, str, str, Any, Any]] = [
    (_PNG_SIG, "png", "PNG image", None, _walk_png),
    (b"\xff\xd8\xff", "jpg", "JPEG image",
     lambda b: len(b) > 3 and b[3] in _JPEG_FIRST_MARKERS, _walk_jpeg),
    (b"GIF87a", "gif", "GIF image", None, _walk_gif),
    (b"GIF89a", "gif", "GIF image", None, _walk_gif),
    (b"PK\x03\x04", "zip", "ZIP archive",
     lambda b: len(b) > 5 and b[5] == 0 and b[4] <= 63, _walk_zip),
    (b"%PDF-", "pdf", "PDF document",
     lambda b: len(b) > 6 and b[5:6] in (b"1", b"2") and b[6:7] == b".", _walk_pdf),
    (b"\x1f\x8b\x08", "gz", "gzip data",
     lambda b: len(b) > 3 and not b[3] & 0xE0, _walk_gzip),
    (b"BM", "bmp", "BMP image", _bmp_ok, _walk_bmp),
    (b"RIFF", "riff", "RIFF container", lambda b: _riff_ext(b) is not None, _walk_riff),
    (b"7z\xbc\xaf\x27\x1c", "7z", "7-Zip archive", None, None),
    (b"Rar!\x1a\x07", "rar", "RAR archive", None, None),
    (b"\x7fELF", "elf", "ELF executable", lambda b: len(b) > 4 and b[4] in (1, 2), None),
    (b"fLaC", "flac", "FLAC audio", None, None),
    (b"OggS\x00", "ogg", "Ogg media", None, None),
    (b"ID3", "mp3", "MP3 audio (ID3 tag)",
     lambda b: len(b) > 4 and b[3] in (2, 3, 4) and b[4] == 0, None),
    (b"II*\x00", "tif", "TIFF image", None, None),
    (b"MM\x00*", "tif", "TIFF image", None, None),
]

_RIFF_DESC = {"wav": "WAV audio", "avi": "AVI video", "webp": "WebP image",
              "riff": "RIFF container"}


def _check_signatures(data: bytes) -> dict[str, Any] | None:
    """Find the earliest validated file signature near the start of data.

    Args:
        data: Extracted bytes.

    Returns:
        The best file candidate, or None.
    """
    best: dict[str, Any] | None = None
    for magic, ext, desc, check, walker in _SIGNATURES:
        limit = min(len(data), _MAX_SIG_OFFSET + len(magic))
        offset = data.find(magic, 0, limit)
        while offset != -1:
            buf = data[offset:]
            if check is None or check(buf):
                break
            offset = data.find(magic, offset + 1, limit)
        if offset == -1:
            continue

        if ext == "riff":
            ext = _riff_ext(buf) or "riff"
            desc = _RIFF_DESC[ext]
        info: dict[str, Any] = {"offset": offset, "ext": ext, "format": desc}
        if walker is None:
            score, status = 0.80, "unknown"
            info.update(complete=None, end_known=False)
            payload = buf
        else:
            status, length, details = walker(buf)
            info.update({k: v for k, v in details.items() if k != "exact"})
            if status == "complete":
                score = 0.97
                info.update(complete=True, end_known=True, valid_structure=True)
                payload = buf[:length]
            elif status == "truncated":
                score = 0.88
                info.update(complete=False, end_known=bool(details.get("exact")),
                            valid_structure=True, needs_bytes=offset + length,
                            needs_exact=bool(details.get("exact")))
                payload = buf
            else:
                score = 0.50
                info.update(complete=None, end_known=False, valid_structure=False)
                payload = buf
        if offset:
            score -= 0.05

        size = ""
        if "width" in info and "height" in info:
            size = f" {info['width']}x{info['height']},"
        state = {"complete": "complete", "truncated": "continues past extracted bytes",
                 "invalid": "structure does not parse", "unknown": "end not located"}[status]
        at = f" at offset {offset}" if offset else ""
        preview = f"{desc}{at},{size} {len(payload)} bytes ({state})"
        cand = _result(score, f"file:{ext}", preview, payload, info)
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# Text and legacy-format detection
# ---------------------------------------------------------------------------

def _check_text(data: bytes) -> dict[str, Any] | None:
    """Score text at the start of data, the legacy format, and long inner runs.

    The leading run is cut at the first NUL byte (UTF-8 text never contains
    one) and at the first byte that is not printable UTF-8. If the run is
    followed by 8 zero bytes it is the legacy format (encoder.py's 64-bit
    delimiter, matched on a byte boundary).

    Args:
        data: Extracted bytes.

    Returns:
        The best text candidate, or None.
    """
    best: dict[str, Any] | None = None
    nul = data.find(b"\x00")
    head = data if nul == -1 else data[:nul]

    try:
        decoded = head.decode("utf-8")
        decode_stopped_at_end = False
    except UnicodeDecodeError as exc:
        decoded = head[:exc.start].decode("utf-8")
        # A multi-byte character cut off by the end of the extracted bytes.
        decode_stopped_at_end = nul == -1 and exc.reason == "unexpected end of data"
    bad = _NON_TEXT.search(decoded)
    text = decoded if bad is None else decoded[:bad.start()]
    text_bytes = text.encode("utf-8")
    whole_head = len(text_bytes) == len(head)
    encoding = "utf-8"

    # Legacy messages from encoder.py store code points 128..255 as single
    # Latin-1 bytes, which is usually invalid UTF-8. That reading is only
    # accepted for mostly-ASCII text (accented letters in a Western
    # language); otherwise saturated bit planes (0xFF 0xFF ...) would pass.
    legacy_tail = data[nul:nul + 8] if nul != -1 else b""
    is_legacy = nul > 0 and legacy_tail == _LEGACY_DELIM
    if is_legacy and not whole_head:
        latin = head.decode("latin-1")
        if _PRINTABLE_LATIN1[np.frombuffer(head, dtype=np.uint8)].all() \
                and _ascii_share(latin) >= 0.6:
            text, text_bytes, whole_head, encoding = latin, head, True, "latin-1"

    if text:
        info: dict[str, Any] = {"offset": 0, "encoding": encoding, "chars": len(text),
                                "bytes": len(text_bytes)}
        if whole_head and is_legacy:
            quality = _text_quality(text, 3)
            if len(text) <= 3:
                # Too short to judge by content: accept plain ASCII only.
                score = 0.15 if text.isascii() and text.isprintable() else 0.0
            else:
                score = 0.15 + 0.80 * quality if quality > 0 else 0.05
            info.update(format="legacy-delimiter", complete=True)
        elif whole_head and nul != -1 and data[nul:] == b"\x00" * (len(data) - nul) \
                and len(data) - nul < 8:
            # NULs run to the end of the extracted bytes: maybe a legacy
            # delimiter that is cut off. Read a little further to find out.
            score = 0.85 * _text_quality(text, 4)
            info.update(format="nul-terminated", complete=False,
                        needs_bytes=nul + 8, needs_exact=True)
        elif whole_head and nul != -1:
            score = 0.85 * _text_quality(text, 4)
            info.update(format="nul-terminated", complete=True)
        elif (whole_head and nul == -1) or decode_stopped_at_end:
            # The text runs to the end of the extracted bytes; there may be more.
            score = 0.85 * _text_quality(text, 6)
            info.update(format="unterminated", complete=False,
                        needs_bytes=2 * len(data), needs_exact=False)
        else:
            score = 0.85 * _text_quality(text, 6)
            info.update(format="unterminated", complete=True)
        if len(text) >= _MIN_LEADING_TEXT or info["format"] == "legacy-delimiter":
            best = _result(score, "text", _safe_preview(text), text_bytes, info)

    # A long printable-ASCII run further in (e.g. after a binary length
    # prefix). Found with a vectorized run-length scan.
    mask = _PRINTABLE_ASCII[np.frombuffer(data, dtype=np.uint8)]
    if mask.any():
        edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        i = int(np.argmax(ends - starts))
        s, e = int(starts[i]), int(ends[i])
        if s > 0 and e - s >= _MIN_INNER_TEXT:
            inner = data[s:e].decode("ascii")
            info = {"offset": s, "encoding": "ascii", "chars": e - s, "bytes": e - s,
                    "format": "inner-run", "complete": e < len(data)}
            if e == len(data):
                info.update(needs_bytes=2 * len(data), needs_exact=False)
            # Less likely to be deliberate than text at the very start.
            score = 0.85 * 0.75 * _text_quality(inner, 6)
            if best is None or score > best["score"]:
                best = _result(score, "text", f"@{s}: {_safe_preview(inner)}",
                               data[s:e], info)
    return best


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidate(data: bytes) -> dict[str, Any]:
    """Score extracted bytes for how much they look like a hidden payload.

    All checks are heuristics (see the module docstring for the score
    bands). The highest-scoring interpretation wins.

    Args:
        data: Bytes extracted under one strategy.

    Returns:
        A dict with:
            score (float): Ranking confidence in [0, 1].
            kind (str): "stga", "file:<ext>" (e.g. "file:png"), "text" or
                "binary".
            preview (str): Short, single-line, printable description.
            data (bytes): The relevant bytes, trimmed: the STGA frame, the
                file (cut at its end marker when found), or the text
                (cut at the first NUL run); the input unchanged for binary.
            info (dict): JSON-serializable details (offsets, header fields,
                encoding, entropy, and ``needs_bytes`` when reading more of
                the image could complete the payload).
    """
    data = bytes(data)
    entropy = _entropy(data)
    distinct = len(set(data[:4096]))
    # Low but non-zero entropy means structure; noise (about 8 bits/byte)
    # and constant runs score 0.
    binary_score = 0.10 * _clamp((7.0 - entropy) / 3.0) if distinct >= 8 else 0.0
    best = _result(
        binary_score, "binary",
        f"binary, entropy {entropy:.2f} bits/byte: {_hex_preview(data)}",
        data, {},
    )
    for check in (_check_stga, _check_signatures, _check_text):
        cand = check(data)
        if cand is not None and cand["score"] > best["score"]:
            best = cand
    best["info"]["entropy"] = round(entropy, 3)
    best["info"]["window_bytes"] = len(data)
    return best


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Only candidates at least this confident are read further than max_bytes.
_EXPAND_MIN_SCORE = 0.5


def bruteforce(
    image: Any,
    top_k: int | None = 10,
    max_bytes: int = 4096,
    max_bit: int = 1,
    *,
    strategies: list[Strategy] | None = None,
    expand: bool = True,
    max_expand_bytes: int = 64 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Try every strategy on an image and return the most promising results.

    Each strategy reads the first ``max_bytes`` bytes, which are scored with
    score_candidate(). If a top candidate says it continues further (an STGA
    frame whose body is longer, a file whose end was not reached, text that
    runs to the end), it is re-read with a larger window, up to the image's
    capacity and ``max_expand_bytes``, and scored again. The window grows to
    the exact frame or file size when the header gives one, otherwise it
    doubles.

    Args:
        image: Path, PIL image, or uint8 array (see _load_array()).
        top_k: How many candidates to return; None returns all.
        max_bytes: Bytes to read per strategy for the first pass.
        max_bit: Highest bit plane to try (0 = LSB only).
        strategies: Explicit strategies to try instead of
            enumerate_strategies(max_bit, has_alpha=...), which includes the
            two-bits-per-sample layouts when max_bit >= 1.
        expand: Whether to re-read promising candidates past max_bytes.
        max_expand_bytes: Upper bound on bytes read by expansion.

    Returns:
        Candidates sorted best first. Each is a dict with ``strategy``
        (Strategy), ``label`` (str), ``score``, ``kind``, ``preview``,
        ``data`` (bytes) and ``info`` (JSON-serializable; ``info["truncated"]``
        is True when the payload is longer than could be read).

    Raises:
        ValueError: If max_bytes is not positive or a strategy needs an
            alpha channel the image does not have.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    arr = _load_array(image)
    has_alpha = arr.shape[2] == 4
    if strategies is None:
        strategies = enumerate_strategies(max_bit, has_alpha=has_alpha)

    candidates = []
    for strategy in strategies:
        window = extract_bytes(arr, strategy, max_bytes)
        cand = score_candidate(window)
        cand["strategy"] = strategy
        candidates.append(cand)
    # Stable sort: ties keep enumerate_strategies() order (common layouts first).
    candidates.sort(key=lambda c: -c["score"])
    if top_k is not None:
        candidates = candidates[:max(int(top_k), 0)]

    if expand:
        for i, cand in enumerate(candidates):
            if cand["score"] >= _EXPAND_MIN_SCORE:
                candidates[i] = _expand(arr, cand, max_bytes, max_expand_bytes)
        candidates.sort(key=lambda c: -c["score"])

    results = []
    for cand in candidates:
        strategy = cand["strategy"]
        info = cand["info"]
        needs = info.get("needs_bytes")
        info["truncated"] = bool(needs and needs > info.get("window_bytes", 0))
        info["capacity_bytes"] = capacity_bytes(arr, strategy)
        results.append({
            "strategy": strategy,
            "label": strategy.label,
            "score": cand["score"],
            "kind": cand["kind"],
            "preview": cand["preview"],
            "data": cand["data"],
            "info": info,
        })
    return results


def _expand(arr: np.ndarray, cand: dict[str, Any], start_bytes: int,
            limit: int) -> dict[str, Any]:
    """Re-read a candidate with larger windows until it is complete.

    Args:
        arr: Image array.
        cand: A scored candidate with a "strategy" key.
        start_bytes: Size of the window it was scored on.
        limit: Largest window to read.

    Returns:
        The final scored candidate (with "strategy" set).
    """
    strategy = cand["strategy"]
    cap = min(capacity_bytes(arr, strategy), limit)
    size = start_bytes
    for _ in range(64):  # doubling from 1 byte reaches any size in < 64 steps
        needs = cand["info"].get("needs_bytes")
        if not needs or needs <= size or size >= cap:
            break
        if cand["info"].get("needs_exact"):
            size = min(cap, needs)
        else:
            size = min(cap, max(needs, 2 * size))
        cand = score_candidate(extract_bytes(arr, strategy, size))
        cand["strategy"] = strategy
    return cand


# ---------------------------------------------------------------------------
# Password list attack on keyed (scattered) embeddings
# ---------------------------------------------------------------------------

def dictionary_attack(
    source: Any,
    passwords: Any,
    *,
    bits_per_sample: tuple[int, ...] = (1, 2),
) -> list[dict[str, Any]]:
    """Try candidate passwords against a payload hidden with a scatter key.

    When stego.py hides data with a password, the password is also the
    scatter key: the carrier visits its samples in a pseudo-random order
    seeded from SHA-256(password), so none of the sequential strategies
    above can find it. This function guesses: for each password it reads
    the first 14 bytes in that password's sample order and checks for a
    valid STGA header (magic, version, known flag bits). Only on a hit does
    it read the rest of the frame, check the CRC and try to unpack it
    (AES-GCM decryption with the same password when the frame is
    encrypted).

    It works on any format the carriers package supports (PNG, BMP, TIFF,
    WAV, ...) because it goes through the carrier registry instead of
    reading pixels itself. The cost is one permutation of all samples per
    password (milliseconds for a typical image), so word lists of thousands
    of entries are practical. Like any offline guessing attack it only
    finds passwords that are on the list.

    Args:
        source: A path to a stego file, or an already loaded carrier
            (anything with ``extract(n_bits, key)``, ``capacity_bits()`` and
            a writable ``bits_per_sample`` attribute).
        passwords: Iterable of candidate passwords (str), e.g. the lines of
            a word list. Trailing newlines are stripped; empty entries are
            skipped because an empty key means "no scatter".
        bits_per_sample: Embedding depths to try for each password.

    Returns:
        One dict per hit, in the order found, with ``password``,
        ``bits_per_sample``, ``score`` (1.0 if the frame unpacked, 0.9 if
        the header and CRC matched but unpacking failed), ``kind``
        ("stga"), ``preview``, ``data`` (the recovered bytes, or the raw
        frame if it could not be unpacked), ``filename`` and ``info``.
        An empty list means no password matched.

    Raises:
        ValueError: If no carrier handles the file's extension.
    """
    # Imported here so the rest of this module works without the carriers
    # package and payload.py's cryptography dependency.
    import carriers
    try:
        import payload as _payload
    except ImportError:  # pragma: no cover - payload.py ships with the repo
        _payload = None

    if isinstance(source, (str, os.PathLike)):
        carrier = carriers.carrier_for_path(source).load(source)
    else:
        carrier = source

    hits: list[dict[str, Any]] = []
    original_bps = carrier.bits_per_sample
    try:
        for password in passwords:
            password = str(password).rstrip("\r\n")
            if not password:
                continue
            for bps in bits_per_sample:
                carrier.bits_per_sample = bps
                hit = _try_scatter_key(carrier, password, _payload)
                if hit is not None:
                    hits.append(hit)
    finally:
        # Leave a caller-supplied carrier as it was given.
        carrier.bits_per_sample = original_bps
    return hits


def _try_scatter_key(carrier: Any, password: str, payload_module: Any) -> dict[str, Any] | None:
    """Test one password at the carrier's current bits_per_sample.

    Args:
        carrier: Loaded carrier (see dictionary_attack()).
        password: The scatter key / decryption password to try.
        payload_module: The imported payload module, or None.

    Returns:
        A hit dict (see dictionary_attack()) or None when the header or CRC
        does not match.
    """
    header_bits = _STGA_HEADER_LEN * 8
    capacity = carrier.capacity_bits()
    if capacity < header_bits:
        return None
    header = np.packbits(carrier.extract(header_bits, key=password)).tobytes()
    magic, version, flags, body_len, crc = _STGA_HEADER.unpack(header)
    if magic != _STGA_MAGIC or version != _STGA_VERSION or flags & ~_STGA_FLAGS_ALL:
        return None
    total_bits = (_STGA_HEADER_LEN + body_len) * 8
    if total_bits > capacity:
        return None
    frame = np.packbits(carrier.extract(total_bits, key=password)).tobytes()
    if zlib.crc32(frame[_STGA_HEADER_LEN:]) & 0xFFFFFFFF != crc:
        return None
    info: dict[str, Any] = {
        "flags": flags,
        "flag_names": [name for bit, name in _STGA_FLAGS.items() if flags & bit],
        "body_len": body_len,
        "frame_len": len(frame),
        "crc_ok": True,
        "encrypted": bool(flags & 0x01),
    }
    hit: dict[str, Any] = {
        "password": password, "bits_per_sample": carrier.bits_per_sample,
        "kind": "stga", "score": 0.9, "data": frame, "filename": None, "info": info,
        "preview": f"STGA frame, {body_len}-byte body (could not unpack)",
    }
    if payload_module is not None:
        try:
            decoded = payload_module.unpack(frame, password=password)
        except Exception as exc:  # damaged body or cipher mismatch
            info["unpack_error"] = str(exc)
        else:
            text = _text_prefix(decoded.data)
            if decoded.filename:
                preview = f"file {decoded.filename!r}, {len(decoded.data)} bytes"
            elif text is not None:
                preview = _safe_preview(text)
            else:
                preview = f"{len(decoded.data)} bytes: {_hex_preview(decoded.data)}"
            info["unpacked"] = True
            hit.update(score=1.0, data=decoded.data,
                       filename=decoded.filename, preview=preview)
    return hit


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _unique_path(path: Path) -> Path:
    """Return path, or path with _2, _3, ... before the suffix if it exists."""
    if not path.exists():
        return path
    for n in range(2, 10000):
        alt = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not alt.exists():
            return alt
    raise FileExistsError(f"too many files named like {path}")


def _safe_name(name: str) -> str:
    """Reduce an untrusted file name to a safe bare name."""
    base = re.split(r"[\\/]", name)[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return base[:100] or "payload"


def save_candidate(candidate: dict[str, Any], out_dir: str | os.PathLike) -> str:
    """Write a candidate's recovered payload to a file.

    * "text": the text, re-encoded as UTF-8, in a ``.txt`` file.
    * "file:<ext>": the recovered bytes with that extension.
    * "stga": if payload.py is importable and the frame is not encrypted,
      the unpacked data (under its stored file name when it has one, else
      ``.txt`` for printable UTF-8 and ``.bin`` otherwise). An encrypted or
      undecodable frame is written raw as ``.stga`` so it can be unpacked
      later with a password. A dictionary_attack() hit is already unpacked
      and is written the same way.
    * "binary": the raw bytes as ``.bin``.

    Existing files are never overwritten; a numeric suffix is added. A file
    name stored inside a payload is untrusted and is reduced to a safe bare
    name first.

    Args:
        candidate: One entry from bruteforce() or dictionary_attack() (or a
            score_candidate() result with a "strategy" key added).
        out_dir: Folder to write into; created if missing.

    Returns:
        The path of the written file, as a string.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strategy = candidate.get("strategy")
    if isinstance(strategy, Strategy):
        stem = f"bruteforce_{strategy.slug}"
    elif isinstance(strategy, dict):
        stem = f"bruteforce_{Strategy(**strategy).slug}"
    else:
        stem = "bruteforce_candidate"
    kind = candidate.get("kind", "binary")
    data = bytes(candidate.get("data", b""))
    info = candidate.get("info", {}) or {}

    if kind == "text":
        text = data.decode(info.get("encoding", "utf-8"), errors="replace")
        path = _unique_path(out / f"{stem}.txt")
        path.write_text(text, encoding="utf-8")
        return str(path)
    if kind.startswith("file:"):
        ext = re.sub(r"[^a-z0-9]", "", kind[5:].lower()) or "bin"
        path = _unique_path(out / f"{stem}.{ext}")
        path.write_bytes(data)
        return str(path)
    if kind == "stga":
        content: bytes | None
        filename: str | None
        if info.get("unpacked"):
            # A dictionary_attack() hit: data is already the unpacked payload.
            content, filename = data, candidate.get("filename")
            if "bits_per_sample" in candidate:
                stem = f"wordlist_{candidate['bits_per_sample']}bit"
        else:
            try:
                # Imported here so the scanner works without payload.py's
                # dependencies; any failure (encrypted, damaged, missing
                # module) falls back to writing the raw frame.
                import payload as _payload
                decoded = _payload.unpack(data)
                content, filename = decoded.data, decoded.filename
            except Exception:
                content, filename = None, None
        if content is None:
            path = _unique_path(out / f"{stem}.stga")
            path.write_bytes(data)
        elif filename:
            path = _unique_path(out / f"{stem}_{_safe_name(filename)}")
            path.write_bytes(content)
        else:
            is_text = _text_prefix(content) is not None
            path = _unique_path(out / f"{stem}.{'txt' if is_text else 'bin'}")
            path.write_bytes(content)
        return str(path)
    path = _unique_path(out / f"{stem}.bin")
    path.write_bytes(data)
    return str(path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see ``python bruteforce.py --help``."""
    parser = argparse.ArgumentParser(
        description="Try many LSB layouts on an image and rank the recovered data.")
    parser.add_argument("image", help="image to scan (any format Pillow reads)")
    parser.add_argument("--top", type=int, default=10, help="candidates to show")
    parser.add_argument("--max-bytes", type=int, default=4096,
                        help="bytes read per strategy before expansion")
    parser.add_argument("--max-bit", type=int, default=1,
                        help="highest bit plane to try (0 = LSB only)")
    parser.add_argument("--save", metavar="DIR",
                        help="write the best candidate's payload into DIR")
    parser.add_argument("--wordlist", metavar="FILE",
                        help="also try each line of FILE as a scatter/decryption "
                             "password (see dictionary_attack)")
    parser.add_argument("--json", action="store_true",
                        help="print JSON instead of a table")
    args = parser.parse_args(argv)

    results = bruteforce(args.image, top_k=args.top, max_bytes=args.max_bytes,
                         max_bit=args.max_bit)
    if args.json:
        print(json.dumps([{k: (v.to_dict() if k == "strategy" else v)
                           for k, v in r.items() if k != "data"} for r in results],
                         indent=2))
    else:
        for r in results:
            print(f"{r['score']:.3f}  {r['label']:<26} {r['kind']:<10} {r['preview']}")
    if args.save and results:
        print(f"saved: {save_candidate(results[0], args.save)}")

    if args.wordlist:
        with open(args.wordlist, encoding="utf-8", errors="replace") as fh:
            hits = dictionary_attack(args.image, fh)
        if not hits:
            print("wordlist: no password matched")
        for hit in hits:
            print(f"wordlist: password {hit['password']!r} "
                  f"({hit['bits_per_sample']} bit/sample): {hit['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
