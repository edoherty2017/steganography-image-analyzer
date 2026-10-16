"""Known-signature scan of the LSB stream (headers, file magic, text).

Unlike the statistical detectors, this one reads the hidden data directly:
it rebuilds the first few kilobytes of the LSB stream in the orders that
simple LSB tools use and looks for things that do not occur by chance:

* This project's payload header: ``b"STGA"`` magic, version, flags and a
  big-endian body length and CRC-32 (the 14-byte frame header written by
  ``payload.pack``; parsed here independently). Version 2, the one
  ``payload.pack`` writes, computes the CRC-32 over the 10 header bytes
  before the CRC field followed by the body; version 1 (development frames)
  computed it over the body alone. A header whose version, flags and length
  are plausible is definitive (score 1.0); when the whole body fits in the
  image its CRC-32 is checked as well.
* Magic numbers of common file formats (PNG, JPEG, GIF, ZIP/DOCX/XLSX, PDF,
  GZIP, BMP, 7z, RAR, XZ, ELF, RIFF/WAV), i.e. a file hidden without any
  wrapper.
* Readable text: the legacy encoder of this project writes the message as
  plain bytes from the first pixel on, followed by 8 zero bytes. A long,
  varied run of printable ASCII (or a short one of 3+ bytes followed by
  that zero marker) at the start of the stream is reported with a preview.
  One or two printable bytes before the marker are ignored: dark images
  produce them by chance.

Orders scanned (row-major pixels; bytes assembled MSB-first unless noted):
``RGB`` (this project's order), ``BGR``, each channel alone, ``RGB/2bit``
(two low bits per sample, bit 1 first, as the 2-bit carrier writes them)
and ``RGB/lsb-first`` (bytes assembled LSB-first).

Payloads scattered with a key (``stego.hide`` with a password) are not in
natural order and are invisible to this scan by design; the statistical
detectors still see them.

Scores (heuristic): valid STGA header 1.0 (0.9 if its CRC fails); file
magic at offset 0 0.9, elsewhere 0.75 for signatures of 6+ bytes and 0.5
for 4-5 bytes (3-byte signatures only count at offset 0); text at the
start 0.6 to 0.85 depending on length and the zero marker (0.45 for a
3-byte run followed by the marker); long text deeper in the stream 0.5.
"""
from __future__ import annotations

import string
import zlib
from typing import Any, Iterable

import numpy as np

from . import register
from ._common import as_channels, channel_labels, make_result

NAME = "signatures"

SCAN_BYTES = 4096

# Local copy of the STGA frame layout (payload.py is the authority; this
# module re-parses it so the detector has no dependency on the encoder).
STGA_MAGIC = b"STGA"
STGA_VERSIONS = (1, 2)  # mirrors payload.VERSION history (see module docstring)
STGA_HEADER_LEN = 14
STGA_PREFIX_LEN = 10    # header bytes before the CRC field (covered by the v2 CRC)
STGA_FLAGS = {0x01: "encrypted", 0x02: "file", 0x04: "compressed"}

# (label, magic bytes, file extension)
FILE_SIGNATURES: tuple[tuple[str, bytes, str], ...] = (
    ("PNG image", b"\x89PNG\r\n\x1a\n", "png"),
    ("JPEG image", b"\xff\xd8\xff", "jpg"),
    ("GIF image", b"GIF87a", "gif"),
    ("GIF image", b"GIF89a", "gif"),
    ("ZIP archive (also DOCX/XLSX/JAR)", b"PK\x03\x04", "zip"),
    ("PDF document", b"%PDF-", "pdf"),
    ("GZIP data", b"\x1f\x8b\x08", "gz"),
    ("7z archive", b"7z\xbc\xaf\x27\x1c", "7z"),
    ("RAR archive", b"Rar!\x1a\x07", "rar"),
    ("XZ data", b"\xfd7zXZ\x00", "xz"),
    ("ELF executable", b"\x7fELF", "elf"),
    ("RIFF container", b"RIFF", "riff"),
)
RIFF_KINDS = {b"WAVE": ("WAV audio", "wav"), b"AVI ": ("AVI video", "avi"), b"WEBP": ("WebP image", "webp")}

# Shortest text run that counts when followed by the legacy zero marker. One
# or two printable bytes then 8 zero bytes occur by chance wherever the LSBs
# are sparse (near-black or clipped regions): a lone 0x20 (space) is a single
# set bit. In simulated sparse LSB streams (bit density 1% to 5%) a printable
# run of any length followed by the marker turns up by chance in up to about
# 1e-2 of scan orders, a varied 2-byte run in about 1e-4 and a varied 3-byte
# run in about 5e-6. Heuristic threshold.
MIN_MARKED_TEXT = 3

_PRINTABLE = frozenset(bytes(string.printable, "ascii")) - frozenset(b"\x0b\x0c")
_TEXTY = frozenset(bytes(string.ascii_letters + string.digits + " .,;:'\"!?-()\n", "ascii"))


def _orders(c: int, labels: list[str]) -> dict[str, tuple[list[int], int, str]]:
    """Scan orders: name -> (channel indices per pixel, bits per sample, bit order)."""
    joined = "".join(labels)
    out: dict[str, tuple[list[int], int, str]] = {joined: (list(range(c)), 1, "big")}
    if c == 3:
        out["".join(reversed(labels))] = [2, 1, 0], 1, "big"
    if c > 1:
        for i, lab in enumerate(labels):
            out[lab] = [i], 1, "big"
    out[joined + "/2bit"] = list(range(c)), 2, "big"
    out[joined + "/lsb-first"] = list(range(c)), 1, "little"
    return out


def lsb_bytes(a: np.ndarray, channels: list[int], bits_per_sample: int = 1,
              bitorder: str = "big", n_bytes: int | None = None) -> bytes:
    """Rebuild bytes from the low bits of an image in row-major order.

    Args:
        a: ``(H, W, C)`` uint8 array.
        channels: Channel indices to read within each pixel, in order.
        bits_per_sample: 1 (bit 0 only) or 2 (bit 1 then bit 0 of each sample).
        bitorder: ``"big"`` (first bit is the byte's MSB) or ``"little"``.
        n_bytes: Bytes to return (default: as many as the image holds).

    Returns:
        The bytes (shorter than ``n_bytes`` if the image is too small).
    """
    samples_total = a.shape[0] * a.shape[1] * len(channels)
    max_bytes = samples_total * bits_per_sample // 8
    want = max_bytes if n_bytes is None else min(n_bytes, max_bytes)
    n_samples = -(-want * 8 // bits_per_sample)
    # Only touch the rows needed for the first n_samples samples.
    per_row = a.shape[1] * len(channels)
    rows = min(a.shape[0], -(-n_samples // per_row)) if per_row else 0
    s = a[:rows][:, :, channels].reshape(-1)[:n_samples]
    if bits_per_sample == 1:
        bits = s & 1
    else:
        bits = np.stack([(s >> 1) & 1, s & 1], axis=-1).reshape(-1)
    bits = bits[: want * 8].astype(np.uint8)
    return np.packbits(bits, bitorder=bitorder).tobytes()


def parse_stga_header(header: bytes) -> dict[str, Any] | None:
    """Parse a 14-byte STGA frame header (local re-implementation).

    Returns:
        Dict with ``version``, ``flags``, ``flag_names``, ``body_len``,
        ``crc32`` and ``plausible`` (a known version and only known flag
        bits), or
        None if ``header`` is too short or does not start with the magic.
    """
    if len(header) < STGA_HEADER_LEN or header[:4] != STGA_MAGIC:
        return None
    version = header[4]
    flags = header[5]
    body_len = int.from_bytes(header[6:10], "big")
    crc = int.from_bytes(header[10:14], "big")
    names = [n for bit, n in STGA_FLAGS.items() if flags & bit]
    return {"version": version, "flags": flags, "flag_names": names, "body_len": body_len,
            "crc32": crc, "plausible": version in STGA_VERSIONS and not (flags & ~0x07)}


def _frame_crc(version: int, prefix: bytes, body: bytes) -> int:
    """CRC-32 of an STGA frame as its version defines it.

    Args:
        version: Header version byte (1 or 2).
        prefix: The STGA_PREFIX_LEN header bytes before the CRC field.
        body: The frame body.

    Returns:
        The unsigned CRC-32 to compare with the header's crc32 field.
    """
    if version == 1:
        return zlib.crc32(body) & 0xFFFFFFFF
    return zlib.crc32(body, zlib.crc32(prefix)) & 0xFFFFFFFF


def _text_run(data: bytes, start: int) -> int:
    """Length of the run of printable ASCII bytes starting at ``start``."""
    i = start
    while i < len(data) and data[i] in _PRINTABLE:
        i += 1
    return i - start


def _texty(run: bytes) -> bool:
    """Heuristic: does a printable run look like natural-language text?"""
    if not run:
        return False
    frac = sum(1 for b in run if b in _TEXTY) / len(run)
    distinct = len(set(run))
    return frac >= 0.75 and distinct >= min(8, max(2, len(run) // 2))


def _longest_text_run(data: bytes) -> tuple[int, int]:
    """(offset, length) of the longest printable run in ``data``."""
    best_off, best_len, i = 0, 0, 0
    n = len(data)
    while i < n:
        if data[i] in _PRINTABLE:
            j = _text_run(data, i)
            if j > best_len:
                best_off, best_len = i, j
            i += j
        else:
            i += 1
    return best_off, best_len


def _preview(raw: bytes, limit: int = 160) -> str:
    """Printable preview of payload bytes."""
    txt = raw[:limit].decode("utf-8", errors="replace")
    return "".join(ch if ch.isprintable() or ch in "\n\t" else "." for ch in txt)


def _scan_stga(a, order, spec, data) -> list[dict[str, Any]]:
    """Find and validate STGA headers in one order's leading bytes."""
    found = []
    capacity = a.shape[0] * a.shape[1] * len(spec[0]) * spec[1] // 8
    start = 0
    while len(found) < 4:
        off = data.find(STGA_MAGIC, start)
        if off < 0:
            break
        start = off + 1
        hdr = parse_stga_header(data[off:off + STGA_HEADER_LEN])
        if hdr is None:
            continue
        fits = off + STGA_HEADER_LEN + hdr["body_len"] <= capacity
        finding = {"kind": "stga", "order": order, "offset": off, "label": "STGA payload header",
                   **hdr, "fits_in_image": fits, "crc_ok": None}
        if hdr["plausible"] and fits:
            total = off + STGA_HEADER_LEN + hdr["body_len"]
            full = data if total <= len(data) else lsb_bytes(a, spec[0], spec[1], spec[2], total)
            body = full[off + STGA_HEADER_LEN: total]
            prefix = full[off:off + STGA_PREFIX_LEN]
            finding["crc_ok"] = _frame_crc(hdr["version"], prefix, body) == hdr["crc32"]
            if not (hdr["flags"] & 0x05):  # neither encrypted nor compressed: readable
                if hdr["flags"] & 0x02 and len(body) >= 2:
                    name_len = int.from_bytes(body[:2], "big")
                    finding["filename"] = body[2:2 + name_len].decode("utf-8", errors="replace")
                    finding["preview"] = _preview(body[2 + name_len:], 80)
                else:
                    finding["preview"] = _preview(body)
            finding["score"] = 1.0 if finding["crc_ok"] else 0.9
        else:
            finding["score"] = 0.8 if off == 0 else 0.5
        found.append(finding)
    return found


def _scan_files(order, data) -> list[dict[str, Any]]:
    """Find file magic numbers in one order's leading bytes."""
    found = []
    for label, magic, ext in FILE_SIGNATURES:
        start = 0
        while True:
            off = data.find(magic, start)
            if off < 0:
                break
            start = off + 1
            if off > 0 and len(magic) < 4:
                continue  # 3-byte magic at an arbitrary offset is too likely by chance
            lab, e = label, ext
            if magic == b"RIFF":
                kind = RIFF_KINDS.get(data[off + 8:off + 12])
                if kind is None:
                    continue
                lab, e = kind
            score = 0.9 if off == 0 else (0.75 if len(magic) >= 6 else 0.5)
            found.append({"kind": "file", "order": order, "offset": off, "label": lab,
                          "extension": e, "magic": magic.hex(), "score": score})
            break  # first occurrence per signature is enough
    # BMP: "BM", 4-byte size, 4 reserved zero bytes; only at offset 0.
    if data[:2] == b"BM" and data[6:10] == b"\0\0\0\0":
        size = int.from_bytes(data[2:6], "little")
        if 26 <= size <= 1 << 30:
            found.append({"kind": "file", "order": order, "offset": 0, "label": "BMP image",
                          "extension": "bmp", "magic": b"BM".hex(), "score": 0.9})
    return found


def _scan_text(order, data) -> list[dict[str, Any]]:
    """Find readable text at the start of (or deep inside) one order's bytes."""
    found = []
    lead = _text_run(data, 0)
    run = data[:lead]
    marker = data[lead:lead + 8] == b"\0" * 8 and lead > 0
    score = 0.0
    if lead >= 16 and _texty(run):
        score = 0.85
    elif lead >= 4 and marker and _texty(run):
        score = 0.8
    elif lead >= 8 and _texty(run):
        score = 0.6
    elif lead >= MIN_MARKED_TEXT and marker and _texty(run):
        # Only reached for a 3-byte run (4+ bytes with the marker score 0.8).
        score = 0.45
    if score:
        found.append({"kind": "text", "order": order, "offset": 0, "length": lead,
                      "label": "readable text" + (" + legacy 64-bit end marker" if marker else ""),
                      "end_marker": marker, "preview": _preview(run), "score": score})
    off, length = _longest_text_run(data)
    if off > 0 and length >= 24 and _texty(data[off:off + length]):
        found.append({"kind": "text", "order": order, "offset": off, "length": length,
                      "label": "readable text inside the stream", "end_marker": False,
                      "preview": _preview(data[off:off + length]), "score": 0.5})
    return found


@register(NAME, "Scans the first few KB of the LSB stream in common orders for this project's "
                "STGA header, common file magic numbers and readable text.")
def signatures(arr: np.ndarray,
               channel_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Scan LSB streams for known signatures.

    Args:
        arr: ``(H, W, C)`` (or ``(H, W)``) uint8 image array.
        channel_names: Labels for the channels (defaults to R, G, B).

    Returns:
        Result dict in the shared detector schema. ``details["findings"]``
        lists every hit (kind, order, offset, label, score and, where
        available, parsed header fields or a text preview);
        ``details["stga"]`` is the best STGA hit or None.
    """
    a = as_channels(arr)
    labels = channel_labels(a.shape[2], channel_names)
    findings: list[dict[str, Any]] = []
    scanned = {}
    for order, spec in _orders(a.shape[2], labels).items():
        data = lsb_bytes(a, *spec, n_bytes=SCAN_BYTES)
        scanned[order] = len(data)
        findings += _scan_stga(a, order, spec, data)
        findings += _scan_files(order, data)
        findings += _scan_text(order, data)
    findings.sort(key=lambda f: (-f["score"], f["offset"]))
    stga = next((f for f in findings if f["kind"] == "stga"), None)
    if not findings:
        return make_result(NAME, 0.0,
                           "No payload header, file signature or readable text in the LSB streams.",
                           {"findings": [], "stga": None, "scanned_bytes": scanned})
    best = findings[0]
    if best["kind"] == "stga":
        flags = ", ".join(best["flag_names"]) or "plain"
        crc = {True: "CRC OK", False: "CRC mismatch", None: "CRC not checked"}[best["crc_ok"]]
        summary = (f"STGA payload header in the {best['order']} LSB stream at byte {best['offset']}: "
                   f"{best['body_len']} byte body ({flags}; {crc}).")
    elif best["kind"] == "file":
        summary = (f"{best['label']} signature in the {best['order']} LSB stream at byte "
                   f"{best['offset']}: a file appears to be hidden in the LSBs.")
    else:
        summary = (f"Readable text in the {best['order']} LSB stream at byte {best['offset']}: "
                   f"\"{best['preview'][:60]}\"")
    return make_result(NAME, best["score"], summary,
                       {"findings": findings, "stga": stga, "scanned_bytes": scanned})
