# decoder.py
"""Recover a text message hidden in an image by LSB steganography.

This is the read side of ``encoder.py``. ``decode()`` keeps the signature
that gui.py and main.py call, and understands two formats:

1. STGA frames (current). ``encoder.encode()`` and ``stego.hide()`` write a
   self-describing frame (payload.py): a 14-byte header with the body length
   and a CRC-32, then the body. ``decode()`` reads it with
   ``stego.reveal()``, which reads exactly the header and then exactly the
   rest of the frame, so nothing has to be guessed about where the message
   ends and a damaged image is detected by the CRC. It tries 1 and 2 bits
   per sample, and with a password also the password-scattered order.
2. The legacy delimiter format (images made by the original encoder and by
   make_all.py): the message bytes followed by 64 zero bits, written into
   the LSB of R, G, B of each pixel in row-major order. ``decode_legacy()``
   reads it. The format has no header and no checksum, so the reader has
   to judge what it finds:

   * Where the message ends. The delimiter is looked for at byte
     boundaries first, so a message whose last character ends in zero bits
     ("d", " ", "@", ".") is not cut short, as it was by the original
     decoder. Only when that finds no message is the first run of 64 zero
     bits at any bit offset used, as the original decoder did; some
     original images need it (make_all.py used to cut its payload at the
     bit level, which puts the delimiter in the middle of a byte).
   * Whether it is a message at all. Any 64 zero bits in the LSB stream
     look like the delimiter, and a flat region (sky, a black border, an
     even-valued block) is full of them. So the bytes before the delimiter
     are accepted only when they read as text (a heuristic, see
     ``find_legacy_message()``). Without that check a clean cover, or a
     password-scattered frame read with the wrong password, "decoded" to
     thousands of bytes of LSB noise.
   * The character encoding. UTF-8 first (a multi-byte character cut off at
     the end of the message is dropped), then Latin-1 for the original
     encoder's single-byte code points 128-255.

   The legacy reading is vectorized with NumPy instead of building a bit
   string pixel by pixel.

How it fits with the other modules:
    * ``encoder.py``: hides a message (the inverse of this module).
    * ``stego.py``: the general hide/reveal API (passwords, files, WAV).
    * ``main.py``: argparse CLI; ``encode`` calls ``stego.hide`` and
      ``decode`` calls ``stego.reveal``, then falls back to the legacy
      format read with this module.
    * ``gui.py``: Tkinter app; its Decode tab does the same.
    * ``report.py``: ``analyze_image`` calls ``decode_legacy()`` when the
      brute force or the signature scan points to a legacy message.
    * ``bruteforce.py``: tries many other bit layouts when the standard ones
      find nothing. Its legacy-format text rule is the one used here.
    * ``detectors/``, ``steganalysis.py`` and ``pairwise_stats.py``: detect
      embedding statistically rather than read it.

This module has no ``__main__`` block. Use it via the CLI::

    python main.py decode -i stego.png

or from Python with ``from decoder import decode``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

import carriers
import stego
from carriers.image_lsb import ImageLSBCarrier

# Legacy end-of-message marker: 64 zero bits (eight 0x00 bytes). Must match
# DELIMITER in encoder.py and DELIM_BITS = 64 in make_all.py.
DELIMITER = "00000000" * 8

# The same marker as bytes, for searching the packed LSB stream.
LEGACY_DELIMITER_BYTES = b"\x00" * (len(DELIMITER) // 8)

# Every run of 64 zero bits, whatever its bit offset, contains at least 7
# whole zero bytes of the packed stream; searching for them finds the
# candidates for a delimiter that is not on a byte boundary.
_SEVEN_ZERO_BYTES = b"\x00" * 7

# Characters that cannot be part of a legacy text message: C0 controls other
# than tab, LF and CR, DEL, the C1 controls, U+FFFD (what invalid UTF-8
# decodes to) and the non-characters U+FFFE and U+FFFF. The same class as
# bruteforce._NON_TEXT, so decode and the brute force agree on what text is.
_NOT_TEXT = re.compile(r"[^\t\n\r\x20-\x7e\xa0-￼\U00010000-\U0010ffff]")

# A reading as Latin-1 (the original encoder's single bytes for code points
# 128-255) is accepted only for mostly-ASCII text, as in bruteforce.py:
# accented letters in a Western-language message. LSB noise read as Latin-1
# is mostly symbols and accented letters, and a flat white region gives
# "ÿÿÿÿ", so this share rejects both.
_LATIN1_MIN_ASCII = 0.6


@dataclass(frozen=True)
class LegacyMessage:
    """A message read in the legacy delimiter format.

    Attributes:
        text: The decoded message ("" when the stream starts with the
            delimiter).
        raw: The whole bytes of the LSB stream before the delimiter. When a
            multi-byte UTF-8 character was cut off at the end, its bytes are
            in ``raw`` but not in ``text``.
        encoding: "utf-8" or "latin-1", how ``raw`` was decoded.
        delimiter_bit: Bit offset in the RGB LSB stream where the 64 zero
            bits start. For the bit-offset fallback this is the start of
            the zero run, which also takes in any trailing zero bits of the
            message's last bits, so it can be a few bits before where the
            encoder wrote the delimiter.
        byte_aligned: True when the delimiter starts on a byte boundary
            (``delimiter_bit == 8 * len(raw)``), as the original encoder
            wrote it. False for the bit-offset fallback, where the 1 to 7
            bits between ``raw`` and the zero run (a character cut at the
            bit level) are dropped.
    """

    text: str
    raw: bytes
    encoding: str
    delimiter_bit: int
    byte_aligned: bool


def decode(image_path: str | os.PathLike, password: str | None = None) -> str | None:
    """Extract a hidden text message from a stego image.

    Tries an STGA frame first (``stego.reveal()`` with 1 or 2 bits per
    sample). If no frame header is found in an image, falls back to the
    legacy delimiter format (``decode_legacy()``).

    Args:
        image_path: Path to the stego file. It must be lossless (PNG, BMP,
            TIFF, or WAV for audio frames); re-saving as JPEG destroys the
            LSBs.
        password: Password for frames hidden with one. Optional; the
            original signature ``decode(image_path)`` still works.

    Returns:
        The message as a ``str``. For an STGA frame this is the body decoded
        as UTF-8 (invalid bytes become U+FFFD, so use ``stego.reveal()`` for
        binary file payloads); an empty frame gives ``""``. For the legacy
        format it is the text before the delimiter, which is ``""`` when the
        LSB stream starts with 64 zero bits (e.g. an all-black image).
        ``None`` if neither a frame nor a legacy message is found. That
        includes a frame scattered with a password that is read with a
        wrong or missing one: the bits come out in the wrong order, which
        cannot be told apart from no frame at all. The legacy reading is a
        heuristic (see ``find_legacy_message()``): it rejects the noise in
        front of a flat region, but a few printable noise bytes right at the
        start of the LSB stream can still pass as a very short message.

    Raises:
        payload.WrongPasswordError: If the image holds an encrypted frame
            and ``password`` is missing or wrong. This is raised rather than
            returning None so the caller can ask for a password; the legacy
            fallback is not tried, since the data is known to be a frame.
        payload.PayloadError: If a frame header was found but the frame is
            damaged (CRC mismatch, or longer than the image after cropping).
        FileNotFoundError: If ``image_path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    try:
        carrier_cls = carriers.carrier_for_path(image_path)
    except ValueError:
        # Unregistered extension: only Pillow (the legacy reader) can try.
        carrier_cls = None

    if carrier_cls is not None:
        try:
            found = stego.reveal(image_path, password=password, bits=None)
        except stego.NoPayloadError:
            # No frame header in any read order: not an STGA image, or a
            # scattered one read with the wrong password. Damaged frames and
            # wrong passwords for unscattered encrypted frames are other
            # PayloadErrors and propagate.
            if not issubclass(carrier_cls, ImageLSBCarrier):
                return None  # the legacy format only ever existed for images
        else:
            return found.text()
    return decode_legacy(image_path)


def decode_legacy(image_path: str | os.PathLike) -> str | None:
    """Read a message in the original delimiter format.

    The image is converted to RGB exactly as the original decoder did (alpha
    dropped), the LSB of every R, G and B value is read in row-major order,
    and the message is the bytes before the delimiter (64 zero bits),
    provided they read as text. See ``find_legacy_message()`` for the
    details and its heuristics.

    Known limits of the format itself (not fixable when reading): NUL
    characters at the end of a legacy message merge with the delimiter and
    are lost; 8 or more NULs in a row inside the text end the message
    there; and the original encoder wrote code points above U+00FF with
    more than 8 bits, which shifts every later byte, so such a message
    reads as noise and gives ``None``. ``encoder.encode()`` writes STGA
    frames, which have none of these problems.

    Args:
        image_path: Path to the image.

    Returns:
        The text (UTF-8, or Latin-1 for the original encoder's single-byte
        code points 128-255); ``""`` if the stream starts with the
        delimiter; ``None`` if no delimiter is found or the bytes before it
        are not text.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    found = find_legacy_message(rgb)
    return None if found is None else found.text


def legacy_text_from_pixels(rgb: np.ndarray) -> str | None:
    """The raw legacy reading of an RGB pixel array, WITHOUT the text check.

    The bytes before the first 8 zero bytes on a byte boundary, decoded as
    UTF-8, or as Latin-1 when they are not valid UTF-8. This is *something*
    for any image with a flat even-valued region, so it is not evidence of
    a message: use ``find_legacy_message()`` (what ``decode_legacy()``
    uses) for that. It is kept, unchanged, for callers that report why a
    reading was rejected (main.py scores these bytes itself).

    Args:
        rgb: ``(H, W, 3)`` uint8 array (row-major pixels, R, G, B).

    Returns:
        The decoded bytes; ``""`` when the stream starts with the delimiter;
        ``None`` when there is no byte-aligned delimiter.
    """
    stream = _lsb_stream(rgb)
    end = stream.find(LEGACY_DELIMITER_BYTES)
    if end == -1:
        return None
    raw = stream[:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Every byte sequence is valid Latin-1, so this cannot fail.
        return raw.decode("latin-1")


def find_legacy_message(rgb: np.ndarray) -> LegacyMessage | None:
    """Find a legacy delimiter-format message in an RGB pixel array.

    The LSB of every R, G and B value, in row-major order, is packed into
    bytes most significant bit first. Two places for the end of the message
    are tried, and the first whose bytes read as text wins:

    1. The first 8 zero bytes on a byte boundary, where the original
       encoder put the delimiter. Matching only there keeps the last
       character of a message that ends in zero bits ("d" is 01100100).
    2. The first run of 64 zero bits at any bit offset, as the original
       decoder searched, taking the whole bytes before it. This finds a
       delimiter that the original make_all.py put in the middle of a byte
       by cutting its payload at the bit level.

    "Reads as text" is a heuristic, the same rule bruteforce.py applies to
    the legacy format: every character is printable, tab, LF or CR after a
    strict UTF-8 decode; or, when the bytes are not valid UTF-8, every byte
    is printable Latin-1 and at least 60% of them are ASCII. A multi-byte
    UTF-8 character cut off at the very end (make_all.py cut long messages
    at the capacity) is dropped first, when the bytes before it contain
    other multi-byte characters or more than one byte of it is left; a
    single trailing byte after pure ASCII is read as Latin-1, since that is
    what the original encoder wrote for "café".

    The check is what separates a message from a clean cover: any flat
    region gives 64 zero bits, and the LSB noise in front of it fails the
    check after a few bytes (a NUL, a C1 control or an invalid UTF-8
    sequence appears almost at once). It cannot reject a few printable
    noise bytes right at the start of the stream, because a one-character
    message such as "d" looks exactly the same.

    Args:
        rgb: ``(H, W, 3)`` uint8 array (row-major pixels, R, G, B).

    Returns:
        A LegacyMessage, or None when no delimiter is found or the bytes
        before it do not read as text.
    """
    stream = _lsb_stream(rgb)

    end = stream.find(LEGACY_DELIMITER_BYTES)
    if end != -1:
        found = _legacy_message(stream[:end], 8 * end)
        if found is not None:
            return found

    start = _first_zero_run(stream)
    # A run that starts within the byte-aligned match gives the same whole
    # bytes, which were just rejected. start // 8 > end cannot happen: the
    # aligned match is itself a run of 64 zero bits.
    if start is not None and (end == -1 or start // 8 < end):
        found = _legacy_message(stream[:start // 8], start)
        if found is not None:
            return found
    return None


def _lsb_stream(rgb: np.ndarray) -> bytes:
    """Pack the R, G, B LSBs of a pixel array into bytes, MSB first.

    A partial last byte is padded with one bits, not the zero bits
    np.packbits would add, so the padding can never fake part of a
    delimiter. The padded byte is never zero, so it cannot be part of a
    byte-aligned match either.

    Args:
        rgb: ``(H, W, 3)`` uint8 array.

    Returns:
        The packed LSB stream.
    """
    lsb = np.asarray(rgb, dtype=np.uint8).reshape(-1) & 1
    pad = -lsb.size % 8
    if pad:
        lsb = np.concatenate([lsb, np.ones(pad, dtype=np.uint8)])
    return np.packbits(lsb).tobytes()


def _first_zero_run(stream: bytes) -> int | None:
    """Bit offset of the first run of at least 64 zero bits, at any offset.

    Every such run contains 7 whole zero bytes, so ``bytes.find`` (fast, in
    C) proposes the candidates and only the bits around each one are
    checked: the trailing zero bits of the byte before it and the leading
    zero bits of the byte after it. This keeps memory at the size of the
    packed stream instead of one integer per bit.

    Args:
        stream: The packed LSB stream from _lsb_stream().

    Returns:
        The bit offset where the run starts, or None if there is none.
    """
    pos = 0
    while True:
        i = stream.find(_SEVEN_ZERO_BYTES, pos)
        if i == -1:
            return None
        # stream[i - 1] is not zero here: otherwise find() would have
        # returned i - 1 (or, after a skip, it is the non-zero byte that
        # ended the previous candidate).
        left = 0
        if i > 0:
            prev = stream[i - 1]
            left = (prev & -prev).bit_length() - 1   # trailing zero bits
        right = 0
        if i + 7 < len(stream):
            right = 8 - stream[i + 7].bit_length()   # leading zero bits (8 if zero)
        if left + 56 + right >= len(DELIMITER):
            return 8 * i - left
        # The run is too short, so stream[i + 7] is not zero and no 7 zero
        # bytes can start at i + 1 ... i + 7.
        pos = i + 8


def _legacy_message(raw: bytes, delimiter_bit: int) -> LegacyMessage | None:
    """Decode the bytes before a delimiter if they read as text (heuristic).

    See find_legacy_message() for the rule.

    Args:
        raw: Whole bytes before the delimiter.
        delimiter_bit: Bit offset where the delimiter starts.

    Returns:
        A LegacyMessage, or None when ``raw`` is not text.
    """
    aligned = delimiter_bit == 8 * len(raw)
    if not raw:
        return LegacyMessage("", raw, "utf-8", delimiter_bit, aligned)
    text: str | None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        text = None
        head = raw[:exc.start]   # valid UTF-8: exc is the first error
        cut_bytes = len(raw) - exc.start
        if exc.reason == "unexpected end of data" and (cut_bytes > 1 or not head.isascii()):
            # A multi-byte character cut off at the end of the message.
            text = head.decode("utf-8")
    if text is not None:
        if _NOT_TEXT.search(text) is not None:
            return None
        return LegacyMessage(text, raw, "utf-8", delimiter_bit, aligned)
    # Not UTF-8: the original encoder's one byte per code point 128-255.
    text = raw.decode("latin-1")
    if _NOT_TEXT.search(text) is not None:
        return None
    ascii_chars = sum(1 for b in raw if b < 0x80)
    if ascii_chars < _LATIN1_MIN_ASCII * len(raw):
        return None
    return LegacyMessage(text, raw, "latin-1", delimiter_bit, aligned)
