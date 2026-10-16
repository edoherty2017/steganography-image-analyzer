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
   reads it. Unlike the original decoder it searches for the delimiter only
   at byte boundaries, so a message whose last character ends in zero bits
   ("d", " ", "@", ".") is no longer cut short; it decodes the bytes as
   UTF-8 (falling back to Latin-1 for the original single-byte encoding of
   code points 128-255); and it is vectorized with NumPy instead of building
   a bit string pixel by pixel.

How it fits with the other modules:
    * ``encoder.py``: hides a message (the inverse of this module).
    * ``stego.py``: the general hide/reveal API (passwords, files, WAV).
    * ``main.py``: argparse CLI; ``encode`` and ``decode`` subcommands call
      ``encoder.encode`` and ``decoder.decode``.
    * ``gui.py``: Tkinter app; its "Decode Image" button calls ``decode``.
    * ``bruteforce.py``: tries many other bit layouts when the standard ones
      find nothing.
    * ``detectors/``, ``steganalysis.py`` and ``pairwise_stats.py``: detect
      embedding statistically rather than read it.

This module has no ``__main__`` block. Use it via the CLI::

    python main.py decode -i stego.png

or from Python with ``from decoder import decode``.
"""
from __future__ import annotations

import os

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
        ``None`` if neither a frame nor a legacy delimiter is found. A
        legacy result does not prove that a message was hidden: any 8 zero
        bytes in the LSB stream, such as a flat black region, look like the
        delimiter.

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
            # No frame header in any read order: not an STGA image. Damaged
            # frames and wrong passwords are other PayloadErrors and propagate.
            if not issubclass(carrier_cls, ImageLSBCarrier):
                return None  # the legacy format only ever existed for images
        else:
            return found.text()
    return decode_legacy(image_path)


def decode_legacy(image_path: str | os.PathLike) -> str | None:
    """Read a message in the original delimiter format.

    The image is converted to RGB exactly as the original decoder did (alpha
    dropped), the LSB of every R, G and B value is read in row-major order,
    and the stream is packed into bytes. The message is the bytes before the
    first 8 zero bytes found at a byte boundary.

    Known limits of the format itself (not fixable when reading): NUL
    characters at the end of a legacy message merge with the delimiter and
    are lost, and 8 or more NULs in a row inside the text end the message
    there. ``encoder.encode()`` writes STGA frames, which have neither
    problem.

    Args:
        image_path: Path to the image.

    Returns:
        The text decoded as UTF-8, or as Latin-1 if it is not valid UTF-8
        (the original encoder stored code points 128-255 as single bytes);
        ``""`` if the stream starts with the delimiter; ``None`` if no
        delimiter is found.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return legacy_text_from_pixels(rgb)


def legacy_text_from_pixels(rgb: np.ndarray) -> str | None:
    """Decode the legacy delimiter format from an RGB pixel array.

    Args:
        rgb: ``(H, W, 3)`` uint8 array (row-major pixels, R, G, B).

    Returns:
        The message, ``""`` or ``None``, as described in decode_legacy().
    """
    lsb = np.asarray(rgb, dtype=np.uint8).reshape(-1) & 1
    # Keep whole bytes only: np.packbits would pad a partial last byte with
    # zero bits, which could fake part of a delimiter.
    whole = lsb.size - lsb.size % 8
    stream = np.packbits(lsb[:whole]).tobytes()
    # bytes.find() works on whole bytes, so every match starts on a byte
    # boundary of the bit stream, where the original encoder put it.
    end = stream.find(LEGACY_DELIMITER_BYTES)
    if end == -1:
        return None
    raw = stream[:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Every byte sequence is valid Latin-1, so this cannot fail.
        return raw.decode("latin-1")
