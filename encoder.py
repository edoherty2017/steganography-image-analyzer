# encoder.py
"""LSB steganography encoder: hide a text message in the pixels of an image.

``encode()`` keeps the signature that gui.py and main.py have always called,
but it is now a thin wrapper around ``stego.hide()``:

1. The message is encoded as UTF-8 and wrapped in an STGA frame by
   payload.py: a 14-byte header (magic ``STGA``, version, flags, body length
   and CRC-32 of the body) followed by the message bytes.
2. The frame's bits (MSB first per byte) replace the least significant bit
   (LSB) of the R, G and B channels, pixel by pixel in row-major order (left
   to right, top to bottom), so each pixel carries 3 bits. This is the same
   layout the original encoder used; only the bits written are different.
   Changing an LSB moves a channel value by at most 1, too little to see.
3. The result is saved losslessly (PNG, BMP or TIFF, from the extension).

What changed from the original encoder, and why:
    * The frame header replaces the 64-zero-bit end delimiter. The decoder
      no longer has to guess where the message ends, so text that ends in a
      character with trailing zero bits ("d", " ", "@", ".") or that
      contains NUL characters comes back exactly. The CRC also detects a
      damaged stego image.
    * Text is stored as UTF-8. The original wrote ``format(ord(c), '08b')``,
      which is wider than 8 bits for code points above 255, so "€" or emoji
      came back garbled.
    * Errors raise instead of printing and returning None, so callers can
      tell failure from success (see Raises in encode()).
    * A lossy output extension (.jpg, .webp, .gif, ...) raises
      ``LossyFormatError`` instead of silently writing PNG data under that
      name.
    * An alpha channel is kept unchanged instead of being dropped.
    * Embedding is vectorized with NumPy instead of a per-pixel Python loop.

Older stego images written in the delimiter format are still readable:
``decoder.decode()`` falls back to it when no STGA frame is found.

How it fits with the other modules:
    encoder.py         This module. ``encode()`` writes the stego image.
    decoder.py         ``decode()`` reads the frame back (and still reads
                       the legacy delimiter format).
    stego.py           The general API underneath: passwords (AES-256-GCM),
                       compression, file payloads, key scatter, 2 bits per
                       sample, and WAV audio as well as images.
    payload.py         The STGA frame format.
    carriers/          Plug-in cover formats (image and WAV LSB carriers).
    main.py            Command-line front end with ``encode`` and ``decode``
                       subcommands that call this module and decoder.py.
    gui.py             Tkinter app. Calls ``encoder.encode()``, then
                       steganalysis.py to compare the original and encoded
                       images; its Decode button calls ``decoder.decode()``.
    steganalysis.py    Plots of cover vs. stego images: the standard
                       histogram and the even/odd pair differences.
    pairwise_stats.py  Numeric even/odd histogram-pair statistics.
    make_all.py        Experiment runner with its own NumPy embedder (UTF-8
                       bytes plus the legacy 64-zero-bit delimiter).

This module has no command-line entry point. Use it through main.py::

    python main.py encode -i cover.png -m "secret" -o stego.png
    python main.py decode -i stego.png

or import it::

    from encoder import encode
    encode("cover.png", "secret", "stego.png")
"""
from __future__ import annotations

import os

import stego

# Legacy end-of-message marker: 64 zero bits (8 zero bytes), as a string of
# '0' characters. encode() no longer writes it (the STGA frame header records
# the length instead); it is kept for code that builds or reads images in the
# original format, such as decoder.decode_legacy() and make_all.py.
DELIMITER = "00000000" * 8


def message_to_binary(message: str) -> str:
    """Convert a text message to the legacy delimiter-format bit string.

    Legacy helper kept for backward compatibility; encode() does not use it.
    The message is encoded as UTF-8 and each byte becomes 8 bits (MSB first),
    followed by DELIMITER. For ASCII text this is bit-for-bit what the
    original encoder wrote. For other characters it writes UTF-8 bytes,
    like make_all.py's embedder, instead of the original
    ``format(ord(char), '08b')``, which emitted more than 8 bits for code
    points above 255 and so could not be decoded. decoder.decode_legacy()
    reads both UTF-8 and the original single-byte (Latin-1) form.

    Args:
        message: The text to convert.

    Returns:
        str: A string of '0' and '1' characters of length
        ``8 * len(message.encode("utf-8")) + 64``.

    Raises:
        TypeError: If ``message`` is not a str.
    """
    if not isinstance(message, str):
        raise TypeError(f"message must be a str, not {type(message).__name__}")
    bits = "".join(format(byte, "08b") for byte in message.encode("utf-8"))
    return bits + DELIMITER


def encode(image_path: str | os.PathLike, message: str,
           output_path: str | os.PathLike) -> str | os.PathLike:
    """Hide ``message`` in the LSBs of an image and save the stego image.

    Uses ``stego.hide()`` with no password, 1 bit per sample and no key
    scatter, so the STGA frame sits at the start of the image in the
    row-major R, G, B order described in the module docstring. Pixels after
    the frame are left unchanged.

    Args:
        image_path: Path to the cover image, in any format Pillow can open
            (a JPEG cover is fine, since it is only read). Palette and
            grayscale images are converted to RGB; an alpha channel is kept
            unchanged and never carries data.
        message: Text to hide. It is stored as UTF-8, so any Unicode text
            round-trips. At most ``(width * height * 3) // 8 - 14`` UTF-8
            bytes fit (see ``stego.capacity_bytes()``).
        output_path: Where to write the stego image. The extension picks a
            lossless format: .png, .bmp, .tif or .tiff. Missing parent
            folders are created and an existing file is overwritten.

    Returns:
        ``output_path``, unchanged, once the file has been written.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        PIL.UnidentifiedImageError: If ``image_path`` is not a readable
            image.
        carriers.LossyFormatError: If ``output_path`` has a lossy or
            unsupported extension such as .jpg (a ValueError).
        carriers.CapacityError: If the message does not fit in the image (a
            ValueError).
        ValueError: If the cover's extension is not a supported image or
            audio format.
        TypeError: If ``message`` is not a str (or bytes).
    """
    stego.hide(image_path, output_path, message, bits=1, key_scatter=False)
    return output_path
