# decoder.py
"""Recover a text message hidden in an image by LSB steganography.

This is the read side of the project's least-significant-bit (LSB) scheme.
``encoder.py`` writes one message bit into the lowest bit of each R, G and B
value (3 bits per pixel), visiting pixels row by row (y outer, x inner) and
channels in R, G, B order, then appends a 64-zero-bit delimiter.
``decode`` reads the low bits back in that same order, cuts the stream at the
first delimiter and turns each 8-bit group before it into a character.

How it fits with the other modules:
    * ``encoder.py``: hides a message (the inverse of this module).
    * ``main.py``: argparse CLI; ``encode`` and ``decode`` subcommands call
      ``encoder.encode`` and ``decoder.decode``.
    * ``gui.py``: Tkinter app; its "Decode Image" button calls ``decode``.
    * ``make_all.py``: experiment runner. It has its own NumPy embedder
      (UTF-8 bytes plus the same 64-zero delimiter) and does not import this
      module.
    * ``steganalysis.py`` (plots) and ``pairwise_stats.py`` (numbers): detect
      embedding rather than read it. Overwriting LSBs moves pixels between the
      values 2k and 2k+1, which tends to equalize the histogram counts of each
      even/odd pair; those tools compare the pairs in cover vs. stego images.

This module has no ``__main__`` block. Use it via the CLI::

    python main.py decode -i stego.png

or from Python with ``from decoder import decode``.
"""
from PIL import Image

# End-of-message marker: 64 zero bits (eight 0x00 bytes). Must match
# DELIMITER in encoder.py and DELIM_BITS = 64 in make_all.py.
DELIMITER = "00000000" * 8

def decode(image_path):
    """Extracts a hidden message from an image.

    Reads the least significant bit of every R, G and B value in row-major
    order, finds the first run of 64 zero bits (``DELIMITER``), and decodes
    the bits before it as 8-bit characters.

    Args:
        image_path: Path to the stego image. It should be lossless (PNG, as
            written by ``encoder.encode``); lossy formats such as JPEG destroy
            the LSBs. Any alpha channel is dropped by ``convert("RGB")``.

    Returns:
        The decoded message as a ``str``, or ``None`` if the file does not
        exist or no delimiter is found. The result is ``""`` whenever fewer
        than 8 bits come before the first 64-zero match: an empty message, a
        cover whose LSB stream starts with 64 zeros (e.g. an all-black
        image), or a one-character message with an even code point such as
        ``"d"`` (see the delimiter NOTE below). ``main.py`` and ``gui.py``
        test truthiness, so for ``""`` the CLI prints nothing and the GUI
        shows its "No hidden message found" error. A non-empty ``str`` does
        not prove a message was hidden: any run of 64 zero LSBs, such as a
        flat black region in an unencoded image, is taken as the delimiter
        and the bits before it come back as noise.

    Side effects:
        Prints an error line to stdout when it returns ``None``.

    Raises:
        Only ``FileNotFoundError`` is caught. Other errors from opening or
        loading the image (for example ``PIL.UnidentifiedImageError`` for a
        non-image file, ``OSError`` from ``convert`` for a truncated one,
        ``IsADirectoryError``, ``PermissionError``) propagate to the caller.
    """
    try:
        # Normalize to 3-channel RGB so every pixel unpacks as (r, g, b),
        # matching how encoder.py wrote the bits.
        image = Image.open(image_path, 'r').convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    binary_data = ""
    pixels = image.load()
    width, height = image.size

    # Same traversal as encoder.py: rows top to bottom, pixels left to right,
    # channels R then G then B, so bit i here is bit i of the embedded stream.
    # Each pixel yields 3 bits, so the scan collects 3 * width * height bits.
    # NOTE: slow on large images. Run time grows with the pixel count, not
    # the message length: this Python loop always visits every pixel and
    # never stops once the delimiter has been read, so a few-byte message in
    # a large photo costs as much to scan as one that fills the image.
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # value & 1 keeps only the least significant bit (0 or 1).
            binary_data += str(r & 1)
            binary_data += str(g & 1)
            binary_data += str(b & 1)

    # Find the delimiter in the extracted binary data
    # NOTE: bug. find() matches at any bit offset, not only at byte
    # boundaries. If the last message character ends in one or more 0 bits
    # (any even code point: space, '@', 'd', '0'), those zeros join the
    # delimiter's 64 zeros and the match starts inside that character, so it
    # is lost ("abcd" -> "abc", "hi " -> "hi", "d" -> ""). Trailing NULs
    # extend the zero run further back and lose more ("ab\x00" -> "a"). Any
    # run of 64 zero bits inside the message itself (e.g. eight NUL
    # characters) also ends it early.
    delimiter_index = binary_data.find(DELIMITER)
    
    if delimiter_index == -1:
        print("Error: Could not find message delimiter in the image.")
        return None
    
    # Get the binary data for the message (everything before the delimiter)
    message_binary = binary_data[:delimiter_index]
    
    message = ""
    # Process the binary string in 8-bit chunks (bytes)
    for i in range(0, len(message_binary), 8):
        byte = message_binary[i:i+8]
        # Ensure we have a full byte before converting
        # NOTE: this is where the delimiter bug above shows up: the partial
        # last byte left by an early delimiter match is silently dropped.
        if len(byte) == 8:
            # Convert the byte to an integer, then to a character
            # int(byte, 2) parses the 8-character bit string as base 2 (0-255).
            # NOTE: bug. chr() of one byte treats each byte as a code point
            # 0-255 (Latin-1), not UTF-8. encoder.py writes
            # format(ord(c), '08b'), which is wider than 8 bits for code points
            # above 255, so later bytes are misaligned ("€5" decodes to
            # "\x82\xb0"). make_all.py embeds UTF-8 bytes, so non-ASCII text
            # from it decodes here as mojibake.
            message += chr(int(byte, 2))
    
    return message