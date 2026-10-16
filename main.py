# main.py
"""Command-line interface for LSB (least significant bit) image steganography.

This module is a thin argparse wrapper around the two core functions:

* ``encoder.encode`` writes each character as ``format(ord(c), '08b')``
  (most significant bit first), appends a delimiter of 64 zero bits, and
  puts that bit stream into the least significant bit of the R, then G,
  then B value of each pixel, starting at the top-left pixel and going row
  by row (3 message bits per pixel). Pixels after the delimiter are left
  unchanged.
* ``decoder.decode`` reads the low bit of every R, G and B value, cuts the
  bit string at the first run of 64 zero bits (searched bit by bit, not on
  byte boundaries), and turns each whole 8-bit group before the cut into
  one character with ``chr()``. It returns None if the file or the
  delimiter is not found.

For characters up to U+00FF (one byte each), capacity is exactly
``(3 * width * height) // 8 - 8`` characters: the whole bytes that fit in
the image, minus the 8-byte delimiter. Characters above U+00FF take more
than 8 bits each, so fewer of them fit.

How it fits with the rest of the project:

* ``encoder.py`` / ``decoder.py``: the embedding and extraction logic.
* ``main.py`` (this file): CLI for encode/decode only; no analysis.
* ``gui.py``: Tkinter front end that imports encoder, decoder and
  steganalysis directly.
* ``steganalysis.py``: histogram plots (standard and even/odd pairwise)
  comparing a cover image with its stego version.
* ``pairwise_stats.py``: the numeric even/odd pair statistics, printed as
  JSON, with optional JSON file and LaTeX table output.
* ``make_all.py``: experiment runner with its own UTF-8 embedder (it does
  not use this CLI, encoder.py or decoder.py). It uses the same bit layout
  and 64-zero delimiter, so for ASCII payloads it writes the same bits that
  encoder.py would. It runs steganalysis.py and pairwise_stats.py as
  subprocesses and writes everything to ``--out`` (default ``results/``).

Usage::

    python main.py encode -i cover.png -m "secret text" -o stego.png
    python main.py decode -i stego.png

The input image may be any format Pillow can open. It is converted to RGB,
so any alpha channel is dropped. The encoded output is always written in
PNG format, whatever the ``-o`` extension. Decode only works on a lossless
file: re-saving the stego image as JPEG destroys the hidden bits.
"""
import argparse
from encoder import encode
from decoder import decode

def main():
    """Parse command-line arguments and run the encode or decode subcommand.

    Reads ``sys.argv`` through argparse. ``encode`` needs ``-i/--image``,
    ``-m/--message`` and ``-o/--output``; ``decode`` needs ``-i/--image``.

    Side effects:
        encode: ``encoder.encode`` writes the stego image to ``--output``
        (PNG format) and prints a success line. If the input file is
        missing or the message is too large, it prints an error line and
        writes nothing.
        decode: prints "Decoded Message Found:" and then the message, but
        only if the decoded message is non-empty. ``decoder.decode`` prints
        its own error line if the file or the delimiter is not found.

    Returns:
        None.

    Raises:
        SystemExit: from argparse, with status 2 when the subcommand or a
            required option is missing or invalid, and status 0 for
            ``-h/--help``.
        Exception: encode/decode catch only FileNotFoundError when opening
            the input, so other errors propagate and exit with status 1,
            e.g. PIL.UnidentifiedImageError for a file that is not an image,
            or FileNotFoundError when the ``--output`` directory does not
            exist.
    """
    parser = argparse.ArgumentParser(description="A tool for LSB steganography.")
    # required=True makes argparse reject a call with no subcommand.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser for the "encode" command
    encode_parser = subparsers.add_parser("encode", help="Encode a message into an image.")
    encode_parser.add_argument("-i", "--image", required=True, help="Path to the input image (PNG).")
    encode_parser.add_argument("-m", "--message", required=True, help="The message string to hide.")
    encode_parser.add_argument("-o", "--output", required=True, help="Path for the output encoded image.")

    # Parser for the "decode" command
    decode_parser = subparsers.add_parser("decode", help="Decode a message from an image.")
    decode_parser.add_argument("-i", "--image", required=True, help="Path to the encoded image.")

    args = parser.parse_args()

    # NOTE: the errors encode/decode handle never reach the exit status
    # (only uncaught ones exit 1; see Raises above). encode() returns None
    # whether or not it worked, and decode() prints an error and returns
    # None, so a missing input file, an over-capacity message, or a missing
    # delimiter all still exit with status 0.
    if args.command == "encode":
        # NOTE: encoder.py turns each character into format(ord(c), '08b'),
        # so any character above U+00FF yields more than 8 bits, the bytes
        # after it go out of alignment, and the message decodes as garbage
        # (e.g. "€x" comes back as "\x82±"). make_all.py encodes UTF-8
        # instead, so the two agree only for ASCII text (U+0080-U+00FF is
        # 1 byte here but 2 bytes in UTF-8).
        encode(image_path=args.image, message=args.message, output_path=args.output)
    elif args.command == "decode":
        # Call the decode function and print the result if successful
        # NOTE: decoder.py ends the message at the first run of 64 zero
        # bits, so trailing zero bits of the last character merge with the
        # delimiter. Any message whose last character has an even code
        # (space, ".", "@", "b", ...) loses that character: "hi@" decodes
        # as "hi", and make_all's "...I am here." comes back without the ".".
        # NOTE: decoder.py also reads the low bits of every pixel in a pure
        # Python loop before searching for the delimiter, even for a short
        # message (about 0.9 s for a 1500x1500 image on one test machine),
        # so it is slow on large images.
        message = decode(image_path=args.image)
        # NOTE: `if message:` also skips "", so an empty result prints
        # nothing at all, neither the message nor an error. That happens for
        # an encoded empty message, for a one-character message eaten by the
        # delimiter bug above (e.g. "@"), and for an image with no message
        # whose first 64 low bits are all zero (e.g. an all-black image).
        if message:
            print("Decoded Message Found:")
            print(message)

# Run the CLI only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
