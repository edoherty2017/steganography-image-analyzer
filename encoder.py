# encoder.py
"""LSB steganography encoder: hide a text message in the pixels of an image.

Each character of the message becomes 8 bits (its code point from ``ord()``;
wider for code points above 255, see message_to_binary()), and a 64-bit
all-zero delimiter is appended. These bits replace the least significant bit
(LSB) of the R, G and B channels, pixel by pixel in row-major order (left to
right, top to bottom), so each pixel carries 3 bits. Changing an LSB moves a
channel value by at most 1, too little to see in normal viewing. The result
is always saved as PNG, because a lossy format such as JPEG would destroy the
LSBs.

How it fits with the other modules:
    encoder.py         This module. ``encode()`` writes the stego image.
    decoder.py         ``decode()`` reads the LSBs of every pixel in the same
                       order, then keeps the bits before the first match of
                       the delimiter. Its DELIMITER must match this one.
    main.py            Command-line front end with ``encode`` and ``decode``
                       subcommands that call this module and decoder.py.
    gui.py             Tkinter app. Calls ``encoder.encode()``, then
                       steganalysis.py to compare the original and encoded
                       images; its Decode button calls ``decoder.decode()``.
    steganalysis.py    Plots of cover vs. stego images: the standard
                       histogram and the even/odd pair differences.
    pairwise_stats.py  Numeric even/odd histogram-pair statistics (L1, L2,
                       median and max of the pair differences) for comparing
                       cover and stego images. It reports numbers only and
                       does not make a detected/not-detected decision.
    make_all.py        Experiment runner that builds all result files from one
                       cover image. It has its own NumPy-array embedder (UTF-8
                       bits), does not import this module, and runs
                       steganalysis.py and pairwise_stats.py as subprocesses.

This module has no command-line entry point. Use it through main.py::

    python main.py encode -i cover.png -m "secret" -o stego.png
    python main.py decode -i stego.png

or import it::

    from encoder import encode
    encode("cover.png", "secret", "stego.png")
"""
from PIL import Image

# End-of-message marker: 64 zero bits (8 zero bytes). decoder.py scans the
# extracted LSB stream for the first occurrence of this pattern.
# NOTE: decoder.py matches the pattern at any bit offset, not only on byte
# boundaries. If the last character ends in zero bits (any even code point,
# e.g. "d", " ", "0", "@"), those zeros plus the delimiter match early and the
# last character is lost ("abcd" decodes as "abc"). A run of NUL characters
# inside the message (8 always; 7 when the neighbouring characters supply 8
# more zero bits, e.g. "@" + 7 NULs + "\x01" decodes as "") is also taken
# for the delimiter, so the text is cut off there, and the character just
# before the run can be lost too ("ab" + 8 NULs + "cd" decodes as "a").
DELIMITER = "00000000" * 8

def message_to_binary(message):
    """Convert a text message to the bit string that encode() embeds.

    Each character becomes ``format(ord(char), '08b')``: its Unicode code point
    in binary, zero-padded to at least 8 digits. DELIMITER is appended at the end.

    Args:
        message: The text to hide.

    Returns:
        str: A string of '0' and '1' characters, the message bits followed by
        the 64 delimiter bits. When every code point is below 256, its length
        is ``8 * len(message) + 64``.

    Raises:
        TypeError: If ``message`` is not an iterable of single characters.
    """
    # '08b' sets a minimum width, not a fixed one.
    # NOTE: code points above 255 (e.g. "€" = 8364) produce more than 8 bits,
    # so decoder.py's 8-bit chunks fall out of alignment and the text comes back
    # garbled ("€5" decodes as "\x82°"). Code points 128-255 do round-trip
    # through decoder.py (apart from the end-marker NOTE at DELIMITER, which
    # drops an even final character), but each is stored as one Latin-1 byte
    # rather than its UTF-8 bytes, so for non-ASCII text the embedded bits
    # differ from what make_all.py's UTF-8 embedder writes.
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += DELIMITER
    return binary_message

def encode(image_path, message, output_path):
    """Hide ``message`` in the LSBs of an image and save the result as PNG.

    The image is converted to RGB. The bits from message_to_binary() are
    written into the LSB of R, then G, then B of each pixel in row-major order.
    Pixels after the last message bit are left unchanged.

    Args:
        image_path: Path to the cover image, in any format Pillow can open.
            It is converted to RGB, so an alpha channel is dropped and palette
            or grayscale images become 3-channel.
        message: Text to hide. See message_to_binary() for the bit format.
            At most ``(width * height * 3) // 8 - 8`` characters fit, because
            8 bytes go to the delimiter (assuming code points below 256).
        output_path: Where to write the stego image. It is always written in
            PNG format, whatever the file extension. An existing file is
            overwritten.

    Returns:
        None, both on success and on the two handled errors below, which are
        reported only by printing. Other errors raise (see Raises).

    Side effects:
        Writes ``output_path`` and prints a success line. If the input file
        does not exist or the message is too large for the image, prints an
        error and returns without writing anything.

    Raises:
        PIL.UnidentifiedImageError: If ``image_path`` is not a readable image.
        OSError: If the input cannot be opened for another reason (e.g.
            permissions, or ``image_path`` is a folder), or ``output_path``
            cannot be written (e.g. its folder does not exist).
        TypeError: From message_to_binary() if ``message`` is not a string
            (or another iterable of single characters).
    """
    try:
        image = Image.open(image_path, 'r').convert("RGB")
    except FileNotFoundError:
        # NOTE: here and in the capacity check below, the error is printed and
        # None is returned instead of raising, so callers cannot tell failure
        # from success. main.py ignores the result, so the CLI exits with
        # status 0 after printing the error.
        # gui.py's try/except never sees it and carries on: if no output file
        # exists it later shows a confusing "error during analysis" from
        # steganalysis.py; if an older results/<name>_encoded.png exists from
        # an earlier run, it silently analyzes that stale file and reports
        # success.
        print(f"Error: Image file not found at {image_path}")
        return

    binary_message = message_to_binary(message)
    width, height = image.size
    # Capacity: one bit in each of R, G and B, so 3 bits per pixel. The // 8
    # rounds down to whole bytes. When every code point is below 256 the bit
    # string is 8 * len(message) + 64 bits, a whole number of bytes, so the
    # rounding never rejects a message that would fit in 3 * width * height
    # bits.
    max_bytes = (width * height * 3) // 8

    # binary_message already includes the 64 delimiter bits, so they count
    # against the capacity too.
    if len(binary_message) > max_bytes * 8:
        print("Error: Message is too large for the image's capacity.")
        return

    # This copy is redundant: convert() already returned a new image, and
    # ``image`` is not used again. The file at image_path is never modified.
    new_image = image.copy()
    # Pixel access object: pixels[x, y] reads and writes (r, g, b) tuples in place.
    pixels = new_image.load()
    # Index of the next bit of binary_message to embed.
    data_index = 0

    # Row-major order (y outer, x inner), the same order decoder.py reads in.
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # Replace the LSB: & 0b11111110 clears bit 0, then | sets it to the
            # message bit. The value changes by at most 1 and stays in its pair
            # (2k, 2k+1). When the message bits are roughly half 0s and half
            # 1s, embedding therefore tends to even out the counts within each
            # even/odd pair of the histogram (in the pixels used). That is the
            # effect that steganalysis.py's pair-difference plots and
            # pairwise_stats.py's statistics measure.
            if data_index < len(binary_message):
                r = (r & 0b11111110) | int(binary_message[data_index])
                data_index += 1
            
            # Same for G and B. Each channel is checked separately because the
            # bits can run out partway through a pixel.
            if data_index < len(binary_message):
                g = (g & 0b11111110) | int(binary_message[data_index])
                data_index += 1

            if data_index < len(binary_message):
                b = (b & 0b11111110) | int(binary_message[data_index])
                data_index += 1
            
            pixels[x, y] = (r, g, b)

            # Stop once every bit is written. break only exits the x loop, so
            # the same check is repeated to exit the y loop.
            if data_index >= len(binary_message):
                break
        if data_index >= len(binary_message):
            break
    
    # Save as PNG: it is lossless, so the modified LSBs survive. The format is
    # forced to PNG even if output_path ends in .jpg or another extension.
    new_image.save(output_path, "PNG")
    print(f"Successfully encoded message in '{output_path}'")