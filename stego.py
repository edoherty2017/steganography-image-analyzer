"""High-level steganography API: hide and reveal payloads in any carrier.

This module is the one place that joins the three lower layers together:

    payload.py   Builds the self-describing STGA frame around the data
                 (magic, length, CRC-32, optional zlib compression and
                 AES-256-GCM encryption with a scrypt-derived key).
    bitutils.py  Turns the frame into a flat MSB-first bit array and back.
    carriers/    Hides those bits in the low bits of a cover file. The
                 registry picks the carrier from the file extension (PNG,
                 BMP and TIFF images, which also read JPEG/GIF covers; PCM
                 WAV audio; and any plug-in registered later), so nothing
                 here is image specific.

Callers that only want text in and text out can keep using the legacy
``encoder.encode()`` and ``decoder.decode()``, which are thin wrappers around
hide() and reveal().

Typical use::

    import stego
    stego.capacity_bytes("cover.png")                    # bytes that fit
    stego.hide("cover.png", "stego.png", "meet at noon", password="pw")
    stego.reveal("stego.png", password="pw").text()      # 'meet at noon'

    stego.hide("cover.png", "stego.png", open("a.pdf", "rb").read(),
               filename="a.pdf", compress=True)
    found = stego.reveal("stego.png")
    stego.save_payload(found, "recovered/")              # recovered/a.pdf

Scatter. When a password is given and ``key_scatter`` is True (the default),
the password is also the carrier's scatter key: the frame bits are spread
over the whole cover in a pseudo-random order derived from the password (see
carriers/base.py) instead of filling the first rows. That defeats a look at
the top rows of the bit plane and the sequential chi-square attack of
Westfeld & Pfitzmann (1999), but not whole-image statistics such as sample
pair analysis or RS analysis. It is obfuscation, not extra secrecy: the
secrecy comes from the encryption. A side effect is that a wrong password
reads the bits in the wrong order, so it cannot be told apart from an image
that holds no payload at all (see reveal()).

Reading order. reveal() reads exactly HEADER_LEN * 8 bits first, checks the
header (magic, version, known flags), and only then reads the rest of the
frame, so it never touches more of the carrier than the frame occupies.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import carriers
import payload
from bitutils import bits_to_bytes, bytes_to_bits
from carriers.base import CapacityError, Carrier, LossyFormatError
from payload import HEADER_LEN, Payload, PayloadError, WrongPasswordError

__all__ = [
    "CapacityError",
    "ENCRYPTION_OVERHEAD",
    "HEADER_LEN",
    "LossyFormatError",
    "NoPayloadError",
    "Payload",
    "PayloadError",
    "WrongPasswordError",
    "capacity_bytes",
    "hide",
    "load_carrier",
    "probe",
    "reveal",
    "save_payload",
]

# Bytes that encryption adds to the body: scrypt salt + GCM nonce + GCM tag.
# AES-GCM is a stream mode, so the ciphertext itself is exactly as long as the
# plaintext and this overhead is exact.
ENCRYPTION_OVERHEAD = payload.SALT_LEN + payload.NONCE_LEN + payload.TAG_LEN

# Bits per sample tried, in order, when reveal()/probe() are called with
# bits=None (auto-detect).
_AUTO_BITS = (1, 2)


class NoPayloadError(PayloadError):
    """No valid STGA frame header was found in the carrier.

    Raised by reveal() when none of the read orders it tries starts with a
    valid header. With a password and key scatter this also covers a wrong
    password, because the wrong key reads the bits in the wrong order.
    decoder.decode() uses this error to decide to try the legacy format.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scatter_key(password: str | None, key_scatter: bool) -> str | None:
    """Return the carrier scatter key for a hide() call, or None.

    Uses the same truthiness rule as payload.pack(): None and "" both mean
    "no password", so hide(password="") and reveal(password="") agree.
    """
    return password if (password and key_scatter) else None


def _stored_filename(filename: str | None) -> str | None:
    """Return the name payload.pack() will store for ``filename``.

    pack() keeps only the last path component, splitting on both "/" and
    "\\"; this mirrors that for the summary returned by hide().
    """
    if not filename:
        return None
    return re.split(r"[\\/]", filename)[-1]


def load_carrier(path: str | os.PathLike, bits: int = 1) -> Carrier:
    """Open a file with the carrier registered for its extension.

    Args:
        path: Cover or stego file.
        bits: Low bits used per sample, 1 or 2.

    Returns:
        Carrier: A loaded carrier instance.

    Raises:
        ValueError: If no carrier handles the extension, or ``bits`` is not
            1 or 2.
        FileNotFoundError: If ``path`` does not exist.
    """
    cls = carriers.carrier_for_path(path)
    return cls.load(path, bits_per_sample=bits)


def _read_header(carrier: Carrier, key: str | None) -> bytes | None:
    """Read HEADER_LEN bytes in the order for ``key`` and validate them.

    Args:
        carrier: A loaded carrier; its current bits_per_sample is used.
        key: Scatter key, or None for natural order.

    Returns:
        The header bytes if they parse as a valid STGA header, else None.
    """
    if carrier.capacity_bits() < HEADER_LEN * 8:
        return None
    header = bits_to_bytes(carrier.extract(HEADER_LEN * 8, key=key))
    try:
        payload.parse_header(header)
    except PayloadError:
        return None
    return header


def _locate(
    carrier: Carrier, password: str | None, bits: int | None
) -> tuple[int, str | None, bytes] | None:
    """Find the read settings under which the carrier starts with a header.

    Tries each bits-per-sample value (``bits``, or 1 then 2 when ``bits`` is
    None) and, for each, the password as a scatter key first (when a
    password is given) and then the natural order. The natural-order retry
    makes frames hidden with ``key_scatter=False`` readable without telling
    reveal() how they were hidden; it costs one extra 14-byte read.

    A random 14-byte read passes parse_header() with probability below
    2**-40 (32 bits of magic, 8 of version, 5 of reserved flag bits), so a
    false match on a clean cover is not a practical concern.

    Args:
        carrier: The loaded carrier. Its bits_per_sample is changed while
            searching and left at the matching value on success.
        password: The password, or None/"".
        bits: 1, 2, or None to try both.

    Returns:
        (bits_per_sample, key, header) for the first match, or None.
    """
    bit_options = _AUTO_BITS if bits is None else (bits,)
    keys: tuple[str | None, ...] = (password, None) if password else (None,)
    for bps in bit_options:
        carrier.bits_per_sample = bps
        for key in keys:
            header = _read_header(carrier, key)
            if header is not None:
                return bps, key, header
    return None


def _check_output_for_carrier(carrier: Carrier, out: Path) -> None:
    """Refuse an output path that this loaded cover cannot be saved to.

    Carrier.check_output_path() only looks at the extension, so it can run
    before the cover is read. One refusal also depends on the cover: BMP
    cannot round-trip an alpha channel through Pillow, so ImageLSBCarrier
    refuses ``.bmp`` for a cover with alpha. The carrier only finds that
    out inside save(), after the frame is built (scrypt), embedded and the
    output folder is created. This mirrors the rule from
    carriers/image_lsb.py so hide() can refuse right after loading. The
    carrier keeps its own check as well.

    Args:
        carrier: The loaded cover.
        out: The output path, already accepted by check_output_path().

    Raises:
        LossyFormatError: If ``out`` is a ``.bmp`` file (any case) and the
            carrier has an alpha channel.
    """
    if out.suffix.lower() == ".bmp" and getattr(carrier, "alpha", None) is not None:
        raise LossyFormatError(
            "BMP output would drop this image's alpha channel; save as .png or .tiff"
        )


def _no_payload_message(path: str | os.PathLike, password: str | None,
                        bits: int | None) -> str:
    """Build the NoPayloadError message for reveal()."""
    where = (f"at {bits} bit(s) per sample" if bits is not None
             else "at 1 or 2 bits per sample")
    msg = f"no STGA payload found in {os.fspath(path)!r} {where}"
    if password:
        msg += (
            "; with key scatter a wrong password reads the bits in the wrong "
            "order, so a wrong password looks the same as no payload"
        )
    else:
        msg += (
            "; if it was hidden with a password (and key scatter), "
            "the same password is needed to find it"
        )
    return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capacity_bytes(
    cover_path: str | os.PathLike,
    bits: int = 1,
    *,
    password: str | None = None,
    filename: str | None = None,
) -> int:
    """Return how many payload bytes fit in a cover after the frame header.

    With no keyword arguments this is the largest body hide() can store:
    ``capacity_bits // 8 - HEADER_LEN``. Passing the same ``password`` and
    ``filename`` that hide() will get subtracts their exact overhead too
    (ENCRYPTION_OVERHEAD bytes for encryption, 2 bytes plus the UTF-8 name
    for a file), so the result is then the largest ``data`` that fits.
    Compression cannot be accounted for in advance: it usually lets more
    text fit, but makes random or already-compressed data slightly larger.

    Args:
        cover_path: The cover file.
        bits: Low bits used per sample, 1 or 2.
        password: Optional; only its truthiness matters here.
        filename: Optional file name that will be stored with the data.

    Returns:
        int: Bytes of data that fit, never negative.

    Raises:
        ValueError: If no carrier handles the extension, or ``bits`` is not
            1 or 2.
        FileNotFoundError: If the cover does not exist.
    """
    carrier = load_carrier(cover_path, bits)
    available = carrier.capacity_bits() // 8 - HEADER_LEN
    if password:
        available -= ENCRYPTION_OVERHEAD
    name = _stored_filename(filename)
    if name:
        available -= 2 + len(name.encode("utf-8"))
    return max(0, available)


def hide(
    cover_path: str | os.PathLike,
    output_path: str | os.PathLike,
    data: bytes | str,
    *,
    password: str | None = None,
    filename: str | None = None,
    compress: bool = False,
    bits: int = 1,
    key_scatter: bool = True,
) -> dict[str, Any]:
    """Hide ``data`` in a cover file and write the stego file.

    Steps: pick the carrier from the cover's extension, refuse a lossy
    output extension before doing any work, load the cover, refuse an
    output format that cannot hold this cover (``.bmp`` for a cover with
    alpha), build the frame with payload.pack(), check that it fits, embed
    its bits, create missing output folders, save.

    Args:
        cover_path: The cover. Any format its carrier can read; for images
            that includes lossy formats such as JPEG, since a cover is only
            read. The file itself is never modified.
        output_path: Where to write the stego file. Its extension must be a
            lossless output format of the same carrier (.png, .bmp, .tif or
            .tiff for images, .wav for audio); .bmp only for covers without
            an alpha channel. Missing parent folders are
            created and an existing file is overwritten.
        data: The bytes to hide; a str is encoded as UTF-8.
        password: If non-empty, encrypt the payload (AES-256-GCM, scrypt
            key) and, when ``key_scatter`` is True, also use it as the
            scatter key. None and "" both mean no password.
        filename: Optional file name stored with the data (only the last
            path component is kept), so reveal() can restore it.
        compress: zlib-compress the payload before encryption.
        bits: Low bits used per sample, 1 (default, least visible) or 2
            (twice the capacity, each sample moves by up to 3).
        key_scatter: When True and a password is given, spread the bits in a
            password-derived order. When False, or with no password, the
            bits fill the samples in natural order (for images: row-major
            pixels, R then G then B), the layout of the legacy encoder.

    Returns:
        dict: A JSON-serializable summary with the keys ``cover``,
        ``output``, ``carrier`` (registry name), ``bits_per_sample``,
        ``scattered``, ``encrypted``, ``compressed``, ``filename`` (as
        stored, or None), ``data_bytes``, ``frame_bytes``, ``bits_used``,
        ``capacity_bits``, ``capacity_bytes`` (largest body that fits) and
        ``utilization`` (bits_used / capacity_bits).

    Raises:
        LossyFormatError: If ``output_path`` has a lossy or unsupported
            extension for this carrier, or is ``.bmp`` for an image cover
            with an alpha channel (BMP would drop the alpha). Raised before
            the frame is built, so nothing is written and no folder is
            created.
        CapacityError: If the frame does not fit in the cover (nothing is
            written and no folder is created).
        ValueError: If no carrier handles the cover's extension, ``bits`` is
            not 1 or 2, or the file name is unusable.
        TypeError: If ``data`` is not bytes-like or str.
        FileNotFoundError: If the cover does not exist.
    """
    cls = carriers.carrier_for_path(cover_path)
    # Fail fast on .jpg and friends before reading the cover or running scrypt.
    out = cls.check_output_path(output_path)
    carrier = cls.load(cover_path, bits_per_sample=bits)
    # Some refusals depend on the loaded cover (BMP cannot keep alpha); make
    # them before scrypt, embedding and creating the output folder.
    _check_output_for_carrier(carrier, out)

    frame = payload.pack(data, password=password, filename=filename,
                         compress=compress)
    capacity = carrier.capacity_bits()
    n_bits = len(frame) * 8
    if n_bits > capacity:
        body_room = max(0, capacity // 8 - HEADER_LEN)
        raise CapacityError(
            f"payload body is {len(frame) - HEADER_LEN} bytes but "
            f"{os.fspath(cover_path)!r} holds at most {body_room} at {bits} "
            f"bit(s) per sample ({len(frame)} bytes needed with the "
            f"{HEADER_LEN}-byte header)"
        )

    key = _scatter_key(password, key_scatter)
    carrier.embed(bytes_to_bits(frame), key=key)
    out.parent.mkdir(parents=True, exist_ok=True)
    carrier.save(out)

    if isinstance(data, str):
        data_len = len(data.encode("utf-8"))
    else:
        data_len = len(bytes(data))
    return {
        "cover": os.fspath(cover_path),
        "output": os.fspath(output_path),
        "carrier": carrier.name,
        "bits_per_sample": carrier.bits_per_sample,
        "scattered": key is not None,
        "encrypted": bool(password),
        "compressed": bool(compress),
        "filename": _stored_filename(filename),
        "data_bytes": data_len,
        "frame_bytes": len(frame),
        "bits_used": n_bits,
        "capacity_bits": capacity,
        "capacity_bytes": max(0, capacity // 8 - HEADER_LEN),
        "utilization": n_bits / capacity if capacity else 0.0,
    }


def reveal(
    stego_path: str | os.PathLike,
    *,
    password: str | None = None,
    bits: int | None = 1,
) -> Payload:
    """Find and decode the STGA frame hidden in a stego file.

    The header is read first (HEADER_LEN * 8 bits). If a password is given,
    the password-scattered order is tried first and the natural order
    second, so frames hidden with or without key scatter are both found.
    Once a valid header is found, exactly the rest of the frame is read and
    passed to payload.unpack(), which checks the CRC, decrypts and
    decompresses.

    Args:
        stego_path: The stego file.
        password: The password used by hide(), if any.
        bits: Low bits per sample used by hide(): 1 (default) or 2. None
            tries 1 and then 2.

    Returns:
        payload.Payload: The data, stored file name and flags.

    Raises:
        NoPayloadError: If no valid header is found. With key scatter this
            includes a wrong password and a missing password, since both
            read the bits in the wrong order; the two cases cannot be told
            apart from "nothing hidden".
        WrongPasswordError: If the frame was found (unscattered) but is
            encrypted and the password is missing or wrong.
        PayloadError: If a header was found but the frame is damaged: CRC
            mismatch, malformed body, or a length longer than the carrier
            (for example after the image was cropped).
        ValueError: If no carrier handles the extension, or ``bits`` is not
            1, 2 or None.
        FileNotFoundError: If the file does not exist.
    """
    carrier = load_carrier(stego_path, 1 if bits is None else bits)
    found = _locate(carrier, password, bits)
    if found is None:
        raise NoPayloadError(_no_payload_message(stego_path, password, bits))
    bps, key, header = found
    carrier.bits_per_sample = bps

    total = payload.frame_len(header)
    capacity = carrier.capacity_bits()
    if total * 8 > capacity:
        raise PayloadError(
            f"frame header claims {total} bytes but the carrier holds only "
            f"{capacity // 8}; the file was probably cropped or resized"
        )
    # Every prefix of the scatter order is stable, so reading the whole frame
    # with the same key repeats the header bits; skip them.
    body_bits = carrier.extract(total * 8, key=key)[HEADER_LEN * 8:]
    frame = header + bits_to_bytes(body_bits)
    return payload.unpack(frame, password=password)


def probe(
    stego_path: str | os.PathLike,
    *,
    password: str | None = None,
    bits: int | None = None,
) -> dict[str, Any] | None:
    """Describe the STGA frame header in a file without decoding the body.

    Useful for reports: it says whether a frame is present and whether it is
    encrypted, without needing (or checking) the password for the body.

    Args:
        stego_path: The file to inspect.
        password: Optional scatter key to try first, as in reveal().
        bits: 1, 2, or None (default) to try both.

    Returns:
        A JSON-serializable dict with ``carrier``, ``bits_per_sample``,
        ``scattered``, ``flags``, ``flag_names``, ``encrypted``,
        ``compressed``, ``has_filename``, ``body_len``, ``frame_bytes``,
        ``crc32`` and ``fits`` (whether the claimed frame fits in the
        carrier), or None if no header is found.

    Raises:
        ValueError: If no carrier handles the extension.
        FileNotFoundError: If the file does not exist.
    """
    carrier = load_carrier(stego_path, 1 if bits is None else bits)
    found = _locate(carrier, password, bits)
    if found is None:
        return None
    bps, key, header = found
    carrier.bits_per_sample = bps
    flags, body_len, crc = payload.parse_header(header)
    names = [name for bit, name in ((payload.FLAG_ENCRYPTED, "encrypted"),
                                    (payload.FLAG_FILE, "file"),
                                    (payload.FLAG_COMPRESSED, "compressed"))
             if flags & bit]
    total = HEADER_LEN + body_len
    return {
        "carrier": carrier.name,
        "bits_per_sample": bps,
        "scattered": key is not None,
        "flags": flags,
        "flag_names": names,
        "encrypted": bool(flags & payload.FLAG_ENCRYPTED),
        "compressed": bool(flags & payload.FLAG_COMPRESSED),
        "has_filename": bool(flags & payload.FLAG_FILE),
        "body_len": body_len,
        "frame_bytes": total,
        "crc32": crc,
        "fits": total * 8 <= carrier.capacity_bits(),
    }


def save_payload(
    found: Payload,
    out_dir: str | os.PathLike,
    *,
    default_stem: str = "payload",
) -> Path:
    """Write a revealed payload to a file in ``out_dir``, never overwriting.

    The stored file name comes from the stego file, so it is untrusted even
    though payload.unpack() already reduced it to a bare name. Here it is
    also stripped of leading dots (no hidden files) and of characters that
    are invalid on Windows. Without a stored name the file is called
    ``<default_stem>.txt`` when the data is valid UTF-8 text and
    ``<default_stem>.bin`` otherwise. If the name is taken, " (1)", " (2)",
    ... is added before the extension.

    Args:
        found: The result of reveal().
        out_dir: Folder to write into; created if missing.
        default_stem: Base name used when no file name was stored.

    Returns:
        Path: The file that was written.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)

    name = None
    if found.filename:
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", found.filename).lstrip(".")
    if not name:
        try:
            found.data.decode("utf-8")
            name = f"{default_stem}.txt"
        except UnicodeDecodeError:
            name = f"{default_stem}.bin"

    stem, ext = os.path.splitext(name)
    candidate = folder / name
    counter = 1
    while True:
        try:
            # "xb" fails if the file exists, so an existing file is never
            # replaced even if another process creates it meanwhile.
            with open(candidate, "xb") as fh:
                fh.write(found.data)
            return candidate
        except FileExistsError:
            candidate = folder / f"{stem} ({counter}){ext}"
            counter += 1
