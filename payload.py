# payload.py
"""Self-describing payload frames: the bytes that a carrier actually hides.

The legacy encoder hid raw text followed by a 64-bit all-zero delimiter,
which cannot carry binary data, cannot tell where a message really ends when
the text itself contains zero bytes, and gives no way to detect corruption.
This module replaces that with a small framed container. A carrier (see the
carriers package) hides the frame's bits; stego.py packs and unpacks frames
around the carrier.

Frame layout (all integers big-endian)::

    offset  size  field
    0       4     MAGIC = b"STGA"
    4       1     version (VERSION = 2)
    5       1     flags (bit set, see below)
    6       4     body_len, uint32: number of body bytes that follow
    10      4     crc32 of header bytes 0-9 followed by the body, uint32
    14      ...   body (body_len bytes)

HEADER_LEN is 14. A reader can therefore pull the first 14 bytes, call
parse_header() / frame_len(), and then read exactly the rest of the frame.
The first HEADER_PREFIX_LEN (10) header bytes, everything before the CRC
field, are "the header prefix" below: the CRC and the GCM tag both cover it.

Flag bits::

    0x01  FLAG_ENCRYPTED   body is AES-256-GCM ciphertext (see below)
    0x02  FLAG_FILE        the plaintext record starts with a file name
    0x04  FLAG_COMPRESSED  the record was zlib-compressed before encryption

Any other flag bit makes the header invalid. Rejecting unknown bits keeps the
false-positive rate low when a scanner looks for b"STGA" in noisy LSB data.

How the body is built (pack) and taken apart (unpack). The steps are nested,
so the file name, when present, is hidden by compression and encryption too::

    record = [name_len (2 bytes) | name (UTF-8)]   only if FLAG_FILE
             + data
    record = zlib.compress(record)                 only if FLAG_COMPRESSED
    body   = encrypt(record)                       only if FLAG_ENCRYPTED
             else record

So "the body begins with the file name length" is literally true only for a
frame that is neither compressed nor encrypted. Code that wants the file name
must call unpack() rather than reading the body itself.

Encryption. The key is derived from the password with scrypt (salt = 16
random bytes, n = 2**14, r = 8, p = 1, 32-byte key) and the record is sealed
with AES-256-GCM using a random 96-bit nonce, with the 10-byte header prefix
(magic, version, flags, body_len) as associated data. The encrypted body is
``salt (16) | nonce (12) | ciphertext + tag (16)``. GCM's tag authenticates
the ciphertext and the header prefix, so a wrong password is detected rather
than producing garbage. The CRC is checked first, which separates the two
failure modes: a damaged frame raises PayloadError, a wrong password raises
WrongPasswordError.

Integrity. The CRC-32 covers the header prefix as well as the body, so
accidental damage to the flags byte or the length (for example one flipped
LSB in a stego image) is caught instead of silently changing how the body is
decoded. The CRC is not a security mechanism: anyone can recompute it. What
the associated data adds is that, whenever decryption runs, the flags and
length it sees are the ones the sender wrote; clearing FLAG_COMPRESSED or
FLAG_FILE on an encrypted frame and fixing up the CRC still fails the tag.
Nothing can stop someone from replacing an encrypted frame with a plaintext
one, though: if FLAG_ENCRYPTED is cleared and the CRC recomputed, decryption
never runs and unpack() returns the raw salt, nonce and ciphertext as
unencrypted data. Callers that expect a secret payload should check
Payload.encrypted.

Version history. Version 1 (used only during development) computed the CRC
over the body alone and passed no associated data to GCM, which left the
flags byte unprotected. Version 1 frames are rejected as unsupported.

Decompression is capped at MAX_DECOMPRESSED_LEN bytes (16 MiB) so a small
"zip bomb" frame cannot make unpack() allocate hundreds of megabytes; pack()
refuses to compress a record larger than the cap, so every frame it writes
can be read back with the default limit.

Typical use::

    import payload
    frame = payload.pack("secret", password="hunter2", compress=True)
    flags, body_len, crc = payload.parse_header(frame[:payload.HEADER_LEN])
    result = payload.unpack(frame, password="hunter2")
    result.text()   # "secret"
"""
from __future__ import annotations

import os
import re
import struct
import zlib
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"STGA"
VERSION = 2
HEADER_LEN = 14

# Header bytes covered by the CRC and used as GCM associated data: magic,
# version, flags and body_len (everything before the CRC field).
HEADER_PREFIX_LEN = 10

# Flag bits stored in byte 5 of the header.
FLAG_ENCRYPTED = 0x01
FLAG_FILE = 0x02
FLAG_COMPRESSED = 0x04
FLAGS_ALL = FLAG_ENCRYPTED | FLAG_FILE | FLAG_COMPRESSED

# MAGIC (4s), version (B), flags (B), body_len (I), crc32 (I), big-endian.
_HEADER_STRUCT = struct.Struct(">4sBBII")
# The same fields without the CRC: the HEADER_PREFIX_LEN-byte prefix.
_PREFIX_STRUCT = struct.Struct(">4sBBI")
_CRC_STRUCT = struct.Struct(">I")

# scrypt and AES-GCM parameters. Changing any of these breaks compatibility
# with frames written earlier, so they would need a new VERSION.
SALT_LEN = 16
NONCE_LEN = 12
TAG_LEN = 16
KEY_LEN = 32
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1

# Largest body a uint32 length field can describe.
MAX_BODY_LEN = 0xFFFFFFFF

# Upper bound on decompressed size. A frame read from an untrusted image could
# be a "zip bomb" (zlib reaches ratios near 1000:1), so unpack() refuses to
# inflate beyond this rather than exhausting memory. 16 MiB is far more than
# any text message or small file needs, and it keeps the worst case for a
# decode (inflated bytes plus a decoded str copy) under about 100 MB. A
# caller that really needs more can pass max_decompressed_len to unpack().
MAX_DECOMPRESSED_LEN = 16 * 1024 * 1024

# Largest file name the 2-byte length prefix can describe, in UTF-8 bytes.
MAX_FILENAME_LEN = 0xFFFF


class PayloadError(ValueError):
    """The bytes are not a valid frame, or the frame is damaged."""


class WrongPasswordError(PayloadError):
    """The frame is encrypted and the password is missing or wrong."""


@dataclass
class Payload:
    """The decoded contents of a frame.

    Attributes:
        data: The hidden bytes, after decryption and decompression.
        filename: The stored file name (a bare name, no directories) when the
            frame carries a file, else None. It comes from data that may have
            been crafted by someone else, so code that writes it to disk
            should still treat it as untrusted.
        encrypted: True if the frame was encrypted.
        compressed: True if the frame was compressed.
    """

    data: bytes
    filename: str | None = None
    encrypted: bool = False
    compressed: bool = False

    def text(self) -> str:
        """Return the data decoded as UTF-8.

        Invalid byte sequences become U+FFFD instead of raising, so this is
        safe to call on binary payloads (the result is just not meaningful).

        Returns:
            The decoded string.
        """
        return self.data.decode("utf-8", errors="replace")


def _clean_filename(name: str) -> str | None:
    """Reduce a file name to its last path component and validate it.

    Both "/" and "\\" are treated as separators so a Windows path written on
    one machine cannot smuggle directories into another.

    Args:
        name: The name as given by the caller or read from a frame.

    Returns:
        The bare file name, or None if nothing usable is left (empty, ".",
        "..", or containing a NUL character).
    """
    base = re.split(r"[\\/]", name)[-1]
    if base in ("", ".", "..") or "\x00" in base:
        return None
    return base


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive the 256-bit AES key from a password with scrypt.

    Args:
        password: The password; encoded as UTF-8.
        salt: SALT_LEN random bytes stored in the encrypted body.

    Returns:
        A KEY_LEN-byte key.
    """
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def _frame_crc(prefix: bytes, body: bytes) -> int:
    """Compute the frame CRC: CRC-32 of the header prefix followed by the body.

    Args:
        prefix: The HEADER_PREFIX_LEN header bytes (magic, version, flags,
            body_len).
        body: The frame body.

    Returns:
        The unsigned 32-bit CRC. Feeding the body as a continuation avoids
        copying a large body just to prepend ten bytes.
    """
    return zlib.crc32(body, zlib.crc32(prefix)) & 0xFFFFFFFF


def _encrypt(record: bytes, password: str, aad: bytes) -> bytes:
    """Seal a record with AES-256-GCM under a password-derived key.

    Args:
        record: The plaintext (possibly already compressed).
        password: The password.
        aad: Associated data bound to the tag: the frame's header prefix.

    Returns:
        salt | nonce | ciphertext+tag.
    """
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = _derive_key(password, salt)
    return salt + nonce + AESGCM(key).encrypt(nonce, record, aad)


def _decrypt(body: bytes, password: str, aad: bytes) -> bytes:
    """Open an encrypted body produced by _encrypt().

    Args:
        body: salt | nonce | ciphertext+tag.
        password: The password to try.
        aad: The frame's header prefix, which must match what was sealed.

    Returns:
        The plaintext record.

    Raises:
        PayloadError: If the body is too short to hold salt, nonce and tag.
        WrongPasswordError: If the GCM tag does not verify. The CRC has
            already ruled out accidental damage, so this means the password
            is wrong or the frame was deliberately altered (for example a
            flag changed and the CRC recomputed).
    """
    if len(body) < SALT_LEN + NONCE_LEN + TAG_LEN:
        raise PayloadError(
            f"encrypted body is {len(body)} bytes, shorter than the "
            f"{SALT_LEN + NONCE_LEN + TAG_LEN}-byte minimum"
        )
    salt = body[:SALT_LEN]
    nonce = body[SALT_LEN:SALT_LEN + NONCE_LEN]
    sealed = body[SALT_LEN + NONCE_LEN:]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, sealed, aad)
    except InvalidTag:
        raise WrongPasswordError(
            "wrong password for encrypted payload (or its header was altered)"
        ) from None


def _decompress(record: bytes, limit: int) -> bytes:
    """Inflate a zlib stream, refusing output larger than ``limit`` bytes.

    zlib never produces more than ``limit`` bytes here, so a "zip bomb"
    costs at most ``limit`` bytes of memory before it is rejected.

    Args:
        record: A complete zlib stream.
        limit: Largest acceptable decompressed size in bytes (at least 1;
            zlib treats 0 as "no limit", so it must never be passed).

    Returns:
        The decompressed bytes.

    Raises:
        PayloadError: If the stream is invalid, incomplete, or would inflate
            past the size limit.
    """
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(record, limit)
        if not inflater.eof and (
            inflater.unconsumed_tail or inflater.decompress(b"", 1)
        ):
            # Either input is left over or zlib still holds pending output:
            # the stream continues past the limit.
            raise PayloadError(f"compressed payload inflates past {limit} bytes")
    except zlib.error as exc:
        raise PayloadError(f"compressed payload is invalid: {exc}") from None
    if not inflater.eof:
        raise PayloadError("compressed payload is truncated")
    return out


def pack(
    data: bytes | str,
    *,
    password: str | None = None,
    filename: str | None = None,
    compress: bool = False,
) -> bytes:
    """Build a frame around some data.

    Args:
        data: The bytes to hide. A str is encoded as UTF-8; bytearray and
            memoryview are accepted too.
        password: If non-empty, encrypt the record with AES-256-GCM using a
            scrypt-derived key. None and "" both mean "no encryption" (an
            empty Tk entry field yields ""); stego.py should apply the same
            rule when it decides whether the password is also a scatter key.
        filename: If non-empty, set FLAG_FILE and store this name so the
            receiver can save the data under it. Only the last path
            component is kept ("dir/a.txt" is stored as "a.txt").
        compress: If True, zlib-compress the record (level 9) before any
            encryption and set FLAG_COMPRESSED. This is done even when it
            makes the data larger, so the flag always matches the request.
            The record (file name plus data) must not exceed
            MAX_DECOMPRESSED_LEN, the most unpack() inflates by default.

    Returns:
        The complete frame: header followed by body.

    Raises:
        TypeError: If data is not bytes-like or str.
        ValueError: If the file name is unusable or too long, a record to be
            compressed is larger than MAX_DECOMPRESSED_LEN (the frame could
            not be read back), or the body would not fit in the 32-bit
            length field.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    else:
        raise TypeError(
            f"data must be bytes or str, not {type(data).__name__}"
        )

    flags = 0
    record = data

    if filename:
        clean = _clean_filename(filename)
        if clean is None:
            raise ValueError(f"unusable file name: {filename!r}")
        name_bytes = clean.encode("utf-8")
        if len(name_bytes) > MAX_FILENAME_LEN:
            raise ValueError(
                f"file name is {len(name_bytes)} bytes in UTF-8; "
                f"the limit is {MAX_FILENAME_LEN}"
            )
        record = struct.pack(">H", len(name_bytes)) + name_bytes + record
        flags |= FLAG_FILE

    if compress:
        if len(record) > MAX_DECOMPRESSED_LEN:
            raise ValueError(
                f"record is {len(record)} bytes; unpack() inflates at most "
                f"{MAX_DECOMPRESSED_LEN} bytes, so hide it without compression"
            )
        record = zlib.compress(record, 9)
        flags |= FLAG_COMPRESSED

    # The header prefix must be final before encrypting, because it is the
    # GCM associated data. GCM ciphertext is as long as its plaintext, so the
    # encrypted body length is known in advance.
    if password:
        flags |= FLAG_ENCRYPTED
        body_len = SALT_LEN + NONCE_LEN + len(record) + TAG_LEN
    else:
        body_len = len(record)
    if body_len > MAX_BODY_LEN:
        raise ValueError(
            f"payload body is {body_len} bytes; the limit is {MAX_BODY_LEN}"
        )
    prefix = _PREFIX_STRUCT.pack(MAGIC, VERSION, flags, body_len)
    body = _encrypt(record, password, prefix) if password else record
    return prefix + _CRC_STRUCT.pack(_frame_crc(prefix, body)) + body


def parse_header(header: bytes) -> tuple[int, int, int]:
    """Parse and validate the fixed 14-byte frame header.

    Args:
        header: At least HEADER_LEN bytes; only the first HEADER_LEN are
            read, so a whole frame may be passed.

    Returns:
        (flags, body_len, crc) as integers.

    Raises:
        PayloadError: If the input is shorter than HEADER_LEN, the magic or
            version is wrong, or an unknown flag bit is set.
    """
    header = bytes(header)
    if len(header) < HEADER_LEN:
        raise PayloadError(
            f"header needs {HEADER_LEN} bytes, got {len(header)}"
        )
    magic, version, flags, body_len, crc = _HEADER_STRUCT.unpack(
        header[:HEADER_LEN]
    )
    if magic != MAGIC:
        raise PayloadError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise PayloadError(f"unsupported version {version}, expected {VERSION}")
    if flags & ~FLAGS_ALL:
        raise PayloadError(f"unknown flag bits 0x{flags & ~FLAGS_ALL:02x}")
    return flags, body_len, crc


def frame_len(header: bytes) -> int:
    """Return the total length in bytes of the frame this header starts.

    Args:
        header: At least HEADER_LEN bytes (see parse_header()).

    Returns:
        HEADER_LEN + body_len.

    Raises:
        PayloadError: If the header is invalid.
    """
    _, body_len, _ = parse_header(header)
    return HEADER_LEN + body_len


def unpack(
    frame: bytes,
    *,
    password: str | None = None,
    max_decompressed_len: int | None = None,
) -> Payload:
    """Validate a frame and recover its contents.

    The steps undo pack() in reverse: check the header, check the CRC (over
    the header prefix and the body), decrypt, decompress, then split off the
    file name.

    Args:
        frame: The frame bytes. Bytes past the end of the frame are ignored,
            so a scanner may pass an oversized buffer.
        password: Needed only for encrypted frames; ignored otherwise.
        max_decompressed_len: Largest decompressed size accepted for a
            compressed frame, in bytes. None (the default) means
            MAX_DECOMPRESSED_LEN. Raise it only for trusted input.

    Returns:
        The decoded Payload.

    Raises:
        WrongPasswordError: If the frame is encrypted and the password is
            missing, empty, or wrong (or the header was deliberately altered
            and the CRC recomputed; see the module docstring).
        PayloadError: If the header is invalid, the frame is truncated, the
            CRC does not match, the body is malformed, or decompression would
            exceed the size limit.
        ValueError: If max_decompressed_len is less than 1.
    """
    limit = MAX_DECOMPRESSED_LEN if max_decompressed_len is None else max_decompressed_len
    if limit < 1:
        raise ValueError(f"max_decompressed_len must be at least 1, got {limit}")
    frame = bytes(frame)
    flags, body_len, crc = parse_header(frame)
    end = HEADER_LEN + body_len
    if len(frame) < end:
        raise PayloadError(
            f"frame is truncated: header says {end} bytes, got {len(frame)}"
        )
    prefix = frame[:HEADER_PREFIX_LEN]
    body = frame[HEADER_LEN:end]
    actual_crc = _frame_crc(prefix, body)
    if actual_crc != crc:
        raise PayloadError(
            f"CRC mismatch: header says 0x{crc:08x}, header prefix and body "
            f"give 0x{actual_crc:08x}"
        )

    encrypted = bool(flags & FLAG_ENCRYPTED)
    compressed = bool(flags & FLAG_COMPRESSED)
    record = body

    if encrypted:
        if not password:
            raise WrongPasswordError("payload is encrypted; a password is required")
        record = _decrypt(record, password, prefix)

    if compressed:
        record = _decompress(record, limit)

    filename = None
    if flags & FLAG_FILE:
        if len(record) < 2:
            raise PayloadError("file payload is too short for its name length")
        (name_len,) = struct.unpack(">H", record[:2])
        if len(record) < 2 + name_len:
            raise PayloadError("file payload is too short for its name")
        try:
            raw_name = record[2:2 + name_len].decode("utf-8")
        except UnicodeDecodeError:
            raise PayloadError("file name is not valid UTF-8") from None
        filename = _clean_filename(raw_name)
        if filename is None or filename != raw_name:
            raise PayloadError(f"unsafe or empty file name {raw_name!r}")
        record = record[2 + name_len:]

    return Payload(
        data=record,
        filename=filename,
        encrypted=encrypted,
        compressed=compressed,
    )
