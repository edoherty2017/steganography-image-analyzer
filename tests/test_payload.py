# tests/test_payload.py
"""Tests for payload.py: frame round trips, header parsing and error paths.

Run from the repository root:

    python -m pytest -q tests/test_payload.py
"""
import os
import struct
import zlib

import pytest

import payload
from payload import (
    FLAG_COMPRESSED,
    FLAG_ENCRYPTED,
    FLAG_FILE,
    HEADER_LEN,
    MAGIC,
    VERSION,
    Payload,
    PayloadError,
    WrongPasswordError,
    frame_len,
    pack,
    parse_header,
    unpack,
)

PASSWORD = "correct horse battery staple"


def _flip_byte(frame: bytes, index: int) -> bytes:
    """Return a copy of frame with one byte inverted."""
    out = bytearray(frame)
    out[index] ^= 0xFF
    return bytes(out)


def _reframe(flags: int, body: bytes) -> bytes:
    """Build a frame with a valid header and CRC around an arbitrary body."""
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(">4sBBII", MAGIC, VERSION, flags, len(body), crc) + body


# --- Constants and error classes ------------------------------------------

def test_contract_constants():
    assert MAGIC == b"STGA"
    assert VERSION == 1
    assert HEADER_LEN == 14
    assert (FLAG_ENCRYPTED, FLAG_FILE, FLAG_COMPRESSED) == (0x01, 0x02, 0x04)


def test_error_hierarchy():
    assert issubclass(PayloadError, ValueError)
    assert issubclass(WrongPasswordError, PayloadError)


# --- Round trips ------------------------------------------------------------

def test_roundtrip_bytes():
    data = bytes(range(256)) * 3
    result = unpack(pack(data))
    assert result == Payload(data=data)


def test_roundtrip_utf8_text_with_emoji():
    text = "Héllo, wörld! \u03c0 \u2248 3.14 \U0001F600\U0001F510 \u65e5\u672c"
    result = unpack(pack(text))
    assert result.data == text.encode("utf-8")
    assert result.text() == text


def test_text_replaces_invalid_utf8():
    result = unpack(pack(b"ok\xff\xfe"))
    assert result.text() == "ok\ufffd\ufffd"


def test_roundtrip_empty_data():
    frame = pack(b"")
    assert len(frame) == HEADER_LEN
    assert parse_header(frame)[1] == 0
    result = unpack(frame)
    assert result.data == b""
    assert result.text() == ""


def test_roundtrip_empty_string():
    assert unpack(pack("")).text() == ""


def test_accepts_bytearray_and_memoryview():
    assert unpack(pack(bytearray(b"abc"))).data == b"abc"
    assert unpack(pack(memoryview(b"xyz"))).data == b"xyz"


def test_rejects_non_bytes_data():
    with pytest.raises(TypeError):
        pack(12345)


def test_roundtrip_large_plain():
    data = os.urandom(1 << 20)
    result = unpack(pack(data))
    assert result.data == data


def test_roundtrip_large_compressed_encrypted():
    data = os.urandom(1 << 20)
    frame = pack(data, password=PASSWORD, compress=True)
    result = unpack(frame, password=PASSWORD)
    assert result.data == data
    assert result.encrypted and result.compressed


def test_roundtrip_filename_sets_file_flag():
    data = b"%PDF-1.4 fake pdf"
    frame = pack(data, filename="report.pdf")
    flags, body_len, _ = parse_header(frame)
    assert flags == FLAG_FILE
    # Plain FILE frame: body begins with the 2-byte name length and the name.
    body = frame[HEADER_LEN:]
    assert body[:2] == struct.pack(">H", len("report.pdf"))
    assert body[2:12] == b"report.pdf"
    assert body_len == 2 + len("report.pdf") + len(data)
    result = unpack(frame)
    assert result.filename == "report.pdf"
    assert result.data == data


def test_filename_unicode():
    result = unpack(pack(b"x", filename="r\u00e9sum\u00e9 \U0001F4C4.txt"))
    assert result.filename == "r\u00e9sum\u00e9 \U0001F4C4.txt"


def test_filename_directories_are_stripped():
    assert unpack(pack(b"x", filename="/tmp/dir/a.txt")).filename == "a.txt"
    assert unpack(pack(b"x", filename="C:\\Users\\me\\b.bin")).filename == "b.bin"


@pytest.mark.parametrize("bad", ["dir/", ".", "..", "a/..", "nul\x00.txt"])
def test_filename_unusable_raises(bad):
    with pytest.raises(ValueError):
        pack(b"x", filename=bad)


def test_filename_too_long_raises():
    with pytest.raises(ValueError):
        pack(b"x", filename="a" * 70000)


def test_no_filename_means_no_file_flag():
    for name in (None, ""):
        frame = pack(b"data", filename=name)
        assert parse_header(frame)[0] & FLAG_FILE == 0
        assert unpack(frame).filename is None


def test_roundtrip_compress():
    data = b"A" * 10000
    frame = pack(data, compress=True)
    assert parse_header(frame)[0] == FLAG_COMPRESSED
    assert len(frame) < len(data)
    result = unpack(frame)
    assert result.data == data
    assert result.compressed and not result.encrypted


def test_compress_flag_set_even_when_data_grows():
    data = os.urandom(64)
    result = unpack(pack(data, compress=True))
    assert result.compressed is True
    assert result.data == data


def test_roundtrip_password():
    frame = pack("attack at dawn", password=PASSWORD)
    flags, body_len, _ = parse_header(frame)
    assert flags == FLAG_ENCRYPTED
    # salt (16) + nonce (12) + ciphertext (14) + tag (16).
    assert body_len == 16 + 12 + len("attack at dawn") + 16
    assert b"attack at dawn" not in frame
    result = unpack(frame, password=PASSWORD)
    assert result.text() == "attack at dawn"
    assert result.encrypted and not result.compressed


def test_encryption_is_randomised():
    a = pack(b"same", password=PASSWORD)
    b = pack(b"same", password=PASSWORD)
    assert a != b
    assert unpack(a, password=PASSWORD).data == unpack(b, password=PASSWORD).data


def test_roundtrip_all_flags():
    data = "the quick brown fox " * 50
    frame = pack(data, password=PASSWORD, filename="fox.txt", compress=True)
    assert parse_header(frame)[0] == FLAG_ENCRYPTED | FLAG_FILE | FLAG_COMPRESSED
    # The file name is inside the ciphertext, not visible in the frame.
    assert b"fox.txt" not in frame
    result = unpack(frame, password=PASSWORD)
    assert result == Payload(
        data=data.encode("utf-8"),
        filename="fox.txt",
        encrypted=True,
        compressed=True,
    )


def test_empty_password_means_no_encryption():
    frame = pack(b"plain", password="")
    assert parse_header(frame)[0] & FLAG_ENCRYPTED == 0
    assert unpack(frame).data == b"plain"


def test_password_ignored_for_unencrypted_frame():
    result = unpack(pack(b"plain"), password="unused")
    assert result.data == b"plain"
    assert not result.encrypted


def test_trailing_bytes_ignored():
    frame = pack(b"hello", filename="h.txt")
    result = unpack(frame + os.urandom(100))
    assert result.data == b"hello"
    assert result.filename == "h.txt"


# --- Password errors --------------------------------------------------------

def test_wrong_password_raises():
    frame = pack(b"secret", password=PASSWORD)
    with pytest.raises(WrongPasswordError):
        unpack(frame, password="not the password")


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_password_raises(missing):
    frame = pack(b"secret", password=PASSWORD)
    with pytest.raises(WrongPasswordError):
        unpack(frame, password=missing)


def test_wrong_password_with_compression_and_file():
    frame = pack(b"secret", password=PASSWORD, filename="s.bin", compress=True)
    with pytest.raises(WrongPasswordError):
        unpack(frame, password=PASSWORD + "!")


# --- Damage and malformed frames -------------------------------------------

def test_tampered_plain_body_fails_crc():
    frame = pack(b"some message body")
    with pytest.raises(PayloadError, match="CRC"):
        unpack(_flip_byte(frame, HEADER_LEN + 3))


def test_tampered_encrypted_body_is_crc_error_not_password_error():
    frame = pack(b"some message body", password=PASSWORD)
    tampered = _flip_byte(frame, len(frame) - 1)
    with pytest.raises(PayloadError, match="CRC") as info:
        unpack(tampered, password=PASSWORD)
    assert not isinstance(info.value, WrongPasswordError)


def test_tampered_crc_field_fails():
    frame = pack(b"some message body")
    with pytest.raises(PayloadError, match="CRC"):
        unpack(_flip_byte(frame, 10))


def test_truncated_frame_raises():
    frame = pack(b"0123456789")
    with pytest.raises(PayloadError, match="truncated"):
        unpack(frame[:-1])


def test_bad_magic_raises():
    frame = _flip_byte(pack(b"hello"), 0)
    with pytest.raises(PayloadError, match="magic"):
        parse_header(frame)
    with pytest.raises(PayloadError):
        unpack(frame)
    with pytest.raises(PayloadError):
        frame_len(frame)


def test_bad_version_raises():
    frame = bytearray(pack(b"hello"))
    frame[4] = VERSION + 1
    with pytest.raises(PayloadError, match="version"):
        parse_header(bytes(frame))
    with pytest.raises(PayloadError):
        unpack(bytes(frame))


def test_unknown_flag_bits_raise():
    frame = bytearray(pack(b"hello"))
    frame[5] |= 0x80
    with pytest.raises(PayloadError, match="flag"):
        parse_header(bytes(frame))


@pytest.mark.parametrize("n", [0, 1, HEADER_LEN - 1])
def test_short_header_raises(n):
    with pytest.raises(PayloadError):
        parse_header(pack(b"hello")[:n])


def test_random_bytes_are_not_a_frame():
    with pytest.raises(PayloadError):
        unpack(b"\x00" * 64)


def test_encrypted_body_too_short_raises():
    with pytest.raises(PayloadError, match="shorter"):
        unpack(_reframe(FLAG_ENCRYPTED, b"\x00" * 20), password=PASSWORD)


def test_invalid_compressed_body_raises():
    with pytest.raises(PayloadError, match="compressed"):
        unpack(_reframe(FLAG_COMPRESSED, b"not a zlib stream"))


def test_truncated_compressed_body_raises():
    stream = zlib.compress(b"x" * 1000)
    with pytest.raises(PayloadError, match="truncated"):
        unpack(_reframe(FLAG_COMPRESSED, stream[:-4]))


def test_decompression_limit(monkeypatch):
    monkeypatch.setattr(payload, "MAX_DECOMPRESSED_LEN", 1000)
    with pytest.raises(PayloadError, match="inflates"):
        unpack(pack(b"\x00" * 5000, compress=True))


@pytest.mark.parametrize(
    "body",
    [
        b"\x00",                                  # too short for name length
        struct.pack(">H", 10) + b"abc",           # name longer than record
        struct.pack(">H", 2) + b"\xff\xfe",       # name is not UTF-8
        struct.pack(">H", 8) + b"../x.txt",       # directory traversal
        struct.pack(">H", 0) + b"data",           # empty name
    ],
)
def test_malformed_file_record_raises(body):
    with pytest.raises(PayloadError):
        unpack(_reframe(FLAG_FILE, body))


# --- parse_header / frame_len consistency -----------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"compress": True},
        {"password": PASSWORD},
        {"filename": "f.bin"},
        {"password": PASSWORD, "filename": "f.bin", "compress": True},
    ],
)
def test_header_and_frame_len_consistent(kwargs):
    data = b"consistency check " * 20
    frame = pack(data, **kwargs)
    header = frame[:HEADER_LEN]
    flags, body_len, crc = parse_header(header)
    assert parse_header(frame) == (flags, body_len, crc)
    assert frame_len(header) == len(frame) == HEADER_LEN + body_len
    assert frame_len(frame) == len(frame)
    assert crc == zlib.crc32(frame[HEADER_LEN:]) & 0xFFFFFFFF
    # Reading exactly frame_len bytes is enough to decode, as stego.reveal does.
    result = unpack(frame[:frame_len(header)], password=kwargs.get("password"))
    assert result.data == data


def test_header_layout_is_big_endian():
    frame = pack(b"abc")
    assert frame[:4] == MAGIC
    assert frame[4] == VERSION
    assert frame[5] == 0
    assert frame[6:10] == (3).to_bytes(4, "big")
    assert frame[10:14] == (zlib.crc32(b"abc") & 0xFFFFFFFF).to_bytes(4, "big")
    assert frame[14:] == b"abc"
