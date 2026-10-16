"""Tests for the legacy entry points encoder.encode() and decoder.decode().

Covers the bugs of the original implementation that the rewrite fixes:

* messages ending in a character with trailing zero bits ("d", " ", "@",
  ".") lost that character, because the 64-zero delimiter was matched at any
  bit offset;
* NUL characters inside the text ended the message early;
* code points above 255 were written with ``format(ord(c), '08b')`` and came
  back garbled;
* errors were printed and swallowed instead of raised.

Legacy-format images (message bytes + 64 zero bits, as written by the
original encoder and by make_all.py) are built inline with the original
algorithm, and the tracked images in results/ are checked against a
reference decoder written here from the original algorithm, so these tests
do not depend on decoder.py to define the legacy format.

Run from the repository root:

    python -m pytest -q tests/test_legacy.py
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import decoder
import encoder
import payload
import stego
from carriers import CapacityError, LossyFormatError
from payload import WrongPasswordError

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# Messages that the original bit-offset delimiter search truncated or the
# original ord()-based encoder garbled, plus ordinary ones.
ROUND_TRIP_MESSAGES = [
    "hello world",
    "end with a period.",
    "end with a space ",
    "abcd",
    "user@",
    "d",
    "@",
    "",
    "ab\x00\x00\x00\x00\x00\x00\x00\x00cd",   # 8 NULs inside the text
    "@\x00\x00\x00\x00\x00\x00\x00\x01",      # 7 NULs plus neighbouring zeros
    "trailing NULs\x00\x00\x00",
    "café",                               # accented Latin-1 letter
    "€5 and \U0001F600",                  # euro sign and an emoji
    "line one\nline two\ttabbed",
    "x" * 1000,
]


# --- reference implementations of the original algorithm ----------------------

def legacy_embed(rgb: np.ndarray, message_bytes: bytes) -> np.ndarray:
    """Original embedding: message bytes + 64 zero bits into R, G, B LSBs.

    Pixels are visited row-major and channels R, G, B, exactly like the
    original encoder.py loop (vectorized: it writes the same values).
    """
    bits = np.unpackbits(np.frombuffer(message_bytes + b"\x00" * 8, dtype=np.uint8))
    out = np.array(rgb, dtype=np.uint8, copy=True)
    flat = out.reshape(-1)
    if bits.size > flat.size:
        raise ValueError("legacy message does not fit")
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return out


def legacy_bits_of_image(path) -> np.ndarray:
    """LSBs of R, G, B of every pixel, row-major, after convert('RGB')."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return rgb.reshape(-1) & 1


def reference_decode_byte_aligned(path):
    """Original decoding, with the delimiter searched at byte boundaries only.

    Bytes before the first 8 zero bytes; UTF-8 if valid, else Latin-1 (what
    chr() of each byte gave in the original). None if no delimiter.
    """
    bits = legacy_bits_of_image(path)
    data = bytes(np.packbits(bits[: bits.size - bits.size % 8]))
    for start in range(len(data) - 7):
        if data[start:start + 8] == b"\x00" * 8:
            raw = data[:start]
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
    return None


def original_bitwise_decode(path):
    """The original decoder.py algorithm, bug included.

    Builds the '0'/'1' string, finds the first 64 zeros at ANY bit offset,
    and turns each whole 8-bit group before it into chr(byte).
    """
    bits = legacy_bits_of_image(path)
    text = (bits + ord("0")).astype(np.uint8).tobytes().decode("ascii")
    idx = text.find("0" * 64)
    if idx == -1:
        return None
    body = text[:idx]
    return "".join(chr(int(body[i:i + 8], 2)) for i in range(0, len(body) - 7, 8))


# --- fixtures ----------------------------------------------------------------

@pytest.fixture
def cover(tmp_path, synthetic_rgb):
    """64x64 synthetic RGB cover as PNG (1536 bytes of 1-bit capacity)."""
    path = tmp_path / "cover.png"
    Image.fromarray(synthetic_rgb(0, size=64)).save(path)
    return str(path)


def write_legacy_image(tmp_path, synthetic_rgb, message_bytes: bytes, name="legacy.png"):
    path = tmp_path / name
    Image.fromarray(legacy_embed(synthetic_rgb(4, size=64), message_bytes)).save(path)
    return str(path)


# --- encode() / decode() round trips -------------------------------------------

@pytest.mark.parametrize("message", ROUND_TRIP_MESSAGES)
def test_encode_decode_round_trip(tmp_path, cover, message):
    out = str(tmp_path / "stego.png")
    assert encoder.encode(cover, message, out) == out
    assert decoder.decode(out) == message


@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif"])
def test_encode_lossless_formats(tmp_path, cover, ext):
    out = tmp_path / f"stego{ext}"
    assert encoder.encode(cover, "format check d", out) == out
    assert decoder.decode(out) == "format check d"


def test_encode_writes_stga_frame_in_row_major_rgb_order(tmp_path, cover):
    out = tmp_path / "stego.png"
    encoder.encode(cover, "café", out)
    frame = payload.pack("café")
    bits = legacy_bits_of_image(out)[: len(frame) * 8]
    assert bytes(np.packbits(bits)) == frame


def test_encode_changes_only_the_lsbs_it_needs(tmp_path, cover):
    out = tmp_path / "stego.png"
    encoder.encode(cover, "short", out)
    with Image.open(cover) as a, Image.open(out) as b:
        before = np.asarray(a).reshape(-1).astype(int)
        after = np.asarray(b).reshape(-1).astype(int)
    used = len(payload.pack("short")) * 8
    assert np.abs(before - after).max() <= 1
    assert np.array_equal(before[used:], after[used:])


def test_encode_keeps_alpha(tmp_path, synthetic_rgb):
    rgba = np.dstack([synthetic_rgb(1, size=32), np.full((32, 32), 200, np.uint8)])
    src = tmp_path / "rgba.png"
    Image.fromarray(rgba).save(src)
    out = tmp_path / "stego.png"
    encoder.encode(src, "alpha stays", out)
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert (np.asarray(img)[..., 3] == 200).all()
    assert decoder.decode(out) == "alpha stays"


# --- encode() errors raise instead of printing -----------------------------------

def test_encode_missing_cover_raises(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        encoder.encode(tmp_path / "missing.png", "x", tmp_path / "out.png")
    assert capsys.readouterr().out == ""


def test_encode_too_large_raises(tmp_path, cover, capsys):
    out = tmp_path / "out.png"
    too_long = "x" * (stego.capacity_bytes(cover) + 1)
    with pytest.raises(CapacityError):
        encoder.encode(cover, too_long, out)
    assert not out.exists()
    assert capsys.readouterr().out == ""
    # Exactly at capacity still fits.
    exact = too_long[:-1]
    encoder.encode(cover, exact, out)
    assert decoder.decode(out) == exact


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".webp", ".gif"])
def test_encode_lossy_output_raises(tmp_path, cover, ext):
    out = tmp_path / f"out{ext}"
    with pytest.raises(LossyFormatError):
        encoder.encode(cover, "x", out)
    assert not out.exists()


def test_encode_success_prints_nothing(tmp_path, cover, capsys):
    encoder.encode(cover, "quiet", tmp_path / "out.png")
    assert capsys.readouterr().out == ""


def test_encode_rejects_non_text(tmp_path, cover):
    with pytest.raises(TypeError):
        encoder.encode(cover, 12345, tmp_path / "out.png")


# --- message_to_binary (legacy helper) ------------------------------------------

def test_message_to_binary_ascii_matches_original():
    message = "Hi there@"
    original = "".join(format(ord(c), "08b") for c in message) + "0" * 64
    assert encoder.message_to_binary(message) == original
    assert encoder.DELIMITER == "0" * 64


def test_message_to_binary_uses_utf8_for_non_ascii():
    bits = encoder.message_to_binary("€")
    assert len(bits) == 3 * 8 + 64
    assert bits[:24] == "".join(format(b, "08b") for b in "€".encode("utf-8"))


# --- legacy-format images -------------------------------------------------------

@pytest.mark.parametrize("message", [
    "abcd",
    "hi ",
    "user@",
    "I am here.",
    "d",
    "plain ascii text that ends in an odd character!",
])
def test_decode_legacy_ascii_including_trailing_zero_bits(tmp_path, synthetic_rgb, message):
    path = write_legacy_image(tmp_path, synthetic_rgb, message.encode("ascii"))
    assert decoder.decode(path) == message
    assert decoder.decode_legacy(path) == message
    assert reference_decode_byte_aligned(path) == message


def test_original_decoder_bug_is_fixed(tmp_path, synthetic_rgb):
    """The original bit-offset search drops a final 'd'; the new one does not."""
    path = write_legacy_image(tmp_path, synthetic_rgb, b"abcd")
    assert original_bitwise_decode(path) == "abc"
    assert decoder.decode(path) == "abcd"


@pytest.mark.parametrize("message", ["café", "naïve résumé € \U0001F600"])
def test_decode_legacy_utf8_like_make_all(tmp_path, synthetic_rgb, message):
    path = write_legacy_image(tmp_path, synthetic_rgb, message.encode("utf-8"))
    assert decoder.decode(path) == message


def test_decode_legacy_latin1_like_original_encoder(tmp_path, synthetic_rgb):
    # The original encoder wrote format(ord(c), '08b'): one byte for U+00E9.
    path = write_legacy_image(tmp_path, synthetic_rgb, "café crème".encode("latin-1"))
    assert decoder.decode(path) == "café crème"
    assert original_bitwise_decode(path) == "café crème"


def test_decode_legacy_trailing_nuls_are_a_format_limit(tmp_path, synthetic_rgb):
    # NULs at the end of a legacy message merge with the delimiter; this is
    # a limit of the format (encode() writes STGA frames, which keep them).
    path = write_legacy_image(tmp_path, synthetic_rgb, b"ab\x00\x00")
    assert decoder.decode(path) == "ab"


def test_decode_legacy_from_rgba_drops_alpha_like_original(tmp_path, synthetic_rgb):
    rgb = legacy_embed(synthetic_rgb(5, size=32), b"with alpha")
    rgba = np.dstack([rgb, np.full((32, 32), 0, np.uint8)])
    path = tmp_path / "legacy_rgba.png"
    Image.fromarray(rgba).save(path)
    assert decoder.decode(path) == "with alpha"


def test_decode_clean_image_returns_none(cover):
    assert decoder.decode(cover) is None
    assert decoder.decode_legacy(cover) is None


def test_decode_black_image_is_falsy(tmp_path):
    # 64 zero LSBs at the start look like an empty legacy message.
    path = tmp_path / "black.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(path)
    assert decoder.decode(path) == ""


def test_decode_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        decoder.decode(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError):
        decoder.decode_legacy(tmp_path / "missing.png")


# --- decode() on frames written by stego.hide() -----------------------------------

def test_decode_encrypted_frame_needs_password(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, "classified", password="pw", key_scatter=False)
    with pytest.raises(WrongPasswordError):
        decoder.decode(out)
    assert decoder.decode(out, password="pw") == "classified"


def test_decode_scattered_frame_with_password(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, "scattered secret", password="pw")
    assert decoder.decode(out, password="pw") == "scattered secret"


def test_decode_two_bit_frame(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, "two bits per sample", bits=2)
    assert decoder.decode(out) == "two bits per sample"


def test_decode_wav_frame_and_clean_wav(tmp_path):
    import wave

    rng = np.random.default_rng(0)
    samples = rng.integers(-3000, 3000, 4000).astype("<i2")
    src = tmp_path / "c.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())
    out = tmp_path / "s.wav"
    stego.hide(src, out, "audio")
    assert decoder.decode(out) == "audio"
    assert decoder.decode(src) is None


def test_decode_damaged_frame_raises(tmp_path, cover):
    out = tmp_path / "s.png"
    encoder.encode(cover, "this frame will be damaged", out)
    with Image.open(out) as img:
        arr = np.array(img)
    arr.reshape(-1)[payload.HEADER_LEN * 8 + 3] ^= 1
    Image.fromarray(arr).save(out)
    with pytest.raises(payload.PayloadError, match="CRC"):
        decoder.decode(out)


# --- tracked legacy images in results/ --------------------------------------------

TRACKED_LEGACY = {
    "stego_encoded.png": "I am a sneaky dino!! Knowbody knows im here!!",
    # Ends in ".", which the original bit-offset decoder dropped.
    "stego_20250811-171038_full-low_enc.png":
        "I am a very sneaky dinosaur... Nobody knows I am here.",
}


@pytest.mark.parametrize("name", ["stego_encoded.png",
                                  "stego_20250811-171038_full-low_enc.png",
                                  "stego_resized_encoded.png"])
def test_tracked_legacy_images_match_reference(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    decoded = decoder.decode(path)
    assert isinstance(decoded, str) and decoded
    assert decoded == reference_decode_byte_aligned(path)
    if name in TRACKED_LEGACY:
        assert decoded == TRACKED_LEGACY[name]


def test_tracked_period_image_shows_original_bug():
    path = RESULTS / "stego_20250811-171038_full-low_enc.png"
    if not path.exists():
        pytest.skip(f"{path} not present")
    expected = TRACKED_LEGACY[path.name]
    assert original_bitwise_decode(path) == expected[:-1]   # "." lost
    assert decoder.decode(path) == expected
