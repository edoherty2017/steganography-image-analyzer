"""Tests for the legacy entry points encoder.encode() and decoder.decode().

Covers the bugs of the original implementation that the rewrite fixes:

* messages ending in a character with trailing zero bits ("d", " ", "@",
  ".") lost that character, because the 64-zero delimiter was matched at any
  bit offset;
* NUL characters inside the text ended the message early;
* code points above 255 were written with ``format(ord(c), '08b')`` and came
  back garbled;
* errors were printed and swallowed instead of raised.

And the regressions found in review of the rewrite's legacy reader:

* a clean cover with a flat region (or a password-scattered frame read with
  the wrong password) "decoded" to the LSB noise in front of the flat
  region, because any 64 zero bits look like the delimiter. Those tests use
  a cover with a flat patch: the noisy synthetic covers alone never contain
  64 zero LSBs in a row, so they cannot show the bug;
* a delimiter off a byte boundary (the original make_all.py cut its payload
  at the bit level) was not found at all;
* one multi-byte character cut off at the end turned a whole UTF-8 message
  into Latin-1 mojibake.

Legacy-format images (message bytes + 64 zero bits, as written by the
original encoder and by make_all.py) are built inline with the original
algorithm, and the tracked images in results/ are checked against a
reference decoder written here from the original algorithm, so these tests
do not depend on decoder.py to define the legacy format.

Run from the repository root:

    python -m pytest -q tests/test_legacy.py
"""
import os
import subprocess
import sys
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


# --- clean covers with a flat region (review regression) ---------------------------

def flat_patch_rgb(synthetic_rgb, seed: int = 0, size: int = 256) -> np.ndarray:
    """A natural-looking cover with one flat 64x64 patch of value 128.

    The flat patch puts thousands of zero LSBs in a row into the stream,
    which is the delimiter of the legacy format. Real photos have such
    regions (sky, black borders, uniform blocks); the noisy synthetic
    covers alone do not.
    """
    arr = synthetic_rgb(seed, size=size)
    arr[100:164, 100:164] = 128
    return arr


@pytest.fixture
def patch_cover(tmp_path, synthetic_rgb):
    path = tmp_path / "patch.png"
    Image.fromarray(flat_patch_rgb(synthetic_rgb)).save(path)
    return path


def test_flat_patch_cover_holds_a_fake_delimiter(patch_cover):
    """Guard for the tests below: the patch really does fool a bare search."""
    noise = reference_decode_byte_aligned(patch_cover)
    assert noise is not None and len(noise) > 1000


def test_clean_cover_with_flat_patch_finds_nothing(patch_cover):
    assert decoder.decode(patch_cover) is None
    assert decoder.decode_legacy(patch_cover) is None
    assert decoder.find_legacy_message(
        np.asarray(Image.open(patch_cover).convert("RGB"))) is None


def test_raw_reading_keeps_its_contract_but_is_not_a_message(patch_cover, tmp_path,
                                                             synthetic_rgb):
    """legacy_text_from_pixels() is the unchecked reading main.py scores itself."""
    rgb = np.asarray(Image.open(patch_cover).convert("RGB"))
    assert decoder.legacy_text_from_pixels(rgb) == reference_decode_byte_aligned(patch_cover)
    assert decoder.find_legacy_message(rgb) is None
    legacy = write_legacy_image(tmp_path, synthetic_rgb, b"plain text")
    rgb = np.asarray(Image.open(legacy).convert("RGB"))
    assert decoder.legacy_text_from_pixels(rgb) == "plain text"
    assert decoder.find_legacy_message(rgb).text == "plain text"


def test_scattered_frame_with_wrong_password_finds_nothing(tmp_path, patch_cover):
    out = tmp_path / "s.png"
    stego.hide(patch_cover, out, "real secret", password="pw")
    with pytest.raises(stego.NoPayloadError):
        stego.reveal(out, password="wrong")
    # Before the fix the legacy fallback returned ~9,600 chars of LSB noise.
    assert decoder.decode(out, password="wrong") is None
    assert decoder.decode(out) is None
    assert decoder.decode(out, password="pw") == "real secret"


def test_cli_wrong_password_on_flat_patch_cover_fails(tmp_path, patch_cover):
    """The headline symptom: main.py printed noise as a message and exited 0."""
    out = tmp_path / "s.png"
    stego.hide(patch_cover, out, "real secret", password="pw")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    for extra in (["--password", "wrong"], []):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "main.py"), "decode", "-i", str(out), *extra],
            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
            env=env, timeout=180, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        assert proc.returncode == 1, (proc.stdout, proc.stderr)
        assert "Decoded Message Found" not in proc.stdout
        assert "Traceback" not in proc.stderr


def test_smooth_blue_reference_image_is_not_a_message():
    """The repo's clean reference image (gitignored, so optional)."""
    path = REPO_ROOT / "test-images" / "smoothBlue.png"
    if not path.exists():
        pytest.skip(f"{path} not present")
    assert decoder.decode(path) is None


@pytest.mark.parametrize("prefix", [
    b"6\x00(4\x88#s\n",          # NUL and C1 bytes, as in the reported noise
    b"\xff\xff\xff\xff\xff\xff",  # a flat white region: "ÿÿÿÿÿÿ" in Latin-1
    b"\xb6\xdbm\xb6\xdbm",        # a flat (odd, even, odd) colour
    b"\xc3(\x80\xa9\x92\x41",     # invalid UTF-8, mostly non-ASCII
])
def test_noise_before_the_delimiter_is_not_text(tmp_path, synthetic_rgb, prefix):
    path = write_legacy_image(tmp_path, synthetic_rgb, prefix)
    assert reference_decode_byte_aligned(path) is not None  # a bare search "finds" it
    assert decoder.decode(path) is None


# --- delimiter off a byte boundary (review regression) -------------------------------

def make_all_bit_cut_image(rgb: np.ndarray, text: str) -> np.ndarray:
    """The original make_all.py case C: payload cut at the BIT level.

    ``bits = to_bits_utf8(text)[:3*W*H - 64]`` then 64 zero bits, filling the
    whole capacity. When ``W*H % 8 != 0`` the delimiter starts mid-byte.
    """
    out = np.array(rgb, dtype=np.uint8, copy=True)
    flat = out.reshape(-1)
    cap = flat.size - 64
    bits = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))[:cap]
    bits = np.concatenate([bits, np.zeros(64, dtype=np.uint8)])
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return out


@pytest.mark.parametrize("width,height", [(101, 99), (33, 33), (101, 101)])
def test_unaligned_delimiter_from_original_make_all(tmp_path, width, height):
    assert (width * height) % 8 != 0
    rng = np.random.default_rng(width)
    cover = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    text = "I am a very sneaky dinosaur... Nobody knows I am here. " * 400
    path = tmp_path / "bitcut.png"
    Image.fromarray(make_all_bit_cut_image(cover, text)).save(path)

    assert reference_decode_byte_aligned(path) is None   # no aligned delimiter
    expected = original_bitwise_decode(path)
    assert expected and text.startswith(expected)
    assert decoder.decode(path) == expected
    assert decoder.decode_legacy(path) == expected

    found = decoder.find_legacy_message(np.asarray(Image.open(path).convert("RGB")))
    assert found is not None and not found.byte_aligned
    # The zero run starts where make_all.py wrote the delimiter, or a few bits
    # earlier when the cut-off last bits of the text are zeros themselves.
    written_at = 3 * width * height - 64
    assert written_at - 8 < found.delimiter_bit <= written_at
    assert found.delimiter_bit % 8 != 0
    assert found.raw == expected.encode("ascii")


def test_unaligned_delimiter_wins_over_a_later_aligned_zero_run(tmp_path, synthetic_rgb):
    """A flat region after the message gives an aligned match full of NULs."""
    rgb = synthetic_rgb(2, size=64)
    rgb[32:] = 128   # flat bottom half: aligned zero bytes after the message
    message = "unaligned end, then a flat area"
    bits = np.unpackbits(np.frombuffer(message.encode(), dtype=np.uint8))
    bits = np.concatenate([bits, [1, 0, 1], np.zeros(64, dtype=np.uint8)])  # 3 stray bits
    flat = rgb.reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    path = tmp_path / "late_flat.png"
    Image.fromarray(rgb).save(path)
    assert decoder.decode(path) == message


def test_aligned_delimiter_still_keeps_trailing_zero_bit_character(tmp_path, synthetic_rgb):
    """The bit-offset fallback must not undo the "abcd" fix."""
    path = write_legacy_image(tmp_path, synthetic_rgb, b"abcd")
    found = decoder.find_legacy_message(np.asarray(Image.open(path).convert("RGB")))
    assert found.text == "abcd" and found.byte_aligned and found.delimiter_bit == 32


def test_original_encoder_wide_code_point_gives_none_not_noise(tmp_path, synthetic_rgb):
    """Documented limit: format(ord('€'), '08b') is 14 bits wide.

    Every byte after it is shifted, so the bytes before the delimiter are
    not text. The original decoder printed 'price 5\\x82°\\x81½'; the new
    one reports nothing rather than noise.
    """
    message = "price 5€ ok"
    bits = "".join(format(ord(c), "08b") for c in message) + "0" * 64
    rgb = synthetic_rgb(3, size=64)
    flat = rgb.reshape(-1)
    arr = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
    flat[: arr.size] = (flat[: arr.size] & 0xFE) | arr
    path = tmp_path / "euro.png"
    Image.fromarray(rgb).save(path)
    assert original_bitwise_decode(path).startswith("price 5\x82")
    assert decoder.decode(path) is None


# --- a multi-byte character cut off at the end (review regression) -------------------

def test_cut_utf8_character_does_not_turn_message_into_mojibake(tmp_path, synthetic_rgb):
    # As in the review repro: the message was cut at capacity inside an "é".
    data = ("café " * 200).encode("utf-8") + b"caf\xc3"
    path = write_legacy_image(tmp_path, synthetic_rgb, data)
    decoded = decoder.decode(path)
    assert decoded == "café " * 200 + "caf"
    assert "Ã" not in decoded
    found = decoder.find_legacy_message(np.asarray(Image.open(path).convert("RGB")))
    assert found.encoding == "utf-8" and found.raw == data


def test_cut_utf8_character_at_end_of_long_make_all_payload(tmp_path, synthetic_rgb):
    """Bit-level cut on a 64x64 cover (aligned) that lands inside 'é'."""
    rgb = synthetic_rgb(4, size=64)
    text = "café " * 50000
    path = tmp_path / "cafe_cut.png"
    Image.fromarray(make_all_bit_cut_image(rgb, text)).save(path)
    decoded = decoder.decode(path)
    assert decoded is not None and decoded.startswith("café café café")
    assert "Ã" not in decoded
    assert text.startswith(decoded)


@pytest.mark.parametrize("cut", [b"\xe2\x82", b"\xf0\x9f\x98"])
def test_cut_multi_byte_tail_after_ascii_is_dropped(tmp_path, synthetic_rgb, cut):
    # Two or three bytes of a cut character cannot be Latin-1 text.
    path = write_legacy_image(tmp_path, synthetic_rgb, b"price 5 " + cut)
    assert decoder.decode(path) == "price 5 "


def test_single_high_byte_after_ascii_reads_as_latin1(tmp_path, synthetic_rgb):
    # The original encoder wrote 'café' as b'caf\xe9'. A cut UTF-8 'é' would
    # also end in one high byte after ASCII; Latin-1 is the better guess.
    path = write_legacy_image(tmp_path, synthetic_rgb, "café".encode("latin-1"))
    assert decoder.decode(path) == "café"
    found = decoder.find_legacy_message(np.asarray(Image.open(path).convert("RGB")))
    assert found.encoding == "latin-1"
