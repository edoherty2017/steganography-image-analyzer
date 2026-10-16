"""Tests for stego.py: hide/reveal round trips, passwords, capacity and errors.

Run from the repository root:

    python -m pytest -q tests/test_stego.py
"""
import json
import wave

import numpy as np
import pytest
from PIL import Image

import payload
import stego
from bitutils import bytes_to_bits
from carriers import CapacityError, LossyFormatError
from carriers.base import Carrier
from payload import HEADER_LEN, PayloadError, WrongPasswordError
from stego import NoPayloadError

PASSWORD = "correct horse battery staple"
MESSAGE = "Meet at the north gate at noon. \u00e9\u20ac\U0001F600"


# --- helpers -----------------------------------------------------------------

def save_image(arr: np.ndarray, path) -> str:
    """Save a uint8 array with Pillow (format from the extension); return str path."""
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
    return str(path)


def make_wav(path, *, sampwidth: int = 2, nchannels: int = 2, nframes: int = 8000,
             rate: int = 8000) -> str:
    """Write a short PCM sine-wave WAV file and return its path as str."""
    t = np.arange(nframes) / rate
    tone = np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(3)
    if sampwidth == 2:
        mono = np.round(tone * 12000 + rng.normal(0, 20, nframes)).astype("<i2")
    else:
        mono = np.clip(np.round(128 + tone * 90 + rng.normal(0, 2, nframes)), 0, 255)
        mono = mono.astype(np.uint8)
    samples = np.repeat(mono[:, None], nchannels, axis=1).ravel()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return str(path)


def lsb_stream(path, n_bytes: int) -> bytes:
    """First n_bytes of the natural-order (row-major R, G, B) LSB stream."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    bits = rgb.reshape(-1)[: n_bytes * 8] & 1
    return np.packbits(bits).tobytes()


@pytest.fixture
def cover(tmp_path, synthetic_rgb):
    """A 96x96 synthetic RGB cover saved as PNG (3456 bytes of 1-bit capacity)."""
    return save_image(synthetic_rgb(0, size=96), tmp_path / "cover.png")


# --- round trips -------------------------------------------------------------

@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif", ".tiff"])
@pytest.mark.parametrize("bits", [1, 2])
def test_image_round_trip(tmp_path, cover, ext, bits):
    out = tmp_path / f"stego{ext}"
    summary = stego.hide(cover, out, MESSAGE, bits=bits)
    assert out.exists()
    assert summary["bits_per_sample"] == bits
    found = stego.reveal(out, bits=bits)
    assert found.text() == MESSAGE
    assert found.filename is None and not found.encrypted and not found.compressed


@pytest.mark.parametrize("sampwidth,nchannels", [(2, 2), (2, 1), (1, 1)])
def test_wav_round_trip(tmp_path, sampwidth, nchannels):
    src = make_wav(tmp_path / "cover.wav", sampwidth=sampwidth, nchannels=nchannels)
    out = tmp_path / "stego.wav"
    summary = stego.hide(src, out, MESSAGE, password=PASSWORD, compress=True)
    assert summary["carrier"] == "wav-lsb"
    assert summary["scattered"] is True
    assert stego.reveal(out, password=PASSWORD).text() == MESSAGE
    with wave.open(src, "rb") as a, wave.open(str(out), "rb") as b:
        assert a.getparams() == b.getparams()


def test_bytes_data_round_trip(tmp_path, cover):
    data = bytes(range(256)) * 4
    stego.hide(cover, tmp_path / "s.png", data)
    assert stego.reveal(tmp_path / "s.png").data == data


def test_empty_message_round_trip(tmp_path, cover):
    stego.hide(cover, tmp_path / "s.png", "")
    assert stego.reveal(tmp_path / "s.png").data == b""


def test_cover_file_is_not_modified(tmp_path, cover):
    before = open(cover, "rb").read()
    stego.hide(cover, tmp_path / "s.png", MESSAGE)
    assert open(cover, "rb").read() == before


def test_output_in_new_folder_is_created(tmp_path, cover):
    out = tmp_path / "a" / "b" / "s.png"
    stego.hide(cover, out, "x")
    assert stego.reveal(out).text() == "x"


def test_unscattered_layout_matches_documented_order(tmp_path, cover):
    """No password: the frame sits at the start, row-major R, G, B LSBs."""
    out = tmp_path / "s.png"
    stego.hide(cover, out, "layout check")
    frame = payload.pack("layout check")  # deterministic without a password
    assert lsb_stream(out, len(frame)) == frame


def test_summary_is_json_serializable(tmp_path, cover):
    summary = stego.hide(cover, tmp_path / "s.png", MESSAGE, password=PASSWORD,
                         filename="dir/notes.txt", compress=True, bits=2)
    json.dumps(summary, allow_nan=False)
    assert summary["filename"] == "notes.txt"
    assert summary["encrypted"] and summary["compressed"] and summary["scattered"]
    assert summary["bits_used"] == summary["frame_bytes"] * 8
    assert summary["capacity_bits"] == 96 * 96 * 3 * 2
    assert 0 < summary["utilization"] <= 1


# --- passwords and scatter ---------------------------------------------------

def test_password_with_scatter_round_trip(tmp_path, cover):
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, MESSAGE, password=PASSWORD)
    assert summary["scattered"] is True
    found = stego.reveal(out, password=PASSWORD)
    assert found.text() == MESSAGE and found.encrypted
    # Scattered: the header is not at the start of the natural-order stream.
    assert lsb_stream(out, 4) != payload.MAGIC


def test_scattered_wrong_or_missing_password_finds_nothing(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, MESSAGE, password=PASSWORD)
    # A wrong key reads the bits in the wrong order, so no header is found.
    with pytest.raises(NoPayloadError, match="wrong password"):
        stego.reveal(out, password="not the password")
    with pytest.raises(NoPayloadError):
        stego.reveal(out)
    assert issubclass(NoPayloadError, PayloadError)


def test_unscattered_password_round_trip_and_wrong_password(tmp_path, cover):
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, MESSAGE, password=PASSWORD, key_scatter=False)
    assert summary["scattered"] is False
    assert lsb_stream(out, 4) == payload.MAGIC
    # reveal() tries the keyed order, then falls back to natural order.
    assert stego.reveal(out, password=PASSWORD).text() == MESSAGE
    with pytest.raises(WrongPasswordError):
        stego.reveal(out, password="wrong")
    with pytest.raises(WrongPasswordError):
        stego.reveal(out)


def test_empty_password_means_no_encryption_and_no_scatter(tmp_path, cover):
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, MESSAGE, password="")
    assert summary["scattered"] is False and summary["encrypted"] is False
    assert lsb_stream(out, 4) == payload.MAGIC
    assert stego.reveal(out, password="").text() == MESSAGE
    assert stego.reveal(out).text() == MESSAGE


def test_scatter_without_password_is_natural_order(tmp_path, cover):
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, MESSAGE, key_scatter=True)
    assert summary["scattered"] is False
    assert lsb_stream(out, 4) == payload.MAGIC


def test_reveal_reads_header_then_exactly_the_frame(tmp_path, cover, monkeypatch):
    out = tmp_path / "s.png"
    stego.hide(cover, out, MESSAGE)
    frame_bits = len(payload.pack(MESSAGE)) * 8
    calls = []
    original = Carrier.extract

    def spy(self, n_bits, key=None):
        calls.append(n_bits)
        return original(self, n_bits, key=key)

    monkeypatch.setattr(Carrier, "extract", spy)
    stego.reveal(out)
    assert calls == [HEADER_LEN * 8, frame_bits]


# --- compression and files ---------------------------------------------------

def test_compress_lets_repetitive_text_fit(tmp_path, cover):
    text = "all work and no play makes jack a dull boy. " * 400  # 17.6 KB
    assert len(text) > stego.capacity_bytes(cover)
    with pytest.raises(CapacityError):
        stego.hide(cover, tmp_path / "raw.png", text)
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, text, compress=True)
    assert summary["compressed"] is True
    found = stego.reveal(out)
    assert found.compressed and found.text() == text


def test_file_payload_with_filename(tmp_path, cover):
    data = np.random.default_rng(7).integers(0, 256, 1500, dtype=np.uint8).tobytes()
    out = tmp_path / "s.png"
    stego.hide(cover, out, data, filename="C:\\Users\\me\\secret.bin",
               password=PASSWORD)
    found = stego.reveal(out, password=PASSWORD)
    assert found.data == data
    assert found.filename == "secret.bin"

    first = stego.save_payload(found, tmp_path / "recovered")
    assert first.name == "secret.bin" and first.read_bytes() == data
    second = stego.save_payload(found, tmp_path / "recovered")
    assert second.name == "secret (1).bin" and second.read_bytes() == data


def test_save_payload_default_names(tmp_path):
    text = stego.save_payload(payload.Payload(b"hello"), tmp_path)
    binary = stego.save_payload(payload.Payload(b"\xff\xfe\x00"), tmp_path)
    assert text.name == "payload.txt" and binary.name == "payload.bin"
    hidden = stego.save_payload(payload.Payload(b"x", filename=".bashrc"), tmp_path)
    assert hidden.name == "bashrc"


# --- capacity ----------------------------------------------------------------

@pytest.mark.parametrize("bits", [1, 2])
def test_capacity_bytes_exact_fit(tmp_path, cover, bits):
    cap = stego.capacity_bytes(cover, bits=bits)
    assert cap == (96 * 96 * 3 * bits) // 8 - HEADER_LEN
    data = b"\xa5" * cap
    stego.hide(cover, tmp_path / "fit.png", data, bits=bits)
    assert stego.reveal(tmp_path / "fit.png", bits=bits).data == data

    too_big = tmp_path / "too_big.png"
    with pytest.raises(CapacityError):
        stego.hide(cover, too_big, data + b"\x00", bits=bits)
    assert not too_big.exists()


def test_capacity_bytes_accounts_for_password_and_filename(tmp_path, cover):
    plain = stego.capacity_bytes(cover)
    cap = stego.capacity_bytes(cover, password=PASSWORD, filename="dir/n.txt")
    assert cap == plain - stego.ENCRYPTION_OVERHEAD - (2 + len("n.txt"))
    data = b"z" * cap
    out = tmp_path / "s.png"
    stego.hide(cover, out, data, password=PASSWORD, filename="dir/n.txt")
    assert stego.reveal(out, password=PASSWORD).data == data
    with pytest.raises(CapacityError):
        stego.hide(cover, tmp_path / "t.png", data + b"z", password=PASSWORD,
                   filename="dir/n.txt")


def test_capacity_bytes_tiny_cover_is_zero(tmp_path):
    tiny = save_image(np.zeros((2, 2, 3), np.uint8), tmp_path / "tiny.png")
    assert stego.capacity_bytes(tiny) == 0
    with pytest.raises(CapacityError):
        stego.hide(tiny, tmp_path / "s.png", "")


def test_capacity_bytes_wav(tmp_path):
    src = make_wav(tmp_path / "c.wav", nframes=1000, nchannels=2)
    assert stego.capacity_bytes(src) == 2000 // 8 - HEADER_LEN


# --- formats -----------------------------------------------------------------

@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".webp", ".gif", ".xyz"])
def test_lossy_output_is_refused(tmp_path, cover, ext):
    out = tmp_path / f"stego{ext}"
    with pytest.raises(LossyFormatError):
        stego.hide(cover, out, MESSAGE)
    assert not out.exists()


def test_wav_cover_cannot_be_saved_as_png(tmp_path):
    src = make_wav(tmp_path / "c.wav")
    with pytest.raises(LossyFormatError):
        stego.hide(src, tmp_path / "s.png", "x")


def test_jpeg_cover_is_accepted(tmp_path, synthetic_rgb):
    jpg = tmp_path / "cover.jpg"
    Image.fromarray(synthetic_rgb(1, size=96)).save(jpg, quality=90)
    out = tmp_path / "s.png"
    stego.hide(jpg, out, MESSAGE, password=PASSWORD)
    assert stego.reveal(out, password=PASSWORD).text() == MESSAGE


def test_rgba_cover_keeps_alpha(tmp_path, synthetic_rgb):
    rgb = synthetic_rgb(2, size=64)
    alpha = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
    src = save_image(np.dstack([rgb, alpha]), tmp_path / "rgba.png")
    out = tmp_path / "s.png"
    stego.hide(src, out, MESSAGE)
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert np.array_equal(np.asarray(img)[..., 3], alpha)
    assert stego.reveal(out).text() == MESSAGE


@pytest.mark.parametrize("ext", [".bmp", ".BMP"])
def test_rgba_cover_to_bmp_fails_before_any_work(tmp_path, synthetic_rgb, monkeypatch, ext):
    """BMP cannot keep alpha; hide() must refuse before scrypt or mkdir.

    Regression: the refusal used to come only from the carrier's save(),
    after payload.pack() (scrypt) and embedding had run and the missing
    output folder had been created.
    """
    rgb = synthetic_rgb(2, size=32)
    alpha = np.full((32, 32), 200, dtype=np.uint8)
    src = save_image(np.dstack([rgb, alpha]), tmp_path / "ca.png")
    out = tmp_path / "sub" / f"o{ext}"

    def no_pack(*args, **kwargs):
        raise AssertionError("payload.pack() ran before the BMP/alpha refusal")

    with monkeypatch.context() as m:
        m.setattr(payload, "pack", no_pack)
        with pytest.raises(LossyFormatError, match="alpha"):
            stego.hide(src, out, "hello", password="pw")
    assert not out.exists()
    assert not out.parent.exists()

    # Positive controls: the same cover still saves losslessly as TIFF, and
    # an RGB cover still saves as BMP.
    tif = tmp_path / "ok" / "o.tif"
    stego.hide(src, tif, "hello", password="pw")
    assert stego.reveal(tif, password="pw").text() == "hello"
    bmp = tmp_path / "ok" / "rgb.bmp"
    stego.hide(save_image(rgb, tmp_path / "c.png"), bmp, "hello")
    assert stego.reveal(bmp).text() == "hello"


def test_unsupported_cover_extension(tmp_path):
    bogus = tmp_path / "cover.xyz"
    bogus.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        stego.hide(bogus, tmp_path / "s.png", "x")


def test_missing_cover_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        stego.hide(tmp_path / "missing.png", tmp_path / "s.png", "x")
    with pytest.raises(FileNotFoundError):
        stego.reveal(tmp_path / "missing.png")


# --- reading errors and auto-detection ---------------------------------------

def test_clean_cover_has_no_payload(cover):
    with pytest.raises(NoPayloadError):
        stego.reveal(cover)
    assert stego.probe(cover) is None


def test_bits_mismatch_and_auto_detect(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, MESSAGE, bits=2)
    with pytest.raises(NoPayloadError):
        stego.reveal(out, bits=1)
    assert stego.reveal(out, bits=None).text() == MESSAGE


def test_damaged_frame_is_a_crc_error_not_no_payload(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, "a message long enough to damage in the middle")
    with Image.open(out) as img:
        arr = np.array(img)
    flat = arr.reshape(-1)
    flat[HEADER_LEN * 8 + 20] ^= 1  # flip one body LSB
    save_image(arr, out)
    with pytest.raises(PayloadError, match="CRC") as info:
        stego.reveal(out)
    assert not isinstance(info.value, NoPayloadError)


def test_cropped_image_reports_truncated_frame(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, b"q" * 3000)  # spans most of the 96 rows
    with Image.open(out) as img:
        save_image(np.asarray(img)[:10], tmp_path / "cropped.png")
    with pytest.raises(PayloadError, match="cropped") as info:
        stego.reveal(tmp_path / "cropped.png")
    assert not isinstance(info.value, NoPayloadError)


def test_probe_reports_encrypted_header_without_password(tmp_path, cover):
    out = tmp_path / "s.png"
    stego.hide(cover, out, MESSAGE, password=PASSWORD, key_scatter=False,
               filename="n.txt", compress=True)
    info = stego.probe(out)
    assert info is not None
    json.dumps(info)
    assert info["encrypted"] and info["compressed"] and info["has_filename"]
    assert info["flag_names"] == ["encrypted", "file", "compressed"]
    assert info["scattered"] is False and info["fits"] is True
    assert info["frame_bytes"] == HEADER_LEN + info["body_len"]

    scattered = tmp_path / "scattered.png"
    stego.hide(cover, scattered, MESSAGE, password=PASSWORD, bits=2)
    assert stego.probe(scattered) is None
    keyed = stego.probe(scattered, password=PASSWORD)
    assert keyed["scattered"] is True and keyed["bits_per_sample"] == 2


def test_invalid_bits_value(tmp_path, cover):
    with pytest.raises(ValueError):
        stego.hide(cover, tmp_path / "s.png", "x", bits=3)
    with pytest.raises(ValueError):
        stego.capacity_bytes(cover, bits=0)


def test_embedded_bits_change_only_lsbs(tmp_path, cover):
    out = tmp_path / "s.png"
    summary = stego.hide(cover, out, MESSAGE * 10, password=PASSWORD)
    with Image.open(cover) as a, Image.open(out) as b:
        diff = np.asarray(a).astype(int) - np.asarray(b).astype(int)
    assert np.abs(diff).max() == 1
    # At most one changed sample per embedded bit (about half of them change).
    assert 0 < np.count_nonzero(diff) <= summary["bits_used"]
    assert summary["bits_used"] == bytes_to_bits(b"\x00" * summary["frame_bytes"]).size
