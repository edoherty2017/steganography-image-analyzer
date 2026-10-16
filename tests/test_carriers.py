"""Tests for the carriers package: image and WAV LSB carriers and the registry.

Run from the repository root:

    python -m pytest -q tests/test_carriers.py
"""
import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import carriers
from carriers import (
    CapacityError,
    Carrier,
    LossyFormatError,
    carrier_for_path,
    list_carriers,
    register,
)
from carriers.base import key_to_seed, scatter_order
from carriers.image_lsb import ImageLSBCarrier
from carriers.wav_lsb import WavLSBCarrier


def random_bits(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def legacy_embed(rgb: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Reference copy of the legacy encoder loop: R, G, B LSBs, row-major."""
    out = rgb.copy()
    height, width, _ = out.shape
    i = 0
    for y in range(height):
        for x in range(width):
            for c in range(3):
                if i < len(bits):
                    out[y, x, c] = (int(out[y, x, c]) & 0b11111110) | int(bits[i])
                    i += 1
            if i >= len(bits):
                return out
    return out


# --- image carrier: round trips ---------------------------------------------

@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif", ".tiff"])
@pytest.mark.parametrize("bps", [1, 2])
@pytest.mark.parametrize("key", [None, "correct horse battery staple"])
def test_image_round_trip(tmp_path, synthetic_rgb, ext, bps, key):
    cover = synthetic_rgb(0, size=64)
    carrier = ImageLSBCarrier(cover, bits_per_sample=bps)
    bits = random_bits(5001)  # odd and not a multiple of 3: partial pixel/sample
    carrier.embed(bits, key=key)
    out = tmp_path / f"stego{ext}"
    carrier.save(out)

    reloaded = ImageLSBCarrier.load(out, bits_per_sample=bps)
    assert reloaded.mode == "RGB"
    assert np.array_equal(reloaded.rgb, carrier.rgb)
    assert np.array_equal(reloaded.extract(len(bits), key=key), bits)

    # Only the low bps bits may change.
    diff = np.abs(reloaded.rgb.astype(int) - cover.astype(int))
    assert diff.max() <= (1 << bps) - 1
    assert np.array_equal(reloaded.rgb >> bps, cover >> bps)


def test_image_capacity(synthetic_rgb):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=32))
    assert carrier.num_samples() == 32 * 32 * 3
    assert carrier.capacity_bits() == 32 * 32 * 3
    carrier.bits_per_sample = 2
    assert carrier.capacity_bits() == 32 * 32 * 3 * 2


@pytest.mark.parametrize("bps", [1, 2])
def test_full_capacity_fits_and_one_more_bit_raises(synthetic_rgb, bps):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=16), bits_per_sample=bps)
    cap = carrier.capacity_bits()
    bits = random_bits(cap, seed=3)
    carrier.embed(bits, key="k")
    assert np.array_equal(carrier.extract(cap, key="k"), bits)

    before = carrier.rgb.copy()
    with pytest.raises(CapacityError) as info:
        carrier.embed(random_bits(cap + 1))
    assert isinstance(info.value, ValueError)
    assert np.array_equal(carrier.rgb, before), "failed embed must not modify pixels"


def test_extract_more_than_capacity_raises(synthetic_rgb):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=8))
    with pytest.raises(CapacityError):
        carrier.extract(carrier.capacity_bits() + 1)
    with pytest.raises(ValueError):
        carrier.extract(-1)
    assert carrier.extract(0).size == 0


def test_embed_rejects_non_binary_bits(synthetic_rgb):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=8))
    with pytest.raises(ValueError, match="0 and 1"):
        carrier.embed(np.array([0, 1, 2], dtype=np.uint8))


@pytest.mark.parametrize("bad", [0, 3, 8])
def test_bits_per_sample_validated(synthetic_rgb, bad):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=8))
    with pytest.raises(ValueError):
        carrier.bits_per_sample = bad


def test_two_bits_per_sample_higher_bit_first():
    carrier = ImageLSBCarrier(np.zeros((1, 2, 3), dtype=np.uint8), bits_per_sample=2)
    carrier.embed(np.array([1, 0, 0, 1, 1, 1, 1], dtype=np.uint8))
    # Samples: R0 <- "10" = 2, G0 <- "01" = 1, B0 <- "11" = 3, R1 <- "1" + old 0.
    assert carrier.rgb[0, 0].tolist() == [2, 1, 3]
    assert carrier.rgb[0, 1].tolist() == [2, 0, 0]


def test_partial_last_sample_keeps_existing_low_bit():
    pixels = np.full((1, 1, 3), 0b11, dtype=np.uint8)
    carrier = ImageLSBCarrier(pixels, bits_per_sample=2)
    carrier.embed(np.array([0], dtype=np.uint8))
    assert carrier.rgb[0, 0].tolist() == [0b01, 0b11, 0b11]


def test_bytes_helpers(synthetic_rgb):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=16))
    carrier.embed_bytes(b"secret payload", key="pw")
    assert carrier.extract_bytes(14, key="pw") == b"secret payload"


# --- image carrier: legacy compatibility ------------------------------------

@pytest.mark.parametrize("n_bits", [1, 8, 100, 3 * 64 * 64])
def test_matches_legacy_encoder_pixels(synthetic_rgb, n_bits):
    cover = synthetic_rgb(4, size=64)
    bits = random_bits(n_bits, seed=n_bits)
    carrier = ImageLSBCarrier(cover)
    carrier.embed(bits)  # 1 bit per sample, no key
    assert np.array_equal(carrier.rgb, legacy_embed(cover, bits))


def test_legacy_reader_order(tmp_path, synthetic_rgb):
    # Reading LSBs pixel by pixel, R then G then B, recovers the bits.
    cover = synthetic_rgb(5, size=32)
    bits = random_bits(1000)
    carrier = ImageLSBCarrier(cover)
    carrier.embed(bits)
    carrier.save(tmp_path / "s.png")
    pixels = np.array(Image.open(tmp_path / "s.png").convert("RGB"))
    read = [int(pixels[y, x, c]) & 1 for y in range(32) for x in range(32) for c in range(3)]
    assert read[:1000] == bits.tolist()


# --- image carrier: key scatter ---------------------------------------------

def test_key_scatter_changes_positions_but_round_trips(synthetic_rgb):
    cover = synthetic_rgb(6, size=64)
    bits = random_bits(2000)
    plain = ImageLSBCarrier(cover)
    plain.embed(bits)
    keyed = ImageLSBCarrier(cover)
    keyed.embed(bits, key="hunter2")

    assert not np.array_equal(plain.rgb, keyed.rgb)
    changed_plain = np.flatnonzero(plain.rgb.reshape(-1) != cover.reshape(-1))
    changed_keyed = np.flatnonzero(keyed.rgb.reshape(-1) != cover.reshape(-1))
    # Sequential embedding touches only the first 2000 samples; scattered
    # embedding reaches far beyond them.
    assert changed_plain.max() < 2000
    assert changed_keyed.max() > 2000
    assert np.array_equal(keyed.extract(2000, key="hunter2"), bits)


def test_wrong_or_missing_key_does_not_recover(synthetic_rgb):
    carrier = ImageLSBCarrier(synthetic_rgb(7, size=64))
    bits = random_bits(4000)
    carrier.embed(bits, key="right")
    assert not np.array_equal(carrier.extract(4000, key="wrong"), bits)
    assert not np.array_equal(carrier.extract(4000), bits)
    # Agreement should be near chance, not near 100%.
    assert np.mean(carrier.extract(4000, key="wrong") == bits) < 0.6


def test_same_key_is_deterministic(synthetic_rgb):
    cover = synthetic_rgb(8, size=32)
    bits = random_bits(500)
    a = ImageLSBCarrier(cover)
    b = ImageLSBCarrier(cover)
    a.embed(bits, key="k")
    b.embed(bits, key="k")
    assert np.array_equal(a.rgb, b.rgb)


def test_scatter_order_matches_contract():
    key = "pässwörd"
    n = 12345
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    expected = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    assert key_to_seed(key) == seed
    assert np.array_equal(scatter_order(n, key), expected)
    carrier = ImageLSBCarrier(np.zeros((5, 823, 3), dtype=np.uint8))  # 12345 samples
    assert np.array_equal(carrier.sample_order(key), expected)


def test_no_key_and_empty_key_use_natural_order():
    assert np.array_equal(scatter_order(10), np.arange(10))
    assert np.array_equal(scatter_order(10, ""), np.arange(10))


@pytest.mark.parametrize("bps", [1, 2])
def test_keyed_prefix_reads_are_consistent(synthetic_rgb, bps):
    # stego.reveal reads the 112-bit header first, then the whole frame.
    carrier = ImageLSBCarrier(synthetic_rgb(9, size=32), bits_per_sample=bps)
    bits = random_bits(3001)
    carrier.embed(bits, key="k")
    head = carrier.extract(112, key="k")
    full = carrier.extract(3001, key="k")
    assert np.array_equal(head, full[:112])
    assert np.array_equal(full, bits)


# --- image carrier: cover formats -------------------------------------------

def test_rgba_alpha_preserved(tmp_path, synthetic_rgb):
    rng = np.random.default_rng(11)
    alpha = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    rgba = np.dstack([synthetic_rgb(10, size=64), alpha])
    Image.fromarray(rgba, "RGBA").save(tmp_path / "cover.png")

    carrier = ImageLSBCarrier.load(tmp_path / "cover.png")
    assert carrier.mode == "RGBA"
    assert carrier.capacity_bits() == 64 * 64 * 3  # alpha carries nothing
    bits = random_bits(carrier.capacity_bits())
    carrier.embed(bits, key="k")
    for ext in (".png", ".tiff"):
        carrier.save(tmp_path / f"stego{ext}")
        with Image.open(tmp_path / f"stego{ext}") as img:
            assert img.mode == "RGBA"
            saved = np.array(img)
        assert np.array_equal(saved[..., 3], alpha)
        assert np.array_equal(ImageLSBCarrier.load(tmp_path / f"stego{ext}").extract(len(bits), key="k"), bits)


def test_rgba_to_bmp_refused(tmp_path, synthetic_rgb):
    rgba = np.dstack([synthetic_rgb(0, size=8), np.full((8, 8), 200, np.uint8)])
    carrier = ImageLSBCarrier(rgba)
    with pytest.raises(LossyFormatError, match="alpha"):
        carrier.save(tmp_path / "out.bmp")
    assert not (tmp_path / "out.bmp").exists()


def test_palette_cover_converted_to_rgb(tmp_path, synthetic_rgb):
    pal = Image.fromarray(synthetic_rgb(12, size=64)).quantize(64)
    assert pal.mode == "P"
    pal.save(tmp_path / "pal.png")
    carrier = ImageLSBCarrier.load(tmp_path / "pal.png")
    assert carrier.mode == "RGB"
    assert np.array_equal(carrier.rgb, np.array(pal.convert("RGB")))
    bits = random_bits(3000)
    carrier.embed(bits)
    carrier.save(tmp_path / "stego.png")
    assert np.array_equal(ImageLSBCarrier.load(tmp_path / "stego.png").extract(3000), bits)


def test_palette_with_transparency_becomes_rgba(tmp_path, synthetic_rgb):
    pal = Image.fromarray(synthetic_rgb(13, size=32)).quantize(16)
    pal.save(tmp_path / "pal_t.png", transparency=0)
    carrier = ImageLSBCarrier.load(tmp_path / "pal_t.png")
    assert carrier.mode == "RGBA"
    with Image.open(tmp_path / "pal_t.png") as img:
        expected = np.array(img.convert("RGBA"))
    assert np.array_equal(carrier.pixels, expected)


@pytest.mark.parametrize("mode", ["L", "LA"])
def test_grayscale_cover_converted(tmp_path, synthetic_rgb, mode):
    gray = synthetic_rgb(14, size=32)[..., 0]
    img = Image.fromarray(gray, "L")
    if mode == "LA":
        img = Image.merge("LA", [img, Image.fromarray(np.full_like(gray, 77))])
    img.save(tmp_path / "gray.png")
    carrier = ImageLSBCarrier.load(tmp_path / "gray.png")
    assert carrier.mode == ("RGBA" if mode == "LA" else "RGB")
    assert np.array_equal(carrier.rgb[..., 0], gray)
    assert np.array_equal(carrier.rgb[..., 1], gray)
    assert np.array_equal(carrier.rgb[..., 2], gray)
    if mode == "LA":
        assert (carrier.alpha == 77).all()


def test_jpeg_cover_accepted_but_jpeg_output_refused(tmp_path, synthetic_rgb):
    Image.fromarray(synthetic_rgb(15, size=64)).save(tmp_path / "cover.jpg", quality=90)
    assert carrier_for_path(tmp_path / "cover.jpg") is ImageLSBCarrier
    carrier = ImageLSBCarrier.load(tmp_path / "cover.jpg")
    assert carrier.mode == "RGB"
    bits = random_bits(2000)
    carrier.embed(bits)

    for bad in ("stego.jpg", "stego.JPEG", "stego.webp", "stego.gif"):
        with pytest.raises(LossyFormatError, match="lossy"):
            carrier.save(tmp_path / bad)
        assert not (tmp_path / bad).exists()

    carrier.save(tmp_path / "stego.png")
    assert np.array_equal(ImageLSBCarrier.load(tmp_path / "stego.png").extract(2000), bits)


@pytest.mark.parametrize("name", ["out.xyz", "out", "out.wav"])
def test_unknown_output_extension_refused(tmp_path, synthetic_rgb, name):
    carrier = ImageLSBCarrier(synthetic_rgb(0, size=8))
    with pytest.raises(LossyFormatError):
        carrier.save(tmp_path / name)
    assert not (tmp_path / name).exists()


def test_check_output_path_accepts_uppercase():
    assert ImageLSBCarrier.check_output_path("A.PNG").name == "A.PNG"


def test_constructor_copies_input(synthetic_rgb):
    cover = synthetic_rgb(0, size=8)
    original = cover.copy()
    ImageLSBCarrier(cover).embed(np.ones(192, dtype=np.uint8))
    assert np.array_equal(cover, original)


def test_constructor_rejects_bad_arrays():
    with pytest.raises(ValueError):
        ImageLSBCarrier(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        ImageLSBCarrier(np.zeros((4, 4, 3), dtype=np.float32))


# --- WAV carrier -------------------------------------------------------------

def write_wav(path, samples: np.ndarray, sampwidth: int, nchannels: int, rate: int = 8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def sine_16bit(n_frames: int = 8000, nchannels: int = 2, rate: int = 8000) -> np.ndarray:
    t = np.arange(n_frames) / rate
    left = 12000 * np.sin(2 * np.pi * 440 * t)
    right = 9000 * np.sin(2 * np.pi * 660 * t + 0.3)
    chans = [left, right][:nchannels]
    return np.rint(np.stack(chans, axis=1)).astype("<i2").reshape(-1)


def sine_8bit(n_frames: int = 8000, rate: int = 8000) -> np.ndarray:
    t = np.arange(n_frames) / rate
    return np.rint(128 + 100 * np.sin(2 * np.pi * 300 * t)).astype(np.uint8)


@pytest.mark.parametrize("bps", [1, 2])
@pytest.mark.parametrize("key", [None, "audio-key"])
def test_wav_16bit_round_trip_and_params(tmp_path, bps, key):
    samples = sine_16bit()
    assert samples.min() < 0  # exercise negative samples
    write_wav(tmp_path / "cover.wav", samples, sampwidth=2, nchannels=2)

    carrier = WavLSBCarrier.load(tmp_path / "cover.wav", bits_per_sample=bps)
    assert carrier.capacity_bits() == samples.size * bps
    assert carrier_for_path(tmp_path / "cover.wav") is WavLSBCarrier
    bits = random_bits(7777)
    carrier.embed(bits, key=key)
    carrier.save(tmp_path / "stego.wav")

    with wave.open(str(tmp_path / "cover.wav")) as a, wave.open(str(tmp_path / "stego.wav")) as b:
        assert a.getparams() == b.getparams()
    reloaded = WavLSBCarrier.load(tmp_path / "stego.wav", bits_per_sample=bps)
    assert np.array_equal(reloaded.extract(len(bits), key=key), bits)
    diff = np.abs(reloaded.samples.astype(int) - samples.astype(int))
    assert diff.max() <= (1 << bps) - 1


@pytest.mark.parametrize("key", [None, "audio-key"])
def test_wav_8bit_round_trip_and_params(tmp_path, key):
    samples = sine_8bit()
    write_wav(tmp_path / "cover8.wav", samples, sampwidth=1, nchannels=1, rate=8000)

    carrier = WavLSBCarrier.load(tmp_path / "cover8.wav")
    assert carrier.samples.dtype == np.uint8
    assert carrier.capacity_bits() == samples.size
    bits = random_bits(samples.size, seed=5)  # full capacity
    carrier.embed(bits, key=key)
    carrier.save(tmp_path / "stego8.wav")

    with wave.open(str(tmp_path / "cover8.wav")) as a, wave.open(str(tmp_path / "stego8.wav")) as b:
        assert a.getparams() == b.getparams()
    reloaded = WavLSBCarrier.load(tmp_path / "stego8.wav")
    assert np.array_equal(reloaded.extract(len(bits), key=key), bits)
    assert np.abs(reloaded.samples.astype(int) - samples.astype(int)).max() <= 1
    with pytest.raises(CapacityError):
        reloaded.embed(random_bits(samples.size + 1))


def test_wav_rejects_lossy_and_unsupported(tmp_path):
    write_wav(tmp_path / "c.wav", sine_16bit(800, 1), sampwidth=2, nchannels=1)
    carrier = WavLSBCarrier.load(tmp_path / "c.wav")
    with pytest.raises(LossyFormatError, match="lossy"):
        carrier.save(tmp_path / "c.mp3")
    with pytest.raises(LossyFormatError):
        carrier.save(tmp_path / "c.png")

    # 24-bit PCM is not supported.
    write_wav(tmp_path / "c24.wav", np.zeros(300, dtype=np.uint8), sampwidth=3, nchannels=1)
    with pytest.raises(ValueError, match="24-bit"):
        WavLSBCarrier.load(tmp_path / "c24.wav")


# --- registry ----------------------------------------------------------------

def test_list_carriers_contains_builtins():
    found = list_carriers()
    assert ImageLSBCarrier in found and WavLSBCarrier in found
    assert all(issubclass(c, Carrier) for c in found)
    assert {c.name for c in found} >= {"image-lsb", "wav-lsb"}


@pytest.mark.parametrize("name, cls", [
    ("a.png", ImageLSBCarrier), ("a.PNG", ImageLSBCarrier), ("a.bmp", ImageLSBCarrier),
    ("a.tif", ImageLSBCarrier), ("a.tiff", ImageLSBCarrier), ("dir/a.jpg", ImageLSBCarrier),
    ("a.jpeg", ImageLSBCarrier), ("a.gif", ImageLSBCarrier), ("a.wav", WavLSBCarrier),
    ("A.WAV", WavLSBCarrier),
])
def test_carrier_for_path(name, cls):
    assert carrier_for_path(name) is cls


@pytest.mark.parametrize("name", ["a.mp3", "a.txt", "noext", "a.mp4"])
def test_carrier_for_path_unsupported(name):
    with pytest.raises(ValueError, match="no carrier"):
        carrier_for_path(name)


def test_register_plugin_carrier(tmp_path, monkeypatch):
    # A complete plug-in: only load, _samples and _write are written; embed,
    # extract, capacity, key scatter and save checks come from the base class.
    monkeypatch.setattr(carriers, "_REGISTRY", list(carriers._REGISTRY))

    @register
    class RawBytesCarrier(Carrier):
        name = "raw-bytes"
        extensions = (".raw",)

        def __init__(self, data, bits_per_sample=1):
            super().__init__(bits_per_sample)
            self.data = np.frombuffer(data, dtype=np.uint8).copy()

        @classmethod
        def load(cls, path, bits_per_sample=1):
            return cls(Path(path).read_bytes(), bits_per_sample)

        def _samples(self):
            return self.data

        def _write(self, path):
            path.write_bytes(self.data.tobytes())

    assert register(RawBytesCarrier) is RawBytesCarrier  # idempotent
    assert list_carriers().count(RawBytesCarrier) == 1
    assert carrier_for_path("blob.RAW") is RawBytesCarrier

    (tmp_path / "cover.raw").write_bytes(bytes(range(256)) * 4)
    plug = carrier_for_path("cover.raw").load(tmp_path / "cover.raw", bits_per_sample=2)
    bits = random_bits(2048)
    plug.embed(bits, key="k")
    plug.save(tmp_path / "stego.raw")
    assert np.array_equal(RawBytesCarrier.load(tmp_path / "stego.raw", 2).extract(2048, key="k"), bits)
    with pytest.raises(LossyFormatError):
        plug.save(tmp_path / "stego.jpg")


def test_registry_restored_after_plugin_test():
    # Runs after test_register_plugin_carrier (file order); the monkeypatched
    # registry must not leak the plug-in into later tests.
    names = [c.name for c in list_carriers()]
    assert "raw-bytes" not in names
    assert names[:2] == ["image-lsb", "wav-lsb"]


def test_register_rejects_non_carrier():
    with pytest.raises(TypeError):
        register(int)


def test_errors_are_value_errors():
    assert issubclass(LossyFormatError, ValueError)
    assert issubclass(CapacityError, ValueError)
