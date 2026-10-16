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
from carriers.base import (
    SCATTER_SCRYPT_N,
    SCATTER_SCRYPT_P,
    SCATTER_SCRYPT_R,
    content_salt,
    key_to_seed,
    scatter_order,
)
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


def reference_salt(samples: np.ndarray) -> bytes:
    """Independent copy of the documented salt: SHA-256 of the high bits."""
    flat = np.asarray(samples).ravel()
    high = (flat >> 2).astype(flat.dtype.newbyteorder("<"))
    tag = f"{high.dtype.kind}{high.dtype.itemsize}".encode()
    data = b"STGA-scatter-salt-v1" + tag + flat.size.to_bytes(8, "big") + high.tobytes()
    return hashlib.sha256(data).digest()[:16]


def reference_seed(key: str, salt: bytes) -> int:
    """Independent copy of the documented seed: scrypt(key, salt), 8 bytes."""
    out = hashlib.scrypt(key.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=8)
    return int.from_bytes(out, "big")


def test_scatter_order_matches_contract():
    key = "pässwörd"
    pixels = np.random.default_rng(5).integers(0, 256, (5, 823, 3), dtype=np.uint8)
    n = pixels.size  # 12345 samples
    salt = reference_salt(pixels)
    seed = reference_seed(key, salt)
    expected = np.random.Generator(np.random.PCG64(seed)).permutation(n)

    assert content_salt(pixels) == salt
    assert key_to_seed(key, salt) == seed
    assert np.array_equal(scatter_order(n, key, salt), expected)
    carrier = ImageLSBCarrier(pixels)
    assert carrier.scatter_salt() == salt
    assert np.array_equal(carrier.sample_order(key), expected)


def test_scatter_seed_is_not_the_old_unsalted_hash():
    # Regression: the seed used to be SHA-256(key)[:8], which let a password
    # guess be checked without running scrypt. It must not come back.
    key = "sunshine"
    salt = reference_salt(np.zeros(100, dtype=np.uint8))
    old_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert key_to_seed(key, salt) != old_seed


def test_no_key_and_empty_key_use_natural_order():
    assert np.array_equal(scatter_order(10), np.arange(10))
    assert np.array_equal(scatter_order(10, ""), np.arange(10))
    assert np.array_equal(scatter_order(10, None, b"salt"), np.arange(10))


def test_keyed_scatter_order_requires_a_salt():
    with pytest.raises(ValueError, match="salt"):
        scatter_order(10, "k")


def test_scatter_kdf_cost_matches_payload_kdf():
    # A guess through the scatter order must cost as much as a guess against
    # the AES-GCM key, so the two scrypt cost settings have to agree.
    payload = pytest.importorskip("payload")
    assert (SCATTER_SCRYPT_N, SCATTER_SCRYPT_R, SCATTER_SCRYPT_P) == (
        payload.SCRYPT_N, payload.SCRYPT_R, payload.SCRYPT_P)


def test_every_password_guess_runs_scrypt_with_the_cover_salt(synthetic_rgb, monkeypatch):
    # Regression for the offline password check: reading the scattered header
    # for a guessed key must go through scrypt (memory-hard, payload cost
    # settings) with this cover's salt, once per guess.
    import carriers.base as base

    calls = []
    real_scrypt = hashlib.scrypt

    def recording_scrypt(password, *, salt, n, r, p, maxmem=0, dklen=64):
        calls.append({"password": password, "salt": salt, "n": n, "r": r, "p": p})
        return real_scrypt(password, salt=salt, n=n, r=r, p=p, maxmem=maxmem, dklen=dklen)

    monkeypatch.setattr(base.hashlib, "scrypt", recording_scrypt)
    carrier = ImageLSBCarrier(synthetic_rgb(10, size=64))
    salt = reference_salt(carrier.rgb)
    guesses = ["123456", "password", "letmein"]
    for guess in guesses:
        carrier.extract(112, key=guess)
        carrier.extract(1000, key=guess)  # header then frame: no second scrypt

    assert [c["password"] for c in calls] == [g.encode() for g in guesses]
    for call in calls:
        assert call["salt"] == salt
        assert (call["n"], call["r"], call["p"]) == (2 ** 14, 8, 1)


def test_same_size_covers_get_different_orders_for_the_same_key(synthetic_rgb):
    # Regression: the order used to depend only on (key, number of samples),
    # so one precomputed table per image size served every stego file of that
    # size. Two covers of the same size must now get unrelated orders.
    a = ImageLSBCarrier(synthetic_rgb(1, size=64))
    b = ImageLSBCarrier(synthetic_rgb(2, size=64))
    assert a.num_samples() == b.num_samples()
    assert a.scatter_salt() != b.scatter_salt()
    order_a = a.sample_order("shared password")
    order_b = b.sample_order("shared password")
    assert np.mean(order_a[:1000] == order_b[:1000]) < 0.01


def test_salt_survives_embedding_at_both_depths_and_saving(tmp_path, synthetic_rgb):
    # The extractor recomputes the salt from the stego file, so embedding
    # (which changes only the two lowest bits) must leave it unchanged.
    cover = synthetic_rgb(11, size=64)
    carrier = ImageLSBCarrier(cover, bits_per_sample=2)
    salt = carrier.scatter_salt()
    order = carrier.sample_order("k").copy()
    bits = random_bits(carrier.capacity_bits(), seed=3)  # every low bit rewritten
    carrier.embed(bits, key="k")
    assert carrier.scatter_salt() == salt
    carrier.bits_per_sample = 1
    assert carrier.scatter_salt() == salt

    out = tmp_path / "full.png"
    carrier.save(out)
    reloaded = ImageLSBCarrier.load(out, bits_per_sample=2)
    assert reloaded.scatter_salt() == salt
    assert np.array_equal(reloaded.sample_order("k"), order)
    assert np.array_equal(reloaded.extract(len(bits), key="k"), bits)


def test_order_cache_follows_the_cover_content(synthetic_rgb):
    # The cached order is reused only while the salt (the high bits) is the
    # same; changing a high bit gives a new salt and a new order.
    carrier = ImageLSBCarrier(synthetic_rgb(12, size=32))
    before = carrier.sample_order("k").copy()
    assert carrier.sample_order("k") is carrier.sample_order("k")
    carrier.rgb[0, 0, 0] ^= 0x80
    after = carrier.sample_order("k")
    assert not np.array_equal(before, after)


def test_wav_salt_is_endian_independent():
    # 16-bit WAV samples are hashed as little-endian, so the salt (and thus a
    # keyed order) does not depend on the machine's byte order.
    samples = np.arange(-500, 500, dtype=np.int16)
    assert content_salt(samples.astype("<i2")) == content_salt(samples.astype(">i2"))
    assert content_salt(samples) == reference_salt(samples)


def test_scattered_header_not_found_through_unsalted_hash(tmp_path, synthetic_rgb, save_png):
    # End-to-end form of the reported attack: hide with a password (which
    # is also the scatter key), then try to find the header the old cheap
    # way, with the order seeded from SHA-256(password) and no scrypt.
    stego = pytest.importorskip("stego")
    payload = pytest.importorskip("payload")
    from bitutils import bits_to_bytes

    cover = save_png(synthetic_rgb(13), "cover.png")
    out = tmp_path / "s.png"
    stego.hide(cover, out, "top secret", password="sunshine")

    samples = ImageLSBCarrier.load(out)._samples()
    old_seed = int.from_bytes(hashlib.sha256(b"sunshine").digest()[:8], "big")
    old_order = np.random.Generator(np.random.PCG64(old_seed)).permutation(samples.size)
    old_header = bits_to_bytes((samples[old_order[:112]] & 1).astype(np.uint8))
    assert old_header[:4] != payload.MAGIC

    # The salted scrypt order (what reveal() uses) still finds it.
    carrier = ImageLSBCarrier.load(out)
    assert carrier.extract_bytes(4, key="sunshine") == payload.MAGIC
    assert stego.reveal(out, password="sunshine").text() == "top secret"


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


def gradient16(size: int = 128) -> np.ndarray:
    """A smooth 0..65535 horizontal gradient, the shape of a 16-bit scan."""
    return np.tile(np.linspace(0, 65535, size), (size, 1)).astype(np.uint16)


@pytest.mark.parametrize("ext", [".png", ".tif"])
def test_16bit_grayscale_cover_is_scaled_not_clipped(tmp_path, ext):
    # Regression: convert("RGB") clipped I;16 values at 255, so the cover
    # (and the stego image) came out ~99% white.
    grad = gradient16()
    Image.fromarray(grad).save(tmp_path / f"g16{ext}")
    with Image.open(tmp_path / f"g16{ext}") as img:
        assert img.mode.startswith("I;16")
    carrier = ImageLSBCarrier.load(tmp_path / f"g16{ext}")
    assert carrier.mode == "RGB"
    for c in range(3):
        assert np.array_equal(carrier.rgb[..., c], (grad >> 8).astype(np.uint8))
    assert (carrier.rgb == 255).mean() < 0.02


def test_16bit_grayscale_stego_end_to_end(tmp_path):
    import stego

    grad = gradient16()
    Image.fromarray(grad).save(tmp_path / "g16.png")
    stego.hide(tmp_path / "g16.png", tmp_path / "out.png", "sixteen bit")
    with Image.open(tmp_path / "out.png") as img:
        assert img.mode == "RGB"
        out = np.array(img)
    assert (out == 255).mean() < 0.02  # was 0.99 before the fix
    expected = (grad >> 8).astype(np.uint8)
    for c in range(3):
        assert np.array_equal(out[..., c] >> 1, expected >> 1)  # only LSBs differ
    assert stego.reveal(tmp_path / "out.png").text() == "sixteen bit"


def test_16bit_pgm_cover_opened_as_I_is_scaled(tmp_path):
    grad = gradient16(64)
    Image.fromarray(grad).save(tmp_path / "g16.pgm")
    with Image.open(tmp_path / "g16.pgm") as img:
        mode = img.mode
    assert mode in ("I", "I;16")  # Pillow 12 opens 16-bit PGM as "I"
    carrier = ImageLSBCarrier.load(tmp_path / "g16.pgm")
    assert np.array_equal(carrier.rgb[..., 0], (grad >> 8).astype(np.uint8))


@pytest.mark.parametrize("values, expected", [
    # "I" values that fit in a byte are used as they are.
    (np.array([[0, 17, 255]], np.int32), [0, 17, 255]),
    # "I" values up to 65535 are treated as 16-bit data.
    (np.array([[0, 4096, 65535]], np.int32), [0, 16, 255]),
    # Anything else is stretched from min..max to 0..255.
    (np.array([[-1000, 0, 1000]], np.int32), [0, 128, 255]),
    (np.array([[0, 100000, 200000]], np.int32), [0, 128, 255]),
    # "F" in 0..1 is a normalised float image.
    (np.array([[0.0, 0.5, 1.0]], np.float32), [0, 128, 255]),
    # "F" in 0..255 is used as is (rounded); NaN counts as 0.
    (np.array([[2.4, 200.6, np.nan]], np.float32), [2, 201, 0]),
])
def test_I_and_F_modes_scaled_to_8bit(values, expected):
    img = Image.fromarray(values)
    assert img.mode in ("I", "F")
    carrier = ImageLSBCarrier.from_image(img)
    assert carrier.mode == "RGB"
    for c in range(3):
        assert carrier.rgb[0, :, c].tolist() == expected


def test_16bit_grayscale_transparency_key_becomes_alpha(tmp_path):
    grad = gradient16(32)
    key = int(grad[0, 5])
    Image.fromarray(grad).save(tmp_path / "g16t.png", transparency=key)
    carrier = ImageLSBCarrier.load(tmp_path / "g16t.png")
    assert carrier.mode == "RGBA"
    assert np.array_equal(carrier.alpha == 0, grad == key)
    assert np.array_equal(carrier.rgb[..., 0], (grad >> 8).astype(np.uint8))


def colour_keyed_rgb_png(path, synthetic_rgb, mode="RGB"):
    """Save a cover whose top-left 8x8 block is a tRNS colour key."""
    arr = synthetic_rgb(16, size=64)
    if mode == "RGB":
        arr[:8, :8] = (0, 255, 0)
        Image.fromarray(arr).save(path, transparency=(0, 255, 0))
        mask = (arr == (0, 255, 0)).all(axis=-1)
    else:
        arr = arr[..., 0].copy()
        arr[:8, :8] = 7
        Image.fromarray(arr, "L").save(path, transparency=7)
        mask = arr == 7  # other pixels may share the key value
    assert mask[:8, :8].all()
    return arr, mask


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_colour_key_transparency_becomes_rgba(tmp_path, synthetic_rgb, mode):
    # Regression: RGB images took an early return and lost their tRNS key.
    path = tmp_path / "keyed.png"
    _, mask = colour_keyed_rgb_png(path, synthetic_rgb, mode)
    carrier = ImageLSBCarrier.load(path)
    assert carrier.mode == "RGBA"
    with Image.open(path) as img:
        assert img.mode == mode
        expected = np.array(img.convert("RGBA"))
    assert np.array_equal(carrier.pixels, expected)
    assert np.array_equal(carrier.alpha == 0, mask)


def test_colour_key_transparency_survives_hide(tmp_path, synthetic_rgb):
    import stego

    path = tmp_path / "keyed.png"
    _, mask = colour_keyed_rgb_png(path, synthetic_rgb)
    # Fill the cover so the keyed pixels' LSBs certainly change; the alpha
    # channel must still mark exactly the originally transparent pixels.
    room = stego.capacity_bytes(path)
    stego.hide(path, tmp_path / "out.png", b"\xa5" * room)
    with Image.open(tmp_path / "out.png") as img:
        assert img.mode == "RGBA"
        alpha = np.array(img)[..., 3]
    assert np.array_equal(alpha == 0, mask)
    assert (alpha[~mask] == 255).all()
    assert stego.reveal(tmp_path / "out.png").data == b"\xa5" * room


def test_image_to_array_matches_load(tmp_path, synthetic_rgb):
    from carriers.image_lsb import image_to_array

    Image.fromarray(gradient16(32)).save(tmp_path / "g16.png")
    _, _ = colour_keyed_rgb_png(tmp_path / "keyed.png", synthetic_rgb)
    for name in ("g16.png", "keyed.png"):
        with Image.open(tmp_path / name) as img:
            arr = image_to_array(img)
        assert arr.dtype == np.uint8 and arr.flags.writeable
        assert np.array_equal(arr, ImageLSBCarrier.load(tmp_path / name).pixels)


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


def write_truncated_wav(tmp_path, samples, *, sampwidth, nchannels, cut, name="trunc.wav"):
    """Write a WAV, then rewrite it without its last ``cut`` bytes."""
    full = tmp_path / ("full-" + name)
    write_wav(full, samples, sampwidth=sampwidth, nchannels=nchannels)
    raw = full.read_bytes()
    path = tmp_path / name
    path.write_bytes(raw[:-cut])
    return path


# (sampwidth, nchannels, frames written, bytes cut, whole frames left)
TRUNCATIONS = [
    (2, 1, 1001, 1, 1000),  # 16-bit mono cut mid-sample (odd byte count)
    (2, 2, 1001, 3, 1000),  # 16-bit stereo cut mid-sample: drop the whole frame
    (2, 2, 1001, 2, 1000),  # 16-bit stereo cut between the two channels
    (1, 2, 1001, 1, 1000),  # 8-bit stereo cut between the two channels
    (2, 1, 1001, 2, 1000),  # cut on a frame boundary: header still says 1001
]


@pytest.mark.parametrize("sampwidth,nchannels,n_frames,cut,left", TRUNCATIONS)
@pytest.mark.parametrize("bps", [1, 2])
def test_wav_truncated_data_drops_partial_frame(tmp_path, sampwidth, nchannels, n_frames,
                                                cut, left, bps):
    # Regression: a file cut off mid-sample used to fail in np.frombuffer with
    # "buffer size must be a multiple of element size".
    if sampwidth == 2:
        samples = sine_16bit(n_frames, nchannels)
    else:
        samples = np.tile(sine_8bit(n_frames)[:, None], (1, nchannels)).reshape(-1)
    path = write_truncated_wav(tmp_path, samples, sampwidth=sampwidth,
                               nchannels=nchannels, cut=cut)

    carrier = WavLSBCarrier.load(path, bits_per_sample=bps)
    assert carrier.params.nframes == left
    assert carrier.samples.size == left * nchannels
    assert np.array_equal(carrier.samples, samples[:left * nchannels])
    assert carrier.capacity_bits() == left * nchannels * bps

    # The truncated file still works as a cover: embed, save, reload, extract.
    bits = random_bits(carrier.capacity_bits(), seed=11)
    carrier.embed(bits, key="cut-key")
    carrier.save(tmp_path / "stego.wav")
    with wave.open(str(tmp_path / "stego.wav")) as w:
        assert w.getnframes() == left
        assert (w.getnchannels(), w.getsampwidth()) == (nchannels, sampwidth)
    reloaded = WavLSBCarrier.load(tmp_path / "stego.wav", bits_per_sample=bps)
    assert reloaded.params == carrier.params
    assert np.array_equal(reloaded.extract(len(bits), key="cut-key"), bits)


@pytest.mark.parametrize("password", [None, "pw"])
def test_truncated_wav_stego_still_reveals(tmp_path, password):
    # End to end through the high-level API named in the bug report: a stego
    # WAV that loses its last byte still reports a capacity and still gives
    # back a payload hidden in samples before the cut. Key scatter is off
    # because its permutation depends on the sample count, so a keyed payload
    # cannot be located once the file is shorter than when it was written.
    import payload
    import stego

    write_wav(tmp_path / "cover.wav", sine_16bit(4000, 2), sampwidth=2, nchannels=2)
    stego.hide(tmp_path / "cover.wav", tmp_path / "stego.wav", "survives the cut",
               password=password, key_scatter=False)
    raw = (tmp_path / "stego.wav").read_bytes()
    cut = tmp_path / "cut.wav"
    cut.write_bytes(raw[:-1])

    # 3999 whole stereo frames are left: 7998 one-bit samples.
    assert stego.capacity_bytes(cut) == 7998 // 8 - payload.HEADER_LEN
    assert stego.reveal(cut, password=password).text() == "survives the cut"


def test_wav_shorter_than_one_frame_has_zero_capacity(tmp_path):
    # Data chunk holds a single byte of a 4-byte stereo frame.
    path = write_truncated_wav(tmp_path, sine_16bit(2, 2), sampwidth=2, nchannels=2, cut=7)
    carrier = WavLSBCarrier.load(path)
    assert carrier.samples.size == 0
    assert carrier.params.nframes == 0
    assert carrier.capacity_bits() == 0
    with pytest.raises(CapacityError):
        carrier.embed(random_bits(1))


@pytest.mark.parametrize("keep", [0, 4, 20, 30])
def test_wav_truncated_header_raises_clear_wave_error(tmp_path, keep):
    # Regression: a file cut inside the RIFF or fmt chunk header made the
    # stdlib raise a bare EOFError (empty message), which the CLI does not
    # catch; it is now a wave.Error that says the file is truncated.
    write_wav(tmp_path / "full.wav", sine_16bit(100, 1), sampwidth=2, nchannels=1)
    path = tmp_path / "cut.wav"
    path.write_bytes((tmp_path / "full.wav").read_bytes()[:keep])
    with pytest.raises(wave.Error, match="truncated WAV"):
        WavLSBCarrier.load(path)


def test_wav_any_header_truncation_is_a_wave_error(tmp_path):
    # Every cut inside the 44-byte canonical header raises wave.Error (some
    # cuts get the stdlib's own message, e.g. "data chunk missing"), never
    # EOFError, so the CLI and GUI handlers for wave.Error always apply.
    write_wav(tmp_path / "full.wav", sine_16bit(100, 1), sampwidth=2, nchannels=1)
    raw = (tmp_path / "full.wav").read_bytes()
    path = tmp_path / "cut.wav"
    for keep in range(44):
        path.write_bytes(raw[:keep])
        with pytest.raises(wave.Error):
            WavLSBCarrier.load(path)


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
