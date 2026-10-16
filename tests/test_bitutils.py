"""Tests for bitutils.py: byte <-> bit conversion, MSB-first.

Run from the repository root:

    python -m pytest -q tests/test_bitutils.py
"""
import numpy as np
import pytest

from bitutils import bits_to_bytes, bytes_to_bits


def test_known_values_msb_first():
    bits = bytes_to_bits(b"\x80\x01A")
    assert bits.tolist() == (
        [1, 0, 0, 0, 0, 0, 0, 0]
        + [0, 0, 0, 0, 0, 0, 0, 1]
        + [0, 1, 0, 0, 0, 0, 0, 1]  # "A" = 0x41
    )


def test_output_dtype_shape_and_writable():
    bits = bytes_to_bits(b"hello")
    assert bits.dtype == np.uint8
    assert bits.shape == (40,)
    assert set(np.unique(bits).tolist()) <= {0, 1}
    bits[0] = 1  # must not raise (fresh, writable array)


def test_matches_legacy_encoder_bit_strings():
    # The legacy encoder used format(ord(c), "08b") per character.
    text = "Hello, World! ~"
    legacy = "".join(format(ord(c), "08b") for c in text)
    assert "".join(map(str, bytes_to_bits(text.encode("ascii")))) == legacy


@pytest.mark.parametrize("n", [0, 1, 7, 255, 4096])
def test_round_trip_random_bytes(n):
    data = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8).tobytes()
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_round_trip_all_byte_values():
    data = bytes(range(256))
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_empty():
    assert bytes_to_bits(b"").size == 0
    assert bits_to_bytes(np.zeros(0, dtype=np.uint8)) == b""


def test_accepts_bytearray_and_memoryview():
    expected = bytes_to_bits(b"abc")
    assert np.array_equal(bytes_to_bits(bytearray(b"abc")), expected)
    assert np.array_equal(bytes_to_bits(memoryview(b"abc")), expected)


def test_rejects_str():
    with pytest.raises(TypeError):
        bytes_to_bits("abc")


def test_bits_to_bytes_accepts_lists_bools_and_wide_ints():
    assert bits_to_bytes([0, 1, 0, 0, 0, 0, 0, 1]) == b"A"
    assert bits_to_bytes(np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=bool)) == b"B"
    assert bits_to_bytes(np.array([0, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)) == b"C"


@pytest.mark.parametrize("n", [1, 7, 9, 15])
def test_length_not_multiple_of_8_raises(n):
    with pytest.raises(ValueError, match="multiple of 8"):
        bits_to_bytes(np.zeros(n, dtype=np.uint8))


@pytest.mark.parametrize("bad", [2, 255, -1])
def test_non_binary_values_raise(bad):
    bits = np.zeros(8, dtype=np.int16)
    bits[3] = bad
    with pytest.raises(ValueError, match="0 and 1"):
        bits_to_bytes(bits)


def test_2d_input_raises():
    with pytest.raises(ValueError, match="1-D"):
        bits_to_bytes(np.zeros((2, 8), dtype=np.uint8))
