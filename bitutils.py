"""Convert between bytes and NumPy bit arrays.

Every part of the project that moves a payload in or out of a carrier works on
flat arrays of 0/1 values (dtype ``uint8``), one element per bit. This module
is the single place where bytes become bits and bits become bytes, so the bit
order is defined once:

    MSB-first within each byte. ``b"A"`` (0x41 = 0b01000001) becomes
    ``[0, 1, 0, 0, 0, 0, 0, 1]``.

That is the same order as ``format(ord(char), '08b')`` in the legacy
``encoder.py``, so a plain ASCII message produces the same bits through either
path.

Example::

    >>> from bitutils import bytes_to_bits, bits_to_bytes
    >>> bits = bytes_to_bits(b"hi")
    >>> bits[:8].tolist()
    [0, 1, 1, 0, 1, 0, 0, 0]
    >>> bits_to_bytes(bits)
    b'hi'
"""
from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Expand bytes into a flat array of bits, MSB-first per byte.

    Args:
        data: The bytes to expand. ``bytearray`` and ``memoryview`` are
            accepted too.

    Returns:
        np.ndarray: A 1-D ``uint8`` array of length ``8 * len(data)`` holding
        only 0 and 1. The array is a fresh copy and is writable.

    Raises:
        TypeError: If ``data`` is not bytes-like (for example a ``str``;
            encode text with ``.encode("utf-8")`` first).
    """
    if isinstance(data, str):
        raise TypeError("bytes_to_bits() needs bytes, not str; encode the text first")
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    # np.unpackbits defaults to bitorder="big", i.e. MSB first.
    return np.unpackbits(buf)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a flat array of bits back into bytes, MSB-first per byte.

    This is the inverse of bytes_to_bits().

    Args:
        bits: A 1-D array-like of 0/1 values (any integer or bool dtype). Its
            length must be a multiple of 8.

    Returns:
        bytes: ``len(bits) // 8`` bytes.

    Raises:
        ValueError: If ``bits`` is not 1-D, its length is not a multiple of 8,
            or it contains a value other than 0 or 1.
    """
    arr = np.asarray(bits)
    if arr.ndim != 1:
        raise ValueError(f"bits must be a 1-D array, got shape {arr.shape}")
    if arr.size % 8 != 0:
        raise ValueError(f"number of bits ({arr.size}) is not a multiple of 8")
    if arr.size and ((arr != 0) & (arr != 1)).any():
        raise ValueError("bits may only contain the values 0 and 1")
    return np.packbits(arr.astype(np.uint8)).tobytes()
