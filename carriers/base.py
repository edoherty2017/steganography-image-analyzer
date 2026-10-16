"""Abstract carrier interface shared by every steganographic cover format.

A *carrier* is a cover medium (an image, a WAV file, ...) seen as a flat
sequence of integer *samples*. For an RGB image the samples are the R, G and B
values of each pixel in row-major order; for a WAV file they are the PCM
sample values. Payload bits are hidden in the lowest ``bits_per_sample`` bits
of those samples (least significant bit, LSB, replacement).

All of the bit handling lives in this base class: capacity, the optional
key-based scatter order, packing bits into samples and reading them back. A
new format only has to say how to read itself, expose its samples and write
itself back out. That is the plug-in contract::

    from carriers import register
    from carriers.base import Carrier

    @register
    class MyCarrier(Carrier):
        name = "my-format-lsb"
        extensions = (".myf",)          # lossless formats save() may write

        @classmethod
        def load(cls, path, bits_per_sample=1):
            ...                         # read the file, keep its samples
            return cls(...)

        def _samples(self):
            return self._data           # 1-D writable integer array (a view)

        def _write(self, path):
            ...                         # write self._data back out

Once registered, ``carriers.carrier_for_path("x.myf")`` finds the class, and
``stego.py`` can hide payloads in it with no other change.

Bit layout:
    * The payload is a flat 0/1 array (see ``bitutils.bytes_to_bits``).
    * Each sample carries ``bits_per_sample`` bits (1 or 2). With 2, payload
      bit ``2*i`` goes into bit 1 of the i-th used sample and bit ``2*i + 1``
      into bit 0 (higher bit first).
    * Without a key, samples are used in their natural order (0, 1, 2, ...).
    * With a key, the order is ``rng.permutation(num_samples)`` where
      ``rng = numpy.random.Generator(numpy.random.PCG64(seed))``. The seed
      comes from a key derivation function, not a plain hash: it is the
      8-byte big-endian output of ``scrypt(key, salt, n=2**14, r=8, p=1)``
      (the same cost parameters payload.py uses for the encryption key), and
      ``salt`` is a 16-byte digest of the carrier's own samples with their
      two lowest bits shifted away (see content_salt()). Embedding changes
      only those two bits, so the cover and the stego file give the same
      salt and nothing extra has to be stored in the carrier.
    * The payload is then spread over the whole cover instead of packed into
      its first rows. That hides it from a simple look at the top rows of
      the LSB plane and weakens the original chi-square attack of Westfeld &
      Pfitzmann (1999), which scans for a sequentially embedded region; it
      does not help against whole-image statistics (sample pair or RS
      analysis) and makes no difference at full capacity. It is an
      obfuscation step, not encryption: anyone who knows the key (or the
      password it came from) can follow the order.
    * Every prefix of the order is stable, so reading the first 112 bits and
      then the whole frame with the same key gives consistent results.

Why the seed goes through scrypt with a per-cover salt. stego.py uses the
payload password as the scatter key. If the seed were a bare hash of the key
(as in an earlier version: the first 8 bytes of SHA-256(key)), a password
guess could be checked by permuting, reading the 112 header bits and looking
for ``b"STGA"``. That check never runs the memory-hard scrypt that protects
the AES key (measured at roughly 18 times cheaper per guess on a 256x256
cover), and because the
order depended only on (key, number of samples), one table of orders per
image size served every stego file of that size. With the salted scrypt seed
each guess costs one scrypt evaluation for that particular cover, the same
as a guess against the encryption, and two different covers of the same size
get different orders for the same key. The salt is not secret (anyone can
recompute it from the file); it only makes the work per file. Weak passwords
stay weak: this removes the shortcut, it does not add strength. Reusing the
same cover with the same key gives the same order.
"""
from __future__ import annotations

import abc
import hashlib
import os
from pathlib import Path

import numpy as np

from bitutils import bits_to_bytes, bytes_to_bits

# Extensions that are known to destroy LSB data on save (lossy compression or
# palette quantization). save() refuses them with a specific message.
LOSSY_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif",
    ".heic", ".heif", ".avif", ".mp3", ".aac", ".m4a", ".ogg", ".opus",
)


class LossyFormatError(ValueError):
    """Raised when a carrier would be saved in a format that loses LSB data.

    Also raised for unknown output extensions, since the carrier cannot
    promise that an arbitrary format preserves the low bits.
    """


class CapacityError(ValueError):
    """Raised when more bits are requested than the carrier can hold."""


# Key derivation for the scatter seed. The scrypt cost matches payload.py's
# encryption key (SCRYPT_N, SCRYPT_R, SCRYPT_P there), so checking a password
# guess through the scatter order is no cheaper than checking it against the
# AES-GCM key. Changing any of these changes every keyed order, which makes
# files hidden with the old values unreadable.
SCATTER_SCRYPT_N = 2 ** 14
SCATTER_SCRYPT_R = 8
SCATTER_SCRYPT_P = 1
# scrypt needs 128 * r * n * p bytes (16 MiB here); allow headroom so builds
# with a small default memory limit still accept these parameters.
_SCATTER_SCRYPT_MAXMEM = 64 * 1024 * 1024
SCATTER_SALT_LEN = 16
# Domain tag hashed in front of the samples, so the salt cannot collide with
# a digest of the same bytes made for any other purpose.
_SALT_DOMAIN = b"STGA-scatter-salt-v1"
# Embedding changes at most the two lowest bits of a sample
# (bits_per_sample is 1 or 2); the salt only looks at the bits above them.
_SALT_SHIFT = 2


def content_salt(samples: np.ndarray) -> bytes:
    """Derive the per-cover scatter salt from a carrier's samples.

    The digest covers every sample with its two lowest bits shifted away.
    embed() only ever changes those two bits (at 1 or 2 bits per sample),
    so a cover and any stego file made from it give the same salt, and
    reveal() can recompute it without the salt being stored anywhere.
    Samples are hashed as little-endian integers of their own width, so the
    salt is the same on every platform.

    Args:
        samples: The carrier samples (any integer dtype, any shape; they are
            read in C order).

    Returns:
        bytes: ``SCATTER_SALT_LEN`` bytes, the start of
        SHA-256(domain tag | dtype | sample count | high bits of the samples).
    """
    arr = np.asarray(samples).ravel()
    high = arr >> _SALT_SHIFT  # arithmetic shift, so signed samples work too
    canonical = np.ascontiguousarray(high.astype(high.dtype.newbyteorder("<"), copy=False))
    digest = hashlib.sha256(_SALT_DOMAIN)
    digest.update(f"{canonical.dtype.kind}{canonical.dtype.itemsize}".encode("ascii"))
    digest.update(int(arr.size).to_bytes(8, "big"))
    digest.update(canonical.view(np.uint8))
    return digest.digest()[:SCATTER_SALT_LEN]


def key_to_seed(key: str | bytes, salt: bytes) -> int:
    """Derive the 64-bit scatter seed from a key and a per-cover salt.

    Uses scrypt (memory-hard, ``n=2**14, r=8, p=1``) rather than a plain
    hash, so testing one password guess against the scatter order costs as
    much as testing it against the payload's encryption key.

    Args:
        key: The scatter key, usually the payload password. ``str`` keys are
            encoded as UTF-8.
        salt: The carrier's salt, normally content_salt() of its samples.

    Returns:
        int: The 8-byte scrypt output read as a big-endian unsigned int.
    """
    raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    derived = hashlib.scrypt(
        raw,
        salt=bytes(salt),
        n=SCATTER_SCRYPT_N,
        r=SCATTER_SCRYPT_R,
        p=SCATTER_SCRYPT_P,
        maxmem=_SCATTER_SCRYPT_MAXMEM,
        dklen=8,
    )
    return int.from_bytes(derived, "big")


def scatter_order(
    n_samples: int,
    key: str | bytes | None = None,
    salt: bytes | None = None,
) -> np.ndarray:
    """Return the order in which samples are used for ``n_samples`` samples.

    Most callers want Carrier.sample_order(key), which supplies the salt
    from the carrier's own samples.

    Args:
        n_samples: Number of samples in the carrier.
        key: Scatter key. ``None`` or an empty string means natural order.
        salt: The carrier's salt (content_salt() of its samples). Required
            when ``key`` is given.

    Returns:
        np.ndarray: A 1-D ``int64`` array that is a permutation of
        ``range(n_samples)``.

    Raises:
        ValueError: If ``key`` is given without a ``salt``.
    """
    if not key:
        return np.arange(n_samples, dtype=np.int64)
    if salt is None:
        raise ValueError(
            "a keyed scatter order needs the carrier's salt; pass "
            "salt=content_salt(samples) or use Carrier.sample_order(key)"
        )
    rng = np.random.Generator(np.random.PCG64(key_to_seed(key, salt)))
    return rng.permutation(n_samples)


def _as_bit_array(bits: np.ndarray) -> np.ndarray:
    """Validate a bit array and return it as a flat ``uint8`` array.

    Raises:
        ValueError: If ``bits`` holds anything other than 0 and 1.
    """
    arr = np.asarray(bits).ravel()
    if arr.size and ((arr != 0) & (arr != 1)).any():
        raise ValueError("bits may only contain the values 0 and 1")
    return arr.astype(np.uint8, copy=False)


class Carrier(abc.ABC):
    """A cover medium that can hide bits in the LSBs of its samples.

    Subclasses set ``name`` and ``extensions`` and implement load(),
    _samples() and _write(). The public methods capacity_bits(), embed(),
    extract() and save() are implemented here once for every format.

    Attributes:
        name: Short identifier, e.g. ``"image-lsb"``.
        description: One-line human description for listings.
        extensions: Lowercase extensions (with the dot) that save() may
            write. These must be lossless for the samples the carrier uses.
        cover_extensions: Extra extensions load() can read as a cover, on
            top of ``extensions`` (an image carrier can read a JPEG cover
            even though it must never write one). all_extensions() returns
            both, and the registry uses that to pick a carrier for a path.
        bits_per_sample: How many low bits of each sample carry payload, 1 or
            2. Changing it changes capacity_bits() and the layout, so embed
            and extract must use the same value.
    """

    name: str = "carrier"
    description: str = ""
    extensions: tuple[str, ...] = ()
    cover_extensions: tuple[str, ...] = ()

    def __init__(self, bits_per_sample: int = 1) -> None:
        self.bits_per_sample = bits_per_sample
        # (key, n_samples, salt, order) of the last keyed order, so reading a
        # header and then the full frame with the same key runs scrypt and
        # the permutation only once.
        self._order_cache: tuple[str | bytes, int, bytes, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Methods a subclass provides
    # ------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | os.PathLike, bits_per_sample: int = 1) -> "Carrier":
        """Read a cover file and return a carrier for it.

        Args:
            path: File to read.
            bits_per_sample: Initial value of ``bits_per_sample``.
        """

    @abc.abstractmethod
    def _samples(self) -> np.ndarray:
        """Return the carrier samples as a 1-D, writable integer array.

        The array must be a view of the carrier's own data (not a copy), in
        the natural sample order, because embed() writes into it in place.
        """

    @abc.abstractmethod
    def _write(self, path: Path) -> None:
        """Write the carrier to ``path``. The extension is already checked."""

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------
    @property
    def bits_per_sample(self) -> int:
        """Number of low bits used in each sample (1 or 2)."""
        return self._bits_per_sample

    @bits_per_sample.setter
    def bits_per_sample(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"bits_per_sample must be 1 or 2, got {value!r}")
        self._bits_per_sample = int(value)

    @classmethod
    def all_extensions(cls) -> tuple[str, ...]:
        """Every extension this carrier understands (output and cover)."""
        return tuple(dict.fromkeys(cls.extensions + cls.cover_extensions))

    @classmethod
    def check_output_path(cls, path: str | os.PathLike) -> Path:
        """Check that ``path`` has an extension this carrier can save losslessly.

        Callers can use this to fail fast before doing any embedding work.

        Args:
            path: Intended output file.

        Returns:
            Path: ``path`` as a ``pathlib.Path``.

        Raises:
            LossyFormatError: If the extension is lossy or not one of
                ``extensions``.
        """
        out = Path(path)
        ext = out.suffix.lower()
        if ext in cls.extensions:
            return out
        allowed = ", ".join(cls.extensions)
        if ext in LOSSY_EXTENSIONS:
            raise LossyFormatError(
                f"{ext} is a lossy format and would destroy the hidden bits; "
                f"save as one of: {allowed}"
            )
        raise LossyFormatError(
            f"cannot save a {cls.name} carrier as {ext or 'a file without an extension'!r}; "
            f"use one of: {allowed}"
        )

    def num_samples(self) -> int:
        """Number of samples available for embedding."""
        return int(self._samples().size)

    def capacity_bits(self) -> int:
        """Maximum number of payload bits this carrier can hold."""
        return self.num_samples() * self.bits_per_sample

    def scatter_salt(self) -> bytes:
        """Return this carrier's scatter salt (content_salt() of its samples).

        It does not change when bits are embedded, so the cover and the
        stego file give the same value.
        """
        return content_salt(self._samples())

    def sample_order(self, key: str | bytes | None = None) -> np.ndarray:
        """Return the sample visiting order for ``key`` (see scatter_order()).

        The salt is recomputed from the samples on every call (a hash of the
        samples, cheap next to the permutation); the scrypt seed and the
        permutation are cached on the instance for the last (key, salt).
        """
        n = self.num_samples()
        if not key:
            return np.arange(n, dtype=np.int64)
        salt = self.scatter_salt()
        cached = self._order_cache
        if (cached is not None and cached[0] == key and cached[1] == n
                and cached[2] == salt):
            return cached[3]
        order = scatter_order(n, key, salt)
        self._order_cache = (key, n, salt, order)
        return order

    def _positions(self, n_used: int, key: str | bytes | None) -> np.ndarray:
        """Indices of the first ``n_used`` samples in the order for ``key``.

        Without a key this is just ``arange(n_used)``, so a short message in
        a large cover does not allocate an index for every sample.
        """
        if not key:
            return np.arange(n_used, dtype=np.int64)
        return self.sample_order(key)[:n_used]

    def embed(self, bits: np.ndarray, key: str | bytes | None = None) -> None:
        """Write ``bits`` into the low bits of the carrier samples, in place.

        Only the lowest ``bits_per_sample`` bits of the used samples change,
        so each sample moves by at most 1 (or 3 with 2 bits per sample).
        Samples past the end of the payload are left untouched. If the last
        used sample is only partly filled (an odd number of bits with
        ``bits_per_sample == 2``), its unused low bit keeps its old value.

        Args:
            bits: 1-D array of 0/1 values to hide.
            key: Optional scatter key. ``None`` or ``""`` uses the natural
                sample order; any other key uses the keyed permutation
                described in the module docstring.

        Raises:
            CapacityError: If ``len(bits)`` exceeds capacity_bits().
            ValueError: If ``bits`` contains values other than 0 and 1.
        """
        payload = _as_bit_array(bits)
        n_bits = payload.size
        capacity = self.capacity_bits()
        if n_bits > capacity:
            raise CapacityError(
                f"payload needs {n_bits} bits but the carrier holds only {capacity} "
                f"({self.num_samples()} samples x {self.bits_per_sample} bit(s))"
            )
        if n_bits == 0:
            return

        bps = self.bits_per_sample
        samples = self._samples()
        n_used = -(-n_bits // bps)  # ceil division
        positions = self._positions(n_used, key)
        current = samples[positions]

        # Arrange the payload as one row of bps bits per sample. A partly
        # filled last row is padded with that sample's existing low bits so
        # they are rewritten unchanged.
        rows = np.empty(n_used * bps, dtype=np.uint8)
        rows[:n_bits] = payload
        if rows.size > n_bits:
            last = int(current[-1])
            existing = [(last >> (bps - 1 - j)) & 1 for j in range(bps)]
            rows[n_bits:] = existing[n_bits - (n_used - 1) * bps:]
        rows = rows.reshape(n_used, bps)
        weights = (1 << np.arange(bps - 1, -1, -1)).astype(np.uint8)  # [2, 1] or [1]
        values = (rows * weights).sum(axis=1).astype(samples.dtype)

        # (x >> bps) << bps clears the low bits for unsigned and signed
        # dtypes alike (arithmetic shift keeps the sign of negative samples).
        samples[positions] = ((current >> bps) << bps) | values

    def extract(self, n_bits: int, key: str | bytes | None = None) -> np.ndarray:
        """Read ``n_bits`` payload bits back from the carrier.

        Args:
            n_bits: How many bits to read.
            key: The scatter key used by embed() (``None`` or ``""`` for the
                natural order). A different key reads unrelated bits.

        Returns:
            np.ndarray: 1-D ``uint8`` array of 0/1 values, length ``n_bits``.

        Raises:
            CapacityError: If ``n_bits`` exceeds capacity_bits(), e.g. when a
                corrupt header claims an impossible payload length.
            ValueError: If ``n_bits`` is negative.
        """
        n_bits = int(n_bits)
        if n_bits < 0:
            raise ValueError(f"n_bits must be non-negative, got {n_bits}")
        capacity = self.capacity_bits()
        if n_bits > capacity:
            raise CapacityError(
                f"asked for {n_bits} bits but the carrier holds only {capacity}"
            )
        if n_bits == 0:
            return np.zeros(0, dtype=np.uint8)

        bps = self.bits_per_sample
        n_used = -(-n_bits // bps)
        positions = self._positions(n_used, key)
        values = self._samples()[positions]
        shifts = np.arange(bps - 1, -1, -1)  # higher bit first
        bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()
        return bits[:n_bits]

    def embed_bytes(self, data: bytes, key: str | bytes | None = None) -> None:
        """Convenience wrapper: embed() the bits of ``data`` (MSB-first)."""
        self.embed(bytes_to_bits(data), key=key)

    def extract_bytes(self, n_bytes: int, key: str | bytes | None = None) -> bytes:
        """Convenience wrapper: extract() ``8 * n_bytes`` bits as bytes."""
        return bits_to_bytes(self.extract(8 * int(n_bytes), key=key))

    def save(self, path: str | os.PathLike) -> None:
        """Write the carrier to ``path`` in a lossless format.

        The format is chosen from the extension, which must be one of
        ``extensions``. Nothing is written if the check fails.

        Raises:
            LossyFormatError: For lossy extensions (.jpg, .jpeg, .webp, .gif,
                ...) and for any extension not in ``extensions``.
        """
        out = self.check_output_path(path)
        self._write(out)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} samples={self.num_samples()} "
            f"bits_per_sample={self.bits_per_sample} capacity_bits={self.capacity_bits()}>"
        )
