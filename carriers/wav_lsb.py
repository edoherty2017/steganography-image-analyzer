"""LSB carrier for uncompressed PCM WAV audio.

Samples are the PCM sample values in file order, i.e. frame by frame with the
channels of each frame interleaved (left, right, left, right, ... for
stereo). Supported sample widths:

    * 8-bit  (sampwidth 1): unsigned, 0..255, silence at 128.
    * 16-bit (sampwidth 2): signed little-endian, -32768..32767.

Changing the LSB of a 16-bit sample alters it by 1/65536 of full scale, far
below audibility; for 8-bit audio the change is 1/256 of full scale, the same
size as the format's own quantization step, so it adds noise on the order of
what 8-bit audio already has. Only the stdlib ``wave``
module and NumPy are used, so there are no audio dependencies. All WAV
parameters (channels, sample width, frame rate, frame count) are written back
unchanged.

Truncated files (an interrupted download or recording) are handled as
follows. If the audio data stops part-way through a frame, the trailing
partial frame is dropped and the whole frames before it are used, so the
file still works as a cover (and a payload hidden before the cut can still
be read). The frame count is then taken from the data actually present, not
from the header, which still declares the original length. If the file is
cut off inside the header itself, load() raises ``wave.Error`` saying so.

This carrier is the proof that the ``Carrier`` interface is not image
specific: the same embed/extract/scatter code from ``carriers/base.py``
works on audio unchanged.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from . import register
from .base import Carrier

# NumPy dtype for each supported WAV sample width (bytes per sample).
_DTYPES = {1: np.dtype(np.uint8), 2: np.dtype("<i2")}


@register
class WavLSBCarrier(Carrier):
    """Hide bits in the low bits of PCM WAV samples.

    Args:
        samples: 1-D array of interleaved samples (``uint8`` for 8-bit,
            ``int16`` for 16-bit). It is copied.
        params: The ``wave._wave_params`` tuple from ``Wave_read.getparams()``.
        bits_per_sample: Low bits used per sample, 1 or 2.

    Attributes:
        samples: The interleaved samples; embed() modifies them in place.
        params: The original WAV parameters, reused by save().
    """

    name = "wav-lsb"
    description = "LSB replacement in 8-bit / 16-bit PCM WAV samples"
    extensions = (".wav",)

    def __init__(self, samples: np.ndarray, params: wave._wave_params,
                 bits_per_sample: int = 1) -> None:
        super().__init__(bits_per_sample)
        if params.sampwidth not in _DTYPES:
            raise ValueError(
                f"unsupported WAV sample width: {params.sampwidth * 8}-bit "
                "(only 8-bit and 16-bit PCM are supported)"
            )
        dtype = _DTYPES[params.sampwidth]
        arr = np.asarray(samples)
        if arr.ndim != 1:
            raise ValueError(f"samples must be 1-D (interleaved), got shape {arr.shape}")
        self.samples = arr.astype(dtype, copy=True)
        self.params = params

    @classmethod
    def load(cls, path: str | os.PathLike, bits_per_sample: int = 1) -> "WavLSBCarrier":
        """Read a PCM WAV file as a cover.

        If the audio data ends part-way through a frame (a truncated file),
        the trailing partial frame is dropped and ``params.nframes`` is set
        to the number of whole frames actually read.

        Args:
            path: The ``.wav`` file.
            bits_per_sample: Low bits used per sample, 1 or 2.

        Returns:
            WavLSBCarrier: A carrier holding a copy of the samples.

        Raises:
            ValueError: For sample widths other than 8 or 16 bits.
            wave.Error: If the file is not a PCM WAV file, or is truncated
                inside its header.
        """
        try:
            with wave.open(os.fspath(path), "rb") as w:
                params = w.getparams()
                if params.sampwidth not in _DTYPES:
                    raise ValueError(
                        f"unsupported WAV sample width: {params.sampwidth * 8}-bit "
                        "(only 8-bit and 16-bit PCM are supported)"
                    )
                frames = w.readframes(params.nframes)
        except EOFError as exc:
            # The stdlib raises a bare EOFError (no message) when the RIFF
            # header or a chunk header is cut short. Callers already handle
            # wave.Error for unreadable WAV files, so report it as one.
            raise wave.Error(
                "truncated WAV file: it ends inside the header, before the audio data"
            ) from exc
        # A file cut off mid-sample or mid-frame returns a byte count that is
        # not a whole number of frames, and np.frombuffer would fail with an
        # unhelpful "buffer size must be a multiple of element size". Keep the
        # whole frames (not just whole samples, so stereo channels stay
        # aligned) and record how many there really are; the header's nframes
        # still states the original, untruncated length.
        frame_size = params.sampwidth * params.nchannels
        whole = len(frames) - len(frames) % frame_size
        if whole != len(frames) or whole // frame_size != params.nframes:
            frames = frames[:whole]
            params = params._replace(nframes=whole // frame_size)
        # frombuffer returns a read-only view of the bytes; __init__ copies it.
        samples = np.frombuffer(frames, dtype=_DTYPES[params.sampwidth])
        return cls(samples, params, bits_per_sample=bits_per_sample)

    @property
    def n_channels(self) -> int:
        """Number of audio channels."""
        return self.params.nchannels

    @property
    def frame_rate(self) -> int:
        """Sample rate in Hz."""
        return self.params.framerate

    def _samples(self) -> np.ndarray:
        return self.samples

    def _write(self, path: Path) -> None:
        with wave.open(os.fspath(path), "wb") as w:
            w.setparams(self.params)
            w.writeframes(self.samples.astype(_DTYPES[self.params.sampwidth]).tobytes())
