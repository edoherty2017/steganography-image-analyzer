"""Carrier plug-in registry.

Each cover format is a ``Carrier`` subclass (see ``carriers/base.py``) that
registers itself with the ``@register`` decorator when its module is
imported. The rest of the project never names a concrete carrier class; it
asks the registry::

    from carriers import carrier_for_path

    carrier = carrier_for_path("cover.png").load("cover.png")
    carrier.embed(bits, key="password")
    carrier.save("stego.png")

Built-in carriers:
    image_lsb.ImageLSBCarrier  PNG / BMP / TIFF output (reads JPEG, GIF and
                               other Pillow formats as covers).
    wav_lsb.WavLSBCarrier      8-bit and 16-bit PCM WAV audio.

Adding a format (for example a video carrier or another LSB variant) means
writing one module with a registered ``Carrier`` subclass and importing it at
the bottom of this file.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import CapacityError, Carrier, LossyFormatError

__all__ = [
    "CapacityError",
    "Carrier",
    "LossyFormatError",
    "carrier_for_path",
    "list_carriers",
    "register",
    "unregister",
]

# Registered carrier classes, in registration order. carrier_for_path()
# returns the first match, so earlier registrations win ties.
_REGISTRY: list[type[Carrier]] = []


def register(cls: type[Carrier]) -> type[Carrier]:
    """Class decorator that adds a ``Carrier`` subclass to the registry.

    Registering the same class twice is a no-op.

    Args:
        cls: The carrier class.

    Returns:
        The class unchanged, so it can be used as ``@register``.

    Raises:
        TypeError: If ``cls`` is not a ``Carrier`` subclass.
    """
    if not (isinstance(cls, type) and issubclass(cls, Carrier)):
        raise TypeError(f"register() expects a Carrier subclass, got {cls!r}")
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def unregister(cls: type[Carrier]) -> None:
    """Remove a carrier class from the registry (no error if absent)."""
    if cls in _REGISTRY:
        _REGISTRY.remove(cls)


def list_carriers() -> list[type[Carrier]]:
    """Return the registered carrier classes, in registration order."""
    return list(_REGISTRY)


def carrier_for_path(path: str | os.PathLike) -> type[Carrier]:
    """Pick the carrier class for a file from its extension.

    Both output extensions (``Carrier.extensions``) and cover-only
    extensions (``Carrier.cover_extensions``) match, so ``"photo.jpg"``
    resolves to the image carrier even though it can only be read as a
    cover. Output extensions are checked first across all carriers.

    Args:
        path: File name or path. The extension is compared case-insensitively.

    Returns:
        The matching ``Carrier`` subclass (not an instance; call ``.load()``).

    Raises:
        ValueError: If no registered carrier handles the extension.
    """
    ext = Path(path).suffix.lower()
    for cls in _REGISTRY:
        if ext in cls.extensions:
            return cls
    for cls in _REGISTRY:
        if ext in cls.cover_extensions:
            return cls
    known = sorted({e for cls in _REGISTRY for e in cls.all_extensions()})
    raise ValueError(
        f"no carrier supports {ext or 'files without an extension'!r} "
        f"(supported: {', '.join(known)})"
    )


# Import the built-in carriers last so they can import register() from this
# module; importing them registers them.
from . import image_lsb, wav_lsb  # noqa: E402,F401
