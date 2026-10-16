"""Tests for bruteforce.py: strategy enumeration, extraction, scoring, saving.

Payloads are embedded with a small NumPy helper written here from the layout
definitions (not with bruteforce.extract_bytes), so a test passing means the
scanner agrees with an independent description of each layout.
"""
from __future__ import annotations

import gzip
import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import bruteforce as bf
from bruteforce import Strategy

_CHANNEL = {"R": 0, "G": 1, "B": 2, "A": 3}


def embed(arr: np.ndarray, data: bytes, strategy: Strategy) -> np.ndarray:
    """Hide ``data`` in a copy of ``arr`` using ``strategy``'s layout.

    Sample k of the stream is pixel k // len(channels), channel
    channels[k % len(channels)]; pixels are numbered row-major for "row"
    (x inner) and column-major for "col" (y inner). Each sample carries
    ``depth`` bits, highest plane first. Bytes become bits MSB-first for
    "msb" and LSB-first for "lsb".
    """
    out = arr.copy()
    height, width = out.shape[:2]
    order = "big" if strategy.bit_order == "msb" else "little"
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder=order)
    depth = strategy.depth
    assert bits.size % depth == 0
    n_ch = len(strategy.channels)
    k = np.arange(bits.size // depth)
    pixel = k // n_ch
    ch = np.array([_CHANNEL[c] for c in strategy.channels])[k % n_ch]
    if strategy.traversal == "row":
        ys, xs = pixel // width, pixel % width
    else:
        ys, xs = pixel % height, pixel // height
    planes = bits.reshape(-1, depth)  # one row per sample, highest plane first
    values = out[ys, xs, ch].astype(np.int64)
    for j in range(depth):
        plane = strategy.bit + depth - 1 - j
        values = (values & ~(1 << plane)) | (planes[:, j].astype(np.int64) << plane)
    out[ys, xs, ch] = values.astype(np.uint8)
    return out


def tiny_png(seed: int = 3, size: int = 8) -> bytes:
    """Encode a small random RGB image as PNG bytes."""
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Strategy space and extraction
# ---------------------------------------------------------------------------

def test_enumerate_strategies_counts_and_order():
    single = bf.enumerate_strategies(1, two_bit=False)
    assert len(single) == 9 * 2 * 2 * 2
    assert single[0] == Strategy("RGB", 0, "msb", "row")  # legacy layout first
    assert len(bf.enumerate_strategies(0)) == 9 * 2 * 2
    with_alpha = bf.enumerate_strategies(1, has_alpha=True, two_bit=False)
    assert {s.channels for s in with_alpha} >= {"RGBA", "BGRA"}
    assert len(with_alpha) == 11 * 2 * 2 * 2
    default = bf.enumerate_strategies()
    assert len(default) == len(single) + 9 * 2 * 2
    assert Strategy("RGB", 0, "msb", "row", depth=2) in default
    assert len(set(default)) == len(default)
    with pytest.raises(ValueError):
        bf.enumerate_strategies(8)


@pytest.mark.parametrize("kwargs", [
    {"channels": ""}, {"channels": "RX"}, {"channels": "RR"}, {"bit": 8},
    {"bit_order": "big"}, {"traversal": "diag"}, {"bit": 7, "depth": 2},
])
def test_strategy_rejects_bad_fields(kwargs):
    with pytest.raises(ValueError):
        Strategy(**kwargs)


def test_strategy_labels():
    s = Strategy("BGR", 0, "lsb", "col")
    assert str(s) == s.label == "BGR bit0 lsb-first col"
    assert Strategy(depth=2).label == "RGB bits1-0 msb-first row"
    assert s.to_dict() == {"channels": "BGR", "bit": 0, "bit_order": "lsb",
                           "traversal": "col", "depth": 1}


@pytest.mark.parametrize("strategy", [
    Strategy("RGB", 0, "msb", "row"),
    Strategy("BGR", 0, "lsb", "col"),
    Strategy("G", 1, "msb", "col"),
    Strategy("BRG", 2, "lsb", "row"),
    Strategy("GBR", 0, "msb", "row", depth=2),
    Strategy("R", 0, "lsb", "col", depth=2),
])
def test_extract_bytes_inverts_reference_embedding(synthetic_rgb, strategy):
    arr = synthetic_rgb(11, 32)
    data = bytes(np.random.default_rng(5).integers(0, 256, 100, dtype=np.uint8))
    stego = embed(arr, data, strategy)
    assert bf.extract_bytes(stego, strategy, len(data)) == data


def test_extract_bytes_caps_at_capacity_and_checks_alpha(synthetic_rgb):
    arr = synthetic_rgb(0, 16)  # 16 * 16 * 3 bits = 96 bytes
    assert len(bf.extract_bytes(arr, Strategy(), 10_000)) == 96
    assert len(bf.extract_bytes(arr, Strategy("G"), 10_000)) == 32
    assert bf.extract_bytes(arr, Strategy(), 0) == b""
    with pytest.raises(ValueError):
        bf.extract_bytes(arr, Strategy("RGBA"), 4)


# ---------------------------------------------------------------------------
# End-to-end recovery
# ---------------------------------------------------------------------------

def test_bruteforce_finds_nondefault_strategy_and_text(synthetic_rgb, save_png):
    text = b"Meet at the old bridge at midnight. Bring the documents."
    strategy = Strategy("BGR", 0, "lsb", "col")
    path = save_png(embed(synthetic_rgb(1), text + b"\x00", strategy), "stego.png")

    results = bf.bruteforce(path, top_k=5)

    top = results[0]
    assert top["strategy"] == strategy
    assert top["kind"] == "text"
    assert top["data"] == text
    assert top["preview"].startswith("Meet at the old bridge")
    assert top["score"] > 0.7
    # The right layout stands out: every other layout reads noise.
    assert results[1]["score"] < 0.3
    # Candidates carry everything a report needs, and info is JSON-friendly.
    assert {"strategy", "label", "score", "kind", "preview", "data", "info"} <= set(top)
    assert top["info"]["format"] == "nul-terminated"


def test_unterminated_text_is_recovered_as_prefix(synthetic_rgb):
    text = b"The quick brown fox jumps over the lazy dog, twice."
    strategy = Strategy("GRB", 0, "msb", "row")
    stego = embed(synthetic_rgb(2), text, strategy)  # no terminator at all

    top = bf.bruteforce(stego, top_k=3)[0]

    assert top["strategy"] == strategy
    assert top["kind"] == "text"
    # The cover's own LSBs follow the message, so a few stray printable
    # bytes may be appended; the message itself must come back intact.
    assert top["data"].startswith(text)


def test_hidden_png_file_is_typed_and_saved(synthetic_rgb, save_png, tmp_path):
    png = tiny_png()
    strategy = Strategy("GRB", 1, "msb", "row")  # bit plane 1, not the LSB
    path = save_png(embed(synthetic_rgb(4), png, strategy), "stego.png")

    top = bf.bruteforce(path, top_k=3)[0]

    assert top["strategy"] == strategy
    assert top["kind"] == "file:png"
    assert top["info"]["complete"] is True
    assert top["data"] == png  # cut exactly at IEND, no trailing noise
    assert (top["info"]["width"], top["info"]["height"]) == (8, 8)

    saved = bf.save_candidate(top, tmp_path / "recovered")
    assert saved.endswith(".png")
    with Image.open(saved) as img:
        img.verify()
    with Image.open(saved) as img, Image.open(io.BytesIO(png)) as original:
        assert np.array_equal(np.asarray(img), np.asarray(original))


def test_file_longer_than_window_is_read_in_full(synthetic_rgb):
    # A 48x48 BMP is about 7 KB, longer than the 4096-byte first pass; its
    # header gives the exact size, so bruteforce() reads exactly that much.
    buf = io.BytesIO()
    Image.fromarray(synthetic_rgb(9, 48)).save(buf, "BMP")
    bmp = buf.getvalue()
    assert len(bmp) > 4096
    strategy = Strategy("RGB", 0, "msb", "col")

    top = bf.bruteforce(embed(synthetic_rgb(5), bmp, strategy), top_k=3)[0]

    assert top["strategy"] == strategy
    assert top["kind"] == "file:bmp"
    assert top["data"] == bmp
    assert top["info"]["truncated"] is False


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_image_scores_low(synthetic_rgb, seed):
    results = bf.bruteforce(synthetic_rgb(seed), top_k=5)
    assert results[0]["score"] < 0.3
    assert all(r["kind"] in ("text", "binary") for r in results)


def test_legacy_format_is_recognized_without_truncation(synthetic_rgb):
    # encoder.py's layout and delimiter: RGB, bit 0, MSB first, row-major,
    # then 64 zero bits. "@" (0x40) ends in six zero bits, which a search at
    # any bit offset would merge into the delimiter and drop.
    message = b"Hello World 1234 @"
    stego = embed(synthetic_rgb(3), message + b"\x00" * 8, Strategy())

    top = bf.bruteforce(stego, top_k=3)[0]

    assert top["strategy"] == Strategy("RGB", 0, "msb", "row")
    assert top["kind"] == "text"
    assert top["info"]["format"] == "legacy-delimiter"
    assert top["data"] == message
    assert top["score"] >= 0.5


def test_legacy_latin1_text(synthetic_rgb, tmp_path):
    # encoder.py stores code points 128..255 as one Latin-1 byte each.
    message = "Café au lait, crème brûlée"
    stego = embed(synthetic_rgb(6), message.encode("latin-1") + b"\x00" * 8, Strategy())

    top = bf.bruteforce(stego, top_k=3)[0]

    assert top["info"]["format"] == "legacy-delimiter"
    assert top["info"]["encoding"] == "latin-1"
    saved = bf.save_candidate(top, tmp_path)
    assert saved.endswith(".txt")
    with open(saved, encoding="utf-8") as fh:
        assert fh.read() == message


def test_stga_frame_is_verified_and_expanded(synthetic_rgb, tmp_path):
    body = ("frame body " * 30).encode()
    frame = struct.pack(">4sBBII", b"STGA", 1, 0, len(body), zlib.crc32(body)) + body
    stego = embed(synthetic_rgb(7), frame, Strategy("RGB", 0, "msb", "row"))

    # A 64-byte first pass sees only the header; the frame length in the
    # header makes bruteforce() read the rest and check the CRC.
    top = bf.bruteforce(stego, top_k=3, max_bytes=64)[0]

    assert top["kind"] == "stga"
    assert top["score"] == 1.0
    assert top["data"] == frame
    assert top["info"]["crc_ok"] is True
    assert top["info"]["body_len"] == len(body)
    assert top["info"]["truncated"] is False
    saved = bf.save_candidate(top, tmp_path)
    assert Path(saved).read_bytes() in (body, frame)  # unpacked when payload.py loads


def test_long_text_is_followed_past_the_window(synthetic_rgb):
    # About 6 KB: more than the 1 KB first pass and the default 4 KB, but
    # within the 8 KB a single channel of a 256x256 image holds.
    text = " ".join(f"line {i}: nothing to see here" for i in range(200)).encode()
    assert 4 * 1024 < len(text) < 8 * 1024
    strategy = Strategy("B", 0, "msb", "row")
    stego = embed(synthetic_rgb(8), text + b"\x00", strategy)

    top = bf.bruteforce(stego, top_k=3, max_bytes=1024)[0]

    assert top["strategy"] == strategy
    assert top["data"] == text


def test_alpha_layouts_are_tried_for_rgba_images(synthetic_rgb, save_png):
    rgba = np.dstack([synthetic_rgb(10), np.full((256, 256), 255, np.uint8)])
    text = b"hidden in the alpha channel too, row by row"
    strategy = Strategy("RGBA", 0, "msb", "row")
    path = save_png(embed(rgba, text + b"\x00", strategy), "stego_rgba.png")

    top = bf.bruteforce(path, top_k=3)[0]

    assert top["strategy"] == strategy
    assert top["data"] == text


def test_two_bit_layout(synthetic_rgb):
    text = b"Two bits per sample, highest plane first, like bits_per_sample=2."
    strategy = Strategy("RGB", 0, "msb", "row", depth=2)
    top = bf.bruteforce(embed(synthetic_rgb(12), text + b"\x00", strategy), top_k=3)[0]
    assert top["strategy"] == strategy
    assert top["data"] == text


# ---------------------------------------------------------------------------
# score_candidate and save_candidate
# ---------------------------------------------------------------------------

def test_score_candidate_noise_is_low_binary():
    noise = bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8))
    result = bf.score_candidate(noise)
    assert result["kind"] == "binary"
    assert result["score"] <= 0.05
    assert result["info"]["entropy"] > 7.9


def test_score_candidate_rejects_repetitive_printable_patterns():
    # Smooth gradients can make an LSB plane decode to repeating printable
    # bytes; those must not look like a message.
    assert bf.score_candidate(b"UUUU" * 256)["score"] < 0.1
    assert bf.score_candidate(b"~@xxHz" * 100)["score"] < 0.1
    assert bf.score_candidate(b"\x00" * 4096)["score"] < 0.1
    assert bf.score_candidate(b"\xff" * 4096)["score"] < 0.1


@pytest.mark.parametrize("data, kind", [
    (b"A perfectly ordinary sentence.\x00", "text"),
    (tiny_png(), "file:png"),
    (b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF\n", "file:pdf"),
    (b"\x1f\x8b\x08\x00" + b"\x00" * 6, "file:gz"),
    (zlib.compress(b"x") + b"\x00", "binary"),
])
def test_score_candidate_kinds_and_safe_previews(data, kind):
    result = bf.score_candidate(data)
    assert result["kind"] == kind
    assert len(result["preview"]) <= 120
    assert result["preview"].isprintable()


def test_gzip_member_is_cut_at_its_end():
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as fh:
        fh.write(b"compressed secret " * 20)
    blob = gz.getvalue()
    result = bf.score_candidate(blob + bytes(range(256)))
    assert result["kind"] == "file:gz"
    assert result["data"] == blob
    assert result["info"]["complete"] is True


def test_save_candidate_never_overwrites(tmp_path):
    cand = {"strategy": Strategy(), "kind": "binary", "data": b"\x01\x02", "info": {}}
    first = bf.save_candidate(cand, tmp_path)
    second = bf.save_candidate(cand, tmp_path)
    assert first != second
    assert first.endswith(".bin") and second.endswith(".bin")


# ---------------------------------------------------------------------------
# Integration with the carriers package and payload.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits_per_sample", [1, 2])
def test_finds_unkeyed_carrier_payload(synthetic_rgb, save_png, tmp_path, bits_per_sample):
    carriers = pytest.importorskip("carriers")
    payload = pytest.importorskip("payload")
    bitutils = pytest.importorskip("bitutils")
    cover = save_png(synthetic_rgb(13), "cover.png")
    frame = payload.pack(tiny_png(), filename="secret.png", compress=True)

    carrier = carriers.carrier_for_path(cover).load(cover)
    carrier.bits_per_sample = bits_per_sample
    carrier.embed(bitutils.bytes_to_bits(frame))
    stego = tmp_path / "stego.png"
    carrier.save(stego)

    top = bf.bruteforce(stego, top_k=3)[0]

    assert top["kind"] == "stga"
    assert top["strategy"] == Strategy("RGB", 0, "msb", "row", depth=bits_per_sample)
    assert top["data"] == frame
    assert top["info"]["has_filename"] and top["info"]["compressed"]
    saved = bf.save_candidate(top, tmp_path / "out")
    assert saved.endswith("secret.png")
    assert Path(saved).read_bytes() == tiny_png()


def test_dictionary_attack_finds_scatter_password(synthetic_rgb, save_png, tmp_path):
    carriers = pytest.importorskip("carriers")
    payload = pytest.importorskip("payload")
    bitutils = pytest.importorskip("bitutils")
    cover = save_png(synthetic_rgb(14), "cover.png")
    password = "hunter2"
    frame = payload.pack("meet me at the usual place", password=password)

    carrier = carriers.carrier_for_path(cover).load(cover)
    carrier.embed(bitutils.bytes_to_bits(frame), key=password)
    stego = tmp_path / "stego.png"
    carrier.save(stego)

    # Scattered with a key, the payload is invisible to sequential layouts.
    assert bf.bruteforce(stego, top_k=1)[0]["score"] < 0.3

    hits = bf.dictionary_attack(stego, ["123456", "password", "", "letmein\n", password])

    assert len(hits) == 1
    hit = hits[0]
    assert hit["password"] == password
    assert hit["bits_per_sample"] == 1
    assert hit["score"] == 1.0
    assert hit["data"] == b"meet me at the usual place"
    assert hit["info"]["encrypted"] is True
    saved = bf.save_candidate(hit, tmp_path / "out")
    assert Path(saved).read_bytes() == b"meet me at the usual place"
    assert bf.dictionary_attack(stego, ["wrong", "guesses"]) == []
