"""Tests for bruteforce.py: strategy enumeration, extraction, scoring, saving.

Payloads are embedded with a small NumPy helper written here from the layout
definitions (not with bruteforce.extract_bytes), so a test passing means the
scanner agrees with an independent description of each layout.
"""
from __future__ import annotations

import gzip
import io
import os
import struct
import zipfile
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


def linearized_pdf(seed: int = 0, body_len: int = 50_000) -> bytes:
    """A PDF shaped like a linearized ("Fast Web View") file.

    The first-page trailer and its "%%EOF" come in the first ~170 bytes;
    the real end of the file is a second "%%EOF" after a long stream.
    """
    head = (b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n"
            b"1 0 obj <</Linearized 1/L 60000/O 3/E 2000/N 1/T 59000/H [500 200]>>\n"
            b"endobj\nxref\n1 1\n0000000016 00000 n \ntrailer <</Size 2/Prev 59000>>\n"
            b"startxref\n0\n%%EOF\n")
    stream = np.random.default_rng(seed).integers(0, 256, body_len, dtype=np.uint8).tobytes()
    return (head + b"2 0 obj <</Length 50000>> stream\n" + stream
            + b"\nendstream endobj\ntrailer <</Size 3>>\nstartxref\n116\n%%EOF\n")


def test_pdf_with_early_eof_marker_is_read_to_the_last_one(synthetic_rgb, tmp_path):
    # Regression: the first-page "%%EOF" of a linearized PDF used to be
    # taken as the end, so a 50 KB PDF was reported complete at 169 bytes.
    pdf = linearized_pdf()
    assert pdf.index(b"%%EOF") < 4096 < len(pdf)
    strategy = Strategy("RGB", 0, "msb", "row")
    stego = embed(synthetic_rgb(8, 512), pdf, strategy)

    top = bf.bruteforce(stego, top_k=1)[0]

    assert top["strategy"] == strategy
    assert top["kind"] == "file:pdf"
    assert top["data"] == pdf
    assert top["info"]["complete"] is True
    assert top["info"]["truncated"] is False
    assert Path(bf.save_candidate(top, tmp_path)).read_bytes() == pdf

    # When expansion is capped before the layout's capacity, the result
    # says it may be truncated instead of claiming to be complete.
    capped = bf.bruteforce(stego, top_k=1, max_expand_bytes=20_000)[0]
    assert capped["kind"] == "file:pdf"
    assert capped["info"]["complete"] is False
    assert capped["info"]["truncated"] is True


def test_pdf_in_a_window_is_tentative_until_the_end():
    pdf = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF\n"
    window = bf.score_candidate(pdf + b"\x00" * 64)
    assert window["kind"] == "file:pdf"
    assert window["data"] == pdf
    assert window["info"]["complete"] is False
    assert window["info"]["needs_bytes"] > window["info"]["window_bytes"]

    final = bf.score_candidate(pdf + b"\x00" * 64, at_end=True)
    assert final["data"] == pdf
    assert final["info"]["complete"] is True
    assert "needs_bytes" not in final["info"]


def nested_zip(seed: int = 0) -> bytes:
    """A stored ZIP whose first member is itself a small ZIP."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("note.txt", "inner archive")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("inner.zip", inner.getvalue())
        zf.writestr("data.bin", np.random.default_rng(seed).integers(
            0, 256, 30_000, dtype=np.uint8).tobytes())
    return outer.getvalue()


def test_zip_containing_a_zip_is_not_cut_at_the_inner_end(synthetic_rgb, tmp_path):
    # Regression: the inner archive's end record (in the first 4096 bytes)
    # used to end the outer one.
    zb = nested_zip()
    strategy = Strategy("BGR", 0, "msb", "col")
    stego = embed(synthetic_rgb(9, 512), zb, strategy)

    top = bf.bruteforce(stego, top_k=1)[0]

    assert top["strategy"] == strategy
    assert top["kind"] == "file:zip"
    assert top["data"] == zb
    assert top["info"]["truncated"] is False
    saved = bf.save_candidate(top, tmp_path)
    with zipfile.ZipFile(saved) as zf:
        assert zf.testzip() is None
        assert sorted(zf.namelist()) == ["data.bin", "inner.zip"]


def test_zip_end_record_that_checks_out_is_final():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("x.txt", "hello " * 100)
        zf.comment = b"archive comment"
    zb = buf.getvalue()
    result = bf.score_candidate(zb + bytes(range(200)))
    assert result["kind"] == "file:zip"
    assert result["data"] == zb  # cut after the comment
    assert result["info"]["complete"] is True


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


@pytest.mark.parametrize("message", [
    "Пароль: hunter2",
    "Встреча в 9:00 у входа 3, код 4417",
    "Κωδικός: 4417, πύλη Β3",
    "كلمة السر: 4417",
])
def test_legacy_mixed_script_messages_are_recognized(synthetic_rgb, message):
    # Regression: non-Latin text mixed with digits or Latin words puts 10% to
    # 65% of its characters in U+0250-U+07FF, which the script-consistency
    # term used to treat as UTF-8 noise and score 0. report.py needs 0.5
    # (its _LEGACY_MIN_SCORE) to recognise the legacy payload.
    data = message.encode("utf-8")
    stego = embed(synthetic_rgb(4), data + b"\x00" * 8, Strategy())

    top = bf.bruteforce(stego, top_k=3)[0]

    assert top["strategy"] == Strategy()
    assert top["info"]["format"] == "legacy-delimiter"
    assert top["data"] == data
    assert top["score"] >= 0.5


def test_scattered_two_byte_characters_still_score_low():
    # The other side of the test above: two-byte characters from random
    # Unicode blocks among ASCII letters and digits (about 35% of the run,
    # with far more adjacent pairs than real noise produces) must not pass
    # as a mixed-script message.
    pool = "abcdefghijklmnopqrstuvwxyz0123456789 "
    for seed in range(200):
        rng = np.random.default_rng(seed)
        text = "".join(chr(int(rng.integers(0x370, 0x800))) if rng.random() < 0.35
                       else pool[int(rng.integers(len(pool)))] for _ in range(40))
        result = bf.score_candidate(text.encode("utf-8") + b"\x00" * 8)
        assert result["score"] < 0.3, text


@pytest.mark.parametrize("version", [1, 2])
def test_stga_frame_is_verified_and_expanded(synthetic_rgb, tmp_path, version):
    body = ("frame body " * 30).encode()
    prefix = struct.pack(">4sBBI", b"STGA", version, 0, len(body))
    # Version 2 (what payload.py writes) checksums the header prefix and the
    # body; version 1 (development frames) checksummed the body alone.
    crc = zlib.crc32(prefix + body) if version == 2 else zlib.crc32(body)
    frame = prefix + struct.pack(">I", crc) + body
    stego = embed(synthetic_rgb(7), frame, Strategy("RGB", 0, "msb", "row"))

    # A 64-byte first pass sees only the header; the frame length in the
    # header makes bruteforce() read the rest and check the CRC.
    top = bf.bruteforce(stego, top_k=3, max_bytes=64)[0]

    assert top["kind"] == "stga"
    assert top["score"] == 1.0
    assert top["data"] == frame
    assert top["info"]["version"] == version
    assert top["info"]["crc_ok"] is True
    assert top["info"]["body_len"] == len(body)
    assert top["info"]["truncated"] is False

    # Exactly one outcome per version. payload.py is part of this repo, so a
    # version 2 frame is always unpacked: the file holds the body (printable,
    # hence .txt), never the frame with its 14-byte header. payload.unpack()
    # rejects version 1, so that frame is kept whole as .stga.
    saved = Path(bf.save_candidate(top, tmp_path))
    if version == 2:
        assert saved.suffix == ".txt"
        assert saved.read_bytes() == body
    else:
        assert saved.suffix == ".stga"
        assert saved.read_bytes() == frame


def test_encrypted_stga_frame_is_saved_raw(tmp_path):
    # Without the password the frame cannot be unpacked, so save_candidate()
    # keeps the whole frame, header included, as .stga for a later unpack.
    import payload

    frame = payload.pack("attack at dawn", password="correct horse")
    cand = bf.score_candidate(frame + b"\x00" * 32)
    assert cand["kind"] == "stga"
    assert cand["data"] == frame
    cand["strategy"] = Strategy("RGB", 0, "msb", "row")

    saved = Path(bf.save_candidate(cand, tmp_path))
    assert saved.suffix == ".stga"
    assert saved.read_bytes() == frame
    assert payload.unpack(saved.read_bytes(), password="correct horse").text() == "attack at dawn"


def test_version_2_frame_crc_covers_the_header(synthetic_rgb):
    # payload.py writes version 2 frames, whose CRC-32 covers the 10 header
    # bytes before the CRC field as well as the body.
    body = np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8).tobytes()
    prefix = struct.pack(">4sBBI", b"STGA", 2, 0, len(body))
    frame = prefix + struct.pack(">I", zlib.crc32(prefix + body)) + body

    result = bf.score_candidate(frame + b"\x00" * 32)
    assert result["kind"] == "stga"
    assert result["score"] == 1.0
    assert result["info"]["version"] == 2
    assert result["data"] == frame

    # A frame whose CRC covers the body alone is not a valid version 2 frame.
    bad = prefix + struct.pack(">I", zlib.crc32(body)) + body
    result = bf.score_candidate(bad + b"\x00" * 32)
    assert result["info"]["crc_ok"] is False
    assert result["score"] < 0.6


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


def test_save_candidate_does_not_follow_a_planted_symlink(tmp_path):
    # Regression: a dangling symlink at the predictable output name made
    # Path.exists() return False and the write follow the link out of
    # out_dir.
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "OUTSIDE.txt"
    link = out / "bruteforce_RGB_bit0_msb_row.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    cand = bf.score_candidate(b"Meet me at the old mill at midnight." + b"\x00" * 9)
    cand["strategy"] = Strategy()
    assert cand["kind"] == "text"

    saved = Path(bf.save_candidate(cand, out))

    assert saved == out / "bruteforce_RGB_bit0_msb_row_2.txt"
    assert not saved.is_symlink()
    assert saved.read_text(encoding="utf-8") == "Meet me at the old mill at midnight."
    assert not target.exists()
    assert link.is_symlink()


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


def scattered_stego(tmp_path: Path, cover: np.ndarray, message: str, password: str) -> Path:
    """Hide an encrypted frame scattered with ``password`` (stego.py's layout)."""
    carriers = pytest.importorskip("carriers")
    payload = pytest.importorskip("payload")
    bitutils = pytest.importorskip("bitutils")
    cover_path = tmp_path / "cover.png"
    Image.fromarray(cover).save(cover_path)
    carrier = carriers.carrier_for_path(cover_path).load(cover_path)
    carrier.embed(bitutils.bytes_to_bits(payload.pack(message, password=password)), key=password)
    stego = tmp_path / "stego.png"
    carrier.save(stego)
    return stego


def test_dictionary_attack_ignores_a_byte_order_mark(synthetic_rgb, tmp_path):
    # Regression: a word list saved as "UTF-8 with BOM" and read as plain
    # UTF-8 starts with U+FEFF, so its first password never matched.
    stego = scattered_stego(tmp_path, synthetic_rgb(2), "attack at dawn", "tiger")
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("tiger\nlion\n", encoding="utf-8-sig")

    with open(wordlist, encoding="utf-8") as fh:
        hits = bf.dictionary_attack(stego, fh)

    assert [h["password"] for h in hits] == ["tiger"]
    assert hits[0]["data"] == b"attack at dawn"


# ---------------------------------------------------------------------------
# Command line (python bruteforce.py ...)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args, message", [
    (["--max-bit", "9"], "--max-bit must be 0..7"),
    (["--max-bit", "-1"], "--max-bit must be 0..7"),
    (["--max-bytes", "0"], "--max-bytes must be positive"),
    (["--top", "0"], "--top must be at least 1"),
])
def test_cli_rejects_out_of_range_options(synthetic_rgb, save_png, capsys, args, message):
    # Regression: these used to end in a ValueError traceback.
    path = save_png(synthetic_rgb(0, 32), "clean.png")
    with pytest.raises(SystemExit) as exc:
        bf._main([str(path), *args])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err and message in err


def test_cli_reports_a_missing_image_without_a_traceback(tmp_path, capsys):
    assert bf._main([str(tmp_path / "missing.png")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_cli_save_with_wordlist_saves_the_password_hit(synthetic_rgb, tmp_path, capsys):
    # Regression: --save wrote the top layout candidate (noise, since the
    # payload is scattered) even when the word list found the password.
    # main.py bruteforce saves the hit; the two entry points now agree.
    stego = scattered_stego(tmp_path, synthetic_rgb(3), "The key is under the mat.",
                            "correct horse")
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("123456\ncorrect horse\n", encoding="utf-8-sig")
    out = tmp_path / "saved"

    code = bf._main([str(stego), "--top", "2", "--save", str(out),
                     "--wordlist", str(wordlist)])

    assert code == 0
    stdout = capsys.readouterr().out
    assert "wordlist: password 'correct horse'" in stdout
    saved = list(out.iterdir())
    assert [p.name for p in saved] == ["wordlist_1bit.txt"]
    assert saved[0].read_text(encoding="utf-8") == "The key is under the mat."


def test_cli_json_output_stays_parseable_with_wordlist(synthetic_rgb, tmp_path, capsys):
    import json

    stego = scattered_stego(tmp_path, synthetic_rgb(5), "json please", "pw1")
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("pw1\n", encoding="utf-8")

    assert bf._main([str(stego), "--top", "2", "--json", "--wordlist", str(wordlist)]) == 0
    captured = capsys.readouterr()
    assert len(json.loads(captured.out)) == 2
    assert "wordlist: password 'pw1'" in captured.err
