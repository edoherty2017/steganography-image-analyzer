"""Tests for report.py: analyze_image(), format_text() and write_report().

Covers are built from the conftest synthetic_rgb factory (256x256, where
every detector reads "clean" on the bare cover; at 128x128 RS analysis
already gives a false "suspicious", so the clean-verdict tests use 256).
"""
from __future__ import annotations

import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import report
import stego
from bitplanes import grid_size_px
from payload import HEADER_LEN

REPO_ROOT = Path(__file__).resolve().parent.parent

TOP_LEVEL_KEYS = {
    "tool", "generated", "image", "file_name", "file_bytes", "format", "mode",
    "size", "channels", "capacity_bytes", "capacity_bytes_2bit", "overall",
    "header", "payload", "signatures", "detectors", "bruteforce", "timings",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strict_json(obj) -> str:
    """json.dumps that fails on NaN/inf and on non-serializable values."""
    return json.dumps(obj, allow_nan=False)


def legacy_image(arr: np.ndarray, message: bytes) -> np.ndarray:
    """Embed ``message`` in the original delimiter format (row-major RGB LSBs)."""
    out = arr.copy()
    bits = np.unpackbits(np.frombuffer(message + b"\x00" * 8, dtype=np.uint8))
    flat = out.reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return out


@pytest.fixture
def cover(synthetic_rgb, save_png) -> Path:
    """A clean 256x256 synthetic RGB cover."""
    return save_png(synthetic_rgb(0), "cover.png")


@pytest.fixture
def plain_stego(cover, tmp_path) -> Path:
    """The cover with an unencrypted STGA text frame in natural order."""
    out = tmp_path / "plain.png"
    stego.hide(cover, out, "meet at noon by the old oak tree")
    return out


# ---------------------------------------------------------------------------
# analyze_image
# ---------------------------------------------------------------------------

def test_clean_cover_schema_and_verdict(cover):
    r = report.analyze_image(cover)
    assert set(r) == TOP_LEVEL_KEYS
    strict_json(r)
    assert r["size"] == {"width": 256, "height": 256}
    assert r["format"] == "PNG" and r["mode"] == "RGB" and r["channels"] == "RGB"
    assert r["header"] is None and r["payload"] is None
    assert r["overall"]["verdict"] == "clean"
    assert r["overall"]["score"] < 0.35
    assert "No detector reached the suspicious level" in r["overall"]["summary"]
    assert set(r["detectors"]) >= {"signatures", "chi_square", "sample_pair_analysis",
                                   "rs_analysis", "histogram_pairs", "lsb_randomness"}
    assert r["signatures"]["stga"] is None


def test_capacity_matches_stego(cover):
    r = report.analyze_image(cover, run_bruteforce=False)
    assert r["capacity_bytes"] == stego.capacity_bytes(cover, 1) == 256 * 256 * 3 // 8 - HEADER_LEN
    assert r["capacity_bytes_2bit"] == stego.capacity_bytes(cover, 2)


def test_bruteforce_candidates_are_json_safe(cover):
    r = report.analyze_image(cover, top_k=4)
    cands = r["bruteforce"]
    assert len(cands) == 4
    for c in cands:
        assert set(c) == {"label", "strategy", "score", "kind", "preview", "info"}
        assert isinstance(c["strategy"], dict) and c["strategy"]["channels"]
    # Best first.
    assert [c["score"] for c in cands] == sorted((c["score"] for c in cands), reverse=True)
    assert "bruteforce" in r["timings"]


def test_no_bruteforce(cover):
    r = report.analyze_image(cover, run_bruteforce=False)
    assert r["bruteforce"] is None
    assert "bruteforce" not in r["timings"]
    assert "bruteforce" not in r["overall"]["flagged"]


def test_plain_payload_is_decoded(plain_stego):
    r = report.analyze_image(plain_stego)
    strict_json(r)
    p = r["payload"]
    assert p["format"] == "stga" and p["decoded"] and p["crc_ok"]
    assert p["is_text"] and p["preview"] == "meet at noon by the old oak tree"
    assert p["data_bytes"] == 32 and p["filename"] is None and not p["encrypted"]
    assert r["header"]["bits_per_sample"] == 1 and not r["header"]["scattered"]
    assert r["signatures"]["stga"]["crc_ok"] is True
    o = r["overall"]
    assert o["score"] == 1.0 and o["verdict"] == "likely_stego" and o["driver"] == "payload"
    assert {"payload", "signatures"} <= set(o["flagged"])
    assert r["bruteforce"][0]["kind"] == "stga"


def test_encrypted_frame_reported_without_password(cover, tmp_path):
    out = tmp_path / "enc.png"
    stego.hide(cover, out, "top secret", password="hunter2", key_scatter=False)
    r = report.analyze_image(out)
    p = r["payload"]
    assert p["encrypted"] and p["note"] == "encrypted"
    assert not p["decoded"] and p["crc_ok"] and not p["password_tried"]
    assert "preview" not in p
    assert r["overall"]["score"] == 1.0 and r["overall"]["driver"] == "payload"
    assert "encrypted" in r["overall"]["summary"]

    r = report.analyze_image(out, password="hunter2")
    assert r["payload"]["decoded"] and r["payload"]["preview"] == "top secret"
    assert "hunter2" not in strict_json(r)  # the password is never stored


def test_wrong_password_on_unscattered_frame(cover, tmp_path):
    out = tmp_path / "enc.png"
    stego.hide(cover, out, "top secret", password="hunter2", key_scatter=False)
    p = report.analyze_image(out, password="nope", run_bruteforce=False)["payload"]
    assert p["note"] == "encrypted" and p["password_tried"] and not p["decoded"]


def test_scattered_payload_needs_its_password(cover, tmp_path):
    out = tmp_path / "scattered.png"
    stego.hide(cover, out, "scattered words", password="pw")
    r = report.analyze_image(out)
    assert r["payload"] is None and r["header"] is None
    assert report.analyze_image(out, password="wrong", run_bruteforce=False)["payload"] is None

    r = report.analyze_image(out, password="pw", run_bruteforce=False)
    assert r["header"]["scattered"] is True
    assert r["payload"]["decoded"] and r["payload"]["preview"] == "scattered words"
    assert r["overall"]["score"] == 1.0


def test_file_payload_reports_name_and_hex(cover, tmp_path):
    out = tmp_path / "file.png"
    data = bytes(range(256)) * 2
    stego.hide(cover, out, data, filename="dir/blob.bin", compress=True, bits=2)
    p = report.analyze_image(out, run_bruteforce=False)["payload"]
    assert p["filename"] == "blob.bin" and p["data_bytes"] == 512
    assert p["bits_per_sample"] == 2 and p["compressed"]
    assert not p["is_text"] and p["preview"].startswith("00 01 02 03")


def test_control_characters_are_not_echoed(cover, tmp_path):
    out = tmp_path / "ansi.png"
    stego.hide(cover, out, "hello \x1b[31mred\x1b[0m world, this is plain text otherwise")
    r = report.analyze_image(out, run_bruteforce=False)
    assert "\x1b" not in r["payload"]["preview"]
    assert "\x1b" not in report.format_text(r)


def test_legacy_format_is_recognised(synthetic_rgb, save_png):
    msg = "an old-style message that ends in d"
    path = save_png(legacy_image(synthetic_rgb(4), msg.encode()), "legacy.png")
    for run_bf in (True, False):  # brute-force evidence, then signature evidence
        r = report.analyze_image(path, run_bruteforce=run_bf)
        p = r["payload"]
        assert p["format"] == "legacy" and p["decoded"]
        assert p["preview"] == msg  # the final "d" is not lost
        assert r["header"] is None
        assert r["overall"]["verdict"] == "likely_stego"
        assert "payload" in r["overall"]["flagged"]


def test_damaged_frame(cover, tmp_path):
    good = tmp_path / "good.png"
    stego.hide(cover, good, "a message that will be damaged")
    arr = np.array(Image.open(good))
    flat = arr.reshape(-1)
    flat[HEADER_LEN * 8 + 5] ^= 1  # one body bit
    bad = tmp_path / "bad.png"
    Image.fromarray(arr).save(bad)

    r = report.analyze_image(bad, run_bruteforce=False)
    p = r["payload"]
    assert p["note"] == "damaged" and not p["decoded"] and not p["crc_ok"]
    assert "CRC" in p["error"] or "crc" in p["error"].lower()
    assert r["overall"]["score"] >= report.DAMAGED_FRAME_SCORE
    assert r["overall"]["verdict"] == "likely_stego"


def test_bruteforce_drives_verdict_for_unusual_layout(synthetic_rgb, save_png):
    """Text in column-major order is missed by the signature scan, found by brute force."""
    text = (b"Column major order hides this sentence from the plain row scan, "
            b"but the brute force tries both traversals and finds it.\x00")
    arr = synthetic_rgb(5)
    cols = np.ascontiguousarray(arr.transpose(1, 0, 2))  # row-major of this = column-major of arr
    bits = np.unpackbits(np.frombuffer(text, dtype=np.uint8))
    flat = cols.reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    path = save_png(np.ascontiguousarray(cols.transpose(1, 0, 2)), "colmajor.png")

    r = report.analyze_image(path)
    top = r["bruteforce"][0]
    assert top["label"] == "RGB bit0 msb-first col" and top["kind"] == "text"
    o = r["overall"]
    assert o["driver"] == "bruteforce"
    assert o["score"] == pytest.approx(top["score"] * report.BRUTEFORCE_WEIGHT)
    assert o["verdict"] in ("suspicious", "likely_stego")
    assert "bruteforce" in o["flagged"]


def test_rgba_cover(synthetic_rgb, tmp_path):
    rgba = np.dstack([synthetic_rgb(6), np.full((256, 256), 200, np.uint8)])
    cover = tmp_path / "rgba.png"
    Image.fromarray(rgba).save(cover)
    out = tmp_path / "rgba_stego.png"
    stego.hide(cover, out, "alpha is kept")
    r = report.analyze_image(out)
    assert r["mode"] == "RGBA"
    assert r["payload"]["preview"] == "alpha is kept"
    strict_json(r)


def test_bad_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.analyze_image(tmp_path / "missing.png")
    not_image = tmp_path / "notes.txt"
    not_image.write_text("not an image")
    with pytest.raises(ValueError, match="not an image"):
        report.analyze_image(not_image)


def test_format_text(plain_stego):
    text = report.format_text(report.analyze_image(plain_stego))
    assert "LIKELY STEGO" in text
    assert "meet at noon by the old oak tree" in text
    assert "Detectors:" in text and "Brute force" in text
    assert "heuristics" in text


def test_report_does_not_import_pyplot():
    """report.py must not switch the GUI's Matplotlib backend."""
    code = "import sys, report; sys.exit('matplotlib.pyplot' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------

def test_write_markdown_with_figures(plain_stego, tmp_path):
    r = report.analyze_image(plain_stego)
    out_dir = tmp_path / "reports" / "nested"
    path = Path(report.write_report(r, out_dir, fmt="md"))
    assert path == out_dir / "plain_report.md"
    md = path.read_text(encoding="utf-8")
    assert md.startswith("# Steganalysis report: plain.png")
    for heading in ("## Verdict", "## Hidden payload", "## Statistical detectors",
                    "## Signature scan", "## Brute-force extraction", "## Figures"):
        assert heading in md
    assert "meet at noon by the old oak tree" in md
    assert "Westfeld & Pfitzmann 1999" in md
    for name in ("plain_bitplanes.png", "plain_lsb.png", "plain_histogram.png"):
        assert f"]({name})" in md
        assert (out_dir / name).is_file()
    with Image.open(out_dir / "plain_bitplanes.png") as im:
        assert im.size == grid_size_px(3, 8)


def test_write_markdown_escapes_table_cells(cover, tmp_path):
    out = tmp_path / "pipes.png"
    stego.hide(cover, out, "alpha|beta|gamma are separated by pipes in this text")
    r = report.analyze_image(out, run_bruteforce=False)
    md = Path(report.write_report(r, tmp_path / "rep", fmt="md", figures=False)).read_text()
    table_rows = [line for line in md.splitlines() if line.startswith("| stga")]
    assert table_rows and "alpha\\|beta\\|gamma" in table_rows[0]
    assert table_rows[0].count(" | ") == 4  # still exactly five cells


def test_write_html_is_self_contained_and_escaped(cover, tmp_path):
    out = tmp_path / "xss.png"
    stego.hide(cover, out, "<script>alert('x')</script> & more text here")
    r = report.analyze_image(out)
    path = Path(report.write_report(r, tmp_path / "rep", fmt="HTML"))
    assert path.name == "xss_report.html"
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert page.count("data:image/png;base64,") == 3
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "likely stego" in page
    HTMLParser().feed(page)  # parses without raising


def test_write_json_round_trips(plain_stego, tmp_path):
    r = report.analyze_image(plain_stego)
    path = Path(report.write_report(r, tmp_path / "rep", fmt="json"))
    assert path.name == "plain_report.json"
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(strict_json(r))
    assert not list(path.parent.glob("*.png"))  # JSON reports never draw figures


def test_write_without_figures(plain_stego, tmp_path):
    r = report.analyze_image(plain_stego, run_bruteforce=False)
    path = Path(report.write_report(r, tmp_path / "rep", fmt="md", figures=False))
    assert "## Figures" not in path.read_text()
    assert "Skipped." in path.read_text()  # brute force section
    assert not list(path.parent.glob("*.png"))


def test_write_rejects_unknown_format(plain_stego, tmp_path):
    r = report.analyze_image(plain_stego, run_bruteforce=False)
    with pytest.raises(ValueError, match="unknown report format"):
        report.write_report(r, tmp_path, fmt="pdf")
