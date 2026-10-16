"""End-to-end tests for the main.py command line.

Every test runs ``python main.py ...`` in a subprocess (the interpreter
running the tests, i.e. the project's .venv) with the working directory set
to tmp_path, exactly as a user would. Covers are small (128x128) so each
call stays fast.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import encoder
import stego
from bitplanes import grid_size_px

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN = REPO_ROOT / "main.py"

SUBCOMMANDS = ("encode", "decode", "capacity", "analyze", "bitplanes",
               "bruteforce", "compare", "plugins")


def run_cli(*args, cwd: Path, input: str | None = None) -> subprocess.CompletedProcess:
    """Run main.py with ``args`` and capture its output.

    ``start_new_session`` detaches the child from the controlling terminal
    and stdin is a pipe (or /dev/null), so a password prompt can never block
    on the test runner's terminal.
    """
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    stdin = {"input": input} if input is not None else {"stdin": subprocess.DEVNULL}
    return subprocess.run(
        [sys.executable, str(MAIN), *map(str, args)],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        env=env, timeout=180, start_new_session=True, **stdin,
    )


def assert_error(proc: subprocess.CompletedProcess, code: int = 1, contains: str = "") -> None:
    """Check a clean failure: nonzero code, message on stderr, no traceback."""
    assert proc.returncode == code, (proc.stdout, proc.stderr)
    assert "error" in proc.stderr
    assert contains in proc.stderr
    assert "Traceback" not in proc.stderr


def legacy_image(arr: np.ndarray, message: bytes) -> np.ndarray:
    """Embed ``message`` in the original delimiter format (row-major RGB LSBs)."""
    out = arr.copy()
    bits = np.unpackbits(np.frombuffer(message + b"\x00" * 8, dtype=np.uint8))
    flat = out.reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return out


@pytest.fixture
def cover(synthetic_rgb, save_png) -> Path:
    """A 128x128 synthetic RGB cover in tmp_path."""
    return save_png(synthetic_rgb(3, size=128), "cover.png")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_help_lists_every_subcommand(tmp_path):
    proc = run_cli("--help", cwd=tmp_path)
    assert proc.returncode == 0
    for name in SUBCOMMANDS:
        assert name in proc.stdout


def test_missing_subcommand_is_usage_error(tmp_path):
    assert_error(run_cli(cwd=tmp_path), code=2)


def test_message_and_file_are_exclusive(cover, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    proc = run_cli("encode", "-i", cover, "-m", "x", "-f", "a.txt", "-o", "o.png", cwd=tmp_path)
    assert_error(proc, code=2, contains="not allowed with")
    assert not (tmp_path / "o.png").exists()


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def test_legacy_syntax_round_trip(cover, tmp_path):
    msg = "Meet me at noon, by the old road"  # ends in "d": trailing zero bits
    enc = run_cli("encode", "-i", "cover.png", "-m", msg, "-o", "stego.png", cwd=tmp_path)
    assert enc.returncode == 0, enc.stderr
    assert enc.stdout.splitlines()[0] == "Successfully encoded message in 'stego.png'"
    dec = run_cli("decode", "-i", "stego.png", cwd=tmp_path)
    assert dec.returncode == 0, dec.stderr
    assert dec.stdout == f"Decoded Message Found:\n{msg}\n"


def test_cli_encode_matches_encoder_module(cover, tmp_path):
    """The CLI writes the same pixels as the legacy encoder.encode()."""
    assert run_cli("encode", "-i", cover, "-m", "same bits", "-o", "cli.png",
                   cwd=tmp_path).returncode == 0
    encoder.encode(cover, "same bits", tmp_path / "lib.png")
    a = np.asarray(Image.open(tmp_path / "cli.png"))
    b = np.asarray(Image.open(tmp_path / "lib.png"))
    assert np.array_equal(a, b)


def test_password_round_trip(cover, tmp_path):
    enc = run_cli("encode", "-i", cover, "-m", "pw secret", "-o", "s.png",
                  "--password", "pw", cwd=tmp_path)
    assert enc.returncode == 0 and "encrypted, scattered" in enc.stdout
    assert_error(run_cli("decode", "-i", "s.png", cwd=tmp_path), contains="no STGA payload")
    assert_error(run_cli("decode", "-i", "s.png", "--password", "nope", cwd=tmp_path))
    dec = run_cli("decode", "-i", "s.png", "--password", "pw", cwd=tmp_path)
    assert dec.returncode == 0 and dec.stdout == "Decoded Message Found:\npw secret\n"


def test_password_prompt_reads_stdin_when_not_a_tty(cover, tmp_path):
    enc = run_cli("encode", "-i", cover, "-m", "prompted", "-o", "s.png",
                  "--password-prompt", cwd=tmp_path, input="s3cret\n")
    assert enc.returncode == 0, enc.stderr
    # The piped line really was the password: --password with it works too.
    dec = run_cli("decode", "-i", "s.png", "--password", "s3cret", cwd=tmp_path)
    assert dec.stdout.endswith("prompted\n")
    dec = run_cli("decode", "-i", "s.png", "--password-prompt", cwd=tmp_path, input="s3cret\n")
    assert dec.returncode == 0 and dec.stdout.endswith("prompted\n")
    assert_error(run_cli("encode", "-i", cover, "-m", "x", "-o", "t.png", "--password-prompt",
                         cwd=tmp_path, input="\n"), contains="empty password")
    assert not (tmp_path / "t.png").exists()


def test_unscattered_encrypted_frame_asks_for_password(cover, tmp_path):
    assert run_cli("encode", "-i", cover, "-m", "x", "-o", "s.png", "--password", "pw",
                   "--no-scatter", cwd=tmp_path).returncode == 0
    assert_error(run_cli("decode", "-i", "s.png", cwd=tmp_path), contains="--password")
    assert_error(run_cli("decode", "-i", "s.png", "--password", "bad", cwd=tmp_path),
                 contains="wrong password")


def test_file_payload_with_compression_and_two_bits(cover, tmp_path):
    data = np.random.default_rng(1).integers(0, 256, 700, dtype=np.uint8).tobytes()
    (tmp_path / "data.bin").write_bytes(data)
    enc = run_cli("encode", "-i", cover, "-f", "data.bin", "-o", "s.png", "--compress",
                  "--bits", "2", cwd=tmp_path)
    assert enc.returncode == 0, enc.stderr
    assert enc.stdout.startswith("Successfully hid file 'data.bin' in 's.png'")
    assert "2 bit(s)/sample" in enc.stdout and "compressed" in enc.stdout

    shown = run_cli("decode", "-i", "s.png", cwd=tmp_path)
    assert shown.returncode == 0
    assert "Hidden file found: 'data.bin' (700 bytes)" in shown.stdout
    assert "--out" in shown.stdout

    saved = run_cli("decode", "-i", "s.png", "--bits", "2", "--out", "rec/", cwd=tmp_path)
    assert saved.returncode == 0, saved.stderr
    assert (tmp_path / "rec" / "data.bin").read_bytes() == data
    again = run_cli("decode", "-i", "s.png", "--out", "rec", cwd=tmp_path)  # existing folder
    assert again.returncode == 0
    assert (tmp_path / "rec" / "data (1).bin").read_bytes() == data  # never overwrites


def test_decode_out_exact_path(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "to a file", "-o", "s.png", cwd=tmp_path)
    proc = run_cli("decode", "-i", "s.png", "--out", "deep/msg.txt", cwd=tmp_path)
    assert proc.returncode == 0 and "Saved 9 bytes" in proc.stdout
    assert (tmp_path / "deep" / "msg.txt").read_text(encoding="utf-8") == "to a file"


def test_decode_legacy_format_image(synthetic_rgb, save_png, tmp_path):
    msg = "An original-format message ending in d"
    save_png(legacy_image(synthetic_rgb(7, size=128), msg.encode()), "old.png")
    auto = run_cli("decode", "-i", "old.png", cwd=tmp_path)
    assert auto.returncode == 0, auto.stderr
    assert auto.stdout == f"Decoded Message Found:\n{msg}\n"
    assert "legacy" in auto.stderr  # the note goes to stderr, stdout stays clean
    forced = run_cli("decode", "-i", "old.png", "--legacy", cwd=tmp_path)
    assert forced.stdout == f"Decoded Message Found:\n{msg}\n"


def test_decode_legacy_flag_on_new_format_fails(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "new format", "-o", "s.png", cwd=tmp_path)
    assert_error(run_cli("decode", "-i", "s.png", "--legacy", cwd=tmp_path),
                 contains="no legacy-format message")


def test_wav_round_trip(tmp_path):
    t = np.arange(8000)
    samples = (8000 * np.sin(2 * np.pi * 440 * t / 8000)).astype("<i2")
    with wave.open(str(tmp_path / "tone.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())
    enc = run_cli("encode", "-i", "tone.wav", "-m", "audio works", "-o", "out.wav", cwd=tmp_path)
    assert enc.returncode == 0 and "wav-lsb" in enc.stdout
    dec = run_cli("decode", "-i", "out.wav", cwd=tmp_path)
    assert dec.stdout == "Decoded Message Found:\naudio works\n"
    # A WAV with no frame has no legacy fallback.
    assert_error(run_cli("decode", "-i", "tone.wav", cwd=tmp_path), contains="no STGA payload")


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------

def _numbers(text: str) -> dict[int, int]:
    """Map bits per sample -> bytes from capacity output."""
    return {int(b): int(n.replace(",", ""))
            for b, n in re.findall(r"(\d) bits?/sample:\s+([\d,]+) bytes", text)}


def test_capacity(cover, tmp_path):
    proc = run_cli("capacity", "-i", cover, cwd=tmp_path)
    assert proc.returncode == 0
    assert "image-lsb" in proc.stdout
    assert _numbers(proc.stdout) == {1: stego.capacity_bytes(cover, 1),
                                     2: stego.capacity_bytes(cover, 2)}
    proc = run_cli("capacity", "-i", cover, "--bits", "2", "--encrypted", cwd=tmp_path)
    assert _numbers(proc.stdout) == {2: stego.capacity_bytes(cover, 2) - stego.ENCRYPTION_OVERHEAD}


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def test_analyze_json(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "find me in the report", "-o", "s.png", cwd=tmp_path)
    proc = run_cli("analyze", "-i", "s.png", "--json", "--top", "3", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)  # stdout is pure JSON
    assert result["overall"]["verdict"] == "likely_stego"
    assert result["payload"]["preview"] == "find me in the report"
    assert len(result["bruteforce"]) == 3


@pytest.mark.parametrize("fmt", ["md", "html", "json"])
def test_analyze_writes_report(cover, tmp_path, fmt):
    run_cli("encode", "-i", cover, "-m", "reported", "-o", "s.png", cwd=tmp_path)
    proc = run_cli("analyze", "-i", "s.png", "--report", "rep", "--format", fmt,
                   "--no-bruteforce", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    report_path = tmp_path / "rep" / f"s_report.{fmt}"
    assert report_path.is_file()
    assert f"Report written to '{Path('rep') / f's_report.{fmt}'}'" in proc.stdout
    assert "LIKELY STEGO" in proc.stdout
    if fmt != "json":
        assert (tmp_path / "rep" / "s_bitplanes.png").is_file()


def test_analyze_password_finds_scattered_payload(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "scattered", "-o", "s.png", "--password", "pw",
            cwd=tmp_path)
    proc = run_cli("analyze", "-i", "s.png", "--no-bruteforce", "--password-prompt",
                   cwd=tmp_path, input="pw\n")
    assert proc.returncode == 0, proc.stderr
    assert "scattered" in proc.stdout and "decrypted" in proc.stdout


def test_analyze_format_needs_report(cover, tmp_path):
    assert_error(run_cli("analyze", "-i", cover, "--format", "html", cwd=tmp_path),
                 code=2, contains="--format needs --report")


# ---------------------------------------------------------------------------
# bitplanes / bruteforce / compare / plugins
# ---------------------------------------------------------------------------

def test_bitplanes_with_diff(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "diff me", "-o", "s.png", cwd=tmp_path)
    proc = run_cli("bitplanes", "-i", "s.png", "-o", "planes/grid.png", "--diff", "cover.png",
                   "--amplify", "planes/lsb.png", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    with Image.open(tmp_path / "planes" / "grid.png") as im:
        assert im.size == grid_size_px(3, 8)
    with Image.open(tmp_path / "planes" / "grid_lsb_diff.png") as im:
        mask = np.asarray(im)
    assert mask.shape == (128, 128) and mask.any()
    assert re.search(rf"{np.count_nonzero(mask)} of 16,384 pixels", proc.stdout)
    assert (tmp_path / "planes" / "lsb.png").is_file()


def test_bruteforce_save_and_json(synthetic_rgb, save_png, tmp_path):
    msg = b"Brute force should rank this hidden sentence first of all layouts"
    save_png(legacy_image(synthetic_rgb(8, size=128), msg), "old.png")
    proc = run_cli("bruteforce", "-i", "old.png", "--top", "3", "--save", "found",
                   cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert msg.decode() in proc.stdout.splitlines()[1]
    saved = list((tmp_path / "found").iterdir())
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == msg.decode()

    proc = run_cli("bruteforce", "-i", "old.png", "--top", "2", "--max-bit", "0", "--json",
                   cwd=tmp_path)
    doc = json.loads(proc.stdout)
    assert len(doc["candidates"]) == 2
    assert doc["candidates"][0]["strategy"] == {"channels": "RGB", "bit": 0, "bit_order": "msb",
                                                "traversal": "row", "depth": 1}


def test_bruteforce_bad_max_bit(cover, tmp_path):
    assert_error(run_cli("bruteforce", "-i", cover, "--max-bit", "9", cwd=tmp_path), code=2)


def test_compare(cover, tmp_path):
    run_cli("encode", "-i", cover, "-m", "compare me", "-o", "s.png", cwd=tmp_path)
    proc = run_cli("compare", "cover.png", "s.png", "-o", "cmp/comp.png", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "cmp" / "comp.png").is_file()
    for name in ("1_original_standard_hist.png", "2_encoded_standard_hist.png",
                 "3_original_difference_analysis.png", "4_encoded_difference_analysis.png"):
        assert (tmp_path / "cmp" / name).is_file()
    assert "cover: L1" in proc.stdout and "stego: L1" in proc.stdout
    assert "have a changed R, G or B LSB" in proc.stdout


def test_plugins(tmp_path):
    proc = run_cli("plugins", cwd=tmp_path)
    assert proc.returncode == 0
    for name in ("image-lsb", "wav-lsb", "signatures", "chi_square", "sample_pair_analysis",
                 "rs_analysis", "histogram_pairs", "lsb_randomness"):
        assert name in proc.stdout
    doc = json.loads(run_cli("plugins", "--json", cwd=tmp_path).stdout)
    assert {c["name"] for c in doc["carriers"]} >= {"image-lsb", "wav-lsb"}
    assert ".wav" in next(c for c in doc["carriers"] if c["name"] == "wav-lsb")["output_extensions"]
    assert doc["report_formats"] == ["md", "json", "html"]


# ---------------------------------------------------------------------------
# Errors: nonzero exit, message on stderr, nothing written
# ---------------------------------------------------------------------------

def test_missing_input(tmp_path):
    assert_error(run_cli("encode", "-i", "nope.png", "-m", "x", "-o", "o.png", cwd=tmp_path),
                 contains="file not found: nope.png")
    assert not (tmp_path / "o.png").exists()
    assert_error(run_cli("decode", "-i", "nope.png", cwd=tmp_path), contains="file not found")
    assert_error(run_cli("analyze", "-i", "nope.png", cwd=tmp_path), contains="file not found")


def test_lossy_output_refused(cover, tmp_path):
    assert_error(run_cli("encode", "-i", cover, "-m", "x", "-o", "o.jpg", cwd=tmp_path),
                 contains="lossy")
    assert not (tmp_path / "o.jpg").exists()


def test_message_too_large(cover, tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (stego.capacity_bytes(cover) + 1))
    assert_error(run_cli("encode", "-i", cover, "-f", big, "-o", "o.png", cwd=tmp_path),
                 contains="holds at most")
    assert not (tmp_path / "o.png").exists()


def test_not_an_image(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    assert_error(run_cli("decode", "-i", "notes.txt", cwd=tmp_path))
    assert_error(run_cli("analyze", "-i", "notes.txt", cwd=tmp_path), contains="not an image")


def test_corrupt_wav(tmp_path):
    (tmp_path / "bad.wav").write_text("not audio")
    assert_error(run_cli("decode", "-i", "bad.wav", cwd=tmp_path), contains="not a readable PCM WAV")
    assert_error(run_cli("capacity", "-i", "bad.wav", cwd=tmp_path), contains="bad.wav")


def test_clean_cover_has_nothing_to_decode(cover, tmp_path):
    assert_error(run_cli("decode", "-i", cover, cwd=tmp_path), contains="no STGA payload")
