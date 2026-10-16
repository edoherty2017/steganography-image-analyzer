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


# ---------------------------------------------------------------------------
# Regression tests: legacy fallback gate, untrusted output, password and
# exit-status contract, WAV word-list attack
# ---------------------------------------------------------------------------

def write_tone(path: Path, n_frames: int = 16000) -> Path:
    """Write a mono 16-bit PCM WAV with a 440 Hz tone."""
    t = np.arange(n_frames)
    samples = (8000 * np.sin(2 * np.pi * 440 * t / 8000)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())
    return path


def test_decode_refuses_legacy_noise_before_a_flat_region(synthetic_rgb, save_png, tmp_path):
    """A clean image with a black border used to "decode" to kilobytes of noise, exit 0."""
    arr = synthetic_rgb(5, size=128)
    save_png(np.vstack([arr, np.zeros((24, 128, 3), np.uint8)]), "letterbox.png")
    # Odd values on top, even below: the LSB stream is 0xFF bytes, then zeros.
    saturated = np.vstack([arr[:64] | 1, arr[64:] & 0xFE])
    save_png(saturated, "saturated.png")
    for name in ("letterbox.png", "saturated.png"):
        proc = run_cli("decode", "-i", name, cwd=tmp_path)
        assert_error(proc, contains="no STGA payload")
        assert "legacy delimiter format was not found" in proc.stderr
        assert "are not text" in proc.stderr
        assert proc.stdout == ""
        # --legacy drops the score floor but still refuses data that is not text.
        forced = run_cli("decode", "-i", name, "--legacy", cwd=tmp_path)
        assert_error(forced, contains="no legacy-format message found")
        assert forced.stdout == ""


def test_decode_legacy_gate_keeps_short_messages(synthetic_rgb, save_png, tmp_path):
    """The gate is bruteforce's legacy floor (0.15), not report's 0.5, so "hello" decodes."""
    save_png(legacy_image(synthetic_rgb(9, size=128), b"hello"), "short.png")
    proc = run_cli("decode", "-i", "short.png", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "Decoded Message Found:\nhello\n"
    assert "heuristic text score" in proc.stderr and "weak evidence" in proc.stderr

    # Repetitive text (what smooth gradients put in an LSB plane) scores 0.05:
    # refused by the automatic fallback, printed when --legacy asks for it.
    save_png(legacy_image(synthetic_rgb(9, size=128), b"UUUUUUUUUUUU"), "rep.png")
    assert_error(run_cli("decode", "-i", "rep.png", cwd=tmp_path),
                 contains="does not look like a message")
    forced = run_cli("decode", "-i", "rep.png", "--legacy", cwd=tmp_path)
    assert forced.returncode == 0, forced.stderr
    assert forced.stdout == "Decoded Message Found:\nUUUUUUUUUUUU\n"


# Control characters a crafted payload could use against the analyst's
# terminal: OSC window title (ESC ] ... BEL), clear screen, SGR colour, a
# carriage return that overwrites the line, a C1 CSI and a right-to-left
# override.
_HOSTILE = "note \x1b]0;pwned-title\x07\x1b[2J\x1b[31mRED\x1b[0m\rfake line \x9b2J ‮end"
_UNSAFE = ("\x1b", "\x07", "\r", "\x9b", "‮")


def test_decode_does_not_echo_terminal_control_characters(cover, tmp_path, capsys):
    """Run in-process: subprocess text mode would turn a raw \\r into \\n."""
    import main

    stego.hide(cover, tmp_path / "esc.png", _HOSTILE)
    assert main.main(["decode", "-i", str(tmp_path / "esc.png")]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("Decoded Message Found:\nnote ")
    assert "RED" in out and "fake line" in out and "end" in out
    for ch in _UNSAFE:
        assert ch not in out
    assert "control characters replaced" in err and "--out" in err
    # --out still writes the exact bytes.
    assert main.main(["decode", "-i", str(tmp_path / "esc.png"),
                      "--out", str(tmp_path / "exact.txt")]) == 0
    assert (tmp_path / "exact.txt").read_bytes() == _HOSTILE.encode("utf-8")
    capsys.readouterr()

    # A stored file name is untrusted too.
    stego.hide(cover, tmp_path / "named.png", b"data", filename="evil\x1b[2J‮.txt")
    assert main.main(["decode", "-i", str(tmp_path / "named.png")]) == 0
    out, _ = capsys.readouterr()
    assert "Hidden file found: 'evil" in out
    for ch in _UNSAFE:
        assert ch not in out


def test_decode_legacy_text_is_sanitized_too(synthetic_rgb, save_png, tmp_path, capsys):
    import main

    # CR and RLO are printable to bruteforce's text heuristic, so this passes
    # the legacy gate and must be cleaned when printed.
    msg = "Account balance is fine\rHACKED: transfer approved ‮now"
    save_png(legacy_image(synthetic_rgb(4, size=128), msg.encode("utf-8")), "cr.png")
    assert main.main(["decode", "-i", str(tmp_path / "cr.png")]) == 0
    out, _ = capsys.readouterr()
    assert "HACKED: transfer approved" in out
    for ch in _UNSAFE:
        assert ch not in out
    # ESC is not text at all, so a legacy reading containing it is refused.
    save_png(legacy_image(synthetic_rgb(4, size=128), b"hello \x1b]0;pwned\x07 world"),
             "esc_legacy.png")
    assert main.main(["decode", "-i", str(tmp_path / "esc_legacy.png")]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "\x1b" not in err


def test_empty_password_is_a_usage_error(cover, tmp_path):
    """--password "" used to write a plaintext, unscattered frame with exit 0."""
    proc = run_cli("encode", "-i", cover, "-m", "secret", "-o", "o.png", "--password", "",
                   cwd=tmp_path)
    assert_error(proc, code=2, contains="--password is empty")
    assert not (tmp_path / "o.png").exists()
    for cmd in ("decode", "analyze"):
        assert_error(run_cli(cmd, "-i", cover, "--password", "", cwd=tmp_path), code=2,
                     contains="--password is empty")


def test_bitplanes_bad_channels_is_a_usage_error(cover, tmp_path):
    for channels in ("XYZ", "RGX", ""):
        proc = run_cli("bitplanes", "-i", cover, "-o", "bp.png", "--channels", channels,
                       cwd=tmp_path)
        assert_error(proc, code=2, contains="--channels takes letters from RGBA")
    assert not (tmp_path / "bp.png").exists()


def test_bruteforce_wordlist_reaches_wav_carrier(tmp_path):
    """The layout brute force opened the WAV with Pillow and failed first."""
    write_tone(tmp_path / "tone.wav")
    enc = run_cli("encode", "-i", "tone.wav", "-m", "audio secret", "-o", "secret.wav",
                  "--password-prompt", cwd=tmp_path, input="correct horse\n")
    assert enc.returncode == 0 and "encrypted, scattered" in enc.stdout
    (tmp_path / "words.txt").write_text("password\ncorrect horse\n", encoding="utf-8")
    (tmp_path / "nowords.txt").write_text("nope\n", encoding="utf-8")

    proc = run_cli("bruteforce", "-i", "secret.wav", "--wordlist", "words.txt", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "password 'correct horse'" in proc.stdout and "audio secret" in proc.stdout
    assert "not an image" in proc.stderr

    doc = json.loads(run_cli("bruteforce", "-i", "secret.wav", "--wordlist", "words.txt",
                             "--json", cwd=tmp_path).stdout)
    assert doc["candidates"] == []
    assert [h["password"] for h in doc["wordlist_hits"]] == ["correct horse"]

    saved = run_cli("bruteforce", "-i", "secret.wav", "--wordlist", "words.txt",
                    "--save", "found", cwd=tmp_path)
    assert saved.returncode == 0, saved.stderr
    assert [p.read_bytes() for p in (tmp_path / "found").iterdir()] == [b"audio secret"]

    assert_error(run_cli("bruteforce", "-i", "secret.wav", cwd=tmp_path),
                 contains="--wordlist")
    assert_error(run_cli("bruteforce", "-i", "secret.wav", "--wordlist", "nowords.txt",
                         cwd=tmp_path), contains="no password in the word list matched")


def test_bruteforce_wordlist_exit_status(cover, tmp_path):
    """Exit 1 only when neither a password nor a layout found anything."""
    (tmp_path / "nowords.txt").write_text("nope\n", encoding="utf-8")
    (tmp_path / "words.txt").write_text("nope\nhunter2\n", encoding="utf-8")
    assert run_cli("encode", "-i", cover, "-m", "scattered secret", "-o", "scat.png",
                   "--password", "hunter2", cwd=tmp_path).returncode == 0

    miss = run_cli("bruteforce", "-i", "scat.png", "--top", "1", "--wordlist", "nowords.txt",
                   cwd=tmp_path)
    assert_error(miss, contains="no password in the word list matched")
    assert "wordlist: no password matched" in miss.stdout  # the table is still printed
    hit = run_cli("bruteforce", "-i", "scat.png", "--top", "1", "--wordlist", "words.txt",
                  cwd=tmp_path)
    assert hit.returncode == 0 and "scattered secret" in hit.stdout

    # An unscattered frame is found by the layouts (score 1.00), so a word
    # list without the (unused) password is not "nothing found".
    assert run_cli("encode", "-i", cover, "-m", "plain sequential message", "-o", "seq.png",
                   cwd=tmp_path).returncode == 0
    seq = run_cli("bruteforce", "-i", "seq.png", "--top", "1", "--wordlist", "nowords.txt",
                  cwd=tmp_path)
    assert seq.returncode == 0, seq.stderr
    # Without --wordlist the brute force only ranks, so a clean cover exits 0.
    assert run_cli("bruteforce", "-i", cover, "--top", "1", cwd=tmp_path).returncode == 0
