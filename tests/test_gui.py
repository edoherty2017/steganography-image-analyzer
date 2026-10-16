"""Tests for gui.py, the Tkinter desktop app.

Every test builds a real Tk root (withdrawn, so no window appears), and the
tests are skipped when Tk cannot open a display, for example on a headless
CI runner. Message boxes are suppressed with ``interactive=False`` and
recorded in ``app.messages``; file dialogs are bypassed by passing paths to
the handlers. Background jobs are waited for by pumping the Tk event loop
with ``root.update()`` until ``app.busy`` is False.

All generated files go to ``tmp_path`` (the app's ``output_dir``), never to
the repository's tracked ``results/`` folder.
"""
from __future__ import annotations

import json
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")

import bitplanes  # noqa: E402
import detectors  # noqa: E402
import gui  # noqa: E402
import stego  # noqa: E402
from bitutils import bytes_to_bits  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Seconds to wait for one batch of background work before failing.
TIMEOUT = 90.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def root():
    """A withdrawn Tk root; skips the test when Tk has no display."""
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk cannot open a display: {exc}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass  # already destroyed by app.close()


@pytest.fixture
def app(root, tmp_path):
    """The application with dialogs suppressed and output under tmp_path."""
    application = gui.SteganoApp(root, output_dir=tmp_path / "out", interactive=False)
    yield application
    application.close()


@pytest.fixture
def cover(synthetic_rgb, save_png) -> Path:
    """A 256x256 natural-looking RGB cover (256 px reads clean on every detector)."""
    return save_png(synthetic_rgb(0), "cover.png")


def pump(app, timeout: float = TIMEOUT):
    """Run the Tk event loop until no background job is left."""
    deadline = time.monotonic() + timeout
    while app.busy:
        if time.monotonic() > deadline:
            pytest.fail("background work did not finish in time")
        app.root.update()
        time.sleep(0.01)
    # One more pass runs the after(10) image renders the callbacks queued.
    app.root.update()


def last_message(app) -> tuple[str, str, str]:
    assert app.messages, "no dialog was recorded"
    return app.messages[-1]


def legacy_image(arr: np.ndarray, text: str) -> np.ndarray:
    """Embed text the original way: UTF-8 bits + 64 zero bits, row-major RGB LSBs."""
    bits = bytes_to_bits(text.encode("utf-8") + b"\x00" * 8)
    flat = arr.copy().reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return flat.reshape(arr.shape)


def write_wav(path: Path, seconds: float = 0.5, rate: int = 8000) -> Path:
    """A 16-bit mono sine-wave WAV cover."""
    t = np.arange(int(seconds * rate)) / rate
    samples = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# Window, menu, file types, helpers
# ---------------------------------------------------------------------------

def test_window_title_tabs_and_menu(app, tmp_path):
    assert app.root.title() == "Steganography Image Analyzer"
    titles = [app.notebook.tab(t, "text") for t in app.notebook.tabs()]
    assert titles == ["Encode", "Decode", "Analyze", "Bitplanes", "Brute force", "Compare"]
    labels = [app.menubar.entrycget(i, "label") for i in range(app.menubar.index("end") + 1)]
    assert labels == ["File", "Help"]
    # Nothing is written until something is generated.
    assert not (tmp_path / "out").exists()


def test_text_widgets_use_explicit_light_colors(app):
    # Without explicit colors, macOS dark mode draws tk.Text white-on-white
    # inside the light ttk theme.
    for widget in (app.encode_tab.message_text, app.decode_tab.output_text,
                   app.analyze_tab.summary_text, app.bruteforce_tab.preview_text):
        assert widget.cget("foreground") == gui.TEXT_COLORS["foreground"]
    assert app.decode_tab.output_text.cget("background") == gui.TEXT_COLORS["background"]
    app.encode_tab.set_payload_file(None)  # File mode greys out the text box
    assert app.encode_tab.message_text.cget("background") == gui.TEXT_DISABLED_BACKGROUND


def test_about_lists_project_and_plugins(app):
    text = app.about_text()
    assert "Steganography Image Analyzer" in text
    assert "least significant bit" in text and "heuristic" in text
    for cls in gui.carriers.list_carriers():
        assert cls.name in text
    for row in detectors.list_detectors():
        assert row["name"] in text
    window = app.show_about()
    app.root.update()
    assert window.winfo_exists()
    window.destroy()


def test_file_dialog_filters_include_lossless_formats_and_wav(app):
    def patterns(types):
        return " ".join(p for _name, p in types)

    covers = patterns(app.encode_tab.open_filetypes())
    for pat in ("*.png", "*.bmp", "*.tif", "*.tiff", "*.wav", "*.jpg"):
        assert pat in covers
    decode = patterns(app.decode_tab.open_filetypes())
    for pat in ("*.png", "*.bmp", "*.tif", "*.tiff", "*.wav"):
        assert pat in decode
    assert "*.jpg" not in decode.split()  # lossy files cannot hold a payload
    for tab in (app.analyze_tab, app.bitplanes_tab, app.bruteforce_tab, app.compare_tab):
        image = patterns(tab.open_filetypes())
        for pat in ("*.png", "*.bmp", "*.tif", "*.tiff"):
            assert pat in image
    assert ".jpg" not in gui.stego_extensions()
    assert {".png", ".bmp", ".tif", ".tiff", ".wav"} <= set(gui.stego_extensions())


def test_describe_error_distinguishes_the_failure_kinds():
    cases = [
        (gui.payload.WrongPasswordError("x"), "Wrong or missing password"),
        (stego.NoPayloadError("x"), "No hidden payload found"),
        (gui.payload.PayloadError("x"), "Damaged payload"),
        (gui.LossyFormatError("x"), "Lossless output needed"),
        (gui.CapacityError("x"), "Payload too large"),
        (FileNotFoundError(2, "No such file", "a.png"), "File not found"),
    ]
    for exc, title in cases:
        assert gui.describe_error(exc)[0] == title


def test_text_helpers():
    assert gui.one_line("a\n\tb\x00  c") == "a b c"
    assert gui.one_line("x" * 200, 10).endswith("…")
    dump = gui.hex_dump(b"AB\x00" * 10, limit=16)
    assert dump.startswith("00000000  41 42 00") and "more bytes" in dump
    assert gui.safe_stem("/a/b/my photo (1).png") == "my_photo__1"
    assert gui.safe_stem("...png") == "image"


# ---------------------------------------------------------------------------
# Encode and Decode
# ---------------------------------------------------------------------------

def test_encode_text_then_decode_tab_recovers_it(app, cover, tmp_path):
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    assert tab.capacity == {1: stego.capacity_bytes(cover, 1), 2: stego.capacity_bytes(cover, 2)}
    message = "Meet at noon. Ends in d"
    tab.set_message(message)
    used, room = tab.update_meter()
    assert (used, room) == (len(message), tab.capacity[1])
    assert f"{len(message)} / {room:,} bytes" in tab.meter_label.cget("text")

    out = tmp_path / "stego.png"
    assert tab.hide(out) is True
    pump(app)
    assert out.exists()
    assert gui.decoder.decode(out) == message
    assert last_message(app)[:2] == ("info", "Saved")
    assert tab.preview_cover.image is not None
    assert tab.preview_stego.image is not None
    assert tab.preview_diff.image is not None
    assert tab.last_summary["data_bytes"] == len(message)
    # The new file is handed to the Decode and Compare tabs.
    assert app.decode_tab.path_var.get() == str(out)
    assert app.compare_tab.stego_var.get() == str(out)

    assert app.decode_tab.decode() is True
    pump(app)
    assert app.decode_tab.recovered_text() == message
    assert app.decode_tab.headline.cget("text").startswith("Hidden message")
    assert message in app.decode_tab.output_text.get("1.0", "end-1c")
    assert app.callback_errors == []


def test_password_compression_two_bits_and_wrong_password(app, cover, tmp_path):
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    tab.set_message("café au lait, encrypted €")
    tab.password_var.set("s3cret")
    tab.compress_var.set(True)
    tab.bits_var.set(2)
    out = tmp_path / "secret.png"
    tab.hide(out)
    pump(app)
    assert tab.last_summary["encrypted"] and tab.last_summary["scattered"]
    assert tab.last_summary["bits_per_sample"] == 2
    assert stego.reveal(out, password="s3cret", bits=None).text() == "café au lait, encrypted €"

    dec = app.decode_tab
    dec.password_var.set("wrong")
    dec.decode(out)
    pump(app)
    # With key scatter a wrong password reads the bits in the wrong order.
    assert dec.headline.cget("text") == "No hidden payload found"
    assert last_message(app)[:2] == ("error", "No hidden payload found")
    assert dec.found is None

    dec.password_var.set("s3cret")
    dec.decode(out)
    pump(app)
    assert dec.recovered_text() == "café au lait, encrypted €"
    detail = dec.detail.cget("text")
    assert "2 bit(s)" in detail and "encrypted" in detail and "compressed" in detail


def test_unscattered_encrypted_frame_reports_wrong_password(app, cover, tmp_path):
    out = tmp_path / "locked.png"
    stego.hide(cover, out, "locked away", password="pw", key_scatter=False)
    app.decode_tab.decode(out)
    pump(app)
    assert app.decode_tab.headline.cget("text") == "Wrong or missing password"
    assert last_message(app)[:2] == ("error", "Wrong or missing password")
    assert not app.decode_tab.busy


def test_file_payload_round_trip_and_save(app, cover, tmp_path):
    secret = bytes(range(256)) * 8
    src = tmp_path / "secret.bin"
    src.write_bytes(secret)
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    tab.set_payload_file(src)
    assert tab.mode_var.get() == "file"
    used, _room = tab.update_meter()
    assert used == len(secret) + 2 + len("secret.bin")

    out = tmp_path / "carrier.bmp"
    tab.hide(out)
    pump(app)
    assert out.exists()

    dec = app.decode_tab
    dec.decode(out)
    pump(app)
    assert dec.headline.cget("text").startswith("Hidden file: secret.bin")
    saved = dec.save_recovered(tmp_path / "recovered")
    assert Path(saved).name == "secret.bin"
    assert Path(saved).read_bytes() == secret
    again = dec.save_recovered(tmp_path / "recovered")
    assert again != saved and Path(saved).read_bytes() == secret  # never overwrites


def test_compressed_file_payload_is_measured_in_background(app, cover, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("repeat me " * 2000)
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    tab.compress_var.set(True)
    tab.set_payload_file(src)
    pump(app)
    used, _room = tab.update_meter()
    assert used == gui.frame_body_len(src.read_bytes(), filename="notes.txt", compress=True)
    assert used < src.stat().st_size


def test_capacity_meter_fill_max_and_over_capacity(app, cover, tmp_path):
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    room = tab.capacity[1]

    app.fill_max_message()
    assert len(tab.get_message()) == room
    assert tab.update_meter() == (room, room)
    assert tab.remaining_label.cget("text") == "Remaining: 0 bytes"

    tab.set_message(tab.get_message() + "!")
    assert tab.update_meter() == (room + 1, room)
    assert "Over capacity by 1 bytes" in tab.remaining_label.cget("text")
    too_big = tmp_path / "too_big.png"
    tab.hide(too_big)
    pump(app)
    assert last_message(app)[:2] == ("error", "Payload too large")
    assert not too_big.exists()

    # With a password the filler leaves room for the 44 bytes of encryption.
    tab.password_var.set("pw")
    app.fill_max_message()
    assert len(tab.get_message()) == room - stego.ENCRYPTION_OVERHEAD
    exact = tmp_path / "exact.png"
    tab.hide(exact)
    pump(app)
    assert exact.exists()
    assert tab.last_summary["utilization"] == pytest.approx(1.0, abs=1e-3)

    # Non-ASCII text counts UTF-8 bytes, not characters.
    tab.password_var.set("")
    tab.set_message("é" * 10)
    assert tab.update_meter()[0] == 20


def test_lossy_output_is_refused_and_nothing_written(app, cover, tmp_path):
    tab = app.encode_tab
    tab.set_cover(cover)
    pump(app)
    tab.set_message("hello")
    out = tmp_path / "stego.jpg"
    tab.hide(out)
    pump(app)
    assert last_message(app)[:2] == ("error", "Lossless output needed")
    assert not out.exists()
    assert "Lossless" in tab.result_label.cget("text")
    assert not tab.busy


def test_wav_cover_round_trip(app, tmp_path):
    wav = write_wav(tmp_path / "tone.wav")
    tab = app.encode_tab
    tab.set_cover(wav)
    pump(app)
    assert tab.capacity[1] == stego.capacity_bytes(wav, 1)
    assert tab.preview_cover.image is None
    assert "WAV audio" in tab.preview_cover.label.cget("text")
    tab.set_message("audio secret")
    out = tmp_path / "tone_stego.wav"
    tab.hide(out)
    pump(app)
    assert out.exists()

    app.decode_tab.decode(out)
    pump(app)
    assert app.decode_tab.recovered_text() == "audio secret"


def test_legacy_format_fallback(app, synthetic_rgb, save_png):
    path = save_png(legacy_image(synthetic_rgb(1), "old style message ends in d"), "legacy.png")
    app.decode_tab.decode(path)
    pump(app)
    assert app.decode_tab.recovered_text() == "old style message ends in d"
    assert app.decode_tab.found_format == "legacy"
    assert "Legacy format" in app.decode_tab.detail.cget("text")


def test_clean_image_and_missing_file_errors(app, cover, tmp_path):
    app.decode_tab.decode(cover)
    pump(app)
    assert app.decode_tab.headline.cget("text") == "No hidden payload found"

    app.decode_tab.decode(tmp_path / "nope.png")
    pump(app)
    assert app.decode_tab.headline.cget("text") == "File not found"
    assert app.decode_tab.save_recovered(tmp_path) is None  # nothing to save
    assert last_message(app)[0] == "warning"


def test_missing_inputs_warn_instead_of_running(app):
    assert app.encode_tab.hide() is False
    assert app.decode_tab.decode() is False
    assert app.analyze_tab.analyze() is False
    assert app.bitplanes_tab.load() is False
    assert app.bruteforce_tab.run() is False
    assert app.compare_tab.compare() is False
    assert [m[0] for m in app.messages] == ["warning"] * 6
    assert not app.busy


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def test_analyze_runs_off_the_ui_thread_and_fills_the_table(app, cover, tmp_path, monkeypatch):
    stego_png = tmp_path / "suspect.png"
    stego.hide(cover, stego_png, "There is a frame in here. " * 20, key_scatter=False)

    threads = []
    real = gui.report.analyze_image

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(gui.report, "analyze_image", spy)
    tab = app.analyze_tab
    assert tab.analyze(stego_png) is True
    # The handler returned before the work was done, and a second run is refused.
    assert app.busy and tab.busy
    assert tab.analyze() is False
    pump(app)

    assert threads and all(t is not threading.main_thread() for t in threads)
    assert len(threads) == 1
    rows = tab.tree.get_children()
    assert set(rows) == {d["name"] for d in detectors.list_detectors()}
    for iid in rows:
        name, score, verdict, summary = tab.tree.item(iid, "values")
        assert name == iid and 0.0 <= float(score) <= 1.0 and summary
    assert tab.result["overall"]["verdict"] == "likely_stego"
    assert tab.banner.cget("text").startswith("Likely stego")
    assert tab.result["payload"]["preview"].startswith("There is a frame")
    assert "suspect.png" in tab.summary_text.get("1.0", "end-1c")  # report.format_text output

    # Selecting a detector shows its full JSON result.
    tab.tree.selection_set("signatures")
    app.root.update()
    assert json.loads(tab.detail_text.get("1.0", "end-1c"))["name"] == "signatures"

    for fmt in ("md", "html", "json"):
        assert tab.export_report(tmp_path / "reports", fmt) is True
        pump(app)
        assert tab.last_report.endswith(f"suspect_report.{fmt}")
        assert Path(tab.last_report).exists()
    data = json.loads(Path(tmp_path / "reports" / "suspect_report.json").read_text())
    assert data["overall"]["verdict"] == "likely_stego"
    assert app.callback_errors == []


def test_analyze_clean_cover_banner_matches_verdict(app, cover):
    tab = app.analyze_tab
    tab.bruteforce_var.set(False)
    tab.analyze(cover)
    pump(app)
    verdict = tab.result["overall"]["verdict"]
    assert verdict == "clean"
    text, fg, bg = gui.VERDICT_STYLE[verdict]
    assert tab.banner.cget("text").startswith(text)
    assert tab.banner.cget("bg") == bg
    assert tab.result["bruteforce"] is None


def test_analyze_failure_is_reported_and_tab_recovers(app, tmp_path):
    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("plain text, not a PNG")
    tab = app.analyze_tab
    tab.analyze(bogus)
    pump(app)
    assert last_message(app)[0] == "error"
    assert tab.result is None and not tab.busy
    assert tab.export_button.instate(["disabled"])
    assert tab.export_report(tmp_path) is False


# ---------------------------------------------------------------------------
# Bitplanes
# ---------------------------------------------------------------------------

def test_bitplanes_tab_shows_selected_plane_amplified_and_grid(app, synthetic_rgb, save_png, tmp_path):
    arr = synthetic_rgb(2, 128)
    arr[..., 2] = (arr[..., 2] & 0xFE) | (np.indices((128, 128)).sum(axis=0) % 2).astype(np.uint8)
    path = save_png(arr, "planes.png")
    tab = app.bitplanes_tab
    assert tab.load(path) is True
    pump(app)
    assert tab.view_amplified.image is not None
    assert tab.view_grid.image is not None
    assert Path(tab.grid_path).exists()
    assert Path(tab.grid_path).parent == tmp_path / "out"
    assert tab.channel_buttons["A"].instate(["disabled"])

    tab.channel_var.set("B")
    tab.bit_var.set(0)
    plane = tab.show_plane()
    np.testing.assert_array_equal(plane, bitplanes.bitplane(arr, 2, 0))
    assert "50.0% of pixels are 1" in tab.plane_info.cget("text")  # the checkerboard
    tab.channel_var.set("R")
    tab.bit_var.set(7)
    np.testing.assert_array_equal(tab.show_plane(), bitplanes.bitplane(arr, 0, 7))
    app.root.update()
    assert tab.view_plane.image is not None


def test_bitplanes_alpha_channel_is_selectable_for_rgba(app, synthetic_rgb, save_png):
    rgba = np.dstack([synthetic_rgb(3, 64), np.full((64, 64), 200, np.uint8)])
    tab = app.bitplanes_tab
    tab.load(save_png(rgba, "alpha.png"))
    pump(app)
    assert tab.channel_buttons["A"].instate(["!disabled"])
    tab.channel_var.set("A")
    tab.bit_var.set(3)
    assert tab.show_plane().max() == 255  # 200 has bit 3 set


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------

def test_bruteforce_tab_finds_text_off_the_ui_thread_and_saves_it(app, synthetic_rgb, save_png,
                                                                  tmp_path, monkeypatch):
    message = "Hidden only in the blue channel, read by brute force."
    arr = synthetic_rgb(4)
    bits = bytes_to_bits(message.encode("utf-8") + b"\x00" * 8)
    blue = arr[..., 2].reshape(-1)
    blue[: bits.size] = (blue[: bits.size] & 0xFE) | bits
    arr[..., 2] = blue.reshape(arr.shape[:2])
    path = save_png(arr, "blue_only.png")

    threads = []
    real = gui.bruteforce.bruteforce

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(gui.bruteforce, "bruteforce", spy)
    tab = app.bruteforce_tab
    tab.top_var.set(5)
    assert tab.run(path) is True
    assert app.busy
    pump(app)
    assert threads and threads[0] is not threading.main_thread()
    assert len(tab.candidates) == 5 == len(tab.tree.get_children())
    best = tab.candidates[0]
    assert best["kind"] == "text" and best["label"].startswith("B bit0")
    assert message[:40] in best["preview"]
    assert tab.tree.selection() == ("0",)
    assert message[:40] in tab.preview_text.get("1.0", "end-1c")

    saved = tab.save_candidate(tmp_path / "saved")
    assert message in Path(saved).read_text(encoding="utf-8")


def test_password_list_attack_finds_scattered_payload(app, cover, tmp_path):
    out = tmp_path / "scattered.png"
    stego.hide(cover, out, "scattered secret", password="hunter2")
    words = tmp_path / "words.txt"
    words.write_text("password\nletmein\nhunter2\n", encoding="utf-8")
    tab = app.bruteforce_tab
    tab.set_path(out)
    assert tab.run_wordlist(words) is True
    pump(app)
    hit = tab.candidates[0]
    assert hit["password"] == "hunter2"
    assert "hunter2" in tab.tree.item("0", "values")[1]
    saved = tab.save_candidate(tmp_path / "hits", index=0)
    assert Path(saved).read_bytes() == b"scattered secret"


# ---------------------------------------------------------------------------
# Compare, toolbar, File > Open, Reset
# ---------------------------------------------------------------------------

def test_compare_tab_histograms_and_lsb_diff(app, cover, tmp_path, synthetic_rgb, save_png):
    tracked = REPO_ROOT / "results" / "comparison_analysis.png"
    before = tracked.stat().st_mtime_ns if tracked.exists() else None

    stego_png = tmp_path / "full.png"
    stego.hide(cover, stego_png, "x" * 20000, key_scatter=False)
    tab = app.compare_tab
    assert tab.compare(cover, stego_png) is True
    pump(app)
    assert Path(tab.figure_path) == tmp_path / "out" / "comparison_analysis.png"
    assert Path(tab.figure_path).exists()
    assert tab.view_hist.image is not None and tab.view_diff.image is not None
    changed, total = tab.changed
    assert total == 256 * 256 and 0 < changed < total
    stats = tab.stats_label.cget("text")
    assert "cover" in stats and "stego" in stats and "changed R, G or B LSB" in stats

    # Different sizes: histograms still drawn, no pixel mask.
    small = save_png(synthetic_rgb(5, 64), "small.png")
    tab.compare(cover, small)
    pump(app)
    assert tab.changed is None
    assert "differ in size" in tab.stats_label.cget("text")
    # The repository's tracked figure was not touched.
    after = tracked.stat().st_mtime_ns if tracked.exists() else None
    assert before == after


def test_resize_for_test_and_reset_all(app, synthetic_rgb, save_png, tmp_path):
    big = save_png(synthetic_rgb(6, 600), "big.png")
    app.encode_tab.set_cover(big)
    pump(app)
    app.resize_image()
    pump(app)
    resized = Path(app.encode_tab.cover_var.get())
    assert resized == tmp_path / "out" / "big_resized.png"
    with Image.open(resized) as img:
        assert max(img.size) == 400
    assert app.encode_tab.capacity[1] == stego.capacity_bytes(resized, 1)

    app.decode_tab.set_path(big)
    app.analyze_tab.set_path(big)
    app.reset_all()
    assert app.encode_tab.cover_var.get() == ""
    assert app.encode_tab.capacity is None
    assert app.decode_tab.path_var.get() == ""
    assert app.analyze_tab.path_var.get() == ""
    assert app.status.cget("text") == "Ready"


def test_file_open_goes_to_the_current_tab(app, cover):
    app.notebook.select(app.analyze_tab)
    app.open_file(cover)
    assert app.analyze_tab.path_var.get() == str(cover)
    app.notebook.select(app.decode_tab)
    app.open_file(cover)
    assert app.decode_tab.path_var.get() == str(cover)
    app.notebook.select(app.encode_tab)
    app.open_file(cover)
    pump(app)
    assert app.encode_tab.cover_var.get() == str(cover)
    assert app.encode_tab.capacity is not None
    # With dialogs suppressed, Open without a path is a cancelled dialog.
    app.open_file()
    assert app.callback_errors == []


def test_import_switches_matplotlib_to_agg():
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
