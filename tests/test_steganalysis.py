"""Tests for steganalysis.py, the cover-vs-stego histogram plots.

The figures themselves are only checked for existence. What matters here is
that every title names the channel whose data is actually plotted, whichever
caller picks the channel (a direct call, ``main.py compare --channel`` or the
GUI's Compare tab), and that the module's description of its callers matches
the code.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: steganalysis imports pyplot

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import main  # noqa: E402
import steganalysis  # noqa: E402

# Each channel draws its values from its own band [base, base + 64), so a
# plotted array shows which channel it came from. The bases are even, so
# flipping an LSB keeps a value inside its band.
BAND = 64
CHANNEL_BASE = {"r": 0, "g": 64, "b": 128}
NAMES = {"r": "Red", "g": "Green", "b": "Blue"}
SINGLE_TITLES = ("Original Image: Standard Histogram",
                 "Encoded Image: Standard Histogram",
                 "Original Image: Difference Analysis",
                 "Encoded Image: Difference Analysis")


@pytest.fixture
def titles(monkeypatch):
    """Record every Axes title and figure suptitle drawn during the test."""
    seen: list[str] = []
    orig_title, orig_sup = Axes.set_title, Figure.suptitle

    def set_title(self, label, *args, **kwargs):
        seen.append(str(label))
        return orig_title(self, label, *args, **kwargs)

    def suptitle(self, t, *args, **kwargs):
        seen.append(str(t))
        return orig_sup(self, t, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", set_title)
    monkeypatch.setattr(Figure, "suptitle", suptitle)
    return seen


@pytest.fixture
def plotted(monkeypatch):
    """Record (title, set of values) for every histogram analyze_comparison draws."""
    seen: list[tuple[str, set[int]]] = []
    for name in ("_std_hist", "_pairwise_hist"):
        orig = getattr(steganalysis, name)

        def spy(ax, chan_data, title, *args, _orig=orig, **kwargs):
            seen.append((title, {int(v) for v in np.unique(chan_data)}))
            return _orig(ax, chan_data, title, *args, **kwargs)

        monkeypatch.setattr(steganalysis, name, spy)
    return seen


@pytest.fixture
def pair(save_png):
    """A cover and a stego image whose R, G and B values lie in disjoint bands."""
    rng = np.random.default_rng(0)
    arr = np.empty((32, 32, 3), dtype=np.uint8)
    for i, ch in enumerate("rgb"):
        arr[..., i] = CHANNEL_BASE[ch] + rng.integers(0, BAND, size=(32, 32))
    # Overwrite every LSB with a random bit, as a saturated embed would.
    stego_arr = (arr & 0xFE) | rng.integers(0, 2, size=arr.shape, dtype=np.uint8)
    return save_png(arr, "cover.png"), save_png(stego_arr, "stego.png")


def _band(channel: str) -> set[int]:
    return set(range(CHANNEL_BASE[channel], CHANNEL_BASE[channel] + BAND))


def _single_plot_titles(seen: list[str]) -> list[str]:
    return [t for t in seen if t.startswith(SINGLE_TITLES)]


@pytest.mark.parametrize("channel", ["r", "g", "b", "R"])
def test_analyze_comparison_titles_name_the_plotted_channel(pair, tmp_path, titles, plotted,
                                                             channel):
    key = channel.lower()
    paths = steganalysis.analyze_comparison(*map(str, pair), output_path=str(tmp_path / "c.png"),
                                            channel=channel)
    for name in ("composite", "p1", "p2", "p3", "p4"):
        assert Path(paths[name]).is_file()

    singles = _single_plot_titles(titles)
    assert len(singles) == 4
    assert all(t.endswith(f"({NAMES[key]} Channel)") for t in singles), singles
    sup = [t for t in titles if t.startswith("Comprehensive Steganalysis")]
    assert sup == [f"Comprehensive Steganalysis: Before vs. After ({NAMES[key]} Channel)"]

    # The data under each single-plot title really is that channel.
    single_data = [vals for title, vals in plotted if title.startswith(SINGLE_TITLES)]
    assert len(single_data) == 4
    assert all(vals <= _band(key) for vals in single_data)


def test_analyze_comparison_unknown_channel_is_labeled_blue(pair, tmp_path, titles, plotted):
    steganalysis.analyze_comparison(*map(str, pair), output_path=str(tmp_path / "c.png"),
                                    channel="x")
    singles = _single_plot_titles(titles)
    assert len(singles) == 4 and all(t.endswith("(Blue Channel)") for t in singles)
    assert plotted and all(vals <= _band("b") for _, vals in plotted)


def test_main_compare_channel_reaches_titles(pair, tmp_path, titles, capsys):
    cover, stego_path = map(str, pair)
    out = tmp_path / "cmp" / "comparison.png"
    assert main.main(["compare", cover, stego_path, "-o", str(out), "--channel", "g"]) == 0
    assert "green channel" in capsys.readouterr().out
    singles = _single_plot_titles(titles)
    assert len(singles) == 4 and all(t.endswith("(Green Channel)") for t in singles), singles


def test_gui_compare_tab_channel_reaches_titles(pair, tmp_path, titles):
    tk = pytest.importorskip("tkinter")
    gui = pytest.importorskip("gui")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk cannot open a display: {exc}")
    root.withdraw()
    app = gui.SteganoApp(root, output_dir=tmp_path / "out", interactive=False)
    try:
        tab = app.compare_tab
        tab.channel_var.set("Red")
        assert tab.compare(*pair) is True
        deadline = time.monotonic() + 60
        while app.busy:
            if time.monotonic() > deadline:
                pytest.fail("compare did not finish in time")
            root.update()
            time.sleep(0.01)
        root.update()
        assert "red channel" in tab.stats_label.cget("text")
        singles = _single_plot_titles(titles)
        assert len(singles) == 4 and all(t.endswith("(Red Channel)") for t in singles), singles
    finally:
        app.close()
        try:
            root.destroy()
        except tk.TclError:
            pass  # already destroyed by app.close()


@pytest.mark.parametrize("stale", [
    "does not use this module",       # main.py compare calls analyze_comparison
    "in a subprocess",                # make_all.py imports and calls analyze()
    "python3 <script>",
    "must be run from the repository root",
    "reaches this through the CLI",
    "gui.py never passes",            # the Compare tab passes its channel
    "always analyzes blue",
])
def test_docstrings_do_not_describe_the_old_wiring(stale):
    docs = " ".join(d for d in (steganalysis.__doc__, steganalysis.analyze.__doc__,
                                steganalysis.analyze_comparison.__doc__) if d)
    assert stale not in " ".join(docs.split())


def test_docstrings_describe_the_current_callers(monkeypatch, tmp_path):
    import make_all

    doc = " ".join(steganalysis.__doc__.split())
    assert "STGA frame" in doc and "main.py compare" in doc and "Compare tab" in doc
    # make_all.run_steganalysis calls analyze() in-process, as documented.
    # (main.py compare and the Compare tab are covered by the title tests.)
    calls = []
    monkeypatch.setattr(steganalysis, "analyze", lambda *a, **k: calls.append((a, k)) or {})
    make_all.run_steganalysis("c.png", "s.png", str(tmp_path), "t", channel="R")
    assert calls == [(("c.png", "s.png"), {"out_dir": str(tmp_path), "tag": "t", "channel": "r"})]
