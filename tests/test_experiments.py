"""Tests for the experiment runner (make_all.py) and the pairwise_stats.py fixes.

Every test writes only under pytest's tmp_path; nothing goes to results/.
The detection runs use --quick settings (two synthetic 128 px covers, four
rates), so the whole module takes a few seconds.
"""
from __future__ import annotations

import csv
import inspect
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: steganalysis imports pyplot

import numpy as np
import pytest
from PIL import Image

import decoder
import detectors
import make_all
import pairwise_stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def _quiet(_msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Detection experiment
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    """One --quick detection run shared by the tests below."""
    out = tmp_path_factory.mktemp("experiments") / "quick"
    summary = make_all.run_detection(out, quick=True, log=_quiet)
    with open(out / "detection.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return out, summary, rows


def _point(summary, variant, det, rate):
    for p in summary["curves"][variant][det]:
        if p["rate"] == rate:
            return p
    raise KeyError((variant, det, rate))


def test_quick_run_writes_csv_json_tex_and_figures(quick_run):
    out, summary, rows = quick_run
    names = [d["name"] for d in detectors.list_detectors()]

    assert (out / "detection.csv").is_file()
    assert (out / "detection_summary.json").is_file()
    assert (out / "detection_table.tex").is_file()
    expected_figs = [f"detection_{n}.png" for n in names + [make_all.OVERALL]]
    expected_figs.append("detection_combined.png")
    for fname in expected_figs:
        path = out / fname
        assert path.is_file(), fname
        with Image.open(path) as im:
            assert im.format == "PNG" and im.size[0] > 300 and im.size[1] > 300
    assert set(summary["files"]["figures"]) == set(names + [make_all.OVERALL, "combined"])

    # Every file the summary lists exists and lives inside the output folder.
    listed = [summary["files"][k] for k in ("csv", "latex", "json")]
    listed += list(summary["files"]["figures"].values())
    for p in listed:
        assert Path(p).is_file()
        assert Path(p).resolve().parent == out.resolve()


def test_quick_run_csv_layout(quick_run):
    out, summary, rows = quick_run
    with open(out / "detection.csv", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert tuple(header) == make_all.CSV_FIELDS

    cfg = summary["config"]
    n_det = len(detectors.list_detectors()) + 1  # + overall
    n_variants = len(cfg["variants"])
    n_nonzero = sum(r > 0 for r in cfg["rates"])
    per_cover = n_variants * (1 + n_nonzero * cfg["trials"]) * n_det
    assert len(rows) == len(cfg["covers"]) * per_cover
    assert {r["detector"] for r in rows} == {d["name"] for d in detectors.list_detectors()} | {"overall"}
    assert {r["variant"] for r in rows} == set(cfg["variants"])
    for r in rows:
        assert 0.0 <= float(r["score"]) <= 1.0
        assert r["verdict"] in detectors.VERDICTS
        if r["detector"] in make_all.ESTIMATE_KEYS:
            assert r["estimate"] != ""


def test_quick_run_json_is_strict_and_complete(quick_run):
    out, summary, _ = quick_run
    on_disk = json.loads((out / "detection_summary.json").read_text())
    json.dumps(on_disk, allow_nan=False)
    assert on_disk["experiment"] == "detection"
    assert on_disk["config"]["quick"] is True
    assert on_disk["config"]["rates"] == list(make_all.QUICK_RATES)
    assert [c["source"] for c in on_disk["config"]["covers"]] == [
        "synthetic:natural", "synthetic:stretched"]
    for variant in on_disk["config"]["variants"]:
        assert set(on_disk["headline"][variant]) >= {"sample_pair_analysis", "overall"}
    # Paths on disk are relative to the output folder, so it can be moved.
    assert on_disk["files"]["csv"] == "detection.csv"
    assert on_disk["files"]["figures"]["combined"] == "detection_combined.png"
    assert all(not Path(p).is_absolute() for p in on_disk["files"]["figures"].values())


def test_quick_run_latex_table(quick_run):
    out, summary, _ = quick_run
    tex = (out / "detection_table.tex").read_text()
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex and r"\end{table}" in tex
    for variant in summary["config"]["variants"]:
        assert make_all.variant_label(variant) in tex
    assert "SPA" in tex and "RS" in tex and "Overall" in tex
    body = "\n".join(line for line in tex.splitlines() if not line.startswith("%"))
    assert "_" not in body.replace(r"\_", "")  # underscores escaped outside comments


def test_true_rate_and_changed_fraction(quick_run):
    _, summary, rows = quick_run
    n = 128 * 128 * 3
    for r in rows:
        nominal, true = float(r["nominal_rate"]), float(r["true_rate"])
        if r["payload"] == "random":
            assert true == pytest.approx(round(nominal * n) / n)
        else:  # encrypted: whole bytes, so at most 7 bits short
            assert nominal - 8 / n <= true <= nominal + 1e-12
        if r["detector"] == "overall" and r["payload"] == "random" and nominal == 1.0:
            # A random bit already matches the LSB half of the time.
            assert 0.45 < float(r["changed_fraction"]) < 0.55


def test_quick_run_detector_behaviour(quick_run):
    """Coarse sanity checks: clean covers stay clean, estimators track the
    rate, and the STGA header is visible only without key scatter."""
    _, summary, _ = quick_run
    for variant in summary["config"]["variants"]:
        clean = summary["headline"][variant]["overall"]
        assert clean["score_at_zero"] < detectors.SUSPICIOUS_AT
        assert clean["flagged_at_zero"] == 0.0
        for det in ("sample_pair_analysis", "rs_analysis"):
            full = _point(summary, variant, det, 1.0)
            assert full["score_mean"] >= detectors.LIKELY_AT
            if variant.startswith("random"):
                # Only the random-bit variants are fully seeded; the encrypted
                # ones get a fresh salt and nonce per run.
                assert full["estimate_mean"] > 0.8
                half = _point(summary, variant, det, 0.5)
                assert abs(half["estimate_mean"] - 0.5) < 0.15

    seq = summary["curves"]["encrypted-sequential"]["signatures"]
    assert all(p["score_mean"] == 1.0 for p in seq if p["rate"] > 0)
    scat = summary["curves"]["encrypted-scattered"]["signatures"]
    assert all(p["score_mean"] < detectors.SUSPICIOUS_AT for p in scat)


def test_cli_quick_without_plots(tmp_path):
    rc = make_all.main(["detection", "--quick", "--out", str(tmp_path), "--tag", "cli",
                        "--rates", "0,1", "--payloads", "random", "--modes", "sequential",
                        "--no-plots"])
    assert rc == 0
    out = tmp_path / "cli"
    assert (out / "detection.csv").is_file()
    assert (out / "detection_summary.json").is_file()
    assert not list(out.glob("*.png"))
    summary = json.loads((out / "detection_summary.json").read_text())
    assert summary["tag"] == "cli"
    assert summary["config"]["variants"] == ["random-sequential"]


def test_cli_reports_bad_arguments(tmp_path, capsys):
    rc = make_all.main(["detection", "--quick", "--out", str(tmp_path), "--rates", "0,1.5"])
    assert rc == 1
    assert "rates must be in [0, 1]" in capsys.readouterr().err


def test_run_detection_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        make_all.run_detection(tmp_path / "a", quick=True, rates=[-0.1], log=_quiet)
    with pytest.raises(ValueError):
        make_all.run_detection(tmp_path / "b", quick=True, payloads=["bogus"], log=_quiet)
    with pytest.raises(ValueError):
        make_all.run_detection(tmp_path / "c", quick=True, modes=["diagonal"], log=_quiet)
    with pytest.raises(ValueError):
        make_all.run_detection(tmp_path / "d", quick=True, trials=0, log=_quiet)
    with pytest.raises(FileNotFoundError):
        make_all.run_detection(tmp_path / "e", covers=[tmp_path / "missing.png"], log=_quiet)


def test_out_dir_has_no_default():
    """The library never picks results/ on its own; only the CLI does."""
    param = inspect.signature(make_all.run_detection).parameters["out_dir"]
    assert param.default is inspect.Parameter.empty


def test_path_cover_with_alpha_text_payload_and_kept_images(tmp_path, synthetic_rgb):
    rgba = np.dstack([synthetic_rgb(3, 64), np.full((64, 64), 200, np.uint8)])
    cover = tmp_path / "rgba.png"
    Image.fromarray(rgba, "RGBA").save(cover)
    out = tmp_path / "run"
    summary = make_all.run_detection(out, covers=[cover], rates=[0, 0.5, 1.0],
                                     payloads=["text", "encrypted"], modes=["sequential"],
                                     keep_images=True, plots=False, log=_quiet)
    assert summary["config"]["covers"][0]["name"] == "rgba"
    kept = sorted((out / "stego").glob("*.png"))
    assert len(kept) == 4  # 2 payloads x 2 nonzero rates
    with Image.open(kept[0]) as im:
        assert im.mode == "RGBA"
    text = summary["curves"]["text-sequential"]["signatures"]
    assert _point(summary, "text-sequential", "signatures", 1.0)["score_mean"] > 0
    assert text[0]["rate"] == 0.0


def test_encrypted_variant_skipped_when_overhead_exceeds_rate(tmp_path, synthetic_rgb):
    cover = make_all.Cover("tiny", "test", synthetic_rgb(1, 32))  # 3072 samples
    summary = make_all.run_detection(tmp_path / "tiny", covers=[cover], rates=[0, 0.01, 1.0],
                                     payloads=["encrypted"], modes=["sequential"],
                                     plots=False, log=_quiet)
    assert summary["skipped"] and summary["skipped"][0]["nominal_rate"] == 0.01
    rates = [p["rate"] for p in summary["curves"]["encrypted-sequential"]["overall"]]
    assert rates == [0.0, 1.0]


def test_synthetic_covers_are_deterministic():
    for kind in make_all.SYNTHETIC_KINDS:
        a = make_all.synthetic_cover(kind, seed=7, size=64)
        b = make_all.synthetic_cover(kind, seed=7, size=64)
        c = make_all.synthetic_cover(kind, seed=8, size=64)
        assert a.shape == (64, 64, 3) and a.dtype == np.uint8
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)
    stretched = make_all.synthetic_cover("stretched", seed=7, size=128)
    # 181 output levels: some byte values never occur (pair-scale structure).
    assert len(np.unique(stretched[..., 0])) <= 181
    with pytest.raises(ValueError):
        make_all.synthetic_cover("plaid")


def test_load_covers_names_are_unique(tmp_path, synthetic_rgb, save_png):
    p = save_png(synthetic_rgb(0, 32), "same.png")
    covers = make_all.load_covers([p, p, "synthetic:natural", "synthetic:natural"], size=32)
    names = [c.name for c in covers]
    assert len(set(names)) == 4
    assert names[0] == "same" and names[1] == "same-2"
    # Same kind, different list position -> different seed -> different image.
    assert not np.array_equal(covers[2].pixels, covers[3].pixels)


# --------------------------------------------------------------------------- #
# Classic experiment
# --------------------------------------------------------------------------- #

def _original_embed_loop(arr: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
    """The pre-rewrite embed_bits_rgb loop, copied verbatim in spirit."""
    flat = arr.copy().reshape(-1, 3)
    stream = np.concatenate([payload_bits, np.zeros(64, dtype=np.uint8)])
    idx = 0
    for i in range(flat.shape[0]):
        for c in range(3):
            if idx >= stream.size:
                break
            flat[i, c] = (flat[i, c] & 0xFE) | stream[idx]
            idx += 1
        if idx >= stream.size:
            break
    return flat.reshape(arr.shape)


@pytest.mark.parametrize("n_bits", [0, 1, 7, 100, 20 * 15 * 3 - 64])
def test_embed_bits_rgb_matches_original_loop(tmp_path, n_bits):
    rng = np.random.default_rng(n_bits)
    arr = rng.integers(0, 256, (15, 20, 3), dtype=np.uint8)
    bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "c.png")
    W, H, n = make_all.embed_bits_rgb(str(tmp_path / "c.png"), str(tmp_path / "s.png"), bits)
    assert (W, H, n) == (20, 15, n_bits + 64)
    got = np.asarray(Image.open(tmp_path / "s.png"))
    assert np.array_equal(got, _original_embed_loop(arr, bits))


def test_embed_bits_rgb_rejects_oversized_payload(tmp_path):
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "c.png")
    with pytest.raises(ValueError):
        make_all.embed_bits_rgb(str(tmp_path / "c.png"), str(tmp_path / "s.png"),
                                np.zeros(4 * 4 * 3 - 63, np.uint8))


def test_fit_text_bits_cuts_at_whole_characters():
    bits = make_all.fit_text_bits("ééé", 4 * 8 - 1)  # room for 3 bytes
    data = np.packbits(bits).tobytes()
    assert data.decode("utf-8") == "é"  # the half 'e-acute' is dropped
    assert make_all.fit_text_bits("abc", 1000).size == 24
    assert make_all.fit_text_bits("abc", 0).size == 0


def test_classic_smoke(tmp_path, synthetic_rgb, save_png):
    cover = save_png(synthetic_rgb(2, 48)[:36], "dino.png")  # 48x36 RGB
    out = tmp_path / "classic"
    res = make_all.run_classic(str(cover), str(out), resize=20, long_multiplier=20,
                               rid="RID", log=_quiet)
    assert res["tags"] == ["RID_full-low", "RID_resized-high", "RID_full-high"]
    assert (out / "dino_RID_resized-high_resized.png").is_file()
    for case in ("full-low", "resized-high", "full-high"):
        assert (out / f"dino_RID_{case}_enc.png").is_file()
        for kind in ("std_cover", "std_stego", "pair_cover", "pair_stego", "comp"):
            assert (out / f"RID_{case}_b_{kind}.png").is_file()
        stats = json.loads((out / f"RID_{case}_pairwise.json").read_text())
        assert stats["channel"] == "B" and stats["tag"] == f"RID_{case}"
        assert set(stats["delta"]) == {"L1_drop", "L2_drop", "median_abs_drop"}
        assert "(blue channel)" in (out / f"RID_{case}_pairwise.tex").read_text()

    # The default message ends in '.', which the old bit-offset decoder
    # dropped; the byte-aligned legacy reader keeps it.
    assert decoder.decode(str(out / "dino_RID_full-low_enc.png")) == make_all.DEFAULT_SHORT
    # The high-rate cases fill the whole capacity: every pixel's LSBs written.
    full = res["cases"]["resized-high"]
    assert full["bits_written"] == 20 * 20 * 3
    assert full["bits_written"] % 8 == 0


def test_classic_cli_old_form_and_channel_caption(tmp_path, synthetic_rgb, save_png):
    cover = save_png(synthetic_rgb(4, 32), "c.png")
    out = tmp_path / "old"
    rc = make_all.main([str(cover), "--out", str(out), "--resize", "16",
                        "--long-multiplier", "5", "--channel", "r"])
    assert rc == 0
    assert len(list(out.glob("*_r_comp.png"))) == 3
    texs = sorted(out.glob("*_pairwise.tex"))
    assert len(texs) == 3
    assert all("(red channel)" in p.read_text() for p in texs)
    assert all("blue" not in p.read_text() for p in texs)


def test_classic_rejects_bad_channel(tmp_path, rgb_png):
    with pytest.raises(ValueError):
        make_all.run_classic(str(rgb_png), str(tmp_path / "x"), channel="a", log=_quiet)


def test_normalize_argv_keeps_old_form():
    assert make_all._normalize_argv(["cover.png", "--out", "x"]) == ["classic", "cover.png", "--out", "x"]
    assert make_all._normalize_argv(["detection", "--quick"]) == ["detection", "--quick"]
    assert make_all._normalize_argv(["classic", "c.png"]) == ["classic", "c.png"]
    assert make_all._normalize_argv(["-h"]) == ["-h"]


@pytest.mark.parametrize("argv", [
    # Regression: the original flat parser accepted options before the cover;
    # these used to reach the subparser as "invalid choice: 'x'" (exit 2).
    ["--out", "x", "--resize", "64", "cover.png"],
    ["--out=x", "cover.png"],
    ["--res", "64", "--long", "5", "cover.png"],  # argparse prefix abbreviations
    ["--short", "classic", "cover.png"],          # a value equal to a subcommand name
    ["--out", "x", "--", "cover.png"],
])
def test_normalize_argv_options_before_cover(argv):
    norm = make_all._normalize_argv(argv)
    assert norm == ["classic", *argv]
    args = make_all.build_parser().parse_args(norm)
    assert args.command == "classic" and args.cover == "cover.png"


def test_normalize_argv_leaves_top_level_cases_alone():
    assert make_all._normalize_argv([]) == []
    assert make_all._normalize_argv(["--help"]) == ["--help"]
    assert make_all._normalize_argv(["classic", "--out", "x", "c.png"]) == ["classic", "--out", "x", "c.png"]
    # Options without a cover go to classic, so the error names the cover.
    assert make_all._normalize_argv(["--out", "x"]) == ["classic", "--out", "x"]


def test_classic_cli_options_before_cover(tmp_path, synthetic_rgb, save_png):
    cover = save_png(synthetic_rgb(4, 32), "c.png")
    out = tmp_path / "opts-first"
    rc = make_all.main(["--out", str(out), "--resize", "16", "--long-multiplier", "5",
                        str(cover)])
    assert rc == 0
    for case in ("full-low", "resized-high", "full-high"):
        assert len(list(out.glob(f"c_*_{case}_enc.png"))) == 1


def test_classic_cli_without_cover_names_it(capsys):
    with pytest.raises(SystemExit) as exc:
        make_all.main(["--out", "x"])
    assert exc.value.code == 2
    assert "required: cover" in capsys.readouterr().err


def _original_cut_stego(cover: Path, out: Path, text: str) -> np.ndarray:
    """Case C as the original script (a5a3774) embedded it: a bit-level cut."""
    bits = make_all.to_bits_utf8(text)[:make_all.capacity_bits(str(cover)) - make_all.DELIM_BITS]
    make_all.embed_bits_rgb(str(cover), str(out), bits)
    return np.asarray(Image.open(out).convert("RGB")).reshape(-1)


@pytest.mark.parametrize("w, h", [(24, 16), (21, 19)])
def test_classic_full_high_vs_original_bit_cut(tmp_path, synthetic_rgb, save_png, w, h):
    """The module docstring's claim about where classic differs from the original.

    With ASCII text, case C is pixel-identical to the original's bit-level cut
    when W*H is a multiple of 8 (24x16). Otherwise (21x19: 1197 samples,
    1133 payload bits available) the message is cut to 1128 bits, the
    delimiter ends 5 samples early, and only the last few samples differ.
    """
    cover = save_png(synthetic_rgb(3, 32)[:h, :w], "cv.png")
    short, mult = "hi", 40
    res = make_all.run_classic(str(cover), str(tmp_path / "run"), resize=16, short=short,
                               long_multiplier=mult, rid="RID", log=_quiet)
    case = res["cases"]["full-high"]
    new = np.asarray(Image.open(case["stego"]).convert("RGB")).reshape(-1)
    orig = _original_cut_stego(cover, tmp_path / "orig.png", (short + " ") * (mult * 10))
    cov = np.asarray(Image.open(cover).convert("RGB")).reshape(-1)

    n = 3 * w * h
    avail = n - make_all.DELIM_BITS
    payload = avail // 8 * 8  # whole bytes only
    assert case["bits_written"] == payload + make_all.DELIM_BITS
    if (w * h) % 8 == 0:
        assert np.array_equal(new, orig)
    else:
        assert case["bits_written"] < n
        # Same up to the byte-aligned cut; the samples past the shorter
        # stream keep their cover values instead of the original's zeros.
        assert np.array_equal(new[:payload], orig[:payload])
        assert np.array_equal(new[case["bits_written"]:], cov[case["bits_written"]:])
        assert not np.array_equal(new, orig)
        assert np.flatnonzero(new != orig).min() >= payload
    assert decoder.decode(case["stego"]) == ((short + " ") * (mult * 10))[:payload // 8]


# --------------------------------------------------------------------------- #
# pairwise_stats.py fixes
# --------------------------------------------------------------------------- #

def _stats_with_l1(l1: int) -> dict:
    h = np.zeros(256, dtype=np.int64)
    h[0] = l1  # one pair (0, 1) with imbalance l1
    return pairwise_stats.pair_stats(h)


def test_latex_change_row_is_encoded_minus_original():
    Sc, Ss = _stats_with_l1(100), _stats_with_l1(150)
    tex = pairwise_stats.latex_table(Sc, Ss, "cap")
    change = next(line for line in tex.splitlines() if line.startswith("Change"))
    assert "Change (E$-$O) & +50 & +50.000" in change
    tex = pairwise_stats.latex_table(Ss, Sc, "cap")
    change = next(line for line in tex.splitlines() if line.startswith("Change"))
    assert "& -50 & -50.000" in change


def test_default_caption_names_channel():
    assert pairwise_stats.default_caption(0) == "Pairwise even/odd statistics (red channel)."
    assert "green channel" in pairwise_stats.default_caption(1)
    assert "blue channel" in pairwise_stats.default_caption(2)


def test_pairwise_stats_cli_unchanged(tmp_path, synthetic_rgb, save_png):
    cover = synthetic_rgb(5, 32)
    stego = cover.copy()
    stego[..., 0] |= 1  # every red value odd: L1 changes, and grows
    c, s = save_png(cover, "c.png"), save_png(stego, "s.png")
    tex, js = tmp_path / "t.tex", tmp_path / "t.json"
    proc = subprocess.run(
        [sys.executable, "pairwise_stats.py", str(c), str(s), "-c", "r",
         "--json", str(js), "--latex", str(tex), "--tag", "t"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    printed = json.loads(proc.stdout)
    assert printed == json.loads(js.read_text())
    assert printed["channel"] == "R"
    # JSON keeps its documented convention: *_drop = cover - stego.
    assert printed["delta"]["L1_drop"] == printed["cover"]["L1"] - printed["stego"]["L1"]
    text = tex.read_text()
    assert "(red channel)" in text
    grow = printed["stego"]["L1"] - printed["cover"]["L1"]
    assert f"Change (E$-$O) & {grow:+d} &" in text

    # An explicit --caption is still used verbatim.
    subprocess.run(
        [sys.executable, "pairwise_stats.py", str(c), str(s), "--latex", str(tex),
         "--caption", "My caption."],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert "My caption." in tex.read_text()
