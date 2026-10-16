"""Consistency tests for the committed pairwise tables in results/.

The ``results/20250811-171038_*_pairwise.tex`` tables were first written by an
older ``pairwise_stats.latex_table`` that printed Original minus Encoded under
the "Change (E$-$O)" label, so every sign in that row was inverted. They were
regenerated from the committed ``*_pairwise.json`` numbers with the current
code. These tests pin that: each committed table must equal what
``latex_table`` produces today from its JSON sibling, and its change row must
be Encoded minus Original (the negative of the JSON ``*_drop`` fields).

The tests only read files under results/; they never write there.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import pairwise_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUN_ID = "20250811-171038"

_JSON_FILES = sorted(RESULTS.glob(f"{RUN_ID}_*_pairwise.json")) if RESULTS.is_dir() else []


def _load_case(json_path: Path) -> tuple[dict, dict, dict, str, str]:
    """Load one committed case.

    Args:
        json_path: Path to a ``*_pairwise.json`` file in results/.

    Returns:
        ``(record, Sc, Ss, caption, tex)``: the parsed JSON, the cover and
        stego ``pair_stats`` dicts (without the ``file`` key), the caption of
        the committed table, and the committed table text.
    """
    tex_path = json_path.with_suffix(".tex")
    if not tex_path.exists():
        pytest.skip(f"no committed table next to {json_path.name}")
    record = json.loads(json_path.read_text())
    Sc = {k: v for k, v in record["cover"].items() if k != "file"}
    Ss = {k: v for k, v in record["stego"].items() if k != "file"}
    tex = tex_path.read_text()
    m = re.search(r"\\caption\{(.*)\}\n", tex)
    assert m, f"{tex_path.name} has no \\caption line"
    return record, Sc, Ss, m.group(1), tex


def _change_cells(tex: str) -> list[str]:
    """Return the three numeric cells of the "Change (E$-$O)" row."""
    line = next(ln for ln in tex.splitlines() if ln.startswith("Change (E$-$O)"))
    cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
    return cells[1:4]


def test_committed_tables_exist():
    if not RESULTS.is_dir():
        pytest.skip("results/ is not present")
    names = {p.name for p in _JSON_FILES}
    for case in ("full-high", "full-low", "resized-high"):
        assert f"{RUN_ID}_{case}_pairwise.json" in names


@pytest.mark.parametrize("json_path", _JSON_FILES, ids=lambda p: p.stem)
def test_committed_table_matches_current_latex_table(json_path):
    _, Sc, Ss, caption, tex = _load_case(json_path)
    assert tex == pairwise_stats.latex_table(Sc, Ss, caption), (
        f"{json_path.with_suffix('.tex').name} is stale: regenerate it from "
        f"{json_path.name} with pairwise_stats.latex_table"
    )


@pytest.mark.parametrize("json_path", _JSON_FILES, ids=lambda p: p.stem)
def test_committed_change_row_is_encoded_minus_original(json_path):
    record, Sc, Ss, _, tex = _load_case(json_path)
    l1, l2, med = _change_cells(tex)
    # Encoded minus Original, i.e. the negative of the JSON "*_drop" fields.
    assert int(l1) == Ss["L1"] - Sc["L1"] == -record["delta"]["L1_drop"]
    assert float(l2) == pytest.approx(-record["delta"]["L2_drop"], abs=5e-4)
    assert float(med) == pytest.approx(-record["delta"]["median_abs_drop"], abs=5e-4)


def test_full_high_l1_rise_is_positive():
    """The docstring example in latex_table: L1 36924 vs 19786 prints +17138."""
    path = RESULTS / f"{RUN_ID}_full-high_pairwise.tex"
    if not path.exists():
        pytest.skip("full-high table is not present")
    l1, l2, med = _change_cells(path.read_text())
    assert (l1, l2, med) == ("+17138", "-4503.167", "+130.500")
