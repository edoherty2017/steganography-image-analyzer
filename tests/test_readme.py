"""Checks that the README quick start can be followed as written.

The README says its quick-start commands "were run as shown". These tests
hold it to that:

* every input file a quick-start command reads (``-i``, ``-f``,
  ``--wordlist``, ``--diff``) is either created by an earlier command in the
  section or is a file committed to the repository (``../results/...``);
* the "Hide a file" walkthrough is executed command by command in a scratch
  ``demo`` folder and its output must match the README text exactly.

The commands and expected outputs are parsed out of README.md rather than
copied here, so the tests guard the document itself.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from payload import HEADER_LEN

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
MAIN = REPO_ROOT / "main.py"

# Flags whose operand is a file the command reads.
INPUT_FLAGS = ("-i", "-f", "--wordlist", "--diff")
# Flags whose operand is a file or folder the command writes.
OUTPUT_FLAGS = ("-o", "--out", "--amplify", "--report", "--save", ">")


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return the body of the Markdown section that starts with ``heading``.

    The section runs until the next heading of the same or a higher level.

    Args:
        text: The whole Markdown document.
        heading: The exact heading line, for example ``"## Quick start"``.
    """
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    start = lines.index(heading)
    body = []
    for line in lines[start + 1:]:
        match = re.match(r"(#+) ", line)
        if match and len(match.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body)


def _code_blocks(section: str) -> list[tuple[str, str]]:
    """Return the ``(language, body)`` of every fenced code block, in order."""
    return re.findall(r"^```(\w*)\n(.*?)^```", section, flags=re.M | re.S)


def _commands(lang: str, body: str) -> list[str]:
    """Return the shell commands in a ``console`` or ``bash`` block."""
    if lang == "console":
        return [line[2:] for line in body.splitlines() if line.startswith("$ ")]
    if lang == "bash":
        return [line for line in body.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    return []


def _console_steps(body: str) -> list[tuple[str, list[str]]]:
    """Split a ``console`` block into ``(command, expected stdout lines)``."""
    steps: list[tuple[str, list[str]]] = []
    for line in body.splitlines():
        if line.startswith("$ "):
            steps.append((line[2:], []))
        elif steps:
            steps[-1][1].append(line.rstrip())
    for _, out in steps:
        while out and not out[-1]:
            out.pop()
    return steps


def _uncreated_inputs(readme: str) -> tuple[list[str], set[str]]:
    """Walk the quick start in order and find inputs nothing created.

    Args:
        readme: The README text.

    Returns:
        ``(missing, seen)``: input operands that no earlier command produced
        and that are not committed files, and every input operand checked
        (so a test can make sure the parser found something).
    """
    quick = _section(readme, "## Quick start")
    created: set[str] = set()
    missing: list[str] = []
    seen: set[str] = set()
    for lang, body in _code_blocks(quick):
        for command in _commands(lang, body):
            # The setup line copies the committed cover into demo/.
            for src, dest in re.findall(r"\bcp (\S+) demo/(\S+)", command):
                assert (REPO_ROOT / src).is_file(), f"setup copies missing {src}"
                created.add(dest)
            # Files written by inline Python, e.g. open('notes.txt', 'wb').
            created.update(re.findall(r"open\('([^']+)',\s*'w", command))
            tokens = shlex.split(command, comments=True)
            for flag, operand in zip(tokens, tokens[1:]):
                if flag in INPUT_FLAGS:
                    seen.add(operand)
                    committed = (operand.startswith("../")
                                 and (REPO_ROOT / operand[3:]).is_file())
                    if operand not in created and not committed:
                        missing.append(operand)
            for flag, operand in zip(tokens, tokens[1:]):
                if flag in OUTPUT_FLAGS:
                    created.add(operand)
    return missing, seen


def test_quickstart_inputs_are_created_before_use():
    missing, seen = _uncreated_inputs(_readme_text())
    assert {"cover.png", "notes.txt", "blob.bin", "words.txt"} <= seen
    assert missing == [], f"quick-start commands read files never created: {missing}"


def test_input_check_catches_a_missing_creation_command():
    """Regression: without its creation line, notes.txt must be reported."""
    readme = _readme_text()
    creation = next(line for line in readme.splitlines()
                    if line.startswith("$ python -c") and "notes.txt" in line)
    missing, _ = _uncreated_inputs(readme.replace(creation + "\n", ""))
    assert "notes.txt" in missing


def _run(command: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run one README command (``python ...``) with this test's interpreter."""
    argv = shlex.split(command)
    assert argv[0] == "python", f"unexpected command in README: {command}"
    argv[0] = sys.executable
    argv = [str(MAIN) if arg == "../main.py" else arg for arg in argv]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8",
                          stdin=subprocess.DEVNULL, timeout=120)


def test_hide_a_file_walkthrough_runs_as_documented(tmp_path):
    readme = _readme_text()
    demo = tmp_path / "demo"
    demo.mkdir()
    shutil.copy(REPO_ROOT / "results" / "stego_resized.png", demo / "cover.png")

    section = _section(readme, "### Hide a file, with compression, at 2 bits per sample")
    (lang, body), = [b for b in _code_blocks(section) if b[0] == "console"]
    steps = _console_steps(body)
    assert any("notes.txt" in cmd and "open(" in cmd for cmd, _ in steps), \
        "the walkthrough must create notes.txt before hiding it"

    for command, expected in steps:
        result = _run(command, demo)
        assert result.returncode == 0, f"{command!r} failed:\n{result.stderr}"
        assert result.stdout.rstrip("\n").splitlines() == expected, command

    original = (demo / "notes.txt").read_bytes()
    assert (demo / "recovered" / "notes.txt").read_bytes() == original

    # The prose after the block quotes the file size and compressed body
    # length; both must agree with what the commands printed.
    frame = int(re.search(r"in a (\d+)-byte frame", body).group(1))
    prose = re.search(r"(\d+) bytes of repeated text\b.*?compresses to a "
                      r"(\d+)-byte body", section, flags=re.S)
    assert prose, "prose after the walkthrough no longer states the sizes"
    assert int(prose.group(1)) == len(original)
    assert int(prose.group(2)) == frame - HEADER_LEN
