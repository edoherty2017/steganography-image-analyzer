# report.py
"""One-image steganalysis report: every check in one place, saved as MD, JSON or HTML.

``analyze_image()`` runs everything the project knows about a single
suspect image and returns one JSON-serializable dict:

* **Detectors** (``detectors.run_all``): the blind statistical detectors,
  namely the chi-square attack (Westfeld & Pfitzmann 1999), sample pair
  analysis (Dumitrescu, Wu & Wang 2003), RS analysis (Fridrich, Goljan & Du
  2001), the even/odd histogram-pair heuristic, the LSB randomness and
  entropy heuristics, and the known-signature scan.
* **Header detection** (``stego.probe``): whether the LSBs start with this
  project's STGA frame header, at 1 or 2 bits per sample, in natural order
  or in the order of a given password.
* **Payload**: the hidden data itself when a frame is found and can be read
  without a password (a text preview, or the stored file name and size).
  An encrypted frame is reported as ``"encrypted"`` and is only decoded
  when the right password is passed. Images in the original delimiter
  format (text followed by 64 zero bits) are read with
  ``decoder.decode_legacy`` when the signature scan or brute force finds
  that layout.
* **Brute force** (``bruteforce.bruteforce``, optional): many other bit
  layouts (channel order, bit plane, bit order, traversal, 2 bits per
  sample), ranked by how much the extracted bytes look like a payload.
* **Capacity**: how many payload bytes the image could hold at 1 and 2
  bits per sample.

``write_report()`` saves that dict as Markdown, JSON or a self-contained
HTML page. Markdown and HTML also get figures: the bit-plane grid (the
visual attack of Westfeld & Pfitzmann 1999, see bitplanes.py), the
amplified LSB image and the per-channel value histogram with even and odd
values in different shades.

The overall verdict is a heuristic, not a calibrated probability. It starts
from ``detectors.combine_results`` (the largest weighted detector score,
raised a little when SPA and RS agree), then:

* a verified STGA frame (CRC checked, decoded or known to be encrypted)
  sets the score to 1.0, since a valid header with a matching CRC does not
  occur by chance; a header whose frame is damaged scores
  ``DAMAGED_FRAME_SCORE``;
* the best brute-force candidate counts with weight ``BRUTEFORCE_WEIGHT``,
  so readable text or a complete hidden file in some other layout raises
  the score even when no statistic moved.

The verdict uses the detectors' cut points: below 0.35 "clean", from 0.65
"likely_stego", "suspicious" in between. A "clean" verdict does not prove
that nothing is hidden: a short payload scattered with a password key is
invisible to the signature scan, to brute force and to the sequential
chi-square attack, and SPA and RS only see payloads above a few percent of
capacity.

Typical use::

    import report
    result = report.analyze_image("suspect.png")
    print(result["overall"]["verdict"], result["overall"]["summary"])
    print(report.format_text(result))                    # terminal summary
    report.write_report(result, "reports/", fmt="html")  # reports/suspect_report.html

or from the command line::

    python main.py analyze -i suspect.png --report reports/ --format html
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import bitplanes
import bruteforce as _bruteforce
import detectors
import stego
from detectors import LIKELY_AT, SUSPICIOUS_AT, verdict_for
from payload import HEADER_LEN, PayloadError, WrongPasswordError

__all__ = [
    "BRUTEFORCE_WEIGHT",
    "DAMAGED_FRAME_SCORE",
    "REPORT_FORMATS",
    "analyze_image",
    "format_text",
    "write_report",
]

# Output formats accepted by write_report().
REPORT_FORMATS = ("md", "json", "html")

# Weight of the best brute-force candidate in the overall score (heuristic).
# Brute-force scores are ranking confidences, not probabilities (see
# bruteforce.py): a complete STGA frame scores 1.0, a hidden file whose
# structure parses to its end 0.97, natural-language text 0.5 to 0.85, and
# the LSB noise of a clean cover below 0.3. Weighting by 0.9 keeps a clean
# cover's best candidate under the "suspicious" cut point while letting a
# recovered file or message reach "likely_stego".
BRUTEFORCE_WEIGHT = 0.9

# Overall score when a valid STGA header is found but the frame is damaged
# (CRC mismatch or longer than the image). Heuristic: a valid header does
# not appear by chance (stego._locate puts the odds below 2**-40), so this
# is still near-certain evidence, just not a readable payload.
DAMAGED_FRAME_SCORE = 0.9

# Brute-force candidates at or above this score are used to recognise the
# legacy delimiter format (see _legacy_payload()).
_LEGACY_MIN_SCORE = 0.5

# Characters of payload text kept in the report.
PREVIEW_CHARS = 400

# Signature-scan findings copied into result["signatures"]["findings"].
_MAX_FINDINGS = 10

# Largest side, in pixels, of the amplified-LSB figure. Bigger images are
# stride-subsampled for display only (a full-size noise PNG of a 12 MP photo
# would be tens of megabytes, and is unreadable at report size anyway).
_LSB_FIGURE_MAX_PX = 1024

# Shades for the even and odd bars of the histogram figure.
_EVEN_COLOR = "#2b5797"
_ODD_COLOR = "#9fb8dc"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    """Return the current UTC time as an ISO 8601 string (seconds)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _capacity(path: str, width: int, height: int) -> tuple[int, int]:
    """Payload bytes the image can hold at 1 and 2 bits per sample.

    Uses the carrier registry (the same numbers as ``stego.capacity_bytes``).
    For an extension no carrier handles, falls back to the image carrier's
    formula: 3 samples (R, G, B) per pixel, minus the 14-byte frame header.
    """
    try:
        samples = stego.load_carrier(path, 1).capacity_bits()
    except ValueError:
        samples = width * height * 3
    return tuple(max(0, samples * bits // 8 - HEADER_LEN) for bits in (1, 2))


def _printable(text: str) -> str:
    """Replace control characters (except newline and tab) with U+FFFD.

    Payload text is untrusted: an escape sequence printed to a terminal or a
    stray control character in a report could change how either renders.
    """
    return "".join(ch if ch.isprintable() or ch in "\n\t" else "�" for ch in text)


def _describe_data(data: bytes) -> dict[str, Any]:
    """Preview payload bytes: text when they read as text, hex otherwise.

    Bytes count as text when they are valid UTF-8 and at least 90% of the
    characters are printable (newlines and tabs included). The preview is
    cut to PREVIEW_CHARS characters and has control characters replaced.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None:
        sample = text[:4096]
        printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in sample)
        if not sample or printable >= 0.9 * len(sample):
            return {"is_text": True, "preview": _printable(text[:PREVIEW_CHARS]),
                    "preview_truncated": len(text) > PREVIEW_CHARS}
    return {"is_text": False, "preview": data[:32].hex(" "),
            "preview_truncated": len(data) > 32}


def _stga_payload(path: str, password: str | None) -> tuple[dict | None, dict | None]:
    """Find and, when possible, decode an STGA frame.

    Args:
        path: The image.
        password: Optional password, used as scatter key and for decryption.

    Returns:
        ``(header, payload)``: the ``stego.probe`` dict (or None when no
        header is found) and the payload description (or None).
    """
    try:
        header = stego.probe(path, password=password, bits=None)
    except ValueError:
        return None, None  # no carrier for this extension
    if header is None:
        return None, None

    info: dict[str, Any] = {
        "found": True,
        "format": "stga",
        "bits_per_sample": header["bits_per_sample"],
        "scattered": header["scattered"],
        "flag_names": header["flag_names"],
        "encrypted": header["encrypted"],
        "compressed": header["compressed"],
        "body_len": header["body_len"],
        "frame_bytes": header["frame_bytes"],
    }
    try:
        found = stego.reveal(path, password=password, bits=header["bits_per_sample"])
    except WrongPasswordError as exc:
        # unpack() checks the CRC before decrypting, so the frame is intact.
        info.update(decoded=False, crc_ok=True, note="encrypted",
                    password_tried=bool(password), error=str(exc))
    except PayloadError as exc:  # CRC mismatch, truncated or malformed body
        info.update(decoded=False, crc_ok=False, note="damaged", error=str(exc))
    else:
        info.update(decoded=True, crc_ok=True, filename=found.filename,
                    data_bytes=len(found.data), **_describe_data(found.data))
        if found.encrypted:
            info["note"] = "decrypted with the given password"
    return header, info


def _legacy_payload(path: str, signature_findings: list[dict],
                    candidates: list[dict] | None) -> dict | None:
    """Read a legacy delimiter-format message when the evidence points to one.

    The legacy format (text bytes then 8 zero bytes, row-major RGB LSBs) has
    no header, and ``decoder.decode_legacy`` returns *something* for any
    image whose LSB stream contains 8 zero bytes. So it is only called when
    the brute force ranks that layout as legacy text (score at least 0.5),
    or the signature scan saw readable text followed by the zero marker at
    the start of the RGB stream.
    """
    strategy = None
    for cand in candidates or []:
        if (cand["info"].get("format") == "legacy-delimiter"
                and cand["score"] >= _LEGACY_MIN_SCORE
                and cand["label"] == "RGB bit0 msb-first row"):
            strategy = cand["label"]
            break
    if strategy is None:
        for f in signature_findings:
            if (f.get("kind") == "text" and f.get("order") == "RGB"
                    and f.get("offset") == 0 and f.get("end_marker")
                    and f.get("score", 0) >= 0.8):
                strategy = "RGB bit0 msb-first row"
                break
    if strategy is None:
        return None

    import decoder  # imported here: only needed for legacy images

    text = decoder.decode_legacy(path)
    if not text:
        return None
    return {
        "found": True,
        "format": "legacy",
        "strategy": strategy,
        "bits_per_sample": 1,
        "scattered": False,
        "encrypted": False,
        "decoded": True,
        "filename": None,
        "data_bytes": len(text.encode("utf-8")),
        "note": "legacy delimiter format (text followed by 64 zero bits)",
        "is_text": True,
        "preview": _printable(text[:PREVIEW_CHARS]),
        "preview_truncated": len(text) > PREVIEW_CHARS,
    }


def _bruteforce_entry(cand: dict) -> dict[str, Any]:
    """Make a bruteforce() candidate JSON-safe (no raw data, strategy as dict)."""
    return {
        "label": cand["label"],
        "strategy": cand["strategy"].to_dict(),
        "score": float(cand["score"]),
        "kind": cand["kind"],
        "preview": cand["preview"],
        "info": cand["info"],
    }


def _overall(dets: dict[str, dict], payload_info: dict | None,
             candidates: list[dict] | None) -> dict[str, Any]:
    """Combine every signal into one score and verdict (heuristic).

    See the module docstring: ``detectors.combine_results`` first, then a
    verified STGA frame lifts the score to 1.0 (``DAMAGED_FRAME_SCORE`` if
    the frame is damaged), and the best brute-force candidate counts with
    weight ``BRUTEFORCE_WEIGHT``.
    """
    combined = detectors.combine_results(dets)
    score = float(combined["score"])
    driver = combined["driver"]
    flagged = list(combined["flagged"])
    if driver is None:
        summary = "No detector found evidence of LSB embedding."
    elif score < SUSPICIOUS_AT:
        summary = (f"No detector reached the suspicious level; the strongest signal "
                   f"was {driver} ({dets[driver]['score']:.2f}): {dets[driver]['summary']}")
    else:
        summary = dets[driver]["summary"]

    if candidates:
        top = candidates[0]
        weighted = float(top["score"]) * BRUTEFORCE_WEIGHT
        if weighted >= SUSPICIOUS_AT:
            flagged.append("bruteforce")
        if weighted > score:
            score, driver = weighted, "bruteforce"
            summary = (f"Brute force: the {top['label']} layout gives "
                       f"{top['kind']} data: {top['preview']}")

    if payload_info and payload_info.get("format") == "stga":
        flagged.append("payload")
        if payload_info.get("crc_ok"):
            score, driver = 1.0, "payload"
            if not payload_info.get("encrypted"):
                what = "a readable"
            elif payload_info.get("decoded"):
                what = "an encrypted (and now decrypted)"
            else:
                what = "an encrypted"
            summary = (f"Found {what} STGA payload: {payload_info['body_len']}-byte "
                       f"body at {payload_info['bits_per_sample']} bit(s) per sample, "
                       f"CRC-32 verified.")
        elif DAMAGED_FRAME_SCORE > score:
            score, driver = DAMAGED_FRAME_SCORE, "payload"
            summary = ("Found an STGA payload header, but the frame is damaged: "
                       f"{payload_info.get('error', '')}")
    elif payload_info and payload_info.get("format") == "legacy":
        flagged.append("payload")

    return {
        "score": min(1.0, max(0.0, score)),
        "verdict": verdict_for(score),
        "driver": driver,
        "summary": summary,
        "flagged": list(dict.fromkeys(flagged)),
        "detector_score": float(combined["score"]),
        "method": ("heuristic: largest weighted detector score (detectors."
                   "combine_results), raised to 1.0 by a CRC-verified STGA frame "
                   f"and including the best brute-force score x {BRUTEFORCE_WEIGHT}"),
    }


def analyze_image(
    path: str | os.PathLike,
    *,
    run_bruteforce: bool = True,
    top_k: int = 5,
    password: str | None = None,
    channels: str = "RGB",
    max_bytes: int = 4096,
    max_bit: int = 1,
) -> dict[str, Any]:
    """Run every steganalysis check on one image.

    Args:
        path: The suspect image (any format Pillow reads; payloads only
            survive in lossless ones).
        run_bruteforce: Also try the brute-force layouts (about 100
            strategies; a second or two on a 12 MP image).
        top_k: Brute-force candidates kept in the result.
        password: Optional password. It is tried as the scatter key when
            looking for the frame header and used to decrypt the payload.
            It is never stored in the result.
        channels: Channels the statistical detectors analyse, as letters
            from "RGB".
        max_bytes: Bytes read per brute-force strategy before expansion.
        max_bit: Highest bit plane the brute force tries (0 = LSB only).

    Returns:
        A JSON-serializable dict (``json.dumps(..., allow_nan=False)``
        works) with the keys:

        ``tool``, ``generated`` (UTC timestamp), ``image`` (the path),
        ``file_name``, ``file_bytes``, ``format`` (Pillow's name), ``mode``,
        ``size`` (``{"width", "height"}``), ``channels``;
        ``capacity_bytes`` and ``capacity_bytes_2bit`` (payload bytes that
        fit after the 14-byte header);
        ``overall`` (``score``, ``verdict``, ``driver``, ``summary``,
        ``flagged``, ``detector_score``, ``method``; see the module
        docstring);
        ``header`` (the ``stego.probe`` dict, or None);
        ``payload`` (None, or a dict with ``found``, ``format`` ("stga" or
        "legacy"), ``decoded``, ``encrypted``, ``note`` ("encrypted" when
        the frame is encrypted and no working password was given,
        "damaged" for a bad frame), ``filename``, ``data_bytes``,
        ``is_text`` and ``preview``);
        ``signatures`` (``summary``, ``stga`` and the top ``findings`` of
        the signature scan);
        ``detectors`` (the ``detectors.run_all`` output);
        ``bruteforce`` (None when skipped, else a list of candidates with
        ``label``, ``strategy`` (dict), ``score``, ``kind``, ``preview`` and
        ``info``; raw data is left out);
        ``timings`` (seconds per stage).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not an image Pillow can read, or an
            argument is out of range.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(2, "No such file", path)
    timings: dict[str, float] = {}
    try:
        with Image.open(path) as img:
            img.load()
            fmt, mode, (width, height) = img.format, img.mode, img.size
            t = time.perf_counter()
            dets = detectors.run_all(img, channels=channels)
            timings["detectors"] = time.perf_counter() - t

            candidates = None
            if run_bruteforce:
                t = time.perf_counter()
                raw = _bruteforce.bruteforce(img, top_k=top_k, max_bytes=max_bytes,
                                             max_bit=max_bit)
                candidates = [_bruteforce_entry(c) for c in raw]
                timings["bruteforce"] = time.perf_counter() - t
    except UnidentifiedImageError as exc:
        raise ValueError(f"{path!r} is not an image Pillow can read") from exc

    t = time.perf_counter()
    header, payload_info = _stga_payload(path, password)
    sig = dets.get("signatures", {})
    findings = (sig.get("details") or {}).get("findings") or []
    if payload_info is None:
        payload_info = _legacy_payload(path, findings, candidates)
    timings["payload"] = time.perf_counter() - t

    cap1, cap2 = _capacity(path, width, height)
    return {
        "tool": "Steganography Image Analyzer (report.py)",
        "generated": _now_utc(),
        "image": path,
        "file_name": os.path.basename(path),
        "file_bytes": os.path.getsize(path),
        "format": fmt,
        "mode": mode,
        "size": {"width": width, "height": height},
        "channels": channels.upper(),
        "capacity_bytes": cap1,
        "capacity_bytes_2bit": cap2,
        "overall": _overall(dets, payload_info, candidates),
        "header": header,
        "payload": payload_info,
        "signatures": {
            "summary": sig.get("summary", ""),
            "stga": (sig.get("details") or {}).get("stga"),
            "findings": findings[:_MAX_FINDINGS],
        },
        "detectors": dets,
        "bruteforce": candidates,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def _payload_line(p: dict | None) -> str:
    """One-line description of result["payload"]."""
    if not p:
        return "none found in natural order (a password-scattered payload needs its password)"
    if p["format"] == "legacy":
        return f"legacy delimiter-format text, {p['data_bytes']} bytes"
    parts = [f"STGA frame, {p['body_len']}-byte body, {p['bits_per_sample']} bit(s)/sample"]
    parts.append("scattered" if p.get("scattered") else "sequential")
    if p.get("flag_names"):
        parts.append("flags: " + ", ".join(p["flag_names"]))
    if p.get("note"):
        parts.append(p["note"])
    if p.get("filename"):
        parts.append(f"file {p['filename']!r}")
    return "; ".join(parts)


def format_text(result: dict[str, Any]) -> str:
    """Render an analyze_image() result as a short plain-text summary.

    Args:
        result: The dict returned by analyze_image().

    Returns:
        Multi-line text for a terminal (no trailing newline).
    """
    o = result["overall"]
    size = result["size"]
    lines = [
        f"Image:    {result['image']} ({result['format']}, {result['mode']}, "
        f"{size['width']}x{size['height']})",
        f"Capacity: {result['capacity_bytes']:,} bytes at 1 bit/sample, "
        f"{result['capacity_bytes_2bit']:,} at 2",
        f"Verdict:  {o['verdict'].replace('_', ' ').upper()} (score {o['score']:.2f}, "
        f"driven by {o['driver'] or 'nothing'})",
        f"          {o['summary']}",
        f"Payload:  {_payload_line(result['payload'])}",
    ]
    p = result["payload"]
    if p and p.get("decoded") and p.get("is_text") and not p.get("filename"):
        preview = p["preview"].replace("\n", " ")
        lines.append(f"          \"{preview[:200]}{'...' if len(preview) > 200 else ''}\"")
    lines.append("")
    lines.append("Detectors:")
    for name, r in result["detectors"].items():
        lines.append(f"  {name:<22} {r['verdict']:<13} {r['score']:.2f}  {r['summary']}")
    if result["bruteforce"] is not None:
        lines.append("")
        lines.append("Brute force (best layouts):")
        if not result["bruteforce"]:
            lines.append("  (no candidates)")
        for c in result["bruteforce"]:
            lines.append(f"  {c['score']:.2f}  {c['label']:<26} {c['kind']:<10} {c['preview'][:80]}")
    lines.append("")
    lines.append("Scores are heuristics, not probabilities; see report.py.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_lsb_figure(arr: np.ndarray, out_path: Path) -> Path:
    """Save the amplified LSB image (subsampled if large) as a PNG."""
    lsb = bitplanes.lsb_amplify(arr)
    step = max(1, -(-max(lsb.shape[:2]) // _LSB_FIGURE_MAX_PX))  # ceil division
    Image.fromarray(np.ascontiguousarray(lsb[::step, ::step])).save(out_path)
    return out_path


def _save_histogram_figure(arr: np.ndarray, out_path: Path, title: str) -> Path:
    """Save per-channel value histograms with even and odd values shaded apart.

    LSB replacement only moves a value within its pair (2k, 2k+1), so it
    evens out the two bars of each pair (random payload) or tilts every pair
    the same way (text payload, which has more 0 bits). Drawn with the
    object-oriented Matplotlib API on an Agg canvas, so pyplot is never
    imported.
    """
    rgb = arr[..., :3]
    fig = Figure(figsize=(9.0, 6.0), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    values = np.arange(256)
    for ax, ch, label in zip(axes, range(3), "RGB"):
        h = np.bincount(rgb[..., ch].ravel(), minlength=256)
        ax.bar(values[0::2], h[0::2], width=1.0, color=_EVEN_COLOR, label="even value")
        ax.bar(values[1::2], h[1::2], width=1.0, color=_ODD_COLOR, label="odd value")
        ax.set_ylabel(label, rotation=0, labelpad=12, va="center")
        ax.set_xlim(-0.5, 255.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("channel value")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, format="png")
    return out_path


def _make_figures(result: dict[str, Any], out: Path, stem: str) -> list[dict[str, str]]:
    """Write the report figures next to the report and describe them.

    Returns:
        ``[{"title", "file", "caption"}]`` with file names relative to
        ``out``.
    """
    src = result["image"]
    arr = detectors.load_rgb(src)  # (H, W, 3); alpha never carries payload
    figs = []

    grid = out / f"{stem}_bitplanes.png"
    # Pass the path, not the array, so the grid title names the file.
    bitplanes.save_bitplane_grid(src, grid, channels="RGB")
    figs.append({
        "title": "Bit planes",
        "file": grid.name,
        "caption": ("Rows are channels, columns bits 7 (MSB) to 0 (LSB); white = 1. "
                    "A band of uniform noise at the top of the LSB column, unlike "
                    "the rest of the plane, suggests sequential embedding (visual "
                    "attack, Westfeld & Pfitzmann 1999). Heuristic: noisy photos "
                    "have noisy LSBs too, and scattered payloads leave no band."),
    })

    lsb = out / f"{stem}_lsb.png"
    _save_lsb_figure(arr, lsb)
    figs.append({
        "title": "Amplified LSBs",
        "file": lsb.name,
        "caption": ("The least significant bit of R, G and B scaled to 0/255 "
                    "(subsampled for display when the image is large)."),
    })

    hist = out / f"{stem}_histogram.png"
    _save_histogram_figure(arr, hist, f"Value histograms of {result['file_name']}")
    figs.append({
        "title": "Value histograms",
        "file": hist.name,
        "caption": ("Dark bars are even values, light bars odd. LSB embedding "
                    "moves values only within each even/odd pair, pulling the "
                    "two bars together or tilting every pair the same way."),
    })
    return figs


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _md_cell(value: Any) -> str:
    """Make untrusted text inert in Markdown prose or one table cell.

    Text from the LSB stream (brute-force and signature previews, and the
    overall summary built from them) is attacker-controlled, so it must
    render as literal characters and never as markup:

    * newlines are flattened so the value stays on one line (one cell);
    * ``&``, ``<`` and ``>`` become HTML entities, so ``<script>`` or
      ``<img onerror=...>`` is not passed through as raw HTML. Entities are
      used rather than ``\\<`` because not every Markdown dialect treats
      ``<`` as backslash-escapable (Python-Markdown does not);
    * backslash, pipe, backtick and square brackets are backslash-escaped,
      so the value cannot split a table cell, open a code span, or form a
      link or image such as ``[x](javascript:...)``.

    Not for use inside a code span, where entities and backslash escapes
    are shown literally; use ``_md_code`` there.

    Args:
        value: Anything; it is converted with ``str()``.

    Returns:
        The escaped single-line text.
    """
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = html.escape(text, quote=False)
    for ch in "\\|`[]":
        text = text.replace(ch, "\\" + ch)
    return text


def _md_code(value: Any, *, in_table: bool = False) -> str:
    """Render untrusted text as one inline Markdown code span.

    Code-span content is literal (no HTML, no entities, no backslash
    escapes), so the only risks are a backtick closing the span early and,
    inside a table, a pipe ending the cell. The delimiter is one backtick
    longer than the longest backtick run in the text (CommonMark code spans
    close only on a run of the same length), padded with spaces when the
    text starts or ends with a backtick.

    Args:
        value: Anything; it is converted with ``str()``.
        in_table: True when the span sits in a GFM table cell, where a pipe
            must be written as ``\\|`` even inside a code span.

    Returns:
        The complete code span, delimiters included.
    """
    text = str(value).replace("\r", " ").replace("\n", " ")
    if in_table:
        text = text.replace("|", "\\|")
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    tick = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{tick}{pad}{text}{pad}{tick}"


def _md_fence(text: str) -> str:
    """Return a code fence longer than any backtick run inside ``text``."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _payload_md(p: dict | None) -> list[str]:
    """Markdown lines for the payload section."""
    if not p:
        return [
            "No STGA frame header was found at 1 or 2 bits per sample in natural "
            "order, and no legacy-format message was recognised. A payload hidden "
            "with a password and key scatter is only found with that password "
            "(`--password`, or `bruteforce.py --wordlist`).",
        ]
    lines = [f"- **Format:** {_md_cell(_payload_line(p))}"]
    if p.get("error"):
        lines.append(f"- **Status:** {_md_cell(p['error'])}")
    if p.get("decoded"):
        lines.append(f"- **Size:** {p['data_bytes']:,} bytes")
        if p.get("filename"):
            lines.append(f"- **Stored file name:** {_md_code(p['filename'])}")
        kind = "Text" if p.get("is_text") else "First bytes (hex)"
        more = " (truncated)" if p.get("preview_truncated") else ""
        fence = _md_fence(p["preview"])
        lines += ["", f"{kind}{more}:", "", f"{fence}text", p["preview"], fence]
    return lines


def _to_markdown(result: dict[str, Any], figs: list[dict[str, str]]) -> str:
    """Render the report as Markdown."""
    o = result["overall"]
    size = result["size"]
    out = [
        f"# Steganalysis report: {_md_cell(result['file_name'])}",
        "",
        f"Generated {result['generated']} by {result['tool']}.",
        "",
        "## Verdict",
        "",
        f"**{o['verdict'].replace('_', ' ').upper()}** (score {o['score']:.2f}, "
        f"driven by `{o['driver'] or 'nothing'}`)",
        "",
        _md_cell(o["summary"]),
        "",
        "| Property | Value |",
        "|---|---|",
        f"| File | {_md_code(result['image'], in_table=True)} ({result['file_bytes']:,} bytes) |",
        f"| Format / mode | {result['format']} / {result['mode']} |",
        f"| Size | {size['width']} x {size['height']} px |",
        f"| LSB capacity | {result['capacity_bytes']:,} bytes at 1 bit/sample, "
        f"{result['capacity_bytes_2bit']:,} bytes at 2 bits/sample |",
        f"| Flagged by | {_md_cell(', '.join(o['flagged']) or 'none')} |",
        "",
        "## Hidden payload",
        "",
        *_payload_md(result["payload"]),
        "",
        "## Statistical detectors",
        "",
        "| Detector | Verdict | Score | Summary |",
        "|---|---|---|---|",
    ]
    for name, r in result["detectors"].items():
        out.append(f"| `{name}` | {r['verdict']} | {r['score']:.2f} | {_md_cell(r['summary'])} |")
    out += ["", "## Signature scan", "", _md_cell(result["signatures"]["summary"]), ""]
    findings = result["signatures"]["findings"]
    if findings:
        out += ["| Kind | Stream | Byte offset | Score | What |", "|---|---|---|---|---|"]
        for f in findings:
            what = f.get("label", "")
            if f.get("preview"):
                what += f": {f['preview'][:80]}"
            out.append(f"| {f.get('kind')} | {_md_cell(f.get('order'))} | {f.get('offset')} "
                       f"| {f.get('score', 0):.2f} | {_md_cell(what)} |")
        out.append("")
    out += ["## Brute-force extraction", ""]
    cands = result["bruteforce"]
    if cands is None:
        out.append("Skipped.")
    elif not cands:
        out.append("No candidates.")
    else:
        out += ["| # | Layout | Score | Kind | Preview |", "|---|---|---|---|---|"]
        for i, c in enumerate(cands, 1):
            out.append(f"| {i} | {_md_cell(c['label'])} | {c['score']:.2f} | {c['kind']} "
                       f"| {_md_cell(c['preview'][:100])} |")
    out.append("")
    if figs:
        out += ["## Figures", ""]
        for f in figs:
            out += [f"### {f['title']}", "", f"![{f['title']}]({f['file']})", "",
                    f["caption"], ""]
    out += [
        "## How to read this report",
        "",
        "- Every score is a heuristic in [0, 1], not a probability. Verdicts use the "
        f"cut points {SUSPICIOUS_AT} (suspicious) and {LIKELY_AT} (likely stego).",
        "- Methods: chi-square attack (Westfeld & Pfitzmann 1999), sample pair analysis "
        "(Dumitrescu, Wu & Wang 2003), RS analysis (Fridrich, Goljan & Du 2001); the "
        "histogram-pair, randomness, signature and brute-force scores are this "
        "project's own heuristics.",
        "- A valid STGA header with a matching CRC-32 is definitive. Short payloads "
        "below a few percent of capacity do not move the statistics, and a payload "
        "scattered with a password key is invisible to the signature scan and brute force.",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_CSS = """
:root { --bg: #ffffff; --fg: #1d1f23; --muted: #5d636e; --line: #d9dde3;
        --card: #f5f7fa; --clean: #1f7a3f; --suspicious: #a45a00; --likely: #b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #15171b; --fg: #e6e8eb; --muted: #9aa1ac; --line: #30343b;
          --card: #1d2026; --clean: #4cc27a; --suspicious: #f0a33a; --likely: #ff6b5e; }
}
body { background: var(--bg); color: var(--fg); margin: 0;
       font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6em; margin: 0 0 4px; overflow-wrap: anywhere; }
h2 { font-size: 1.2em; margin: 32px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
.badge { display: inline-block; font-weight: 700; padding: 2px 10px; border-radius: 999px;
         color: #fff; letter-spacing: .02em; }
.clean { background: var(--clean); } .suspicious { background: var(--suspicious); }
.likely_stego { background: var(--likely); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code, pre { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 10px;
      white-space: pre-wrap; overflow-wrap: anywhere; }
td { overflow-wrap: anywhere; }
figure { margin: 16px 0; } figure img { max-width: 100%; height: auto; border: 1px solid var(--line); }
figcaption { color: var(--muted); font-size: 14px; }
"""


def _esc(value: Any) -> str:
    """HTML-escape any value."""
    return html.escape(str(value), quote=True)


def _badge(verdict: str) -> str:
    """HTML badge for a verdict."""
    return f'<span class="badge {_esc(verdict)}">{_esc(verdict.replace("_", " "))}</span>'


def _payload_html(p: dict | None) -> str:
    """HTML for the payload section."""
    if not p:
        return ("<p>No STGA frame header was found at 1 or 2 bits per sample in natural "
                "order, and no legacy-format message was recognised. A payload hidden with "
                "a password and key scatter is only found with that password.</p>")
    parts = [f"<p><strong>{_esc(_payload_line(p))}</strong></p>"]
    if p.get("error"):
        parts.append(f"<p class=\"muted\">{_esc(p['error'])}</p>")
    if p.get("decoded"):
        parts.append(f"<p>{p['data_bytes']:,} bytes"
                     + (f", stored file name <code>{_esc(p['filename'])}</code>"
                        if p.get("filename") else "") + "</p>")
        kind = "Text" if p.get("is_text") else "First bytes (hex)"
        more = " (truncated)" if p.get("preview_truncated") else ""
        parts.append(f"<p class=\"muted\">{kind}{more}:</p><pre>{_esc(p['preview'])}</pre>")
    return "\n".join(parts)


def _to_html(result: dict[str, Any], figs: list[dict[str, str]], out: Path) -> str:
    """Render the report as one self-contained HTML page (figures inlined)."""
    o = result["overall"]
    size = result["size"]
    rows = "\n".join(
        f"<tr><td><code>{_esc(n)}</code></td><td>{_badge(r['verdict'])}</td>"
        f"<td class=\"num\">{r['score']:.2f}</td><td>{_esc(r['summary'])}</td></tr>"
        for n, r in result["detectors"].items())
    findings = result["signatures"]["findings"]
    find_rows = "\n".join(
        f"<tr><td>{_esc(f.get('kind'))}</td><td>{_esc(f.get('order'))}</td>"
        f"<td class=\"num\">{_esc(f.get('offset'))}</td><td class=\"num\">{f.get('score', 0):.2f}</td>"
        f"<td>{_esc(f.get('label', ''))}"
        + (f": <code>{_esc(f['preview'][:80])}</code>" if f.get("preview") else "")
        + "</td></tr>"
        for f in findings)
    cands = result["bruteforce"]
    if cands is None:
        bf_html = "<p>Skipped.</p>"
    elif not cands:
        bf_html = "<p>No candidates.</p>"
    else:
        bf_rows = "\n".join(
            f"<tr><td class=\"num\">{i}</td><td><code>{_esc(c['label'])}</code></td>"
            f"<td class=\"num\">{c['score']:.2f}</td><td>{_esc(c['kind'])}</td>"
            f"<td><code>{_esc(c['preview'][:120])}</code></td></tr>"
            for i, c in enumerate(cands, 1))
        bf_html = ("<div class=\"table-wrap\"><table><tr><th>#</th><th>Layout</th><th>Score</th>"
                   f"<th>Kind</th><th>Preview</th></tr>{bf_rows}</table></div>")
    fig_html = ""
    if figs:
        items = []
        for f in figs:
            data = base64.b64encode((out / f["file"]).read_bytes()).decode("ascii")
            items.append(f"<figure><figcaption><strong>{_esc(f['title'])}</strong></figcaption>"
                         f"<img alt=\"{_esc(f['title'])}\" src=\"data:image/png;base64,{data}\">"
                         f"<figcaption>{_esc(f['caption'])}</figcaption></figure>")
        fig_html = "<h2>Figures</h2>\n" + "\n".join(items)
    find_html = (f"<div class=\"table-wrap\"><table><tr><th>Kind</th><th>Stream</th><th>Byte offset</th>"
                 f"<th>Score</th><th>What</th></tr>{find_rows}</table></div>" if findings else "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steganalysis report: {_esc(result['file_name'])}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<main>
<h1>Steganalysis report: {_esc(result['file_name'])}</h1>
<p class="muted">Generated {_esc(result['generated'])} by {_esc(result['tool'])}.</p>

<h2>Verdict</h2>
<div class="card">
<p>{_badge(o['verdict'])} &nbsp; score <strong>{o['score']:.2f}</strong>,
driven by <code>{_esc(o['driver'] or 'nothing')}</code></p>
<p>{_esc(o['summary'])}</p>
</div>
<div class="table-wrap"><table>
<tr><th>File</th><td><code>{_esc(result['image'])}</code> ({result['file_bytes']:,} bytes)</td></tr>
<tr><th>Format / mode</th><td>{_esc(result['format'])} / {_esc(result['mode'])}</td></tr>
<tr><th>Size</th><td>{size['width']} x {size['height']} px</td></tr>
<tr><th>LSB capacity</th><td>{result['capacity_bytes']:,} bytes at 1 bit/sample,
{result['capacity_bytes_2bit']:,} bytes at 2 bits/sample</td></tr>
<tr><th>Flagged by</th><td>{_esc(', '.join(o['flagged']) or 'none')}</td></tr>
</table></div>

<h2>Hidden payload</h2>
{_payload_html(result['payload'])}

<h2>Statistical detectors</h2>
<div class="table-wrap"><table>
<tr><th>Detector</th><th>Verdict</th><th>Score</th><th>Summary</th></tr>
{rows}
</table></div>

<h2>Signature scan</h2>
<p>{_esc(result['signatures']['summary'])}</p>
{find_html}

<h2>Brute-force extraction</h2>
{bf_html}

{fig_html}

<h2>How to read this report</h2>
<ul>
<li>Every score is a heuristic in [0, 1], not a probability. Verdicts use the cut points
{SUSPICIOUS_AT} (suspicious) and {LIKELY_AT} (likely stego).</li>
<li>Methods: chi-square attack (Westfeld &amp; Pfitzmann 1999), sample pair analysis
(Dumitrescu, Wu &amp; Wang 2003), RS analysis (Fridrich, Goljan &amp; Du 2001); the
histogram-pair, randomness, signature and brute-force scores are this project's own
heuristics.</li>
<li>A valid STGA header with a matching CRC-32 is definitive. Short payloads below a few
percent of capacity do not move the statistics, and a payload scattered with a password
key is invisible to the signature scan and brute force.</li>
</ul>
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _safe_stem(name: str) -> str:
    """Reduce an image file name to a safe stem for report file names."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem).strip("._")
    return stem or "image"


def write_report(result: dict[str, Any], out_dir: str | os.PathLike,
                 fmt: str = "md", figures: bool = True) -> str:
    """Save an analyze_image() result as a report file.

    The report is ``<out_dir>/<image stem>_report.<fmt>``; an existing
    report of the same name is replaced. With ``figures`` (Markdown and
    HTML only) three PNGs are written beside it: ``<stem>_bitplanes.png``
    (bitplanes.save_bitplane_grid), ``<stem>_lsb.png`` (amplified LSBs) and
    ``<stem>_histogram.png`` (per-channel histograms). Markdown links to
    them by relative path; HTML embeds them as base64 data URIs, so the
    page is a single self-contained file. JSON is the result dict exactly
    (UTF-8, indented) and never has figures.

    Args:
        result: The dict returned by analyze_image(). The image it names
            must still exist when figures are drawn.
        out_dir: Folder to write into; created if missing.
        fmt: "md", "json" or "html" (case-insensitive).
        figures: Draw the figures (md and html only).

    Returns:
        The report's path, as a string.

    Raises:
        ValueError: If ``fmt`` is not a known format.
        FileNotFoundError: If figures are requested and the image is gone.
    """
    fmt = str(fmt).lower().lstrip(".")
    if fmt == "markdown":
        fmt = "md"
    if fmt not in REPORT_FORMATS:
        raise ValueError(f"unknown report format {fmt!r}; use one of {', '.join(REPORT_FORMATS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(result.get("file_name") or result.get("image") or "image")
    path = out / f"{stem}_report.{fmt}"

    if fmt == "json":
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")
        return str(path)

    figs = _make_figures(result, out, stem) if figures else []
    if fmt == "md":
        path.write_text(_to_markdown(result, figs), encoding="utf-8")
    else:
        path.write_text(_to_html(result, figs, out), encoding="utf-8")
    return str(path)
