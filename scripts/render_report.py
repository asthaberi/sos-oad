"""Render a Markdown report to PDF.

    python scripts/render_report.py reports/panel-report.md

The Markdown file is the editable source and the PDF is a build artefact, so the author can
rewrite any paragraph in their own words and re-render. That matters for a document the
author will be questioned on: it has to be theirs.

Supports the subset the reports actually use -- headings, paragraphs, bullet and numbered
lists, pipe tables, ``**bold**``, ``*italic*``, ``` ``code`` ```, horizontal rules and block
quotes. Anything else is rendered as plain text rather than silently dropped.

Requires reportlab, which is a reporting tool rather than a dependency of the thesis
pipeline and is therefore not in requirements.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def inline(text: str) -> str:
    """Markdown inline spans to reportlab's mini-HTML, escaping everything else first."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', out)
    # Markdown links: keep the text, drop the target -- these are printed documents.
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    return out


def parse_blocks(lines: list[str]):
    """Yield (kind, payload) blocks. Deliberately small and explicit."""
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            yield ("h", (level, stripped.lstrip("#").strip()))
            index += 1
            continue

        if stripped in ("---", "***", "___"):
            yield ("rule", None)
            index += 1
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                # The |---|---| separator row carries no content.
                if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    rows.append(cells)
                index += 1
            if rows:
                yield ("table", rows)
            continue

        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            items: list[str] = []
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            while index < len(lines):
                current = lines[index].strip()
                if re.match(r"^[-*+]\s+", current) or re.match(r"^\d+\.\s+", current):
                    items.append(re.sub(r"^([-*+]|\d+\.)\s+", "", current))
                elif current and lines[index].startswith((" ", "\t")) and items:
                    items[-1] += " " + current  # continuation line
                else:
                    break
                index += 1
            yield ("list", (ordered, items))
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            yield ("quote", " ".join(quote))
            continue

        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if (
                not current
                or current.startswith(("#", "|", ">"))
                or current in ("---", "***", "___")
                or re.match(r"^([-*+]|\d+\.)\s+", current)
            ):
                break
            paragraph.append(current)
            index += 1
        if paragraph:
            yield ("p", " ".join(paragraph))


def build(source: Path, out: Path, *, subtitle: str | None = None) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    base = getSampleStyleSheet()
    ink = colors.HexColor("#161616")
    muted = colors.HexColor("#5b6572")
    rule = colors.HexColor("#c6ccd5")
    accent = colors.HexColor("#1f3a63")

    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=9.6, leading=14.2,
                          alignment=TA_JUSTIFY, textColor=ink, spaceAfter=7)
    heads = {
        1: ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, leading=20,
                          textColor=accent, spaceBefore=4, spaceAfter=10, alignment=TA_CENTER),
        2: ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, leading=17,
                          textColor=accent, spaceBefore=16, spaceAfter=7),
        3: ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.8, leading=14,
                          textColor=ink, spaceBefore=11, spaceAfter=4),
        4: ParagraphStyle("h4", parent=base["Heading4"], fontSize=9.8, leading=13,
                          textColor=ink, spaceBefore=9, spaceAfter=3),
    }
    cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11.6, alignment=0,
                          spaceAfter=0)
    quote = ParagraphStyle("quote", parent=body, leftIndent=10, rightIndent=10,
                           textColor=muted, borderPadding=4)
    italic = ParagraphStyle("it", parent=body, alignment=TA_CENTER, textColor=muted,
                            spaceAfter=12)

    story: list = []
    for kind, payload in parse_blocks(source.read_text(encoding="utf-8").splitlines()):
        if kind == "h":
            level, text = payload
            story.append(Paragraph(inline(text), heads.get(level, heads[4])))
        elif kind == "p":
            style = italic if payload.startswith("*") and payload.endswith("*") else body
            story.append(Paragraph(inline(payload), style))
        elif kind == "rule":
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.7, color=rule))
            story.append(Spacer(1, 7))
        elif kind == "quote":
            story.append(Paragraph(inline(payload), quote))
        elif kind == "list":
            ordered, items = payload
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(i), body), leftIndent=12) for i in items],
                    bulletType="1" if ordered else "bullet",
                    bulletFontSize=8 if ordered else 6,
                    leftIndent=14, bulletOffsetY=0 if ordered else 1,
                )
            )
            story.append(Spacer(1, 5))
        elif kind == "table":
            rows = payload
            width = (A4[0] - 44 * mm) / max(len(rows[0]), 1)
            data = [[Paragraph(inline(c), cell) for c in row] for row in rows]
            table = Table(data, colWidths=[width] * len(rows[0]), hAlign="LEFT", repeatRows=1)
            table.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff4")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, rule),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, rule),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f8f9fb")]),
                ])
            )
            story.append(table)
            story.append(Spacer(1, 9))

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    stamp = f"{subtitle or 'Interim progress report'} · {dt.date.today().isoformat()} · rev {commit}"

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(muted)
        canvas.drawString(22 * mm, 11 * mm, stamp)
        canvas.drawRightString(A4[0] - 22 * mm, 11 * mm, str(document.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
        title=source.stem.replace("-", " ").title(), author="",
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="markdown file to render")
    parser.add_argument("--out", type=Path, default=None, help="output PDF path")
    parser.add_argument("--subtitle", default=None, help="footer label")
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"no such file: {args.source}")
        return 1
    out = args.out or args.source.with_suffix(".pdf")
    build(args.source, out, subtitle=args.subtitle)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
