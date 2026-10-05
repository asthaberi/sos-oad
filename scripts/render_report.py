"""Render a Markdown report to PDF.

    python scripts/render_report.py reports/panel-report.md
    python scripts/render_report.py reports/private/study-guide.md --subtitle "Study guide"

The Markdown file is the editable source and the PDF is a build artefact, so the author can
rewrite any paragraph in their own words and re-render. That matters for a document the
author will be questioned on: it has to be theirs.

Supported Markdown -- the subset the reports actually use:

* ``#`` headings. The first level-1 heading is the centred document title; any later
  level-1 heading starts a new section, left-aligned.
* paragraphs, ``-``/``*`` bullets and ``1.`` numbered lists
* pipe tables, with column widths proportional to their content
* fenced code blocks (triple backticks), whitespace preserved, never wrapped
* ``> `` block quotes, rendered as tinted boxes. The opening bold label picks the colour:
  ``**Analogy**`` blue, ``**Say this**`` green, ``**Avoid**`` / ``**Trap**`` red,
  ``**Note**`` / anything else grey
* ``**bold**``, ``*italic*``, ``` ``code`` ```, ``---`` rules, and ``<!-- pagebreak -->``

Anything else renders as plain text rather than being silently dropped.

Fonts: Arial and Consolas are used when present (they ship with Windows), because the
standard PDF fonts cover only Latin-1 -- arrows, inequalities and Greek letters would
otherwise render as empty boxes. Helvetica and Courier are the fallback.

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

#: Longest code line that fits the text width at the code font size. Longer lines are
#: reported, because a code block never wraps and would otherwise run off the page.
CODE_LINE_LIMIT = 96

CALLOUTS = {
    "analogy": ("EAF2FC", "2B5CA8"),
    "say this": ("E8F5EC", "1E7B4F"),
    "avoid": ("FDECEE", "B3263A"),
    "trap": ("FDECEE", "B3263A"),
    "note": ("F1F3F6", "5B6572"),
}


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def register_fonts() -> tuple[str, str, str]:
    """Return (body, bold, mono) font names, registering TTFs when available."""
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = Path("C:/Windows/Fonts")
    wanted = {
        "Body": "arial.ttf",
        "Body-Bold": "arialbd.ttf",
        "Body-Italic": "ariali.ttf",
        "Body-BoldItalic": "arialbi.ttf",
        "Mono": "consola.ttf",
    }
    if all((fonts / f).is_file() for f in wanted.values()):
        for name, file in wanted.items():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
        addMapping("Body", 0, 0, "Body")
        addMapping("Body", 1, 0, "Body-Bold")
        addMapping("Body", 0, 1, "Body-Italic")
        addMapping("Body", 1, 1, "Body-BoldItalic")
        return "Body", "Body-Bold", "Mono"
    return "Helvetica", "Helvetica-Bold", "Courier"


def inline(text: str, mono: str) -> str:
    """Markdown inline spans to reportlab's mini-HTML, escaping everything else first."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+?)`", rf'<font face="{mono}">\1</font>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\*\w])\*([^*\n]+?)\*(?![\*\w])", r"<i>\1</i>", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    return out


def parse_blocks(lines: list[str]):
    """Yield (kind, payload) blocks. Deliberately small and explicit."""
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            body: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                body.append(lines[index].rstrip("\n").rstrip())
                index += 1
            index += 1  # closing fence
            yield ("code", body)
            continue

        if stripped == "<!-- pagebreak -->":
            yield ("pagebreak", None)
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
                    items[-1] += " " + current
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
            # Blank quote lines separate paragraphs inside one callout.
            paragraphs, current_para = [], []
            for q in quote:
                if q:
                    current_para.append(q)
                elif current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
            if current_para:
                paragraphs.append(" ".join(current_para))
            yield ("quote", paragraphs)
            continue

        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if (
                not current
                or current.startswith(("#", "|", ">", "```"))
                or current in ("---", "***", "___", "<!-- pagebreak -->")
                or re.match(r"^([-*+]|\d+\.)\s+", current)
            ):
                break
            paragraph.append(current)
            index += 1
        if paragraph:
            yield ("p", " ".join(paragraph))


def column_widths(rows: list[list[str]], total: float) -> list[float]:
    """Widths proportional to content, so a short label column does not get half the page."""
    columns = max(len(r) for r in rows)
    weights = []
    for c in range(columns):
        longest = max((len(re.sub(r"[*`]", "", r[c])) for r in rows if c < len(r)), default=4)
        # Floor of 10: a column holding '0.402' still needs room for the cell padding, and
        # a narrower one wraps a five-character number onto two lines.
        weights.append(min(max(longest, 10), 58))
    scale = total / sum(weights)
    return [w * scale for w in weights]


def build(source: Path, out: Path, *, subtitle: str | None = None) -> list[str]:
    """Render, returning any warnings (code lines too long to fit)."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    body_font, bold_font, mono = register_fonts()
    base = getSampleStyleSheet()
    ink = colors.HexColor("#161616")
    muted = colors.HexColor("#5b6572")
    rule = colors.HexColor("#c6ccd5")
    accent = colors.HexColor("#1f3a63")
    width = A4[0] - 44 * mm

    body = ParagraphStyle("body", parent=base["BodyText"], fontName=body_font, fontSize=9.6,
                          leading=14.2, alignment=TA_JUSTIFY, textColor=ink, spaceAfter=7)
    title = ParagraphStyle("title", parent=body, fontName=bold_font, fontSize=19, leading=24,
                           textColor=accent, alignment=TA_CENTER, spaceAfter=10)
    heads = {
        1: ParagraphStyle("h1", parent=body, fontName=bold_font, fontSize=17, leading=22,
                          textColor=accent, alignment=TA_LEFT, spaceBefore=2, spaceAfter=12),
        2: ParagraphStyle("h2", parent=body, fontName=bold_font, fontSize=13, leading=17,
                          textColor=accent, alignment=TA_LEFT, spaceBefore=15, spaceAfter=7),
        3: ParagraphStyle("h3", parent=body, fontName=bold_font, fontSize=10.8, leading=14,
                          textColor=ink, alignment=TA_LEFT, spaceBefore=11, spaceAfter=4),
        4: ParagraphStyle("h4", parent=body, fontName=bold_font, fontSize=9.8, leading=13,
                          textColor=ink, alignment=TA_LEFT, spaceBefore=8, spaceAfter=3),
    }
    cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11.4, alignment=TA_LEFT,
                          spaceAfter=0)
    italic = ParagraphStyle("it", parent=body, alignment=TA_CENTER, textColor=muted,
                            spaceAfter=12)
    callout = ParagraphStyle("callout", parent=body, fontSize=9.3, leading=13.6,
                             alignment=TA_LEFT, spaceAfter=3)
    code_style = ParagraphStyle("code", fontName=mono, fontSize=7.6, leading=9.6,
                                textColor=colors.HexColor("#1b2330"))

    warnings: list[str] = []
    story: list = []
    seen_title = False

    for kind, payload in parse_blocks(source.read_text(encoding="utf-8").splitlines()):
        if kind == "h":
            level, text = payload
            if level == 1 and not seen_title:
                story.append(Paragraph(inline(text, mono), title))
                seen_title = True
            else:
                story.append(Paragraph(inline(text, mono), heads.get(level, heads[4])))

        elif kind == "p":
            # A subtitle is wrapped in single asterisks. A paragraph that is entirely **bold**
            # also starts and ends with an asterisk, and must not be mistaken for one.
            is_subtitle = (payload.startswith("*") and payload.endswith("*")
                           and not payload.startswith("**"))
            style = italic if is_subtitle else body
            story.append(Paragraph(inline(payload, mono), style))

        elif kind == "pagebreak":
            story.append(PageBreak())

        elif kind == "rule":
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.7, color=rule))
            story.append(Spacer(1, 7))

        elif kind == "code":
            for number, line in enumerate(payload, 1):
                if len(line) > CODE_LINE_LIMIT:
                    warnings.append(f"code line {len(line)} chars > {CODE_LINE_LIMIT}: {line[:50]}")
            block = Preformatted("\n".join(payload) or " ", code_style)
            frame = Table([[block]], colWidths=[width], hAlign="LEFT")
            frame.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F5F8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDE2EA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(frame)
            story.append(Spacer(1, 8))

        elif kind == "quote":
            paragraphs = payload
            first = paragraphs[0] if paragraphs else ""
            label = re.match(r"\*\*([^*]+?)\*\*", first)
            key = label.group(1).strip().rstrip(":").lower() if label else "note"
            fill, edge = next(
                (v for k, v in CALLOUTS.items() if key.startswith(k)), CALLOUTS["note"]
            )
            flow = [Paragraph(inline(p, mono), callout) for p in paragraphs]
            box = Table([[flow]], colWidths=[width], hAlign="LEFT")
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#" + fill)),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#" + edge)),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([box]))
            story.append(Spacer(1, 8))

        elif kind == "list":
            ordered, items = payload
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(i, mono), body), leftIndent=12) for i in items],
                    bulletType="1" if ordered else "bullet",
                    bulletFontName=body_font,
                    bulletFontSize=body.fontSize if ordered else 6,
                    bulletFormat="%s." if ordered else None,
                    leftIndent=16, bulletOffsetY=0 if ordered else 1,
                )
            )
            story.append(Spacer(1, 4))

        elif kind == "table":
            rows = payload
            columns = max(len(r) for r in rows)
            rows = [r + [""] * (columns - len(r)) for r in rows]
            data = [[Paragraph(inline(c, mono), cell) for c in row] for row in rows]
            table = Table(data, colWidths=column_widths(rows, width), hAlign="LEFT",
                          repeatRows=1)
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eceff4")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, rule),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, rule),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f8f9fb")]),
            ]))
            story.append(table)
            story.append(Spacer(1, 9))

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    stamp = f"{subtitle or 'Interim progress report'} · {dt.date.today().isoformat()} · rev {commit}"

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFont(body_font, 7.4)
        canvas.setFillColor(muted)
        canvas.drawString(22 * mm, 11 * mm, stamp)
        canvas.drawRightString(A4[0] - 22 * mm, 11 * mm, str(document.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
        title=source.stem.replace("-", " ").title(), author="",
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)
    return warnings


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
    warnings = build(args.source, out, subtitle=args.subtitle)
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
