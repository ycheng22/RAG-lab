"""Build the fictional Token Hotels & Resorts corpus from facts.yaml + templates.

    uv run python data/_build/build_corpus.py

Writes every file in data/corpus/. Re-running is deterministic: same facts, same bytes
(the PDF carries a fixed creation date so its bytes don't change between builds).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from pathlib import Path

import yaml
from fpdf import FPDF
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus"
FIXED_DATE = dt.datetime(2025, 3, 1, tzinfo=dt.timezone.utc)

TEXT_FILES = {
    "dining_guide.md": "dining_guide.md.j2",
    "faq.html": "faq.html.j2",
    "cancellation_policy_2024.md": "cancellation_policy_2024.md.j2",
    "cancellation_policy_2025.md": "cancellation_policy_2025.md.j2",
    "loyalty_program.md": "loyalty_program.md.j2",
    "loyalty_program_reprint.md": "loyalty_program_reprint.md.j2",
    "guest_reviews.md": "guest_reviews.md.j2",
}


def round5(x: float) -> int:
    return int(5 * round(x / 5))


def tidy_markdown(text: str) -> str:
    """Blank line before every heading, at most one blank line in a row."""
    text = re.sub(r"\n(#{1,3} )", r"\n\n\1", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def write_text(path: Path, text: str) -> None:
    if path.suffix == ".md":
        text = tidy_markdown(text)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def build_rates(facts: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["hotel", "city", "room_type", "season", "nightly_rate_usd", "max_guests"])
    for hotel in facts["hotels"]:
        for room in facts["room_types"]:
            for season in facts["seasons"]:
                rate = round5(hotel["price_per_night"] * room["multiplier"] * season["multiplier"])
                writer.writerow([hotel["name"], hotel["city"], room["name"], season["name"], rate, room["max_guests"]])
    return out.getvalue()


def latin1(text: str) -> str:
    """The PDF uses a built-in font (Latin-1 only): swap the few characters outside it."""
    return text.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"')


def markdown_to_pdf(markdown: str, path: Path) -> None:
    """Render the tiny Markdown subset used by hotel_profiles.md.j2: #, ##, paragraphs, pipe tables."""
    pdf = FPDF(format="A4")
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_title("Token Hotels & Resorts - Property Guide")
    pdf.set_author("Token Hotels & Resorts (fictional)")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            pdf.set_font("Helvetica", size=10.5)
            pdf.multi_cell(0, 5.5, latin1(" ".join(paragraph)), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            paragraph.clear()

    def flush_table() -> None:
        if table:
            pdf.set_font("Helvetica", size=9.5)
            with pdf.table(col_widths=(35, 65), text_align="LEFT", line_height=6) as t:
                for row in table:
                    r = t.row()
                    for cell in row:
                        r.cell(latin1(cell))
            pdf.ln(4)
            table.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            flush_paragraph()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= {"-", " ", ":"} for c in cells):
                table.append(cells)
            continue
        flush_table()
        if not line:
            flush_paragraph()
        elif line.startswith("## "):
            flush_paragraph()
            if pdf.get_y() > 220:
                pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.multi_cell(0, 8, latin1(line[3:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif line.startswith("# "):
            flush_paragraph()
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.multi_cell(0, 10, latin1(line[2:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        else:
            paragraph.append(line)
    flush_paragraph()
    flush_table()
    pdf.output(str(path))


def main() -> None:
    facts = yaml.safe_load((HERE / "facts.yaml").read_text(encoding="utf-8"))
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    CORPUS.mkdir(parents=True, exist_ok=True)

    for out_name, template in TEXT_FILES.items():
        write_text(CORPUS / out_name, env.get_template(template).render(**facts))

    profiles_md = env.get_template("hotel_profiles.md.j2").render(**facts)
    markdown_to_pdf(profiles_md, CORPUS / "hotel_profiles.pdf")

    (CORPUS / "room_rates.csv").write_text(build_rates(facts), encoding="utf-8", newline="\n")

    for f in sorted(CORPUS.iterdir()):
        print(f"{f.name:<32}{f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
