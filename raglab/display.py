"""Small helpers for showing things nicely inside notebooks."""

from __future__ import annotations

import html
import textwrap
from collections.abc import Sequence

from raglab.config import get_settings

SHOWN_METADATA = ("source_file", "page", "doc_type", "hotel_name", "section", "subsection", "page_title")


def show_chunks(docs: Sequence, n: int = 3, width: int = 300) -> None:
    """Print the first `n` chunks: where they came from, a few metadata fields, and a text preview."""
    docs = list(docs)
    print(f"{len(docs)} chunk(s)" + (f", showing {min(n, len(docs))}" if len(docs) > n else ""))
    for i, doc in enumerate(docs[:n], start=1):
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", str(doc))
        tags = " | ".join(f"{k}={meta[k]}" for k in SHOWN_METADATA if meta.get(k) not in (None, ""))
        preview = text if len(text) <= width else text[:width].rstrip() + " …"
        print(f"\n[{i}] {tags}  ({len(text)} chars)")
        print(textwrap.indent(textwrap.fill(" ".join(preview.split()), 100), "    "))


def show_diagram(name: str, width: int = 900, interactive: bool = False) -> None:
    """Show an Archify diagram.

    The PNG is embedded in the cell output, so it renders on GitHub, in Colab and in Kaggle.
    With interactive=True the self-contained interactive HTML is shown inline instead
    (works in Jupyter, VS Code, Colab and Kaggle; GitHub's static viewer cannot run it).
    """
    from IPython.display import HTML, Image, Markdown, display

    diagrams = get_settings().diagrams_dir
    png, page = diagrams / f"{name}.png", diagrams / f"{name}.html"
    if interactive and page.exists():
        srcdoc = html.escape(page.read_text(encoding="utf-8"), quote=True)
        display(HTML(f'<iframe srcdoc="{srcdoc}" style="width:100%;height:720px;border:0"></iframe>'))
        return
    if png.exists():
        display(Image(filename=str(png), width=width))
    else:
        print(f"(diagram '{name}' not found in {diagrams})")
    if page.exists():
        display(Markdown(f"[Open interactive version](../diagrams/{name}.html)"))
