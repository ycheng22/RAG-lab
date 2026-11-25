"""Turn files into Documents (text + metadata).

Notebook 02 writes the loaders by hand (pypdf, BeautifulSoup, csv). Notebook 04 adds the
LangChain equivalents. Both return `langchain_core.documents.Document`, which is just a small
data class with `page_content` and `metadata`. We borrow it so the two halves of the series
speak the same type.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

from raglab.config import get_settings


def load_pdf(path: Path) -> list[Document]:
    """One Document per page. Metadata: source_file, page (1-based)."""
    path = Path(path)
    reader = PdfReader(path)
    return [
        Document(page_content=page.extract_text() or "", metadata={"source_file": path.name, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


def load_markdown(path: Path) -> list[Document]:
    """The whole file as one Document (Markdown is already plain text)."""
    path = Path(path)
    return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source_file": path.name})]


def load_html(path: Path) -> list[Document]:
    """Visible text of the page as one Document. The raw HTML is kept in metadata["raw_html"]
    for structure-aware chunking later. Scripts and styles are dropped; nav/footer are not."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return [Document(page_content=text, metadata={"source_file": path.name, "raw_html": raw})]


def load_csv(path: Path) -> list[Document]:
    """One Document per row, rendered as 'column: value' lines. Metadata: source_file, row (1-based)."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Document(
            page_content="\n".join(f"{key}: {value}" for key, value in row.items()),
            metadata={"source_file": path.name, "row": i},
        )
        for i, row in enumerate(rows, start=1)
    ]


LOADERS = {".pdf": load_pdf, ".md": load_markdown, ".html": load_html, ".csv": load_csv}


def load_corpus(include: set[str] | None = None) -> list[Document]:
    """Load every supported file in data/corpus (or only the file names in `include`), sorted by name."""
    corpus = get_settings().corpus_dir
    docs: list[Document] = []
    for path in sorted(corpus.iterdir()):
        if include is not None and path.name not in include:
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader:
            docs.extend(loader(path))
    return docs


# ── LangChain loaders (added in notebook 04) ─────────────────────────────────


def load_corpus_lc(
    include: set[str] | None = None, *, corpus_dir: Path | None = None, skip_errors: bool = False
) -> list[Document]:
    """Same job as `load_corpus`, using LangChain's document loaders. Every Document gets the same
    `source_file` metadata; HTML documents also keep `raw_html` for the HTML header splitter.
    `corpus_dir` (added in notebook 09) reads another folder instead of data/corpus.
    `skip_errors` (added in notebook 11) skips files that fail to load (wrong encoding, broken PDF)
    with a warning instead of raising: one bad file must not break a whole service."""
    corpus = Path(corpus_dir) if corpus_dir else get_settings().corpus_dir
    docs: list[Document] = []
    for path in sorted(corpus.iterdir()):
        if include is not None and path.name not in include:
            continue
        try:
            loaded = _load_one_lc(path)
        except Exception as err:
            if not skip_errors:
                raise
            warnings.warn(f"skipped {path.name}: {type(err).__name__}: {err}", stacklevel=2)
            continue
        if loaded is None:
            continue
        for d in loaded:
            d.metadata.pop("source", None)  # absolute local path; source_file is enough
            d.metadata["source_file"] = path.name
        docs.extend(loaded)
    return docs


def _load_one_lc(path: Path) -> list[Document] | None:
    """One file with the matching LangChain loader; None for file types we don't index."""
    from langchain_community.document_loaders import BSHTMLLoader, CSVLoader, PyPDFLoader, TextLoader

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".md":
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".html":
        loaded = BSHTMLLoader(str(path), open_encoding="utf-8", get_text_separator=chr(10)).load()
        for d in loaded:
            d.metadata["raw_html"] = path.read_text(encoding="utf-8")
        return loaded
    if suffix == ".csv":
        return CSVLoader(str(path), encoding="utf-8").load()
    return None
