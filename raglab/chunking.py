"""Split Documents into chunks, and enrich chunk metadata.

Notebook 02 implements the chunkers by hand so you can see exactly what they do.
Notebook 04 adds `chunk_corpus_lc`, which does the same with LangChain's text splitters.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from raglab.config import hotel_metadata

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

DOC_TYPES = {
    "hotel_profiles": "profile",
    "dining_guide": "dining",
    "faq": "faq",
    "cancellation_policy": "policy",
    "room_rates": "rates",
    "loyalty_program": "loyalty",
    "guest_reviews": "reviews",
}


def _copy(doc: Document, text: str, **extra) -> Document:
    return Document(page_content=text, metadata={**doc.metadata, **extra})


# ── fixed-size ───────────────────────────────────────────────────────────────


def chunk_fixed(docs: list[Document], size: int = 500, overlap: int = 0) -> list[Document]:
    """Cut every `size` characters, ignoring words and sentences. Simple, and often wrong."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks = []
    for doc in docs:
        text = doc.page_content
        for start in range(0, max(len(text), 1), size - overlap):
            piece = text[start : start + size]
            if piece.strip():
                chunks.append(_copy(doc, piece))
            if start + size >= len(text):
                break
    return chunks


# ── recursive ────────────────────────────────────────────────────────────────


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Split on the first separator present; recurse with finer separators on pieces still too big."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep == "":
            return [text[j : j + size] for j in range(0, len(text), size)]
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep the separator with its piece
            out: list[str] = []
            for piece in pieces:
                if not piece:
                    continue
                out.extend(_split(piece, size, separators[i + 1 :]) if len(piece) > size else [piece])
            return out
    return [text]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks of at most `size` chars; start each new chunk with the
    trailing pieces of the previous one (up to `overlap` chars) so context carries over."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) > size:
            chunks.append("".join(current).strip())
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev)
            current, length = carry, carried
        current.append(piece)
        length += len(piece)
    if current and "".join(current).strip():
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_recursive(
    docs: list[Document],
    size: int = 800,
    overlap: int = 150,
    separators: list[str] = DEFAULT_SEPARATORS,
) -> list[Document]:
    """Split at the biggest natural boundary that works (paragraph, line, sentence, word),
    then pack the pieces back into chunks of up to `size` characters with `overlap`."""
    chunks = []
    for doc in docs:
        for text in _merge(_split(doc.page_content, size, list(separators)), size, overlap):
            chunks.append(_copy(doc, text))
    return chunks


# ── header-aware (Markdown) ──────────────────────────────────────────────────

_LEVEL_KEYS = {1: "title", 2: "section", 3: "subsection", 4: "subsubsection"}


def chunk_markdown_headers(docs: list[Document], levels: tuple[int, ...] = (2, 3)) -> list[Document]:
    """Start a new chunk at every Markdown header of the given levels. The header text stays in the
    chunk, and the header path is copied into metadata (## → "section", ### → "subsection")."""
    chunks = []
    for doc in docs:
        path: dict[int, str] = {}
        lines: list[str] = []

        def flush() -> None:
            text = "\n".join(lines).strip()
            only_headers = all(re.match(r"^#{1,6}\s", ln) or not ln.strip() for ln in lines)
            if text and not only_headers:  # a header with nothing under it isn't worth a chunk
                meta = {_LEVEL_KEYS[lvl]: title for lvl, title in sorted(path.items())}
                chunks.append(_copy(doc, text, **meta))

        for line in doc.page_content.splitlines():
            match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if match and len(match.group(1)) in levels:
                flush()
                lines = []
                level = len(match.group(1))
                path = {lvl: t for lvl, t in path.items() if lvl < level}
                path[level] = match.group(2).strip()
            lines.append(line)
        flush()
    return chunks


# ── metadata ─────────────────────────────────────────────────────────────────


def doc_type_for(source_file: str) -> str:
    stem = source_file.rsplit(".", 1)[0]
    for prefix, kind in DOC_TYPES.items():
        if stem.startswith(prefix):
            return kind
    return "other"


def enrich_metadata(docs: list[Document]) -> list[Document]:
    """Add `doc_type` (from the file name) and, when the chunk is about exactly one hotel, that
    hotel's fields (hotel_name, city, continent, star_rating, price_per_night, has_pool, has_spa).
    A hotel counts as mentioned if its name is in the text or in a header (section/subsection)."""
    hotels = hotel_metadata()
    out = []
    for doc in docs:
        meta = dict(doc.metadata)
        meta["doc_type"] = doc_type_for(meta.get("source_file", ""))
        headers = " ".join(str(meta.get(k, "")) for k in ("section", "subsection", "page_title"))
        named_in_headers = [name for name in hotels if name in headers]
        named = named_in_headers or [name for name in hotels if name in doc.page_content]
        if len(named) == 1:
            meta.update(hotels[named[0]])
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out


def add_context_prefix(docs: list[Document]) -> list[Document]:
    """Prepend "<hotel> — <doc type>" to chunks that belong to one hotel, so the embedding
    knows what the chunk is about even when the text itself never names the hotel."""
    out = []
    for doc in docs:
        hotel = doc.metadata.get("hotel_name")
        text = f"{hotel} — {doc.metadata.get('doc_type', '')}\n\n{doc.page_content}" if hotel else doc.page_content
        out.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return out


# ── LangChain splitters (added in notebook 04) ───────────────────────────────


def chunk_corpus_lc(
    docs: list[Document], size: int = 800, overlap: int = 150, context_prefix: bool = True
) -> list[Document]:
    """Chunk with LangChain's splitters, choosing one per file type:
    - Markdown: MarkdownHeaderTextSplitter (## → section, ### → subsection), then recursive for long sections
    - HTML: HTMLHeaderTextSplitter on the raw HTML (h1 → page_title, h2 → section)
    - PDF and CSV: RecursiveCharacterTextSplitter
    Then enrich metadata and (optionally) add the context prefix."""
    from langchain_text_splitters import (
        HTMLHeaderTextSplitter,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    recursive = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=DEFAULT_SEPARATORS)
    markdown = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section"), ("###", "subsection")], strip_headers=False
    )
    html = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "page_title"), ("h2", "section")])

    chunks: list[Document] = []
    for doc in docs:
        name = doc.metadata.get("source_file", "")
        base = {k: v for k, v in doc.metadata.items() if not k.startswith("raw_")}
        if name.endswith(".md"):
            for part in markdown.split_text(doc.page_content):
                part.metadata = {**base, **part.metadata}
                chunks.extend(recursive.split_documents([part]))
        elif name.endswith(".html"):
            for part in html.split_text(doc.metadata.get("raw_html", doc.page_content)):
                text = part.page_content.strip()
                headers = {str(v) for v in part.metadata.values()}
                if text and text not in headers:  # drop empty and header-only pieces
                    part.metadata = {**base, **part.metadata}
                    chunks.extend(recursive.split_documents([part]))
        else:
            chunks.extend(recursive.split_documents([Document(page_content=doc.page_content, metadata=base)]))
    chunks = enrich_metadata(chunks)
    return add_context_prefix(chunks) if context_prefix else chunks
