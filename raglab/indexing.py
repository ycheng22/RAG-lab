"""Production indexing (added in notebook 09): incremental sync, de-duplication, document versions.

The earlier notebooks delete the index and rebuild it from scratch. That's fine for nine files; for
nine thousand it means re-embedding everything whenever anything changes, and serving an empty or
half-built index while you do. `sync_index` keeps a manifest of what's already indexed and touches
only what changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from raglab import llm
from raglab.evaluation import normalise
from raglab.store_chroma import chunk_id, clean_metadata


@dataclass
class SyncReport:
    """Files: added / updated / deleted / unchanged. Chunks: skipped_duplicates (identical chunks
    within a file, stored once), chunks_written (sent to the store, i.e. embedded unless cached)
    and chunks_removed."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped_duplicates: int = 0
    chunks_written: int = 0
    chunks_removed: int = 0


def _file_hash(docs: list[Document]) -> str:
    """Fingerprint of one file as loaded: its text and metadata (so a metadata-only change, such as
    a newer version arriving and flipping `is_latest`, also counts as a change)."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.page_content.encode("utf-8"))
        meta = {k: v for k, v in d.metadata.items() if not k.startswith("raw_")}
        h.update(json.dumps(meta, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def sync_index(
    store,
    docs: list[Document],
    manifest_path: Path,
    *,
    chunker: Callable[[list[Document]], list[Document]],
) -> SyncReport:
    """Bring `store` in line with `docs` (every loaded document of the corpus, grouped by
    `source_file`), touching only what changed since the last sync.

    The manifest (JSON) maps source_file -> {"hash": ..., "chunk_ids": [...]}.
    - new file: chunk it and add its chunks
    - changed file: re-chunk it; remove chunks that disappeared, add chunks that are new, and
      re-write chunks whose text is the same but whose metadata changed (from the embedding cache)
    - file no longer present: delete its chunks
    - unchanged file: nothing (not even chunking)
    Chunk ids are deterministic (`store_chroma.chunk_id`: sha256 of source_file + text), so an
    unchanged chunk keeps its id and is never embedded again.
    """
    manifest_path = Path(manifest_path)
    manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    by_file: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        by_file[d.metadata["source_file"]].append(d)

    report = SyncReport()
    for source_file, file_docs in sorted(by_file.items()):
        file_hash = _file_hash(file_docs)
        old = manifest.get(source_file)
        if old and old["hash"] == file_hash:
            report.unchanged += 1
            continue

        chunks: dict[str, Document] = {}
        for c in chunker(file_docs):
            cid = chunk_id(c)
            if cid in chunks:
                report.skipped_duplicates += 1
                continue
            chunks[cid] = Document(page_content=c.page_content, metadata=clean_metadata(c.metadata))

        # A chunk is kept only if the same text (same id) is already stored with the same metadata.
        new_keys = {cid: _chunk_key(cid, c) for cid, c in chunks.items()}
        old_ids = set(old["chunk_ids"]) if old else set()
        old_keys = old.get("chunk_keys", {}) if old else {}
        keep = {cid for cid in old_ids & set(chunks) if old_keys.get(cid) == new_keys[cid]}
        to_remove = sorted(old_ids - keep)
        to_add = [cid for cid in chunks if cid not in keep]

        if to_remove:
            store.delete(ids=to_remove)
        if to_add:
            store.add_documents([chunks[cid] for cid in to_add], ids=to_add)
        report.chunks_removed += len(to_remove)
        report.chunks_written += len(to_add)
        if old:
            report.updated += 1
        else:
            report.added += 1
        manifest[source_file] = {"hash": file_hash, "chunk_ids": list(chunks), "chunk_keys": new_keys}

    for source_file in sorted(set(manifest) - set(by_file)):
        ids = manifest.pop(source_file)["chunk_ids"]
        if ids:
            store.delete(ids=ids)
        report.deleted += 1
        report.chunks_removed += len(ids)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return report


def _chunk_key(cid: str, chunk: Document) -> str:
    meta = json.dumps(chunk.metadata, sort_keys=True, default=str)
    return hashlib.sha256((cid + meta).encode("utf-8")).hexdigest()[:16]


# ── de-duplication ───────────────────────────────────────────────────────────


def dedupe(chunks: list[Document], *, threshold: float = 0.97) -> tuple[list[Document], list[Document]]:
    """Return (kept, dropped). First drop exact duplicates (after lower-casing and collapsing
    whitespace), then near-duplicates: chunks whose embedding has cosine similarity >= `threshold`
    with a chunk already kept. The first chunk seen always wins. Embeddings come from the cache."""
    kept: list[Document] = []
    dropped: list[Document] = []
    seen: set[str] = set()
    for c in chunks:
        key = normalise(c.page_content)
        if key in seen:
            dropped.append(c)
        else:
            seen.add(key)
            kept.append(c)
    if len(kept) < 2:
        return kept, dropped

    vectors = np.asarray(llm.embed([c.page_content for c in kept]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    final: list[int] = []
    for i in range(len(kept)):
        if final and float((vectors[final] @ vectors[i]).max()) >= threshold:
            dropped.append(kept[i])
        else:
            final.append(i)
    return [kept[i] for i in final], dropped


# ── document versions ────────────────────────────────────────────────────────

_VERSION = re.compile(r"^(?P<family>.+)_(?P<year>(19|20)\d{2})$")


def tag_versions(docs: list[Document]) -> list[Document]:
    """Set `is_latest` on every document, and `doc_version` (the year) on versioned ones.

    A versioned file ends in _YYYY (cancellation_policy_2025.md); its family is the name without
    the suffix (cancellation_policy). Within a family only the highest year is the latest.
    Unversioned documents are always `is_latest=True`. Returns new Documents; inputs are untouched."""
    parsed = []
    latest: dict[str, int] = {}
    for d in docs:
        match = _VERSION.match(Path(d.metadata.get("source_file", "")).stem)
        if match:
            family, year = match["family"], int(match["year"])
            latest[family] = max(latest.get(family, year), year)
            parsed.append((d, family, year))
        else:
            parsed.append((d, None, None))
    out = []
    for d, family, year in parsed:
        meta = dict(d.metadata)
        if family is None:
            meta["is_latest"] = True
        else:
            meta["doc_version"] = year
            meta["is_latest"] = year == latest[family]
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out
