"""Chroma, the vector database used for most of the series.

Notebook 03 uses the raw `chromadb` client so you can see what a vector store stores and how a
query works. Notebook 04 adds the LangChain wrapper (`create_lc_store` / `load_lc_store`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
import numpy as np
from langchain_core.documents import Document

from raglab import llm
from raglab.config import get_settings

ADD_BATCH = 500


def get_client(path: Path | None = None) -> chromadb.ClientAPI:
    """A persistent Chroma client storing its files under .cache/chroma (or `path`)."""
    path = Path(path or get_settings().chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))


def chunk_id(doc: Document) -> str:
    """Deterministic id: the same chunk text from the same file always gets the same id."""
    key = f"{doc.metadata.get('source_file', '')}\n{doc.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def clean_metadata(meta: dict) -> dict:
    """Chroma accepts only str/int/float/bool values. Drop None and bulky raw_* fields."""
    out = {}
    for key, value in meta.items():
        if value is None or key.startswith("raw_"):
            continue
        out[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out


def create_collection(
    name: str,
    docs: list[Document],
    *,
    path: Path | None = None,
    reset: bool = True,
    metadata: dict | None = None,
) -> chromadb.Collection:
    """Embed `docs` (via the cached llm.embed) and store them in a cosine-distance collection."""
    client = get_client(path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # the collection didn't exist yet
            pass
    collection = client.get_or_create_collection(
        name, metadata=metadata or None, configuration={"hnsw": {"space": "cosine"}}
    )
    unique = {chunk_id(d): d for d in docs}  # identical chunks are stored once
    ids, chunks = list(unique), list(unique.values())
    vectors = llm.embed([d.page_content for d in chunks])
    for i in range(0, len(ids), ADD_BATCH):
        collection.add(
            ids=ids[i : i + ADD_BATCH],
            embeddings=vectors[i : i + ADD_BATCH],
            documents=[d.page_content for d in chunks[i : i + ADD_BATCH]],
            metadatas=[clean_metadata(d.metadata) or None for d in chunks[i : i + ADD_BATCH]],
        )
    return collection


def _to_docs(result: dict) -> list[tuple[Document, float]]:
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    return [(Document(page_content=t, metadata=m or {}), float(d)) for t, m, d in zip(docs, metas, dists)]


def query(collection: chromadb.Collection, text: str, k: int = 4, where: dict | None = None) -> list[tuple[Document, float]]:
    """The k nearest chunks to `text` as (Document, cosine distance) pairs, nearest first.
    Distance = 1 - cosine similarity, so smaller is closer."""
    result = collection.query(
        query_embeddings=[llm.embed_one(text)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _to_docs(result)


def query_mmr(
    collection: chromadb.Collection,
    text: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    where: dict | None = None,
) -> list[Document]:
    """Maximal Marginal Relevance: fetch `fetch_k` candidates, then pick k that are relevant to the
    query AND different from each other. lambda_mult=1 is pure relevance; 0 is pure diversity."""
    q = np.asarray(llm.embed_one(text))
    result = collection.query(
        query_embeddings=[q.tolist()],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "embeddings"],
    )
    texts, metas = result["documents"][0], result["metadatas"][0]
    vecs = np.asarray(result["embeddings"][0], dtype=float)
    if len(texts) == 0:
        return []
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    q = q / np.linalg.norm(q)
    relevance = vecs @ q
    selected: list[int] = []
    candidates = list(range(len(texts)))
    while candidates and len(selected) < k:
        if selected:
            redundancy = (vecs[candidates] @ vecs[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(candidates))
        scores = lambda_mult * relevance[candidates] - (1 - lambda_mult) * redundancy
        best = candidates[int(np.argmax(scores))]
        selected.append(best)
        candidates.remove(best)
    return [Document(page_content=texts[i], metadata=metas[i] or {}) for i in selected]


# ── LangChain vector store (added in notebook 04) ────────────────────────────


def create_lc_store(
    chunks: list[Document],
    collection_name: str,
    *,
    path: Path | None = None,
    reset: bool = True,
    collection_metadata: dict | None = None,
):
    """Build a LangChain `Chroma` store from chunks (cosine distance, deterministic ids)."""
    from langchain_chroma import Chroma

    client = get_client(path)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    store = Chroma(
        collection_name=collection_name,
        embedding_function=llm.CachedEmbeddings(),
        client=client,
        collection_metadata=collection_metadata or None,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    unique = {chunk_id(d): d for d in chunks}
    ids = list(unique)
    docs = [Document(page_content=d.page_content, metadata=clean_metadata(d.metadata)) for d in unique.values()]
    for i in range(0, len(docs), ADD_BATCH):
        store.add_documents(docs[i : i + ADD_BATCH], ids=ids[i : i + ADD_BATCH])
    return store


def load_lc_store(collection_name: str, *, path: Path | None = None):
    """Open an existing collection as a LangChain `Chroma` store."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=llm.CachedEmbeddings(),
        client=get_client(path),
        create_collection_if_not_exists=False,
    )
