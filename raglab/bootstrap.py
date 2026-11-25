"""Build (or reuse) the index a notebook needs, so every notebook runs on its own.

Each named profile is frozen: same files, same chunking, same collection name. A profile's index is
rebuilt automatically when the corpus or the profile changes (detected by a fingerprint stored in
the collection), and reused otherwise. Embeddings are cached on disk, so rebuilds are cheap.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

from langchain_core.documents import Document

from raglab import chunking, loaders
from raglab.config import get_settings

# Bump when the chunking code changes in a way that should invalidate existing indexes.
BUILD_VERSION = "2"


@dataclass(frozen=True)
class IndexProfile:
    name: str
    chunker: Literal["hand_recursive", "lc", "fixed"]
    chunk_size: int
    overlap: int
    context_prefix: bool
    exclude_files: tuple[str, ...] = ()


PROFILES: dict[str, IndexProfile] = {
    "nb03_raw": IndexProfile("nb03_raw", "hand_recursive", 800, 150, True),
    "nb04_baseline": IndexProfile("nb04_baseline", "lc", 800, 150, True),
}


def _profile(profile: str | IndexProfile) -> IndexProfile:
    if isinstance(profile, IndexProfile):
        return profile
    if profile not in PROFILES:
        raise KeyError(f"Unknown index profile {profile!r}. Known: {', '.join(PROFILES)}")
    return PROFILES[profile]


def _included_files(p: IndexProfile) -> list:
    corpus = get_settings().corpus_dir
    return [f for f in sorted(corpus.iterdir()) if f.is_file() and f.name not in p.exclude_files]


def corpus_fingerprint(profile: str | IndexProfile) -> str:
    """sha256 over the profile's settings and the name + bytes of every file it indexes."""
    p = _profile(profile)
    h = hashlib.sha256(f"{BUILD_VERSION}|{sorted(asdict(p).items())}".encode())
    for f in _included_files(p):
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:20]


def _hand_chunks(docs: list[Document], size: int, overlap: int) -> list[Document]:
    """Notebook 02's chunkers: Markdown by header then recursively; everything else recursively.
    CSV rows are already small and stay one chunk each."""
    out: list[Document] = []
    for doc in docs:
        name = doc.metadata.get("source_file", "")
        if name.endswith(".csv"):
            out.append(doc)
        elif name.endswith(".md"):
            out.extend(chunking.chunk_recursive(chunking.chunk_markdown_headers([doc]), size, overlap))
        else:
            out.extend(chunking.chunk_recursive([doc], size, overlap))
    return out


def build_chunks(profile: str | IndexProfile) -> list[Document]:
    """Load and chunk the corpus exactly as the profile says (no API calls)."""
    p = _profile(profile)
    include = {f.name for f in _included_files(p)}
    if p.chunker == "hand_recursive":
        chunks = _hand_chunks(loaders.load_corpus(include), p.chunk_size, p.overlap)
        chunks = chunking.enrich_metadata(chunks)
    elif p.chunker == "fixed":  # added in notebook 08: blind fixed-size cuts, no structure
        docs = loaders.load_corpus_lc(include)
        for d in docs:
            d.metadata.pop("raw_html", None)
        chunks = chunking.enrich_metadata(chunking.chunk_fixed(docs, p.chunk_size, p.overlap))
    else:
        chunks = chunking.chunk_corpus_lc(  # added in notebook 04
            loaders.load_corpus_lc(include), size=p.chunk_size, overlap=p.overlap, context_prefix=False
        )
    return chunking.add_context_prefix(chunks) if p.context_prefix else chunks


def _needs_build(collection_meta: dict | None, fingerprint: str, count: int | None = None) -> str | None:
    if collection_meta is None:
        return "building index"
    if "fingerprint" not in collection_meta or count == 0:
        return "rebuilding (the previous build did not finish)"
    if collection_meta.get("fingerprint") != fingerprint:
        return "rebuilding (corpus changed)"
    return None


def ensure_raw_collection(profile: str = "nb03_raw"):
    """Return the raw chromadb collection for `profile`, building it if missing or stale."""
    from raglab import store_chroma

    p = _profile(profile)
    fingerprint = corpus_fingerprint(p)
    client = store_chroma.get_client()
    try:
        existing = client.get_collection(p.name)
        meta = existing.metadata or {}
    except Exception:
        existing, meta = None, None
    count = existing.count() if existing is not None else None
    reason = _needs_build(meta if existing is not None else None, fingerprint, count)
    if reason is None:
        print(f"[{p.name}] reusing index ({existing.count()} chunks)")
        return existing
    print(f"[{p.name}] {reason}...")
    chunks = build_chunks(p)
    collection = store_chroma.create_collection(p.name, chunks, metadata={"fingerprint": fingerprint})
    print(f"[{p.name}] {collection.count()} chunks indexed")
    return collection


# ── LangChain indexes (added in notebook 04) ─────────────────────────────────


def get_chunks(profile: str) -> list:
    """The exact chunks indexed for `profile` (re-chunked deterministically; no API calls)."""
    return build_chunks(profile)


def ensure_index(profile: str = "nb04_baseline"):
    """Return a LangChain Chroma store for `profile`, building it if missing or stale."""
    from raglab import store_chroma

    p = _profile(profile)
    fingerprint = corpus_fingerprint(p)
    client = store_chroma.get_client()
    try:
        collection = client.get_collection(p.name)
        meta, count = collection.metadata or {}, collection.count()
    except Exception:
        meta, count = None, None
    reason = _needs_build(meta, fingerprint, count)
    if reason is None:
        store = store_chroma.load_lc_store(p.name)
        print(f"[{p.name}] reusing index ({store._collection.count()} chunks)")
        return store
    print(f"[{p.name}] {reason}...")
    store = store_chroma.create_lc_store(build_chunks(p), p.name, collection_metadata={"fingerprint": fingerprint})
    print(f"[{p.name}] {store._collection.count()} chunks indexed")
    return store


# ── tuning profiles (added in notebook 08) ───────────────────────────────────

# A deliberately broken index for notebook 08's diagnosis exercise: blind 2,500-character cuts,
# no overlap, no context prefix, and one policy file missing (a "stale" index).
PROFILES["nb08_broken"] = IndexProfile(
    "nb08_broken", "fixed", 2500, 0, False, exclude_files=("cancellation_policy_2025.md",)
)
# The same bad chunking with every file indexed (step 1 of notebook 08's fixes).
PROFILES["nb08_broken_all_files"] = IndexProfile("nb08_broken_all_files", "fixed", 2500, 0, False)


def make_profile(chunk_size: int, overlap: int, context_prefix: bool = True) -> str:
    """Register a LangChain-chunked profile for a tuning grid and return its name,
    e.g. make_profile(400, 100) -> "grid_cs400_ov100_p1"."""
    name = f"grid_cs{chunk_size}_ov{overlap}_p{int(context_prefix)}"
    PROFILES.setdefault(name, IndexProfile(name, "lc", chunk_size, overlap, context_prefix))
    return name


# ── incremental indexing (added in notebook 09) ──────────────────────────────

# Same chunking as the baseline; notebook 09 fills it incrementally with `indexing.sync_index`
# in its own folder (.cache/chroma_nb09), so it never disturbs the other notebooks' indexes.
PROFILES["nb09_incremental"] = IndexProfile("nb09_incremental", "lc", 800, 150, True)


def profile_chunker(profile: str | IndexProfile):
    """A function docs -> chunks that chunks already-loaded documents the way `profile` says
    (LangChain chunkers only). Handy as `sync_index(..., chunker=profile_chunker("nb09_incremental"))`."""
    p = _profile(profile)
    if p.chunker != "lc":
        raise ValueError(f"profile_chunker supports 'lc' profiles only, not {p.chunker!r}")

    def chunker(docs: list[Document]) -> list[Document]:
        chunks = chunking.chunk_corpus_lc(docs, size=p.chunk_size, overlap=p.overlap, context_prefix=False)
        return chunking.add_context_prefix(chunks) if p.context_prefix else chunks

    return chunker
