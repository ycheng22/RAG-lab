"""PostgreSQL + pgvector as the vector store (added in notebook 09; optional `[prod]` extra).

Chroma is perfect for learning and for a single machine. In production you usually want your vectors
next to the rest of your data, with backups, access control and SQL. pgvector adds a vector type and
nearest-neighbour indexes to Postgres; Neon's free tier includes it.
"""

from __future__ import annotations

from typing import Literal

from raglab import llm
from raglab.config import get_settings

SETUP_HELP = """pgvector is optional. To try it:
  1. uv sync --extra prod
  2. Get a Postgres with pgvector, either:
     - Neon (free tier): create a project at neon.tech, then run `CREATE EXTENSION IF NOT EXISTS vector;`
       in its SQL editor, or
     - Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
  3. Put the connection string in .env, e.g. DATABASE_URL=postgresql://user:password@host/dbname
  4. Restart the kernel and re-run this notebook."""


def _sqlalchemy_url(url: str) -> str:
    """langchain-postgres talks to Postgres through SQLAlchemy + psycopg 3."""
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _plain_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def pg_available() -> bool:
    """True when the `[prod]` extra is installed, DATABASE_URL is set and a connection succeeds
    within 5 seconds. Never raises."""
    try:
        import langchain_postgres  # noqa: F401
        import psycopg

        url = get_settings().database_url
        if not url:
            return False
        with psycopg.connect(_plain_url(url), connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _ensure_extension(url: str) -> None:
    import psycopg

    with psycopg.connect(_plain_url(url), connect_timeout=5, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def load_pg_store(collection_name: str):
    """Open (or create) a pgvector-backed LangChain store named `collection_name`."""
    from langchain_postgres import PGVector

    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set.\n" + SETUP_HELP)
    _ensure_extension(url)
    return PGVector(
        embeddings=llm.CachedEmbeddings(),
        collection_name=collection_name,
        connection=_sqlalchemy_url(url),
        use_jsonb=True,
    )


def create_pg_store(chunks, collection_name: str, *, reset: bool = True):
    """Build a pgvector store from chunks (deterministic ids, like the Chroma helpers)."""
    from langchain_core.documents import Document

    from raglab.store_chroma import chunk_id, clean_metadata

    store = load_pg_store(collection_name)
    if reset:
        store.delete_collection()
        store.create_collection()
    unique = {chunk_id(d): d for d in chunks}
    docs = [Document(page_content=d.page_content, metadata=clean_metadata(d.metadata)) for d in unique.values()]
    store.add_documents(docs, ids=list(unique))
    return store


def get_store(kind: Literal["chroma", "pgvector"] | None = None, collection_name: str = "token"):
    """The vector store the app should use: `kind`, or RAGLAB_STORE from .env (default chroma).
    Chroma collections are created empty if missing; fill them with `indexing.sync_index`."""
    kind = kind or get_settings().store
    if kind == "pgvector":
        return load_pg_store(collection_name)
    if kind == "chroma":
        from raglab.store_chroma import open_lc_store

        return open_lc_store(collection_name)
    raise ValueError(f"Unknown store kind {kind!r}: use 'chroma' or 'pgvector'")
