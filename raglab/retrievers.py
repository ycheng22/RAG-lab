"""Retrievers beyond plain similarity search (notebooks 06 and 07).

MultiQuery, ContextualCompression and SelfQuery come from `langchain_classic`: in LangChain 1.x
they are legacy APIs, still maintained but no longer where new features land. We use them because
they are compact, widely used examples of each idea.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from raglab import llm
from raglab.config import hotel_metadata

DOCUMENT_DESCRIPTION = (
    "Chunks of documents from Token Hotels & Resorts, a hotel group with 8 hotels: property profiles, "
    "restaurant guides, FAQs, cancellation policies, room rates, loyalty programme rules and guest reviews."
)


def _field_info():
    from langchain_classic.chains.query_constructor.schema import AttributeInfo

    hotels = hotel_metadata()
    names = ", ".join(f"'{n}'" for n in hotels)
    cities = ", ".join(f"'{h['city']}'" for h in hotels.values())
    continents = ", ".join(sorted({f"'{h['continent']}'" for h in hotels.values()}))
    prices = [h["price_per_night"] for h in hotels.values()]
    return [
        AttributeInfo(name="hotel_name", type="string", description=f"The hotel the chunk is about. One of: {names}"),
        AttributeInfo(name="city", type="string", description=f"City of the hotel. One of: {cities}"),
        AttributeInfo(name="continent", type="string", description=f"Continent of the hotel. One of: {continents}"),
        AttributeInfo(name="star_rating", type="integer", description="Hotel star rating, 3 to 5"),
        AttributeInfo(
            name="price_per_night",
            type="integer",
            description=f"Lowest nightly rate in US dollars, from {min(prices)} to {max(prices)}",
        ),
        AttributeInfo(name="has_pool", type="boolean", description="Whether the hotel has a swimming pool (true/false)"),
        AttributeInfo(name="has_spa", type="boolean", description="Whether the hotel has a spa (true/false)"),
        AttributeInfo(
            name="doc_type",
            type="string",
            description="Kind of document: 'profile' (hotel facts, amenities, check-in), 'dining' (restaurants), "
            "'faq', 'policy' (cancellation), 'rates' (room prices), 'loyalty', 'reviews'",
        ),
    ]


SELF_QUERY_FIELDS = _field_info


def _typed_translator():
    """Chroma's translator, plus type coercion: models sometimes write filter values as strings
    ('250', 'true'), which Chroma rejects for numeric and boolean fields."""
    from langchain_community.query_constructors.chroma import ChromaTranslator

    types = {f.name: f.type for f in SELF_QUERY_FIELDS()}

    def coerce(attribute, value):
        kind = types.get(attribute)
        try:
            if kind == "integer" and not isinstance(value, bool):
                return int(float(value))
            if kind == "boolean" and isinstance(value, str):
                return value.strip().lower() in ("true", "yes", "1")
        except (TypeError, ValueError):
            pass
        return value

    class TypedChromaTranslator(ChromaTranslator):
        def visit_comparison(self, comparison):
            comparison.value = coerce(comparison.attribute, comparison.value)
            return super().visit_comparison(comparison)

    return TypedChromaTranslator()


MODES = ("similarity", "mmr", "multiquery", "compression", "selfquery")


def get_retriever(mode: str, store, *, k: int = 4, chunks: list[Document] | None = None, **kw) -> BaseRetriever:
    """One factory for every retriever in the series.

    similarity  plain top-k (the notebook 05 baseline)
    mmr         top-k with diversity (notebook 03)
    multiquery  LLM writes alternative phrasings; results are merged (legacy API)
    compression retrieve k, then an LLM extracts only the relevant sentences (legacy API)
    selfquery   LLM turns the question into a search query + a metadata filter (legacy API)
    """
    if mode == "similarity":
        return store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if mode == "mmr":
        return store.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": kw.get("fetch_k", 20), "lambda_mult": kw.get("lambda_mult", 0.5)}
        )
    if mode == "multiquery":
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        return MultiQueryRetriever.from_llm(
            retriever=store.as_retriever(search_kwargs={"k": k}), llm=llm.get_lc_chat(), include_original=True
        )
    if mode == "compression":
        from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor

        return ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(llm.get_lc_chat()),
            base_retriever=store.as_retriever(search_kwargs={"k": k}),
        )
    if mode == "selfquery":
        from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

        return SelfQueryRetriever.from_llm(
            llm=llm.get_lc_chat(),
            vectorstore=store,
            structured_query_translator=_typed_translator(),  # the built-in lookup fails on langchain-community 0.4
            document_contents=DOCUMENT_DESCRIPTION,
            metadata_field_info=SELF_QUERY_FIELDS(),
            enable_limit=False,  # keep k fixed so comparisons with other retrievers are fair
            search_kwargs={"k": k},
        )
    raise ValueError(f"Unknown retriever mode {mode!r}. Choose from: {', '.join(MODES)}")


# ── keyword, hybrid and reranked retrieval (added in notebook 07) ────────────

import importlib.util  # noqa: E402
import re  # noqa: E402

from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODES = MODES + ("bm25", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for BM25 ("Check-in" -> ["check", "in"])."""
    return re.findall(r"\w+", text.lower())


def _doc_key(doc: Document) -> tuple[str, str]:
    return (doc.metadata.get("source_file", ""), doc.page_content)


def rrf_fuse(result_lists: list[list[Document]], *, k_rrf: int = 60, top_n: int = 4) -> list[Document]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k_rrf + rank), rank starting at 1.
    Documents ranked high in several lists win; raw scores from different systems are never compared."""
    scores: dict[tuple[str, str], float] = {}
    first_seen: dict[tuple[str, str], Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            first_seen.setdefault(key, doc)
    best = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [first_seen[key] for key in best]


def has_rerank_extra() -> bool:
    """True when sentence-transformers (the [rerank] extra) is installed."""
    return importlib.util.find_spec("sentence_transformers") is not None


_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        if not has_rerank_extra():
            raise ImportError("Reranking needs the [rerank] extra: run `uv sync --extra rerank` (downloads PyTorch).")
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANK_MODEL, device="cpu")
    return _cross_encoder


def rerank(question: str, docs: list[Document], top_n: int = 4) -> list[Document]:
    """Score every (question, chunk) pair with a local cross-encoder and keep the top_n."""
    if not docs:
        return []
    scores = _get_cross_encoder().predict([(question, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return [docs[i] for i in order[:top_n]]


class HybridRetriever(BaseRetriever):
    """Vector search + BM25 keyword search, fused with RRF; optionally reranked by a cross-encoder."""

    vector: BaseRetriever
    keyword: BaseRetriever
    k: int = 4
    fetch_k: int = 20
    use_rerank: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        fused = rrf_fuse(
            [self.vector.invoke(query), self.keyword.invoke(query)],
            top_n=self.fetch_k if self.use_rerank else self.k,
        )
        return rerank(query, fused, top_n=self.k) if self.use_rerank else fused


def _bm25(chunks: list[Document] | None, k: int):
    if chunks is None:
        raise ValueError("BM25 needs the chunks: pass chunks=bootstrap.get_chunks(profile)")
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)


_base_get_retriever = get_retriever


def get_retriever(mode: str, store, *, k: int = 4, chunks: list[Document] | None = None, **kw) -> BaseRetriever:  # noqa: F811
    """One factory for every retriever in the series (notebook 07 adds three modes).

    bm25           keyword search (BM25) over the same chunks; needs chunks=...
    hybrid         vector top-fetch_k + BM25 top-fetch_k, fused with RRF, keep k
    hybrid_rerank  hybrid with fetch_k candidates, then a local cross-encoder keeps the best k
    """
    if mode not in ("bm25", "hybrid", "hybrid_rerank"):
        return _base_get_retriever(mode, store, k=k, chunks=chunks, **kw)
    fetch_k = kw.get("fetch_k", 20)
    if mode == "bm25":
        return _bm25(chunks, k)
    if mode == "hybrid_rerank" and not has_rerank_extra():
        raise ImportError("Reranking needs the [rerank] extra: run `uv sync --extra rerank` (downloads PyTorch).")
    return HybridRetriever(
        vector=store.as_retriever(search_kwargs={"k": fetch_k}),
        keyword=_bm25(chunks, fetch_k),
        k=k,
        fetch_k=fetch_k,
        use_rerank=(mode == "hybrid_rerank"),
    )


get_retriever.__doc__ = (_base_get_retriever.__doc__ or "") + (get_retriever.__doc__ or "")
