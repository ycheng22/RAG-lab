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
