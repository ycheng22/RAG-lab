# %% [markdown]
# # 06 · Advanced retrievers
#
# Notebook 05 showed where the baseline struggles: questions that need facts from *several*
# chunks (comparative, multi-hop) and questions that are really *filters* ("hotels in Asia with a
# pool"). This notebook tries three classic fixes, each adding an LLM call somewhere in retrieval:
#
# | Retriever | Idea | Extra LLM calls per question |
# |---|---|---|
# | **MultiQuery** | Ask the question several ways and merge the results | 1 (writes the variants) |
# | **Contextual compression** | Retrieve, then keep only the relevant sentences | 1 per retrieved chunk |
# | **SelfQuery** | Turn the question into a search query + a metadata filter | 1 (writes the query) |
#
# > **Legacy API note.** In LangChain 1.x these three retrievers live in the `langchain_classic`
# > package. They still work and are widely used, but new LangChain features land elsewhere
# > (notebook 12 builds the same ideas as an agent). Learn the *ideas* here; the API may move.
#
# Every retriever is scored against the notebook 05 baseline with the same evaluation set.

# %%
# <<SETUP>>

# %%
# Setup 2/3: the baseline index (notebook 04)
from raglab.bootstrap import ensure_index

store = ensure_index("nb04_baseline")

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %%
from raglab.display import show_diagram

show_diagram("advanced_retrievers")

# %% [markdown]
# ## 1. MultiQuery: more ways to ask
#
# A question's wording might not match the document's wording. MultiQuery asks the LLM for a few
# alternative phrasings, searches with each one (plus the original), and merges the unique results.
# Turn on its logger to see the generated queries:

# %%
import logging
import sys

from raglab import retrievers

handler = logging.StreamHandler(sys.stdout)
for name in ("langchain_classic.retrievers.multi_query", "langchain_classic.retrievers.self_query.base"):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(handler)

multiquery = retrievers.get_retriever("multiquery", store, k=4)
docs = multiquery.invoke("Which hotel is best for a relaxing holiday?")
print(f"\n{len(docs)} unique chunks from", sorted({d.metadata.get('hotel_name', '-') for d in docs}))

# %% [markdown]
# Note the result size: up to 4 chunks *per query*, so usually more than 4 in total. More chunks
# mean more chances to include the evidence, and also a longer, more expensive prompt. Keep that in
# mind when comparing scores.
#
# ## 2. Contextual compression: less noise
#
# Retrieved chunks contain a lot of text that has nothing to do with the question. A
# **compressor** asks the LLM to extract only the relevant parts of each chunk (and drop chunks
# with nothing relevant).

# %%
question = "What time is check-in at Token Vancouver?"
plain = store.as_retriever(search_kwargs={"k": 4}).invoke(question)
compression = retrievers.get_retriever("compression", store, k=4)
compressed = compression.invoke(question)
print("plain:     ", [len(d.page_content) for d in plain], "characters")
print("compressed:", [len(d.page_content) for d in compressed], "characters")
for d in compressed:
    print("  →", d.page_content[:200].replace("\n", " "))

# %% [markdown]
# Shorter context means a cheaper, more focused generation step. The price is one LLM call per
# retrieved chunk *during retrieval*, plus the risk that the extractor drops something the answer needed.
#
# ## 3. SelfQuery: let the LLM write the filter
#
# In notebook 03 we wrote metadata filters by hand. SelfQuery gives the LLM a description of the
# metadata fields (`retrievers.SELF_QUERY_FIELDS()`) and asks it to translate the question into a
# search string **and** a filter.

# %%
for field in retrievers.SELF_QUERY_FIELDS()[:4]:
    print(f"{field.name:<16}{field.type:<9}{field.description[:90]}")

# %%
selfquery = retrievers.get_retriever("selfquery", store, k=4)
for q in ["Which Token hotels in Asia have a swimming pool?", "Cheap hotels under $250 a night with a pool"]:
    structured = selfquery.query_constructor.invoke({"query": q})
    print(f"Q: {q}\n   search text: {structured.query!r}\n   filter: {structured.filter}")
    print("   results:", [(d.metadata.get("hotel_name"), d.metadata["source_file"]) for d in selfquery.invoke(q)], "\n")

# %% [markdown]
# Two things to notice:
#
# - The filter only works on chunks that *have* the metadata. A PDF chunk that `enrich_metadata`
#   couldn't pin to a single hotel has no `continent`, so the filter silently excludes it (notebook 03).
# - Models sometimes write filter values as strings (`'250'`, `'true'`). `raglab` coerces them to
#   the right type before Chroma sees them (`retrievers._typed_translator`); without that, Chroma
#   raises an error. Expect this kind of glue code whenever an LLM writes structured queries.

# %%
# Quiet the loggers again before scoring 35 questions.
for name in ("langchain_classic.retrievers.multi_query", "langchain_classic.retrievers.self_query.base"):
    logging.getLogger(name).removeHandler(handler)

# %% [markdown]
# ## 4. Score them all
#
# Now the real test: the same 35 questions as the baseline. Retrieval metrics need no answer
# generation, but three of these retrievers make LLM calls *while retrieving*, so this cell costs a
# little. As before, the notebook shows the saved full-run results when they exist.

# %%
import pandas as pd

from raglab import evaluation as ev
from raglab import llm

items = ev.load_eval("full")
modes = ["similarity", "multiquery", "compression", "selfquery"]
meta = {"profile": "nb04_baseline", "k": 4}

retrieval = {}
for mode in modes:
    name = "nb05_baseline_retrieval" if mode == "similarity" else f"nb06_{mode}_retrieval"
    before = cost.total()
    retrieval[mode] = ev.run_or_load(
        name, lambda m=mode: ev.retrieval_metrics(retrievers.get_retriever(m, store, k=4), items), meta={**meta, "retriever": mode}
    )
    print(f"  {mode}: ${cost.total() - before:.4f} in this run")

ev.compare(retrieval)[["evidence_recall", "hit", "rr", "n_docs", "latency_ms"]]

# %% [markdown]
# And by category, where the differences live:

# %%
hit_by_category = pd.DataFrame({m: df.groupby("category")["hit"].mean() for m, df in retrieval.items()}).round(2)
hit_by_category

# %% [markdown]
# ## 5. Does better retrieval give better answers?
#
# Retrieval metrics are a proxy. What we ultimately care about is correct answers. The quick subset
# runs live; the full-run table shows all 35 questions.

# %%
from raglab import chains

settings = get_settings()
answer_items = items if settings.full_run else ev.load_eval("quick")
answers = {}
for mode in modes:
    name = "nb05_baseline_answers" if mode == "similarity" else f"nb06_{mode}_answers"
    chain = chains.build_rag_chain_with_sources(retrievers.get_retriever(mode, store, k=4))
    answers[mode] = ev.run_or_load(
        name, lambda c=chain: ev.score_answers(ev.run_rag(c, answer_items)), meta={**meta, "retriever": mode}
    )

pd.DataFrame({m: {"accuracy": df["correct"].mean(), "questions": len(df)} for m, df in answers.items()}).T.round(2)

# %% [markdown]
# ## 6. What the numbers say
#
# In the run saved here none of the three is a clear win on this corpus, and that is the lesson:
#
# - **MultiQuery** nudged evidence recall up, but it also retrieved about 6 chunks per question
#   instead of 4, cost one LLM call per question and added seconds of latency. Part of the gain may
#   come simply from *more chunks* (check with `k=8` in the exercises). When we logged the generated
#   queries for all 35 questions, the rewriter sometimes read "Token" as crypto *tokens* ("hotels
#   that use tokens"). Query rewriting needs domain context.
# - **Compression** lowered evidence recall. It can only *remove* text from what the base retriever
#   found, and on multi-hop questions it removed exactly the sentences linking two facts. Its benefit
#   is shorter prompts; its risk is dropping what the answer needed.
# - **SelfQuery** scored *worst*, especially on the metadata-filter questions it was meant to help.
#   Two reasons, both visible in section 3: the evidence for those questions sits in PDF profile
#   chunks that span two hotels and so carry **no** hotel metadata (the filter excludes them), and
#   the generated search text is often empty (`' '`), so results within the filter come back in
#   arbitrary order: rate rows instead of profiles.
#
# The fix for SelfQuery isn't a better retriever, it's **better metadata**: chunk the PDF so every
# chunk belongs to exactly one hotel. Retrieval techniques can't recover information the index
# doesn't carry. Notebook 07 tries a different angle (keyword search plus reranking), and notebook 08
# shows how chunking changes these numbers.
#
# ## Your turn
#
# 1. Score plain similarity with `k=8` (`retrievers.get_retriever("similarity", store, k=8)`).
#    Does it match MultiQuery's evidence recall? What does that tell you about where MultiQuery's gain came from?
# 2. Find a question where SelfQuery's filter is wrong. Use `selfquery.query_constructor.invoke(...)`
#    to look at the filter. How would you improve the field descriptions?
# 3. Build a chain with the compression retriever and compare the prompt token count (`cost.report()`)
#    with the baseline chain on the same five questions.
#
# ## Key takeaways
#
# - Each advanced retriever spends LLM calls to fix a specific weakness: wording (MultiQuery),
#   noise (compression), structure (SelfQuery).
# - Measure them on the same evaluation set, per category. Averages hide where each one helps or hurts.
# - Check what you're really comparing: number of chunks, cost and latency differ between retrievers.
#
# **Added to raglab:** `retrievers.get_retriever` (modes `similarity`, `mmr`, `multiquery`,
# `compression`, `selfquery`), `retrievers.SELF_QUERY_FIELDS`, `retrievers.DOCUMENT_DESCRIPTION`.

# %%
cost.report()
