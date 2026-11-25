# %% [markdown]
# # 09 · Production indexing: incremental updates, duplicates and versions
#
# So far every notebook builds its index with "delete everything, embed everything". That works for
# nine files. In production, documents change every day, and rebuilding means:
#
# - **re-embedding everything** for every small edit (time and money, growing with the corpus),
# - **serving an empty or half-built index** while the rebuild runs, or keeping two copies,
# - **no record** of which version of which file the index holds.
#
# **In this notebook you will:**
# - keep the index in sync with a folder of documents, touching only what changed,
# - remove duplicate content before it crowds out other results,
# - tag document versions so only the current policy is retrieved,
# - see LangChain's built-in equivalent, and (optionally) move the index to Postgres with pgvector.
#
# Cost: well under $0.01 (a few embeddings and three short answers).

# %%
# <<SETUP>>

# %%
# Setup 2/3: a private working copy of the corpus, and a fresh index in its own folder
import shutil

from raglab import bootstrap, chains, indexing, loaders, store_chroma
from raglab.display import show_chunks, show_diagram

settings = get_settings()
work = settings.cache_dir / "nb09"
corpus = work / "corpus"  # we edit this copy; data/corpus is never touched
manifest = work / "manifest.json"
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(settings.corpus_dir, corpus)

chroma_dir = settings.cache_dir / "chroma_nb09"
try:
    store_chroma.get_client(chroma_dir).delete_collection("nb09_incremental")
except Exception:
    pass  # first run: nothing to delete
store = store_chroma.open_lc_store("nb09_incremental", path=chroma_dir)
chunker = bootstrap.profile_chunker("nb09_incremental")  # the baseline's chunking: 800 / 150, with prefix
print(sorted(p.name for p in corpus.iterdir()))

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. Sync instead of rebuild
#
# `indexing.sync_index` keeps a **manifest**: for every file, a hash of its content and the ids of
# its chunks. Chunk ids are deterministic (a hash of file name + chunk text), so the same chunk always
# gets the same id. On each sync it compares the folder with the manifest:

# %%
show_diagram("incremental_indexing")

# %%
def load():
    """Load the working corpus and tag document versions (section 4 explains the tags)."""
    return indexing.tag_versions(loaders.load_corpus_lc(corpus_dir=corpus))


report = indexing.sync_index(store, load(), manifest, chunker=chunker)
print(report)
print("chunks in the index:", store._collection.count())

# %% [markdown]
# Every file is new, so all 9 are **added**. (The embeddings come from the on-disk cache that
# notebook 04 filled, so even this first build is free.) Now sync again without changing anything:

# %%
print(indexing.sync_index(store, load(), manifest, chunker=chunker))

# %% [markdown]
# Nothing to do: 9 files **unchanged**, no chunking, no embedding calls.
#
# ## 2. One file changes
#
# Token Marrakech's restaurant moves dinner to 7:00 PM. We edit the dining guide **in the working copy**:

# %%
dining = corpus / "dining_guide.md"
text = dining.read_text(encoding="utf-8")
dining.write_text(text.replace("7:30 PM - 11:30 PM", "7:00 PM - 11:30 PM", 1), encoding="utf-8")

report = indexing.sync_index(store, load(), manifest, chunker=chunker)
print(report)

# %% [markdown]
# **1 updated, 8 unchanged.** And inside the dining guide, only the chunk whose text changed was
# replaced (`chunks_written=1`, `chunks_removed=1`): the other chunks kept their ids and their vectors.
# A full rebuild would have re-embedded all of the index's chunks. Does retrieval see the new time?

# %%
hits = store.similarity_search("When does Dar Zitoun open for dinner?", k=1)
show_chunks(hits, n=1, width=400)

# %% [markdown]
# ## 3. A file is removed

# %%
(corpus / "guest_reviews.md").unlink()
report = indexing.sync_index(store, load(), manifest, chunker=chunker)
print(report)
print("chunks in the index:", store._collection.count())

# %% [markdown]
# Its chunks are **deleted** from the index too. Without this step, a "deleted" document keeps
# answering questions forever. That's a classic production bug: removed content that the assistant
# still quotes.
#
# ## 4. Duplicates
#
# Real corpora are full of copies: the same text in a PDF and a web page, a reprint, a forwarded email.
# Duplicates waste index space and, worse, crowd the top-k with the **same** information, pushing
# out other useful chunks. Our corpus has a reprint of the loyalty programme: the same text with a
# reworded opening. `indexing.dedupe` drops exact duplicates first (ignoring case and spacing), then
# near-duplicates whose embeddings are almost identical (cosine similarity ≥ 0.97):

# %%
loyalty_docs = loaders.load_corpus_lc({"loyalty_program.md", "loyalty_program_reprint.md"}, corpus_dir=corpus)
loyalty_chunks = chunker(loyalty_docs)
kept, dropped = indexing.dedupe(loyalty_chunks)
print(f"{len(loyalty_chunks)} chunks -> kept {len(kept)}, dropped {len(dropped)}")
for d in dropped:
    print(f"  dropped from {d.metadata['source_file']}: {d.page_content[:70]!r}")

# %%
# How close is "close"? The similarity of each dropped chunk to its best match among the kept ones:
import numpy as np

from raglab import llm

kept_vecs = np.asarray(llm.embed([d.page_content for d in kept]))
for d in dropped:
    v = np.asarray(llm.embed_one(d.page_content))
    sims = kept_vecs @ v / (np.linalg.norm(kept_vecs, axis=1) * np.linalg.norm(v))
    print(f"{sims.max():.3f}  {d.page_content[:60]!r}")

# %% [markdown]
# All five dropped chunks are word-for-word copies (similarity 1.000, caught by the exact check).
# The reprint's reworded opening ("costs nothing to join" instead of "is free to join") was kept:
# its similarity is below 0.97. Near-duplicate thresholds are a judgement call: too low, and you merge
# chunks that differ in an important detail (a price, a date). **Look at what you drop.**
#
# In a real pipeline you'd run `dedupe` inside the chunker passed to `sync_index`, so duplicates
# never reach the index.
#
# ## 5. Versions: only the current policy should answer
#
# The corpus has a 2024 and a 2025 cancellation policy. The 2025 one changed free cancellation from
# 48 to 72 hours. `indexing.tag_versions` reads the `_YYYY` suffix from file names and marks only the
# newest file of each family with `is_latest=True` (files without a version are always latest):

# %%
for d in load():
    if d.metadata["source_file"].startswith("cancellation"):
        print(d.metadata["source_file"], {k: d.metadata[k] for k in ("doc_version", "is_latest")})

# %% [markdown]
# Ask the cancellation question with and without a metadata filter. We use the **basic** prompt
# here, which has no rule about versions, to see what retrieval alone does:

# %%
question = "How many hours before arrival can I cancel for free?"
for label, kwargs in [("no filter", {"k": 4}), ("is_latest only", {"k": 4, "filter": {"is_latest": True}})]:
    retriever = store.as_retriever(search_kwargs=kwargs)
    sources = [d.metadata["source_file"] for d in retriever.invoke(question)]
    answer = chains.build_rag_chain(retriever, prompt=chains.BASIC_PROMPT).invoke(question)
    print(f"[{label}] sources: {sources}\n  answer: {answer}\n")

# %% [markdown]
# Without the filter, both policies reach the prompt and the model has to guess which one applies.
# With the filter, the outdated policy can't be retrieved at all. The grounded prompt's "prefer the
# most recent passage" rule is a useful second line of defence, but a filter is **deterministic**.
#
# What happens when a 2026 policy arrives? The 2025 file's text doesn't change, but its `is_latest`
# tag does, so `sync_index` re-writes its chunks with the new metadata (embeddings from the cache):

# %%
policy_2025 = (corpus / "cancellation_policy_2025.md").read_text(encoding="utf-8")
policy_2026 = policy_2025.replace("2025", "2026").replace("72 hours", "96 hours")
(corpus / "cancellation_policy_2026.md").write_text(policy_2026, encoding="utf-8")

print(indexing.sync_index(store, load(), manifest, chunker=chunker))
latest = store.as_retriever(search_kwargs={"k": 2, "filter": {"is_latest": True}}).invoke(question)
print([d.metadata["source_file"] for d in latest])

# %% [markdown]
# **1 added** (the 2026 file) and **1 updated** (the 2025 file, now `is_latest=False`). The
# filtered retriever now returns the 2026 policy.
#
# ## 6. The framework version: LangChain's `index()`
#
# LangChain ships the same idea as `langchain_core.indexing.index` with a **record manager** that
# remembers what was indexed. `cleanup="incremental"` deletes the old chunks of any source that
# changed. Production setups use a SQL-backed record manager; the in-memory one is enough to see it work:

# %%
from langchain_core.indexing import InMemoryRecordManager, index
from langchain_core.vectorstores import InMemoryVectorStore

record_manager = InMemoryRecordManager(namespace="nb09")
record_manager.create_schema()
memory_store = InMemoryVectorStore(llm.CachedEmbeddings())
chunks = chunker(load())

options = {"cleanup": "incremental", "source_id_key": "source_file", "key_encoder": "sha256"}
print("first run: ", index(chunks, record_manager, memory_store, **options))
print("second run:", index(chunks, record_manager, memory_store, **options))

# %% [markdown]
# The second run skips everything. Our `sync_index` does the same job in about 60 lines you can read:
# use whichever you prefer, but **never** "delete and rebuild" in production.
#
# ## 7. Optional: Postgres + pgvector
#
# Chroma is ideal on one machine. In production you often want vectors **in your main database**:
# backups, access control, transactions and SQL joins come for free. `pgvector` adds vector columns
# and nearest-neighbour indexes to Postgres, and Neon's free tier includes it. Set `DATABASE_URL`
# and install the `[prod]` extra to run this section; otherwise it explains how and skips.

# %%
from raglab import store_pg

if store_pg.pg_available():
    pg_store = store_pg.create_pg_store(chunks, "nb09_demo")
    for d in pg_store.similarity_search(question, k=2, filter={"is_latest": True}):
        print(d.metadata["source_file"], "|", d.page_content[:80].replace("\n", " "))
else:
    print("Skipping: no Postgres configured.\n")
    print(store_pg.SETUP_HELP)

# %% [markdown]
# `store_pg.get_store()` returns whichever store `RAGLAB_STORE` in `.env` names (`chroma` or
# `pgvector`), so the service in notebook 11 runs on either without code changes.
#
# ## Your turn
#
# 1. Change **two** facts in two different files of the working copy and sync. Predict the report
#    before you run it.
# 2. Lower the `dedupe` threshold to 0.90. Does the reworded opening get dropped now? Compute its
#    similarity to the original's opening, and decide whether you'd want it dropped.
# 3. Rename a file in the working copy (same content, new name). What does `sync_index` report, and why?
#
# ## Key takeaways
#
# - Index incrementally: hash files, give chunks deterministic ids, and only embed what changed.
# - Deletions must reach the index too, or removed content keeps answering questions.
# - De-duplicate before indexing, and look at what you drop.
# - Tag versions in metadata and filter on `is_latest`: deterministic beats "the prompt will sort it out".
#
# **Added to raglab:** `indexing.sync_index / SyncReport / dedupe / tag_versions`,
# `store_pg.pg_available / create_pg_store / load_pg_store / get_store`, `store_chroma.open_lc_store`,
# `bootstrap.profile_chunker`, the profile `nb09_incremental`, and `loaders.load_corpus_lc(corpus_dir=...)`.

# %%
cost.report()
