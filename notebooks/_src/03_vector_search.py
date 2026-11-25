# %% [markdown]
# # 03 · Vector search for real
#
# In notebook 01 we kept eight vectors in a numpy array and compared the question against all of
# them. That doesn't scale to millions of chunks, and it forgets everything when the kernel stops.
# A **vector database** stores chunks with their vectors and metadata on disk, and finds nearest
# neighbours quickly. We'll use **Chroma**, which runs locally with no server.
#
# **In this notebook you will:**
# - see what a stored record looks like (id, vector, text, metadata),
# - query by meaning and read the distances,
# - filter by metadata ("only hotels in Asia"),
# - compare plain top-k with **MMR**, which trades a little relevance for variety,
# - understand how every notebook gets its index without re-running earlier notebooks.

# %%
# <<SETUP>>

# %%
# Setup 2/3: build (or reuse) this notebook's index: the whole corpus, chunked as in notebook 02
from raglab.bootstrap import ensure_raw_collection

collection = ensure_raw_collection("nb03_raw")

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. What's in a collection?
#
# A **collection** is like a table. Each record has an `id`, an `embedding` (the vector), a
# `document` (the chunk text) and `metadata` (whatever we attached in notebook 02).

# %%
print(f"{collection.count()} records")
record = collection.get(limit=1, offset=40, include=["documents", "metadatas", "embeddings"])
print("id:       ", record["ids"][0])
print("metadata: ", record["metadatas"][0])
print("document: ", record["documents"][0][:200].replace("\n", " "), "…")
print("embedding:", len(record["embeddings"][0]), "numbers, starting", [round(float(x), 3) for x in record["embeddings"][0][:4]])

# %% [markdown]
# The id is a hash of the file name and chunk text (`store_chroma.chunk_id`), so the same chunk
# always gets the same id. That will matter in notebook 09, when we update the index instead of
# rebuilding it.

# %%
from raglab.display import show_diagram

show_diagram("vector_search")

# %% [markdown]
# ## 2. Query by meaning
#
# `store_chroma.query` embeds the question and asks Chroma for the k nearest records. Chroma
# reports a **distance**: here `1 - cosine similarity`, so **smaller is closer**.

# %%
from raglab import store_chroma


def show_hits(hits):
    for doc, distance in hits:
        meta = doc.metadata
        label = meta.get("hotel_name") or "-"
        print(f"{distance:.3f}  {meta['source_file']:<22}{label:<18}{doc.page_content[:70].replace(chr(10), ' ')}")


show_hits(store_chroma.query(collection, "What time is check-in at Token Kyoto?", k=4))

# %% [markdown]
# The right chunk comes first. Compare the distances of a question that's related to our
# documents with one that isn't:

# %%
for q in ["Which restaurant serves Moroccan food?", "How do I change a bicycle tyre?"]:
    best = store_chroma.query(collection, q, k=1)[0][1]
    print(f"{best:.3f}  {q}")

# %% [markdown]
# The unrelated question's *best* match is much further away. Distance is a useful signal for
# "we probably don't have this", but the right threshold depends on the embedding model and the
# corpus, so measure it before relying on it.
#
# Notice also how many results above come from `room_rates.csv`. Each of its 48 rows is a chunk
# that names a hotel, so rate rows crowd into results for any hotel-specific question. Structured
# data like this often deserves its own tool rather than the vector index (notebook 12).

# %% [markdown]
# ## 3. Filter by metadata
#
# Because we attached metadata in notebook 02, we can restrict the search. Chroma applies the
# filter during the search, so you still get k results that all match it.

# %%
show_hits(store_chroma.query(collection, "a hotel with a spa", k=4, where={"continent": "Asia"}))

# %% [markdown]
# Filters can combine conditions with `$and` / `$or` and compare numbers with `$lt`, `$gte`, and so on:

# %%
cheap_with_pool = {"$and": [{"has_pool": True}, {"price_per_night": {"$lt": 300}}]}
show_hits(store_chroma.query(collection, "swimming pool", k=3, where=cheap_with_pool))

# %% [markdown]
# A filter is only as good as the metadata behind it. Chunks that `enrich_metadata` couldn't assign
# to a single hotel (for example a PDF chunk spanning two hotels) have no `continent` or
# `has_pool`, so a filter silently excludes them.

# %% [markdown]
# ## 4. Persistence
#
# The collection lives in `.cache/chroma` on disk. A brand-new client sees the same data, with no
# re-embedding needed:

# %%
fresh_client = store_chroma.get_client()
print(fresh_client.get_collection("nb03_raw").count(), "records found by a new client")

# %% [markdown]
# ## 5. Top-k vs MMR: relevance vs variety
#
# Plain top-k returns the k *most similar* chunks, and they're often near-duplicates of each other.
# For a broad question ("which hotels have a spa?") you want **coverage**: several different hotels,
# not three slices of the same one.
#
# **Maximal Marginal Relevance (MMR)** picks results one at a time. Each pick maximises
#
# `λ · similarity(question, chunk) − (1 − λ) · max similarity(chunk, already picked)`
#
# so a chunk that repeats what we already have gets penalised. `λ = 1` is plain top-k.

# %%
def sources(docs):
    return [d.metadata["source_file"] for d in docs]


question = "What is Token Marrakech like?"
plain = [d for d, _ in store_chroma.query(collection, question, k=5)]
mmr = store_chroma.query_mmr(collection, question, k=5, fetch_k=25, lambda_mult=0.5)
print("top-k:", sources(plain))
print("MMR:  ", sources(mmr))

# %% [markdown]
# Plain top-k spends three of its five slots on near-identical rows of `room_rates.csv`. MMR keeps
# the best match, then swaps the repeats for *different* documents (dining, reviews, FAQ), which is a
# better basis for a "what is it like?" answer.
#
# MMR isn't always better. When the question really is about those rows, repetition is what you want:

# %%
question = "How much does a room at Token Singapore cost?"
print("top-k:", sources([d for d, _ in store_chroma.query(collection, question, k=5)]))
print("MMR:  ", sources(store_chroma.query_mmr(collection, question, k=5, fetch_k=25, lambda_mult=0.5)))

# %% [markdown]
# Here MMR throws away rate rows (different room types and seasons) in favour of loosely related
# chunks. Use MMR for broad, exploratory questions and plain top-k for precise lookups, or better,
# *measure* which works for your questions (notebook 05).

# %% [markdown]
# ## 6. How every notebook gets its index
#
# Setup cell 2 called `ensure_raw_collection("nb03_raw")`. It:
#
# 1. computes a **fingerprint**: a hash of the profile's settings (chunk size, overlap…) and of every corpus file,
# 2. reuses the collection if its stored fingerprint matches,
# 3. otherwise rebuilds it. Thanks to the embedding cache, a rebuild only pays for chunks it has never seen.
#
# That's why you can open notebook 07 without running notebooks 03–06 first. Each notebook asks
# for the index it needs by name, and names never change meaning.

# %%
from raglab import bootstrap

print("fingerprint:", bootstrap.corpus_fingerprint("nb03_raw"))
print("stored:     ", collection.metadata.get("fingerprint"))

# %% [markdown]
# ## Your turn
#
# 1. Query `"late checkout"` with `k=5`. Which documents answer it? Do any results look like noise?
# 2. Find all 5-star hotels with a pool using only a `where` filter (any query text will do).
#    Which hotels do you get, and are any missing? Why might they be?
# 3. Run the Marrakech MMR comparison with `lambda_mult=0.2` and `0.9`. What changes?
#
# <details><summary>Hints</summary>
#
# 2. `where={"$and": [{"star_rating": 5}, {"has_pool": True}]}`. Expect Token Marrakech and Token
#    Singapore. Hotels can only appear if at least one of their chunks carries the metadata; see section 3.
# 3. Lower λ → more variety (and more off-topic chunks); higher λ → closer to plain top-k.
# </details>
#
# ## Key takeaways
#
# - A vector store keeps id + vector + text + metadata and answers "nearest neighbours" fast.
# - Distances tell you how close a match is; metadata filters narrow the search.
# - MMR balances relevance and diversity, which helps broad questions.
# - Named, fingerprinted index profiles make every notebook standalone.
#
# **Added to raglab:** `store_chroma.get_client / create_collection / query / query_mmr / chunk_id`,
# `bootstrap.IndexProfile / PROFILES / corpus_fingerprint / build_chunks / ensure_raw_collection`.

# %%
cost.report()
