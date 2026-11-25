# %% [markdown]
# # 00 · Setup and costs
#
# Welcome to **RAG-lab**, a hands-on series that takes you from *"what is RAG?"* to a production
# RAG service and an agent that decides when to search.
#
# **In this notebook you will:**
# - check that your OpenAI key works,
# - make your first chat call and your first embedding call,
# - see what an embedding actually is (spoiler: a long list of numbers),
# - learn how the series is organised and how much it costs to run.
#
# > Every notebook in this series is self-contained: the first three cells set everything up,
# > so you can open any notebook and run it top to bottom.

# %%
# <<SETUP>>

# %%
# Setup 2/3: nothing to prepare for this notebook (later notebooks build or load an index here).

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## What is RAG?
#
# A language model only knows what was in its training data. Ask it about *your* documents
# (a company handbook, a product manual, a hotel group's policies) and it either refuses or,
# worse, invents a confident answer.
#
# **Retrieval-Augmented Generation (RAG)** fixes this in two stages:
#
# 1. **Indexing (once):** split your documents into small chunks and turn each chunk into a
#    vector (an *embedding*) that captures its meaning. Store the vectors.
# 2. **Querying (every question):** embed the question the same way, find the chunks whose vectors
#    are closest, and paste those chunks into the prompt. The model answers from them.

# %%
from raglab.display import show_diagram

show_diagram("rag_overview")

# %% [markdown]
# ## The documents we'll use
#
# All notebooks use the documents of **Token Hotels & Resorts**, a fictional hotel group with 8
# hotels on six continents. Because the company is made up, the model can't know these facts from
# its training data, so any correct answer *must* come from retrieval. That makes it a fair test.
#
# The corpus lives in `data/corpus/` and has deliberately varied formats and problems:

# %%
for path in sorted(settings.corpus_dir.iterdir()):
    print(f"{path.name:<32}{path.stat().st_size:>8,} bytes")

# %% [markdown]
# ## Your first chat call
#
# `raglab.llm` is the only module that talks to OpenAI. `chat()` sends one request and returns the text.

# %%
from raglab import llm

print(llm.chat("In one sentence, what is retrieval-augmented generation?"))

# %% [markdown]
# Now ask about something the model can't know:

# %%
print(llm.chat("What time is check-in at the Token Kyoto hotel?"))

# %% [markdown]
# Depending on the model and the day, you'll see a refusal ("I don't have information about…") or
# a confident guess, sometimes about a *real* hotel with a similar name. Token Kyoto's actual check-in
# time (from its documents) is 4:00 PM. Without the documents the model can't know that; the rest of
# the series fixes this.

# %% [markdown]
# ## Your first embedding
#
# An **embedding** turns text into a vector of numbers. Texts with similar meanings get vectors
# that point in similar directions. That is all retrieval needs.

# %%
vector = llm.embed_one("Check-in starts at 4:00 PM.")
print(f"dimensions: {len(vector)}")
print(f"first five values: {[round(v, 4) for v in vector[:5]]}")

# %% [markdown]
# Let's compare a few sentences. We'll use **cosine similarity**: 1.0 means "same direction",
# around 0 means unrelated. (Notebook 01 explains the maths.)

# %%
import numpy as np

sentences = [
    "Check-in starts at 4:00 PM.",
    "You can get your room key from four in the afternoon.",
    "The restaurant serves a seven-course tasting menu.",
]
vectors = np.array(llm.embed(sentences))
unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
similarity = unit @ unit.T
for i, s in enumerate(sentences):
    print(f"{s:<58} " + "  ".join(f"{similarity[i, j]:.2f}" for j in range(len(sentences))))

# %% [markdown]
# The first two sentences share almost no words, yet their similarity is much higher than either
# one's similarity to the restaurant sentence. Embeddings capture *meaning*, not keywords.

# %% [markdown]
# ## Embeddings are cached
#
# Every embedding is saved in `.cache/embeddings.sqlite`, keyed by the model and the exact text.
# Re-running a notebook doesn't pay for the same embedding twice. Watch the call count:

# %%
before = cost.calls()
llm.embed(sentences)  # same texts as above
print(f"API calls made by that line: {cost.calls() - before}")

# %% [markdown]
# ## How much does this series cost?
#
# The defaults are chosen to be cheap:
#
# | Model | Used for | Price per 1M tokens (in / out) |
# |---|---|---|
# | `text-embedding-3-small` | embeddings | $0.02 |
# | `gpt-6-luna` | chat (answers, judges) | $0.10 / $0.50 |
#
# A default run of any notebook costs **about $0.05 or less**. `llm.estimate_cost()` lets you
# check before you run something big:

# %%
corpus_chars = sum(p.stat().st_size for p in settings.corpus_dir.iterdir())
corpus_tokens = corpus_chars // 4  # rough rule: ~4 characters per token in English
print(f"Embedding the whole corpus once: ~${llm.estimate_cost('text-embedding-3-small', 1, corpus_tokens, 0):.5f}")
print(f"1,000 short questions answered: ~${llm.estimate_cost(settings.chat_model, 1000, 1500, 150):.2f}")

# %% [markdown]
# ## How the series is organised
#
# 13 notebooks in three parts. Each notebook adds a few functions to the shared **`raglab`**
# package (shown under each title), and later notebooks import them.

# %%
show_diagram("series_map")

# %% [markdown]
# **One rule keeps this safe:** once a notebook adds a function to `raglab`, later notebooks never
# change what it does. Improvements arrive as *new* functions or *optional* arguments. So notebook
# 04 behaves the same whether or not you've run notebook 09.
#
# ## Your turn
#
# 1. Ask `llm.chat()` another question about Token Hotels (for example, "Does Token Cusco have a
#    pool?"). Does the model admit it doesn't know?
# 2. Add a fourth sentence to the similarity example that should be close to the restaurant
#    sentence. Check your intuition against the numbers.
# 3. Use `llm.estimate_cost()` to estimate what 10,000 questions would cost with
#    `gpt-5.4-mini` instead of `gpt-6-luna`.
#
# <details><summary>Solution for 3</summary>
#
# ```python
# llm.estimate_cost("gpt-5.4-mini", 10_000, 1500, 150)  # about $18.00, vs about $2.25 for gpt-6-luna
# ```
# </details>
#
# ## Key takeaways
#
# - RAG = retrieve relevant text, then let the model answer from it.
# - Embeddings map meaning to vectors; similar meanings → similar vectors.
# - The series runs on cheap models, and every notebook reports what it spent.
#
# **Added to raglab:** `config.get_settings`, `llm.chat`, `llm.embed`, `llm.embed_one`, `llm.cost`,
# `llm.estimate_cost`, `llm.check_budget`, `display.show_chunks`, `display.show_diagram`.

# %%
cost.report()
