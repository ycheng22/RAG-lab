# %% [markdown]
# # 01 · RAG from scratch
#
# Before using any framework, we'll build the whole RAG loop by hand in about 50 lines: embed
# some passages, find the closest ones with a bit of numpy, and paste them into a prompt.
#
# **In this notebook you will:**
# - see the model fail on a question it can't know,
# - compute cosine similarity yourself and retrieve the top passages,
# - build the augmented prompt and watch the same question get answered correctly,
# - see what happens when the answer isn't in the passages.

# %%
# <<SETUP>>

# %%
# Setup 2/3: nothing to prepare. This notebook uses 8 short passages written below.

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. Without retrieval, the model guesses
#
# Token Hotels & Resorts is fictional, so the model has never seen its documents.

# %%
from raglab import llm

question = "What time is check-in at Token Kyoto?"
print(llm.chat(question))

# %% [markdown]
# Whatever it says, it can't be grounded in Token's real policy. Now let's give it the facts.
#
# ## 2. A tiny knowledge base
#
# Real documents are long; we'll get to loading and chunking them in notebook 02. For now,
# here are eight short passages, the kind of chunks a real pipeline would produce.

# %%
passages = [
    "Check-in at Token Lisbon starts at 3:00 PM and check-out is at 11:00 AM. Guests can use a heated rooftop pool.",
    "Check-in at Token Kyoto starts at 4:00 PM and check-out is at 11:00 AM. The hotel has a private onsen-style spa.",
    "Token Vancouver has 210 rooms, an indoor saltwater pool and a full-service spa. Check-in starts at 3:30 PM.",
    "Hanare at Token Kyoto serves kaiseki. Reservations are required at least 3 days ahead.",
    "Token Singapore is the group's flagship tower on Marina Bay, with a 60-metre infinity pool on the 52nd floor.",
    "Wi-Fi is free for every guest at every Token Hotels & Resorts property.",
    "Token Cusco sits at 3,400 metres. Pickup from Cusco airport is free, and coca tea is served on arrival.",
    "Token Circle members earn 10 points for every dollar spent; 10,000 points buy one free night.",
]
len(passages)

# %% [markdown]
# ## 3. Embed everything once
#
# Each passage becomes a vector. We stack them into a matrix with one row per passage.

# %%
import numpy as np

passage_vecs = np.array(llm.embed(passages))
print(passage_vecs.shape)  # (number of passages, dimensions)

# %% [markdown]
# ## 4. Cosine similarity, by hand
#
# Two vectors are similar when they point in the same direction. **Cosine similarity** is the
# cosine of the angle between them:
#
# $$\text{cos}(q, d) = \frac{q \cdot d}{\lVert q \rVert \, \lVert d \rVert}$$
#
# If we first scale every vector to length 1, it's just a dot product, and one matrix
# multiplication scores the question against every passage at once.

# %%
question_vec = np.array(llm.embed_one(question))

q = question_vec / np.linalg.norm(question_vec)
m = passage_vecs / np.linalg.norm(passage_vecs, axis=1, keepdims=True)
scores = m @ q

import pandas as pd

table = pd.DataFrame({"score": scores.round(3), "passage": [p[:80] + "…" for p in passages]})
table.sort_values("score", ascending=False)

# %% [markdown]
# The Kyoto check-in passage wins clearly. Next come the Lisbon check-in passage (same *topic*,
# different hotel) and the Kyoto restaurant passage (same *hotel*, different topic). Passages about
# Wi-Fi, loyalty points or Singapore score lowest.
#
# ## 5. Retrieve the top 3
#
# `raglab.scratch` wraps the same maths in two small functions, `cosine_sim` and `top_k`:

# %%
from raglab import scratch

hits = scratch.top_k(question_vec, passage_vecs, k=3)
for index, score in hits:
    print(f"{score:.3f}  {passages[index]}")

# %% [markdown]
# ## 6. Augment the prompt and generate
#
# "Augmented generation" just means pasting the retrieved passages into the prompt and telling
# the model to stick to them. Here is the full prompt the model will see:

# %%
contexts = [passages[i] for i, _ in hits]
prompt = scratch.build_prompt(question, contexts)
print(prompt)

# %%
print(llm.chat(prompt))

# %% [markdown]
# Same model, same question, now a correct, grounded answer. **That's all RAG is.**
#
# Here's the flow we just ran, step by step:

# %%
from raglab.display import show_diagram

show_diagram("scratch_flow")

# %% [markdown]
# `scratch.answer()` runs the whole loop in one call:

# %%
result = scratch.answer("Is Wi-Fi free?", passages, k=3)
print(result["answer"])
print([round(s, 3) for s in result["scores"]])

# %% [markdown]
# ## 7. When the answer isn't there
#
# Retrieval always returns *something*: the top 3 passages exist even when none of them is
# relevant. A good prompt tells the model to say so instead of guessing.

# %%
result = scratch.answer("Can I bring my dog to Token Lisbon?", passages, k=3)
print(result["answer"])
print("best score:", round(result["scores"][0], 3))

# %% [markdown]
# Notice the best score: lower than for the questions that *were* answerable. Similarity scores
# are a useful signal, but there's no universal cut-off; we'll measure retrieval properly in notebook 05.
#
# ## Your turn
#
# 1. Ask "Which hotels have a pool?" with `k=1` and then `k=4`. Which answer is more complete, and why?
# 2. Add a *distractor* passage such as `"Token Kyoto's sister hotel opens its doors at 2 PM."`
#    and ask the check-in question again. Does retrieval still pick the right passage? Does the answer change?
# 3. Remove the "say I don't know" sentence from `scratch.PROMPT_TEMPLATE` (copy it into a new
#    string first) and ask about dogs again.
#
# <details><summary>What to expect</summary>
#
# 1. With `k=1` only one hotel's passage reaches the model, so it can name at most one pool.
#    Questions that need several facts need a larger `k` (or a smarter retriever, notebook 06).
# 2. Similar-sounding distractors can outrank the true passage. This is why real systems
#    evaluate retrieval (notebook 05) and why wording in documents matters.
# 3. Without the instruction, models are more likely to guess, the classic hallucination.
# </details>
#
# ## Key takeaways
#
# - Retrieval = embed the question, rank passages by cosine similarity, keep the top k.
# - Generation = put those passages in the prompt and instruct the model to use only them.
# - Retrieval always returns something; the prompt must handle "not in the passages".
#
# **Added to raglab:** `scratch.cosine_sim`, `scratch.top_k`, `scratch.build_prompt`, `scratch.answer`.

# %%
cost.report()
