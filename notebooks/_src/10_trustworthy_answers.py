# %% [markdown]
# # 10 · Answers you can trust
#
# A RAG system that answers well 90% of the time is still dangerous if you can't tell which 10% is
# wrong. This notebook adds the layers that make answers **checkable**:
#
# - **citations** a program can verify, not just a list of file names,
# - **refusals** a program can detect, through a structured `found` flag,
# - a defence against **prompt injection** hidden inside your own documents,
# - **streaming**, so users see the answer while it's being written.
#
# Cost: about $0.005.

# %%
# <<SETUP>>

# %%
# Setup 2/3: the baseline index and a retriever
from raglab import answering, chains, retrievers
from raglab import evaluation as ev
from raglab.bootstrap import ensure_index
from raglab.display import show_diagram

store = ensure_index("nb04_baseline")
retriever = retrievers.get_retriever("similarity", store, k=4)

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# Every layer in this notebook, from ingest to reply:

# %%
show_diagram("trust_layers")

# %% [markdown]
# ## 1. Structured answers with citations
#
# Instead of free text, we ask the model to fill in a schema (a Pydantic model), using LangChain's
# `with_structured_output`:
#
# ```python
# class Citation(BaseModel):
#     source_file: str   # which document
#     quote: str         # copied word for word
#
# class Answer(BaseModel):
#     answer: str
#     found: bool        # False = the documents don't contain the answer
#     citations: list[Citation]
# ```
#
# The prompt (`answering.SAFE_PROMPT`) wraps each retrieved chunk in
# `<document source="...">...</document>` tags and asks for verbatim quotes:

# %%
answer, docs = answering.answer_with_citations("What time is check-in at Token Kyoto?", retriever)
print("answer:", answer.answer)
print("found: ", answer.found)
for c in answer.citations:
    print(f"  [{c.source_file}] {c.quote!r}")

# %% [markdown]
# ## 2. Verify every citation
#
# A citation is only useful if it's real. Models sometimes paraphrase a "quote", attribute it to the
# wrong file, or invent it. `answering.verify_citations` checks each quote is really in a retrieved
# chunk from that file (ignoring case and spacing). It's plain string matching: free and instant.

# %%
print("verified:", answering.verify_citations(answer, docs))

# %% [markdown]
# Now a forged answer, built by hand: one invented quote, and one real quote attributed to the wrong file.

# %%
from raglab.answering import Answer, Citation

forged = Answer(
    answer="Check-in at Token Kyoto is at 4:00 PM, and late arrivals get a free upgrade.",
    found=True,
    citations=[
        Citation(source_file="hotel_profiles.pdf", quote="Check-in at Token Kyoto starts at 4:00 PM"),
        Citation(source_file="hotel_profiles.pdf", quote="late arrivals receive a free room upgrade"),
        Citation(source_file="faq.html", quote="Check-in at Token Kyoto starts at 4:00 PM"),
    ],
)
for c, ok in zip(forged.citations, answering.verify_citations(forged, docs)):
    print("OK  " if ok else "FAIL", f"[{c.source_file}] {c.quote}")

# %% [markdown]
# The invented quote and the wrong attribution both fail. In an app you'd drop unverified
# citations, or refuse to show the answer at all. How often do the model's real citations verify?

# %%
quick = ev.load_eval("quick")
checked = []
for item in quick:
    ans, ctx = answering.answer_with_citations(item["question"], retriever)
    flags = answering.verify_citations(ans, ctx)
    checked.append({"question": item["question"], "found": ans.found, "citations": len(flags), "verified": sum(flags)})

import pandas as pd

pd.DataFrame(checked)

# %% [markdown]
# In the run saved here every real citation verified: asked for verbatim quotes, a good model
# complies. The check exists for the times it doesn't, and it costs nothing to run on every answer.
#
# ## 3. Refusals a program can detect
#
# Notebook 08 showed that free-text refusals are hard to detect in code. The structured `found` flag
# fixes that. Let's compare on the 5 unanswerable questions: the grounded prompt from notebook 04
# (refusal = the exact sentence in the text) and the safe prompt (refusal = `found=False`).

# %%
unanswerable = ev.load_eval("full", categories={"unanswerable"})
grounded_chain = chains.build_rag_chain(retriever)
rows = []
for item in unanswerable:
    grounded = grounded_chain.invoke(item["question"])
    safe, _ = answering.answer_with_citations(item["question"], retriever)
    rows.append({
        "question": item["question"],
        "grounded refused": chains.NOT_FOUND in grounded,
        "safe found=False": not safe.found,
    })
refusals = pd.DataFrame(rows)
refusals

# %%
refusals[["grounded refused", "safe found=False"]].mean().rename("share refused")

# %% [markdown]
# Both prompts refuse here; the difference is **how you find out**. With `found` you don't depend on
# the model copying a sentence exactly, and you can branch on it: show a "contact the hotel" button,
# log the question as a gap in your documents, or let an agent search elsewhere (notebook 12).
#
# ## 4. Prompt injection: instructions hidden in your data
#
# Anything in your documents ends up in the prompt. If a document contains *instructions*, a model
# may follow them. One of our guest reviews contains a planted line (as a malicious review might):

# %%
for line in (store.get(where={"source_file": "guest_reviews.md"})["documents"]):
    if "NOTE TO AI" in line:
        print("...", line[line.index("NOTE TO AI"):][:120])

# %% [markdown]
# A guest asks the question that retrieves that review:

# %%
question = "Are rooms free tonight at Token Marrakech?"
retrieved = retriever.invoke(question)
print("injected chunk retrieved:", any("NOTE TO AI" in d.page_content for d in retrieved))
print()
print("BASIC prompt:", chains.build_rag_chain(retriever, prompt=chains.BASIC_PROMPT).invoke(question))
print()
safe, _ = answering.answer_with_citations(question, retriever)
print("SAFE prompt: ", safe.answer, f"(found={safe.found})")

# %% [markdown]
# In the run saved here the basic prompt was **not** fooled: current models resist an injection
# this crude. Don't take comfort from that. The injected instruction did reach the model, so the
# basic prompt depends entirely on the model's training, and subtler injections (fake policy text,
# "updated pricing") get through far more often than this one. The safe prompt adds three things:
# documents are tagged as data, the prompt says instructions inside them must be ignored, and the
# answer must cite a quote that really exists. Be clear about what that last check proves: that the
# quote is **in the documents**, not that it is true or official. A fooled model could quote the
# planted review line word for word and pass verification, which is why the injection has to be
# stopped earlier too.
#
# ### Screen at ingest time too
#
# The cheapest defence is not to index the attack at all. `answering.screen_injection` looks for
# phrases that have no business in hotel documents. It doesn't block anything: it flags documents
# for a human to review.

# %%
from raglab import loaders

# Screen the extracted text, not the raw file: a PDF's text is compressed inside the file.
texts = {}
for d in loaders.load_corpus_lc():
    name = d.metadata["source_file"]
    texts[name] = texts.get(name, "") + "\n" + d.page_content
for name, text in sorted(texts.items()):
    hits = answering.screen_injection(text)
    print(f"{name:32} {hits if hits else 'clean'}")

# %% [markdown]
# A regex list is easy to evade (another language, a paraphrase), which is exactly why it's only
# one layer. **Defence in depth:**
#
# | Layer | Stops | Misses |
# |---|---|---|
# | Ingest screen | Obvious planted instructions | Paraphrased or translated ones |
# | Tagged documents + safe prompt | Most attempts to take over the answer | A determined, subtle attack |
# | Structured answer + citation check | Claims that aren't quoted from a real document | A real quote used misleadingly |
# | Exact refusals / `found` flag | Guesses presented as answers | Wrong answers the model is sure of |
#
# No single layer is enough; together they make a wrong answer much less likely to reach a guest
# unnoticed.
#
# ## 5. Streaming
#
# A full answer can take a few seconds. Streaming sends tokens as they're generated, so users start
# reading almost immediately. `answering.stream_answer` yields text pieces:

# %%
import time

started = time.perf_counter()
first_token_at = None
pieces = []
for piece in answering.stream_answer("What can I do at Token Queenstown in winter?", retriever):
    if first_token_at is None and piece:
        first_token_at = time.perf_counter() - started
    pieces.append(piece)
    print(piece, end="", flush=True)
total = time.perf_counter() - started
print(f"\n\nfirst text after {first_token_at:.1f}s, complete after {total:.1f}s, {len(pieces)} pieces")

# %% [markdown]
# Streaming and structured output pull in different directions: you can't verify citations until
# the whole answer exists. A common pattern is to stream the text for the user and verify the
# citations when it's done, flagging the answer if any fail. Notebook 11's API offers both.
#
# ## Your turn
#
# 1. Add a pattern to `answering.INJECTION_PATTERNS` that catches "forget everything above". Test it
#    on a sentence of your own.
# 2. Write a review line that might fool the **basic** prompt (for example, a fake "updated policy"
#    that sounds official). Put it in a Document, pass it as context, and compare both prompts.
# 3. Make `verify_citations` stricter: require each quote to be at least 20 characters. Why might
#    very short quotes be a problem?
#
# ## Key takeaways
#
# - Ask for structured answers: a `found` flag and citations turn trust into something code can check.
# - Verify every quote against the retrieved text. It's free, and it catches invented citations.
# - Treat documents as untrusted input: tag them as data, tell the model to ignore instructions inside
#   them, and screen new documents at ingest time.
# - Layer the defences: each one misses something the others catch.
#
# **Added to raglab:** `answering.Answer / Citation / SAFE_PROMPT / answer_with_citations /
# verify_citations / screen_injection / INJECTION_PATTERNS / stream_answer / format_docs_tagged`.

# %%
cost.report()
