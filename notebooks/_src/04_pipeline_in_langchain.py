# %% [markdown]
# # 04 · The same pipeline, in LangChain
#
# Part 1 built every piece by hand. From here on we use **LangChain**, the most widely used
# framework for LLM applications. You now know what each LangChain class does underneath, so none
# of it should feel like magic.
#
# **In this notebook you will:**
# - map each hand-written step from notebooks 01–03 to its LangChain equivalent,
# - build the full index of the Token Hotels corpus with LangChain loaders and splitters,
# - compose a RAG chain with **LCEL** (LangChain Expression Language) and run it stage by stage,
# - harden the prompt so the model refuses instead of guessing,
# - return the retrieved sources along with the answer (needed for evaluation next).

# %%
# <<SETUP>>

# %%
# Setup 2/3: build (or reuse) the baseline index used by notebooks 04-08
from raglab.bootstrap import ensure_index

store = ensure_index("nb04_baseline")

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. Hand-written → LangChain
#
# | Step | Part 1 (by hand) | LangChain |
# |---|---|---|
# | Document type | `Document` (borrowed) | `langchain_core.documents.Document` |
# | Load PDF / Markdown / HTML / CSV | `loaders.load_pdf` … | `PyPDFLoader`, `TextLoader`, `BSHTMLLoader`, `CSVLoader` |
# | Recursive chunking | `chunking.chunk_recursive` | `RecursiveCharacterTextSplitter` |
# | Header-aware chunking | `chunking.chunk_markdown_headers` | `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter` |
# | Embeddings | `llm.embed` | an `Embeddings` class (ours wraps `llm.embed`, so the cache still works) |
# | Vector store | raw `chromadb` | `langchain_chroma.Chroma` |
# | Top-k search | `store_chroma.query` | `store.as_retriever(search_kwargs={"k": 4})` |
# | Prompt | f-string in `scratch.build_prompt` | `ChatPromptTemplate` |
# | Chat call | `llm.chat` | `ChatOpenAI` (ours: `llm.get_lc_chat()`) |
# | The loop | `scratch.answer` | an **LCEL chain** |
#
# `raglab.chunking.chunk_corpus_lc` picks a splitter per file type: header-aware for Markdown and HTML,
# recursive for PDF and CSV. The setup cell above used it to build the index.

# %%
from collections import Counter

from raglab.bootstrap import get_chunks

chunks = get_chunks("nb04_baseline")
print(f"{len(chunks)} chunks")
Counter(c.metadata["source_file"] for c in chunks)

# %% [markdown]
# Notice the HTML FAQ: `HTMLHeaderTextSplitter` cut it at every `<h2>` and recorded the heading
# as `section`, something our hand-written loader in notebook 02 couldn't do:

# %%
from raglab.display import show_chunks

faq_chunks = [c for c in chunks if c.metadata["source_file"] == "faq.html"]
show_chunks([c for c in faq_chunks if "section" in c.metadata], n=2, width=200)

# %% [markdown]
# But look at the very first FAQ chunk. The navigation bar became a chunk of its own, and the
# splitter can't know it's boilerplate:

# %%
show_chunks(faq_chunks[:1])

# %% [markdown]
# ## 2. A retriever
#
# A **retriever** is anything that takes a question and returns Documents. The simplest one comes
# straight from the vector store:

# %%
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
docs = retriever.invoke("What time is check-in at Token Kyoto?")
show_chunks(docs, n=4, width=120)

# %% [markdown]
# ## 3. The chain, one stage at a time
#
# LCEL chains are built with the `|` operator: the output of one step is the input of the next.
# Here is the RAG chain from `raglab.chains.build_rag_chain`:
#
# ```python
# chain = (
#     {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# ```

# %%
from raglab.display import show_diagram

show_diagram("lcel_chain")

# %% [markdown]
# Let's run each stage by hand to see what flows between them.
#
# **Stage 1, the dict step:** runs the retriever and the passthrough *in parallel* on the same input.

# %%
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from raglab import chains

question = "What time is check-in at Token Kyoto?"
step1 = {"context": retriever | RunnableLambda(chains.format_docs), "question": RunnablePassthrough()}
from langchain_core.runnables import RunnableParallel

inputs = RunnableParallel(step1).invoke(question)
print("keys:", list(inputs))
print(inputs["context"][:500], "…")

# %% [markdown]
# **Stage 2, the prompt:** fills `{context}` and `{question}` and produces chat messages.

# %%
messages = chains.GROUNDED_PROMPT.invoke(inputs)
for m in messages.to_messages():
    print(f"--- {m.type} ---\n{m.content[:400]}\n")

# %% [markdown]
# **Stages 3 and 4, the model and the parser:** the model returns a message object, and the parser
# pulls out its text.

# %%
from langchain_core.output_parsers import StrOutputParser

from raglab.llm import get_lc_chat

llm = get_lc_chat()
message = llm.invoke(messages)
print(type(message).__name__, "→", StrOutputParser().invoke(message))

# %% [markdown]
# And the whole chain in one call:

# %%
rag = chains.build_rag_chain(retriever)
rag.invoke(question)

# %% [markdown]
# ## 4. Prompts matter: basic vs grounded
#
# `chains.BASIC_PROMPT` just says "use the context". `chains.GROUNDED_PROMPT` adds a role, forbids
# outside knowledge, tells the model to prefer the most recent document when two disagree, and
# gives it an **exact sentence** to use when the answer isn't there.

# %%
print(chains.GROUNDED_PROMPT.messages[0].prompt.template)

# %%
basic = chains.build_rag_chain(retriever, chains.BASIC_PROMPT)
grounded = chains.build_rag_chain(retriever, chains.GROUNDED_PROMPT)

for q in [
    "How many hours before arrival can I cancel for free?",  # two policy versions disagree
    "Does Token Kyoto have a gym?",  # not in the documents
    "Who is the general manager of Token Lisbon?",  # not in the documents
]:
    print(f"Q: {q}\n  basic:    {basic.invoke(q)}\n  grounded: {grounded.invoke(q)}\n")

# %% [markdown]
# Read the answers closely (the run saved in this notebook is shown above; yours may word things differently):
#
# - **Refusals.** Modern models often refuse sensibly even with a basic prompt, but in their own
#   words. The grounded prompt makes the refusal an *exact, known sentence* (`chains.NOT_FOUND`),
#   so code can detect it: show a fallback, log the gap, and count refusals in evaluation.
# - **Conflicting versions.** Whether the answer is 48 hours (2024 policy) or 72 hours (2025 policy)
#   depends on which policy chunks were retrieved, and on whether the prompt says how to resolve
#   conflicts. Notebook 09 fixes this properly with version metadata.
#
# ## 5. Return the sources too
#
# For evaluation (next notebook) and citations (notebook 10) we need the retrieved chunks *as well
# as* the answer. `build_rag_chain_with_sources` retrieves once and returns both:

# %%
rag_with_sources = chains.build_rag_chain_with_sources(retriever)
result = rag_with_sources.invoke("Can I bring my dog to Token Lisbon?")
print(result["answer"])
print("\nsources:", [d.metadata["source_file"] for d in result["contexts"]])

# %% [markdown]
# ## Your turn
#
# 1. Write a prompt that answers in bullet points and ends every bullet with its source in
#    brackets, e.g. `[faq.html]`. Build a chain with it and ask about pets and parking.
# 2. Change the retriever to `k=1`. Which of the questions above break?
# 3. Use `search_type="mmr"` in `as_retriever` (LangChain's built-in MMR) and compare the sources
#    returned for "What is Token Marrakech like?" with plain similarity.
#
# <details><summary>A starting point for 1</summary>
#
# ```python
# from langchain_core.prompts import ChatPromptTemplate
# bullets = ChatPromptTemplate.from_messages([
#     ("system", "Answer only from the numbered context passages, as short bullet points. "
#                "End each bullet with the source file of the passage it came from in [brackets]. "
#                f'If the answer is not there, reply exactly: "{chains.NOT_FOUND}"'),
#     ("human", "Context:\n{context}\n\nQuestion: {question}"),
# ])
# chains.build_rag_chain(retriever, bullets).invoke("Can I bring my dog, and is there parking?")
# ```
# </details>
#
# ## Key takeaways
#
# - LangChain packages the same steps you wrote by hand: loaders, splitters, vector store, retriever, prompt, model.
# - LCEL composes them with `|`; a dict step runs branches in parallel on the same input.
# - A grounded prompt with an exact refusal sentence makes behaviour predictable and measurable.
# - Return the contexts with the answer; evaluation and citations depend on them.
#
# **Added to raglab:** `llm.CachedEmbeddings / get_lc_chat`, `loaders.load_corpus_lc`,
# `chunking.chunk_corpus_lc`, `store_chroma.create_lc_store / load_lc_store`,
# `bootstrap.ensure_index / get_chunks`, `chains.NOT_FOUND / format_docs / BASIC_PROMPT /
# GROUNDED_PROMPT / build_rag_chain / build_rag_chain_with_sources`.

# %%
cost.report()
