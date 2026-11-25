"""RAG from scratch (notebook 01): no framework, just embeddings, numpy and one chat call."""

from __future__ import annotations

import numpy as np

from raglab import llm

PROMPT_TEMPLATE = """You are a concierge for Token Hotels & Resorts.
Answer the question using only the numbered passages below.
If the passages do not contain the answer, say "I don't know based on these passages."

Passages:
{context}

Question: {question}
Answer:"""


def cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of `matrix` (one score per row)."""
    q = np.asarray(query, dtype=float)
    m = np.asarray(matrix, dtype=float)
    q = q / np.linalg.norm(q)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return m @ q


def top_k(query_vec, doc_vecs, k: int = 3) -> list[tuple[int, float]]:
    """Indexes and scores of the k rows most similar to the query, best first."""
    scores = cosine_sim(np.asarray(query_vec), np.asarray(doc_vecs))
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]


def build_prompt(question: str, contexts: list[str]) -> str:
    """Number the passages and drop them into the prompt template."""
    context = "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer(question: str, passages: list[str], k: int = 3) -> dict:
    """The whole RAG loop: embed, retrieve the top-k passages, build the prompt, generate."""
    passage_vecs = llm.embed(passages)
    question_vec = llm.embed_one(question)
    hits = top_k(question_vec, passage_vecs, k=k)
    contexts = [passages[i] for i, _ in hits]
    reply = llm.chat(build_prompt(question, contexts))
    return {"answer": reply, "contexts": contexts, "scores": [score for _, score in hits]}
