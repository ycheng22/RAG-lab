"""Tuning grids and failure triage (added in notebook 08).

- `run_grid` measures every combination of chunking, k, retriever and prompt on the eval set.
- `broken_system` is a deliberately faulty RAG system to practise diagnosis on.
- `triage` labels each question's outcome: ok, retrieval_failure, generation_failure or refusal_failure.
"""

from __future__ import annotations

import concurrent.futures
import itertools

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from raglab import bootstrap, chains, evaluation, llm, retrievers
from raglab.evaluation import EvalItem, normalise

# A prompt with three classic mistakes: it treats the context as optional, it forbids "I don't know",
# and it invites the model to fill gaps from general knowledge.
BROKEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly assistant for a hotel chain. Here is some information that may help:

{context}

Guest question: {question}

Always give the guest a specific, confident answer: guests hate hearing "I don't know". If the
information above is incomplete, use your general knowledge of hotels to fill the gaps."""
)

PROMPTS = {"basic": chains.BASIC_PROMPT, "grounded": chains.GROUNDED_PROMPT, "broken": BROKEN_PROMPT}

GRID_DEFAULTS = {"chunk_size": [800], "overlap": [150], "k": [4], "retriever": ["similarity"], "prompt": ["grounded"]}


def broken_system():
    """The "mystery system" of notebook 08: question (str) -> {"answer", "contexts"}."""
    store = bootstrap.ensure_index("nb08_broken")
    retriever = retrievers.get_retriever("similarity", store, k=1)
    return chains.build_rag_chain_with_sources(retriever, BROKEN_PROMPT)


class _MeasuringRetriever:
    """Wraps a retriever and remembers how much text it returned (a proxy for prompt cost)."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.chars: list[int] = []

    def invoke(self, question: str):
        docs = self.retriever.invoke(question)
        self.chars.append(sum(len(d.page_content) for d in docs))
        return docs


def run_grid(grid: dict[str, list], items: list[EvalItem], *, judge: bool = False) -> pd.DataFrame:
    """One row per combination of the grid's values.

    Grid keys: chunk_size, overlap, k, retriever, prompt ("basic" | "grounded"); missing keys use
    GRID_DEFAULTS. Retrieval metrics are always computed (no LLM calls). With judge=True every
    combination also answers each question and an LLM judge scores the answers, which costs money:
    the `prompt` key only matters then. Combinations where overlap >= chunk_size are skipped.

    Columns: the grid keys, evidence_recall, hit_rate, mrr, context_tokens (≈ chars / 4 of the
    retrieved text, i.e. what each answer would pay for), accuracy (judge only), latency_ms
    (retrieval) and cost_usd (API cost of this combination, excluding index building).
    """
    grid = {**GRID_DEFAULTS, **grid}
    prompts = grid["prompt"] if judge else [None]
    index_sizes = sorted({(cs, ov) for cs in grid["chunk_size"] for ov in grid["overlap"] if ov < cs})

    # Build (or reuse) every index first, so index cost isn't charged to the first combination.
    stores = {}
    for cs, ov in index_sizes:
        profile = bootstrap.make_profile(cs, ov)
        stores[(cs, ov)] = (bootstrap.ensure_index(profile), bootstrap.get_chunks(profile))

    rows = []
    for (cs, ov), k, mode, prompt in itertools.product(index_sizes, grid["k"], grid["retriever"], prompts):
        store, chunks = stores[(cs, ov)]
        before = llm.cost.total()
        retriever = retrievers.get_retriever(mode, store, k=k, chunks=chunks)
        measuring = _MeasuringRetriever(retriever)
        metrics = evaluation.retrieval_metrics(measuring, items)
        row = {
            "chunk_size": cs,
            "overlap": ov,
            "k": k,
            "retriever": mode,
            **({"prompt": prompt} if judge else {}),
            "evidence_recall": metrics["evidence_recall"].mean(),
            "hit_rate": metrics["hit"].mean(),
            "mrr": metrics["rr"].mean(),
            "context_tokens": round(sum(measuring.chars) / max(len(measuring.chars), 1) / 4),
            "latency_ms": metrics["latency_ms"].mean(),
        }
        if judge:
            chain = chains.build_rag_chain_with_sources(retriever, PROMPTS[prompt])
            row["accuracy"] = evaluation.score_answers(evaluation.run_rag(chain, items))["correct"].mean()
        row["cost_usd"] = round(llm.cost.total() - before, 5)
        rows.append(row)
    return pd.DataFrame(rows).round(3)


LABELS = ("ok", "retrieval_failure", "generation_failure", "refusal_failure")


def triage(rows: list[dict], items: list[EvalItem]) -> pd.DataFrame:
    """Label every row from `evaluation.run_rag`:

    - refusal_failure: an unanswerable question that got an answer instead of `chains.NOT_FOUND`
    - retrieval_failure: some evidence string is missing from the retrieved contexts
    - generation_failure: all evidence was retrieved, but the judge says the answer is wrong
    - ok: everything else

    Columns: id, category, question, label, evidence_found, correct, answer.
    The judge runs on every answerable row (one cheap LLM call each), so `correct` is always filled.
    """
    by_id = {i["id"]: i for i in items}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        verdicts = list(
            pool.map(lambda r: evaluation.judge_correct(r["question"], r["answer"], r["ground_truth"], r["category"]), rows)
        )
    out = []
    for row, correct in zip(rows, verdicts):
        item = by_id[row["id"]]
        refused = evaluation.is_not_found(row["answer"])
        joined = normalise(" ".join(row["contexts"]))
        evidence_found = all(normalise(e) in joined for e in item["evidence"])
        if item["category"] == "unanswerable":
            label = "ok" if refused else "refusal_failure"
        elif not evidence_found:
            label = "retrieval_failure"
        else:
            label = "ok" if correct else "generation_failure"
        out.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "label": label,
                "evidence_found": evidence_found,
                "correct": correct,
                "answer": row["answer"],
            }
        )
    return pd.DataFrame(out)

