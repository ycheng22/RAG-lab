"""Measure a RAG system (notebook 05): retrieval metrics by hand, an LLM judge, RAGAS, saved runs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import json
import time
import warnings
from collections.abc import Callable
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from raglab import llm
from raglab.chains import NOT_FOUND
from raglab.config import get_settings

warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm in notebooks without ipywidgets


class EvalItem(TypedDict):
    id: int
    category: str
    question: str
    ground_truth: str
    source_documents: list[str]
    evidence: list[str]
    hallucination_prone: bool
    quick: bool


def load_eval(subset: Literal["quick", "full"] = "quick", categories: set[str] | None = None) -> list[EvalItem]:
    """Load data/eval/questions.json. "quick" = the 5 flagged items; "full" = everything except
    `no_search_needed` (those exist for the agent notebook) unless you ask for that category."""
    path = get_settings().eval_dir / "questions.json"
    items: list[EvalItem] = json.loads(path.read_text(encoding="utf-8"))
    if subset == "quick":
        items = [i for i in items if i["quick"]]
    if categories is not None:
        items = [i for i in items if i["category"] in categories]
    elif subset == "full":
        items = [i for i in items if i["category"] != "no_search_needed"]
    return items


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace, so evidence matches across PDF line wraps."""
    return " ".join(text.split()).lower()


# ── retrieval metrics (no LLM needed) ────────────────────────────────────────


def retrieval_metrics(retriever, items: list[EvalItem]) -> pd.DataFrame:
    """One row per item that has evidence strings:
    - evidence_recall: share of the item's evidence strings found in the retrieved texts
    - hit: True when ALL evidence was retrieved
    - rr: reciprocal rank, 1 / position of the first chunk containing any evidence (0 if none)
    """
    rows = []
    for item in items:
        if not item["evidence"]:
            continue
        started = time.perf_counter()
        docs = retriever.invoke(item["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        texts = [normalise(d.page_content) for d in docs]
        joined = " ".join(texts)
        evidence = [normalise(e) for e in item["evidence"]]
        found = [e in joined for e in evidence]
        first = next((rank for rank, t in enumerate(texts, start=1) if any(e in t for e in evidence)), None)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "evidence_recall": sum(found) / len(found),
                "hit": all(found),
                "rr": 1 / first if first else 0.0,
                "n_docs": len(docs),
                "latency_ms": round(latency_ms),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.Series:
    """Means of the numeric columns (hit → hit rate, rr → MRR)."""
    numeric = df.drop(columns=[c for c in ("id",) if c in df.columns]).select_dtypes(include=["number", "bool"])
    return numeric.astype(float).mean().round(3)


# ── generation ───────────────────────────────────────────────────────────────


def run_rag(chain_with_sources, items: list[EvalItem], max_concurrency: int = 4) -> list[dict]:
    """Run a chain from `chains.build_rag_chain_with_sources` over the items (in parallel)."""
    outputs = chain_with_sources.batch([i["question"] for i in items], config={"max_concurrency": max_concurrency})
    return [
        {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": out["answer"],
            "contexts": [d.page_content for d in out["contexts"]],
            "ground_truth": item["ground_truth"],
        }
        for item, out in zip(items, outputs)
    ]


JUDGE_PROMPT = """You grade answers from a hotel concierge assistant.
Question: {question}
Reference answer: {ground_truth}
Assistant's answer: {answer}

Does the assistant's answer contain the key facts of the reference answer, without contradicting it?
Extra correct detail is fine. Reply with exactly one word: yes or no."""


def judge_correct(question: str, answer: str, ground_truth: str, category: str | None = None) -> bool:
    """A cheap LLM judge. For `unanswerable` items the only correct answer is the exact
    refusal sentence `chains.NOT_FOUND` (that's what the grounded prompt asks for)."""
    if category == "unanswerable":
        return NOT_FOUND.lower() in answer.lower()
    if NOT_FOUND.lower() in answer.lower():
        return False
    verdict = llm.chat(JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer))
    return verdict.strip().lower().startswith("yes")


def score_answers(rows: list[dict], max_workers: int = 4) -> pd.DataFrame:
    """Judge every row from `run_rag`: id, category, correct, answer."""
    with concurrent.futures.ThreadPoolExecutor(max_workers) as pool:
        verdicts = list(
            pool.map(lambda r: judge_correct(r["question"], r["answer"], r["ground_truth"], r["category"]), rows)
        )
    return pd.DataFrame(
        [{"id": r["id"], "category": r["category"], "correct": v, "answer": r["answer"]} for r, v in zip(rows, verdicts)]
    )


# ── RAGAS ────────────────────────────────────────────────────────────────────

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
# Rough number of LLM calls each metric makes per question (context_precision checks every chunk).
_CALLS_PER_ITEM = {"faithfulness": 2, "answer_relevancy": 1, "context_precision": 4, "context_recall": 1}
RAGAS_SKIP_CATEGORIES = {"unanswerable", "no_search_needed"}


def estimate_ragas_cost(n_items: int, metrics=RAGAS_METRICS) -> float:
    calls = n_items * sum(_CALLS_PER_ITEM[m] for m in metrics)
    return llm.estimate_cost(get_settings().chat_model, calls, avg_in=1200, avg_out=150)


def _ragas_client():
    """An AsyncOpenAI client whose chat and embedding calls are added to `llm.cost`."""
    from openai import AsyncOpenAI

    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.openai_api_key, max_retries=5, default_headers={"Accept-Encoding": "identity"}
    )
    chat_create, embed_create = client.chat.completions.create, client.embeddings.create

    async def tracked_chat(*args, **kwargs):
        response = await chat_create(*args, **kwargs)
        if getattr(response, "usage", None):
            llm.cost.add(kwargs.get("model", "?"), response.usage.prompt_tokens, response.usage.completion_tokens)
        return response

    async def tracked_embed(*args, **kwargs):
        response = await embed_create(*args, **kwargs)
        if getattr(response, "usage", None):
            llm.cost.add(kwargs.get("model", "?"), response.usage.prompt_tokens)
        return response

    client.chat.completions.create = tracked_chat
    client.embeddings.create = tracked_embed
    return client


def _ragas_metric_objects(metrics):
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    settings = get_settings()
    client = _ragas_client()
    extra = {"reasoning_effort": settings.reasoning_effort} if llm._is_reasoning(settings.chat_model) else {}
    judge = llm_factory(settings.chat_model, client=client, **extra)
    embeddings = embedding_factory("openai", model=settings.embed_model, client=client)
    build = {
        "faithfulness": lambda: Faithfulness(llm=judge),
        "answer_relevancy": lambda: AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=judge),
        "context_recall": lambda: ContextRecall(llm=judge),
    }
    return {name: build[name]() for name in metrics}


def ragas_scores(rows: list[dict], metrics=RAGAS_METRICS, max_concurrency: int = 4) -> pd.DataFrame:
    """Score rows from `run_rag` with RAGAS 0.4 metrics (one row per question, one column per metric).
    Unanswerable / no-search rows are skipped: these metrics assume a real answer and reference."""
    warnings.filterwarnings("ignore", module="ragas")
    rows = [r for r in rows if r["category"] not in RAGAS_SKIP_CATEGORIES]
    objects = _ragas_metric_objects(metrics)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(metric_name: str, row: dict) -> float:
        kwargs = {
            "faithfulness": dict(user_input=row["question"], response=row["answer"], retrieved_contexts=row["contexts"]),
            "answer_relevancy": dict(user_input=row["question"], response=row["answer"]),
            "context_precision": dict(
                user_input=row["question"], reference=row["ground_truth"], retrieved_contexts=row["contexts"]
            ),
            "context_recall": dict(
                user_input=row["question"], retrieved_contexts=row["contexts"], reference=row["ground_truth"]
            ),
        }[metric_name]
        async with semaphore:
            try:
                return float((await objects[metric_name].ascore(**kwargs)).value)
            except Exception:  # a failed judgement shouldn't sink the whole run
                return float("nan")

    async def run_all():
        tasks = [one(m, r) for r in rows for m in metrics]
        return await asyncio.gather(*tasks)

    # Jupyter already runs an event loop, so run ours in a worker thread.
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        values = pool.submit(asyncio.run, run_all()).result()
    table = np.array(values, dtype=float).reshape(len(rows), len(metrics)) if rows else np.empty((0, len(metrics)))
    df = pd.DataFrame(table, columns=list(metrics)).round(3)
    df.insert(0, "category", [r["category"] for r in rows])
    df.insert(0, "id", [r["id"] for r in rows])
    return df


# ── saved runs ───────────────────────────────────────────────────────────────


def save_run(name: str, df: pd.DataFrame, meta: dict) -> None:
    folder = get_settings().results_dir
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_run(name: str) -> tuple[pd.DataFrame, dict] | None:
    folder = get_settings().results_dir
    csv_path, meta_path = folder / f"{name}.csv", folder / f"{name}.json"
    if not csv_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return pd.read_csv(csv_path), meta


def run_or_load(name: str, compute: Callable[[], pd.DataFrame], *, meta: dict) -> pd.DataFrame:
    """- RAGLAB_FULL_RUN=1: compute, save to results/<name>.csv (+ .json metadata), return.
    - otherwise, if a saved run exists: return it (no API cost).
    - otherwise: compute without saving (your `compute` decides how big that is)."""
    settings = get_settings()
    saved = None if settings.full_run else load_run(name)
    if saved is not None:
        df, saved_meta = saved
        print(f"showing saved full-run results '{name}' from {saved_meta.get('date', '?')} "
              f"({saved_meta.get('n_items', len(df))} items, cost ${saved_meta.get('cost_usd', 0):.4f})")
        return df
    before = llm.cost.total()
    df = compute()
    if settings.full_run:
        full_meta = {**meta, "n_items": len(df), "cost_usd": round(llm.cost.total() - before, 5),
                     "date": dt.date.today().isoformat(), "chat_model": settings.chat_model}
        save_run(name, df, full_meta)
        print(f"saved results/{name}.csv")
    return df


def compare_runs(names: list[str]) -> pd.DataFrame:
    """One row per saved run: the mean of every numeric column."""
    rows = {}
    for name in names:
        saved = load_run(name)
        if saved is not None:
            rows[name] = summarize(saved[0])
    return pd.DataFrame(rows).T


def compare(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Like compare_runs, for results already in memory: {"name": df} -> one row of means per name."""
    return pd.DataFrame({name: summarize(df) for name, df in results.items()}).T
