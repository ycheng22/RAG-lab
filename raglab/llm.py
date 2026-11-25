"""The only place that talks to OpenAI.

- `chat()` and `embed()` are thin wrappers around the OpenAI SDK.
- `embed()` keeps an on-disk cache, so re-running a notebook does not pay for the same embeddings twice.
- `cost` tracks every call's tokens and dollars, so each notebook can print what it spent.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections import defaultdict
from functools import lru_cache

import numpy as np
from openai import OpenAI

from raglab.config import get_settings

# USD per 1M tokens: (input, output).
# Checked 2026-11-24 at https://developers.openai.com/api/docs/pricing (Standard tier).
PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "text-embedding-3-small": (0.02, 0.0),
}

# Reasoning models take `reasoning_effort` and reject `temperature`.
REASONING_MODELS: set[str] = {"gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-nano"}

MISSING_KEY_MESSAGE = "Set OPENAI_API_KEY in .env (see README)"

EMBED_BATCH = 256


# ── client ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def client() -> OpenAI:
    """A shared OpenAI client. Raises a one-line RuntimeError if the key is missing."""
    key = get_settings().openai_api_key.strip()
    if not key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    # Identity encoding sidesteps a decompression bug seen with some SDK/httpx2 builds.
    return OpenAI(api_key=key, max_retries=5, default_headers={"Accept-Encoding": "identity"})


# ── cost tracking ────────────────────────────────────────────────────────────


class CostTracker:
    """Counts calls, tokens and dollars per model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # calls, in, out

    def add(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        with self._lock:
            row = self._rows[model]
            row[0] += 1
            row[1] += int(input_tokens or 0)
            row[2] += int(output_tokens or 0)

    @staticmethod
    def price_of(model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = PRICES.get(model) or PRICES.get(_base_model(model))
        if price is None:
            return None
        return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000

    def total(self) -> float:
        return sum(self.price_of(m, r[1], r[2]) or 0.0 for m, r in self._rows.items())

    def calls(self) -> int:
        return sum(r[0] for r in self._rows.values())

    def snapshot(self) -> dict[str, tuple[int, int, int]]:
        return {m: tuple(r) for m, r in self._rows.items()}

    def report(self) -> None:
        if not self._rows:
            print("Cost: no API calls in this notebook ($0.0000)")
            return
        print(f"{'model':<26}{'calls':>7}{'in tokens':>12}{'out tokens':>12}{'USD':>10}")
        for model, (n, tin, tout) in sorted(self._rows.items()):
            usd = self.price_of(model, tin, tout)
            usd_s = f"{usd:.4f}" if usd is not None else "unknown"
            print(f"{model:<26}{n:>7}{tin:>12,}{tout:>12,}{usd_s:>10}")
        print(f"{'total':<26}{self.calls():>7}{'':>24}{self.total():>10.4f}")


def _base_model(model: str) -> str:
    """'gpt-6-luna-2026-05-01' -> 'gpt-6-luna' (dated snapshots share the family price)."""
    parts = model.split("-")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "-".join(parts)


cost = CostTracker()


def estimate_cost(model: str, n_calls: int, avg_in: int, avg_out: int) -> float:
    """Rough USD estimate for `n_calls` calls of the given average size."""
    return CostTracker.price_of(model, n_calls * avg_in, n_calls * avg_out) or 0.0


def check_budget(estimated_usd: float, *, limit_usd: float | None = None) -> bool:
    """Print an estimate; return True if it fits the per-notebook budget (or a full run was requested)."""
    settings = get_settings()
    limit = settings.notebook_budget_usd if limit_usd is None else limit_usd
    ok = estimated_usd <= limit or settings.full_run
    verdict = "OK" if ok else "over budget — skipping (set RAGLAB_FULL_RUN=1 to run anyway)"
    print(f"Estimated cost: ${estimated_usd:.4f} (limit ${limit:.2f}) → {verdict}")
    return ok


# ── chat ─────────────────────────────────────────────────────────────────────


def _is_reasoning(model: str) -> bool:
    return model in REASONING_MODELS or _base_model(model) in REASONING_MODELS


def chat(
    prompt: str | list[dict],
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    **kw,
) -> str:
    """Send one chat request and return the reply text.

    `prompt` is either a plain string (sent as one user message) or a list of
    {"role": ..., "content": ...} messages.
    """
    settings = get_settings()
    model = model or settings.chat_model
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    if _is_reasoning(model):
        kw.pop("temperature", None)
        kw.setdefault("reasoning_effort", reasoning_effort or settings.reasoning_effort)
    else:
        kw.setdefault("temperature", 0)
    response = client().chat.completions.create(model=model, messages=messages, **kw)
    if response.usage is not None:
        cost.add(model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return (response.choices[0].message.content or "").strip()


# ── embeddings (with an on-disk cache) ───────────────────────────────────────

_db_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cache_db() -> sqlite3.Connection:
    path = get_settings().cache_dir / "embeddings.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, h TEXT, vec BLOB, PRIMARY KEY (model, h))")
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed many texts. Cached by (model, sha256(text)); only cache misses hit the API."""
    model = model or get_settings().embed_model
    if not texts:
        return []
    hashes = [_hash(t) for t in texts]
    found: dict[str, list[float]] = {}
    with _db_lock:
        db = _cache_db()
        unique = list(dict.fromkeys(hashes))
        for i in range(0, len(unique), 500):
            part = unique[i : i + 500]
            rows = db.execute(
                f"SELECT h, vec FROM emb WHERE model = ? AND h IN ({','.join('?' * len(part))})",
                [model, *part],
            ).fetchall()
            for h, blob in rows:
                found[h] = np.frombuffer(blob, dtype=np.float32).tolist()

    missing = [(h, t) for h, t in dict(zip(hashes, texts)).items() if h not in found]
    for i in range(0, len(missing), EMBED_BATCH):
        batch = missing[i : i + EMBED_BATCH]
        response = client().embeddings.create(model=model, input=[t for _, t in batch])
        if response.usage is not None:
            cost.add(model, response.usage.prompt_tokens)
        new_rows = []
        for (h, _), item in zip(batch, response.data):
            found[h] = item.embedding
            new_rows.append((model, h, np.asarray(item.embedding, dtype=np.float32).tobytes()))
        with _db_lock:
            db = _cache_db()
            db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", new_rows)
            db.commit()

    return [found[h] for h in hashes]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


# ── LangChain adapters (added in notebook 04) ────────────────────────────────
# They route LangChain's calls through the same cache and cost meter as the functions above.

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class CachedEmbeddings(Embeddings):
    """LangChain Embeddings backed by `llm.embed` (disk cache + cost tracking)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_settings().embed_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(list(texts), model=self.model)

    def embed_query(self, text: str) -> list[float]:
        return embed([text], model=self.model)[0]


class CostCallbackHandler(BaseCallbackHandler):
    """Adds the token usage of every LangChain chat call to `llm.cost`."""

    def __init__(self, model: str) -> None:
        self.model = model

    def on_llm_end(self, response, **kwargs) -> None:
        counted = False
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    cost.add(self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                    counted = True
        if not counted and response.llm_output:  # some providers report usage here instead
            usage = response.llm_output.get("token_usage") or {}
            if usage:
                cost.add(self.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def get_lc_chat(model: str | None = None, temperature: float = 0.0):
    """A LangChain ChatOpenAI wired to the cost meter. Reasoning models get `reasoning_effort`
    instead of `temperature` (they reject temperature)."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    model = model or settings.chat_model
    if not settings.openai_api_key.strip():
        raise RuntimeError(MISSING_KEY_MESSAGE)
    kwargs = {"reasoning_effort": settings.reasoning_effort} if _is_reasoning(model) else {"temperature": temperature}
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        max_retries=5,
        default_headers={"Accept-Encoding": "identity"},
        callbacks=[CostCallbackHandler(model)],
        **kwargs,
    )
