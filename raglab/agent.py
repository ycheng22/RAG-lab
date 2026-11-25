"""Agentic RAG (added in notebook 12): let the model decide whether, what and how often to search.

Two implementations of the same idea:
- `run_tool_loop`: the plain OpenAI SDK with function tools, in about 40 lines.
- `build_graph` / `run_graph`: a LangGraph state machine that adds query rewriting and a
  self-check (needs the `[agent]` extra).
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd
from langchain_core.documents import Document

from raglab import llm
from raglab.chains import NOT_FOUND
from raglab.config import get_settings, hotel_metadata


@dataclass
class ToolCall:
    name: str
    args: dict
    result_preview: str


@dataclass
class AgentResult:
    answer: str
    steps: list[ToolCall] = field(default_factory=list)
    searched: bool = False
    latency_ms: int = 0


# ── tools ────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _hybrid():
    from raglab import bootstrap, retrievers

    store = bootstrap.ensure_index("nb04_baseline")
    return retrievers.get_retriever("hybrid", store, k=12, chunks=bootstrap.get_chunks("nb04_baseline"))


def _canonical_hotel(name: str | None) -> str | None:
    """'kyoto', 'Token Kyoto' -> 'Token Kyoto' (None if it isn't one of our hotels)."""
    if not name:
        return None
    wanted = name.lower().replace("token", "").strip()
    for hotel in hotel_metadata():
        if hotel.lower().replace("token", "").strip() == wanted:
            return hotel
    return None


def search_docs(query: str, hotel: str | None = None, k: int = 4) -> list[dict]:
    """Hybrid search over the hotel documents. With `hotel`, keep only chunks about that hotel
    (by metadata, or because the text names it: PDF chunks can span two hotels)."""
    canonical = _canonical_hotel(hotel)
    docs: list[Document] = _hybrid().invoke(f"{canonical} {query}" if canonical else query)
    if canonical:
        docs = [d for d in docs if d.metadata.get("hotel_name") == canonical or canonical in d.page_content]
    return [{"source_file": d.metadata.get("source_file", "?"), "text": d.page_content} for d in docs[:k]]


@lru_cache(maxsize=1)
def _rates() -> pd.DataFrame:
    df = pd.read_csv(get_settings().corpus_dir / "room_rates.csv")
    meta = hotel_metadata()
    df["continent"] = df["hotel"].map(lambda h: meta.get(h, {}).get("continent", ""))
    return df


def lookup_rates(hotel: str | None = None, room_type: str | None = None, season: str | None = None) -> list[dict]:
    """Rows of the rate table (hotel, city, continent, room_type, season, nightly_rate_usd,
    max_guests), filtered by any of the arguments (case-insensitive). No hotel = all hotels."""
    df = _rates()
    if hotel:
        canonical = _canonical_hotel(hotel)
        df = df[df["hotel"] == canonical] if canonical else df.iloc[0:0]
    if room_type:
        df = df[df["room_type"].str.lower() == room_type.lower()]
    if season:
        df = df[df["season"].str.lower() == season.lower()]
    return df.to_dict("records")


def list_hotels() -> list[dict]:
    """One row per hotel from the hotel database: city, country, continent, stars, rooms,
    check-in time, price from, pool, spa. For comparing or filtering hotels."""
    import yaml

    from raglab.config import FACTS_FILE

    facts = yaml.safe_load(FACTS_FILE.read_text(encoding="utf-8"))
    extra = {h["name"]: h for h in facts["hotels"]}
    rows = []
    for name, meta in hotel_metadata().items():
        rows.append({**meta, "rooms": extra[name].get("rooms"), "check_in": extra[name].get("check_in")})
    return rows


TOOLS = {"search_docs": search_docs, "lookup_rates": lookup_rates, "list_hotels": list_hotels}

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the Token Hotels documents: hotel profiles, dining guide, FAQ, policies, "
            "loyalty programme and guest reviews. Returns matching passages with their source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, as a short search query"},
                    "hotel": {"type": "string", "description": "Optional hotel name, e.g. 'Token Kyoto'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_rates",
            "description": "Nightly room rates in USD. Room types: Standard, Deluxe, Suite. Seasons: low, high. "
            "Every argument is optional; leave hotel out to get all hotels. Rows include the continent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel": {"type": "string"},
                    "room_type": {"type": "string", "enum": ["Standard", "Deluxe", "Suite"]},
                    "season": {"type": "string", "enum": ["low", "high"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hotels",
            "description": "The hotel database: one row per Token hotel with city, country, continent, star "
            "rating, rooms, check-in time, starting price, pool and spa. Use it to compare or filter hotels.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = f"""You are the concierge assistant for Token Hotels & Resorts, a hotel chain.
You have tools for Token-specific facts: search_docs for anything in the hotel documents,
lookup_rates for room prices, and list_hotels to compare or filter hotels.
Answer general questions (arithmetic, unit conversions, translations, geography) directly,
without tools. Never answer a Token-specific question from memory: use the tools. A question may
need several tool calls (for example, find which hotel matches, then search that hotel).
If the tools don't give you the answer, reply exactly: "{NOT_FOUND}"
If two documents disagree (for example two versions of a policy), use the most recent one.
Keep answers short and factual."""


def _preview(result) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= 160 else text[:157] + "..."


def _call_tool(name: str, args: dict):
    try:
        return TOOLS[name](**args)
    except Exception as err:  # a bad tool call is reported back to the model, not raised
        return {"error": f"{type(err).__name__}: {err}"}


# ── 1. the plain tool loop ───────────────────────────────────────────────────


def run_tool_loop(question: str, *, max_steps: int = 5, model: str | None = None, verbose: bool = False) -> AgentResult:
    """Ask the model; run every tool it calls; send the results back; repeat until it answers
    without calling a tool (or `max_steps` rounds of tool calls have happened)."""
    settings = get_settings()
    model = model or settings.chat_model
    extra = {"reasoning_effort": "none"} if llm._is_reasoning(model) else {"temperature": 0}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[ToolCall] = []
    started = time.perf_counter()
    for round_ in range(max_steps + 1):
        # After max_steps rounds of tool calls, the model must answer with what it has.
        choice = "auto" if round_ < max_steps else "none"
        response = llm.client().chat.completions.create(
            model=model, messages=messages, tools=TOOL_SPECS, tool_choice=choice, **extra
        )
        if response.usage is not None:
            llm.cost.add(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        message = response.choices[0].message
        if not message.tool_calls:
            answer = (message.content or "").strip()
            break
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = _call_tool(call.function.name, args)
            steps.append(ToolCall(call.function.name, args, _preview(result)))
            if verbose:
                print(f"  -> {call.function.name}({args})  =>  {_preview(result)[:100]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)})
    else:
        answer = NOT_FOUND
    latency = round((time.perf_counter() - started) * 1000)
    return AgentResult(answer=answer, steps=steps, searched=bool(steps), latency_ms=latency)


# ── 2. a LangGraph state machine (needs the [agent] extra) ───────────────────

ROUTE_PROMPT = """Classify the guest question for a hotel chain's assistant. Reply with exactly one word:
- rates: it asks about room prices or rates
- search: it needs any other fact about Token Hotels (policies, hotels, dining, loyalty, reviews)
- answer_directly: general knowledge that needs no hotel information (maths, translations, geography)

Question: {question}"""

GRADE_PROMPT = """Do these passages contain the information needed to answer the question?
Reply with exactly one word: yes or no.

Question: {question}

Passages:
{context}"""

REWRITE_PROMPT = """The search query below didn't find the answer in a hotel chain's documents.
Write one better search query: name the specific facts to look for, and expand any reference
("the cheapest hotel", "the hotel in Japan") into terms the documents would use.
Reply with the query only.

Question: {question}
Previous query: {query}"""

GENERATE_PROMPT = """You are the concierge assistant for Token Hotels & Resorts.
Answer using ONLY the information below. If it doesn't contain the answer, reply exactly:
"{not_found}". If two passages disagree, use the most recent version. Keep the answer short.

Information:
{context}

Question: {question}"""

DIRECT_PROMPT = """Answer this general question briefly and accurately: {question}"""

CHECK_PROMPT = """Is every fact in the answer supported by the information? Reply with exactly one word: yes or no.

Information:
{context}

Answer: {answer}"""


def _context_text(state: dict) -> str:
    if state.get("rates"):
        return "Room rates (USD per night):\n" + pd.DataFrame(state["rates"]).to_string(index=False)
    return "\n\n".join(f"[{i}] ({d['source_file']}) {d['text']}" for i, d in enumerate(state.get("docs", []), 1))


def build_graph():
    """route -> (answer_directly | rates | retrieve -> grade [-> rewrite -> retrieve]) -> generate -> self_check."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        question: str
        route: str
        query: str
        docs: list[dict]
        rates: list[dict]
        relevant: bool
        rewrites: int
        answer: str
        steps: list

    def step(state, name, args, result):
        return [*state.get("steps", []), ToolCall(name, args, _preview(result))]

    def route(state):
        word = llm.chat(ROUTE_PROMPT.format(question=state["question"])).strip().lower()
        choice = next((c for c in ("rates", "answer_directly", "search") if c in word), "search")
        return {"route": choice, "query": state["question"], "rewrites": 0, "steps": step(state, "route", {}, choice)}

    def answer_directly(state):
        answer = llm.chat(DIRECT_PROMPT.format(question=state["question"]))
        return {"answer": answer, "steps": step(state, "answer_directly", {}, answer)}

    def rates(state):
        rows = lookup_rates()  # the whole table is small: let the generator filter it
        return {"rates": rows, "steps": step(state, "lookup_rates", {}, f"{len(rows)} rows")}

    def retrieve(state):
        docs = search_docs(state["query"], k=6)
        return {"docs": docs, "steps": step(state, "search_docs", {"query": state["query"]}, [d["source_file"] for d in docs])}

    def grade(state):
        verdict = llm.chat(GRADE_PROMPT.format(question=state["question"], context=_context_text(state)))
        relevant = verdict.strip().lower().startswith("yes")
        return {"relevant": relevant, "steps": step(state, "grade", {}, "relevant" if relevant else "not relevant")}

    def rewrite(state):
        query = llm.chat(REWRITE_PROMPT.format(question=state["question"], query=state["query"]))
        return {"query": query, "rewrites": state.get("rewrites", 0) + 1, "steps": step(state, "rewrite", {}, query)}

    def generate(state):
        prompt = GENERATE_PROMPT.format(not_found=NOT_FOUND, context=_context_text(state), question=state["question"])
        answer = llm.chat(prompt)
        return {"answer": answer, "steps": step(state, "generate", {}, answer)}

    def self_check(state):
        from raglab.evaluation import is_not_found

        if is_not_found(state["answer"]):
            return {"steps": step(state, "self_check", {}, "refusal: nothing to check")}
        verdict = llm.chat(CHECK_PROMPT.format(context=_context_text(state), answer=state["answer"]))
        if verdict.strip().lower().startswith("yes"):
            return {"steps": step(state, "self_check", {}, "supported")}
        return {"answer": NOT_FOUND, "steps": step(state, "self_check", {}, "NOT supported: replaced by the refusal")}

    graph = StateGraph(State)
    for name, fn in [("route", route), ("answer_directly", answer_directly), ("rates", rates), ("retrieve", retrieve),
                     ("grade", grade), ("rewrite", rewrite), ("generate", generate), ("self_check", self_check)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", lambda s: {"search": "retrieve"}.get(s["route"], s["route"]), ["retrieve", "rates", "answer_directly"]
    )
    graph.add_edge("answer_directly", END)
    graph.add_edge("rates", "generate")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", lambda s: "generate" if s["relevant"] or s.get("rewrites", 0) >= 1 else "rewrite", ["generate", "rewrite"]
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "self_check")
    graph.add_edge("self_check", END)
    return graph.compile()


@lru_cache(maxsize=1)
def _graph():
    return build_graph()


def run_graph(question: str) -> AgentResult:
    """Run the LangGraph agent. `steps` lists every node that ran, in order."""
    started = time.perf_counter()
    state = _graph().invoke({"question": question})
    return AgentResult(
        answer=state.get("answer", NOT_FOUND),
        steps=state.get("steps", []),
        searched=state.get("route") != "answer_directly",
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


# ── comparison ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _pipeline_chain():
    from raglab import bootstrap, chains, retrievers

    store = bootstrap.ensure_index("nb04_baseline")
    return chains.build_rag_chain(retrievers.get_retriever("similarity", store, k=4))


def _pipeline(question: str) -> AgentResult:
    """Notebook 04's fixed pipeline: always retrieve (k=4), then answer with the grounded prompt."""
    started = time.perf_counter()
    answer = _pipeline_chain().invoke(question)
    return AgentResult(answer=answer, steps=[], searched=True, latency_ms=round((time.perf_counter() - started) * 1000))


def compare_with_pipeline(
    items: list, systems: tuple[str, ...] = ("pipeline", "tool_loop", "graph"), max_workers: int = 4
) -> pd.DataFrame:
    """Run every item through each system and judge the answers (evaluation.judge_correct).
    Columns: id, category, system, correct, searched, latency_ms, cost_usd (measured per run), answer."""
    from raglab.evaluation import judge_correct

    runners = {"pipeline": _pipeline, "tool_loop": run_tool_loop, "graph": run_graph}

    def one(job):
        item, system = job
        meter = llm.start_meter()  # this worker thread's own meter: runs don't mix their costs
        result = runners[system](item["question"])
        spent = llm.meter_totals(meter)["cost_usd"]
        correct = judge_correct(item["question"], result.answer, item["ground_truth"], item["category"])
        return {"id": item["id"], "category": item["category"], "system": system, "correct": correct,
                "searched": result.searched, "latency_ms": result.latency_ms, "cost_usd": spent,
                "answer": result.answer}

    _hybrid(), _pipeline_chain()  # load the index once, before the threads start
    jobs = [(item, system) for item in items for system in systems]
    with concurrent.futures.ThreadPoolExecutor(max_workers) as pool:
        rows = list(pool.map(one, jobs))
    return pd.DataFrame(rows)
