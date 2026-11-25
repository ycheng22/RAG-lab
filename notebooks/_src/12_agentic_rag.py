# %% [markdown]
# # 12 · Agentic RAG: let the model decide when to search
#
# Every system so far retrieves on **every** question, exactly once, with the question as the query.
# That's wasteful for "What's 15% of 200?", and too little for "What's the signature dish at the
# restaurant of the cheapest hotel?", which needs two lookups, the second depending on the first.
#
# An **agent** gives the model tools and lets it decide: answer directly, search, look something up
# in a table, or search again with a better query.
#
# **In this notebook you will:**
# - give the model three tools and run the classic **tool loop** with the plain OpenAI SDK,
# - build the same behaviour as an explicit **LangGraph** state machine, with query rewriting and a self-check,
# - compare both against the fixed pipeline on accuracy, latency and cost.
#
# Cost: about $0.01.

# %%
# <<SETUP>>

# %%
# Setup 2/3: the agent module (it builds or reuses the baseline index on first use)
import importlib.util
import json

import pandas as pd

from raglab import agent
from raglab import evaluation as ev
from raglab.display import show_diagram

settings = get_settings()
pd.set_option("display.max_colwidth", 100)

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. What an agent adds: decisions

# %%
show_diagram("agent_loop")

# %% [markdown]
# ## 2. The tools
#
# A tool is a plain Python function plus a JSON description the model reads. We have three:
#
# | Tool | Data | Good for |
# |---|---|---|
# | `search_docs(query, hotel=None)` | hybrid search over the documents (notebook 07) | policies, dining, FAQ, anything written in prose |
# | `lookup_rates(hotel, room_type, season)` | the rate table, filtered with pandas | prices: exact numbers, no retrieval guesswork |
# | `list_hotels()` | the hotel database (one row per hotel) | comparisons and filters: "earliest", "cheapest", "in Asia with a pool" |
#
# The last two matter: notebooks 06 and 07 showed that questions comparing **all** hotels are the
# hardest for retrieval, because no single chunk contains the answer. A table does.

# %%
print(agent.search_docs("pets allowed", hotel="Token Lisbon", k=1))
print(agent.lookup_rates("Token Cusco", season="low"))
pd.DataFrame(agent.list_hotels())

# %% [markdown]
# This is what the model sees for one tool, an OpenAI function-tool schema:

# %%
print(json.dumps(agent.TOOL_SPECS[1], indent=2))

# %% [markdown]
# ## 3. The tool loop
#
# The whole agent is a loop (`agent.run_tool_loop`):
#
# ```python
# messages = [system_prompt, question]
# for each round (at most max_steps):
#     reply = model(messages, tools=TOOL_SPECS)
#     if reply has no tool calls: return reply.text        # the model decided it's done
#     for each tool call: run the function, append its result as a "tool" message
# ```
#
# The system prompt tells it when to use tools: Token-specific facts always come from tools, general
# knowledge never needs them. Watch the decisions (`verbose=True` prints every tool call):

# %%
demo_questions = [
    "What is 15% of 200?",
    "What time is check-in at Token Cusco?",
    "What is the cheapest suite in Asia in low season?",
    "What is the signature dish at the restaurant of the cheapest Token hotel?",
]
for q in demo_questions:
    print(f"Q: {q}")
    result = agent.run_tool_loop(q, verbose=True)
    print(f"A: {result.answer}\n   searched={result.searched}, tool calls={len(result.steps)}, {result.latency_ms} ms\n")

# %% [markdown]
# Four questions, four strategies, all chosen by the model:
#
# - the arithmetic needs **no** tool, so it costs one short call and no retrieval;
# - the check-in time and the suite price come from tools. Which tool the model picks can change
#   between runs (both `search_docs` and `list_hotels` know the check-in time), and for the price it
#   typically uses the structured ones, `list_hotels` to find the Asian hotels and `lookup_rates`;
# - the last question is **multi-hop**: the model first finds the cheapest hotel, then looks up its
#   restaurant. A fixed pipeline can't do that: it searches once, with the original question.
#
# ## 4. The same idea as a graph (LangGraph)
#
# The tool loop is flexible, but its behaviour lives inside the model's head: hard to test, hard to
# bound. **LangGraph** makes the control flow explicit. Each node is a function, edges decide what
# runs next, and you can add steps the model would skip on its own:
#
# - `route`: one cheap call decides *search*, *rates* or *answer directly*,
# - `grade`: do the retrieved passages actually answer the question? If not, `rewrite` the query
#   and search once more,
# - `self_check`: is every fact in the answer supported by the context? If not, refuse.
#
# LangGraph needs the optional `[agent]` extra.

# %%
HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
if HAS_LANGGRAPH:
    graph = agent.build_graph()
    print(graph.get_graph().draw_mermaid())  # the graph as Mermaid text (paste it into any Mermaid viewer)
else:
    print("Skipping the LangGraph half: install it with `uv sync --extra agent` and restart the kernel.")

# %%
if HAS_LANGGRAPH:
    for q in demo_questions + ["Who is the general manager of Token Lisbon?"]:
        result = agent.run_graph(q)
        print(f"Q: {q}\nA: {result.answer}")
        print("   path:", " -> ".join(s.name for s in result.steps), f"({result.latency_ms} ms)\n")

# %% [markdown]
# Read the paths. The graph routes the arithmetic straight to an answer and the price question to
# the rate table, and it refuses the unanswerable question after one rewrite. But the multi-hop
# question shows the price of a fixed graph: it only has `search_docs` and `lookup_rates`, so
# "the cheapest hotel" has to be found by searching, and the rewrite may or may not get there.
# The tool loop simply called `list_hotels` first. **Explicit control flow is easier to test and
# bound; a free tool loop is more flexible.** Real systems often combine them: a graph whose nodes
# are small tool loops.
#
# ## 5. Is it worth it? Measure.
#
# The default run compares the three systems on the 5 quick questions plus the 5 questions that need
# no search at all. The full run (`RAGLAB_FULL_RUN=1`) covers all 40 questions.

# %%
items = ev.load_eval("quick") + ev.load_eval("full", categories={"no_search_needed"})
if settings.full_run:
    items = ev.load_eval("full") + ev.load_eval("full", categories={"no_search_needed"})
systems = ("pipeline", "tool_loop", "graph") if HAS_LANGGRAPH else ("pipeline", "tool_loop")

compare = ev.run_or_load(
    "nb12_agent_compare",
    lambda: agent.compare_with_pipeline(items, systems=systems),
    meta={"systems": list(systems)},
)
summary = compare.groupby("system").agg(
    accuracy=("correct", "mean"),
    searched=("searched", "mean"),
    median_latency_ms=("latency_ms", "median"),
    cost_per_1000_questions=("cost_usd", lambda c: c.mean() * 1000),
)
summary.round(3)

# %%
compare.pivot_table(index="category", columns="system", values="correct", aggfunc="mean").round(2)

# %% [markdown]
# ## 6. What the numbers say
#
# In the run saved here:
#
# - **The pipeline fails every "no search needed" question**: its grounded prompt refuses anything
#   that isn't in the documents, so "What is 15% of 200?" gets the refusal sentence. Both agents
#   answer those directly, without searching (that's the 0.5 in `searched`: half the questions).
# - **On the hotel questions** the default run has only **one question per category**, so a single
#   miss swings a category between 1.0 and 0.0. Read those rows as anecdotes; the full run (all
#   40 questions) is the fair test. Agents run several model calls per question, so their answers
#   also vary more from run to run than the pipeline's.
# - **The price is latency and cost**: each agent answer takes several model calls instead of one,
#   so it is slower and costs two to three times as much per question as the pipeline (compare
#   `median_latency_ms` and `cost_per_1000_questions`). The graph's grade and self-check calls
#   make it the slowest.
#
# **When is an agent worth it?** When questions vary: some need no search, some need several, some
# need structured data. When every question is "look this up in the documents", a well-tuned
# pipeline (notebooks 07 and 08) is faster, cheaper and easier to test. Measure on *your* questions
# before you choose.
#
# ## Your turn
#
# 1. Add a `max_steps=2` limit to the multi-hop question. What does the tool loop do when it runs out of steps?
# 2. Give the graph a `list_hotels` route. Does the multi-hop question improve?
# 3. Ask the tool loop something with a trap, e.g. "What's the price of a suite at Token Paris?" Does it refuse?
#
# ## Key takeaways
#
# - An agent's value is **decisions**: whether to search, which tool, and whether to search again.
# - Give agents structured tools for structured questions; retrieval is weakest at "compare all X".
# - A free tool loop is flexible; an explicit graph is testable and bounded. Both are a few dozen lines.
# - Agents cost more calls per question. Measure accuracy, latency and cost against a plain pipeline.
#
# **Added to raglab:** `agent.search_docs / lookup_rates / list_hotels / TOOL_SPECS / run_tool_loop /
# build_graph / run_graph / compare_with_pipeline / ToolCall / AgentResult`.
#
# # Series wrap-up
#
# You started with RAG in 40 lines of plain Python (notebook 01) and ended with an evaluated,
# defended, deployable service and an agent. The three things this series set out to give you:
#
# 1. **Judge and tune someone else's RAG system.** Build an eval set with unanswerable and
#    conflicting questions (05), measure retrieval separately from generation (05, 08), triage every
#    failure (08), and try cheap fixes before clever ones (06, 07).
# 2. **A working service you can adapt and deploy.** Incremental indexing and versions (09),
#    verifiable answers and injection defence (10), and a FastAPI service with auth, rate limits,
#    cost logging and a Cloud Run deploy guide (11).
# 3. **An agent that decides when to search.** Tools, a tool loop and a graph, measured against
#    the pipeline (12).
#
# The habit that ties them together: **measure, change one thing, measure again.**

# %%
cost.report()
