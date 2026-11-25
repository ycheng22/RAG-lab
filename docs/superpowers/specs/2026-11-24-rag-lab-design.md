# RAG-lab — Design

Date: 2026-11-24
Status: draft for approval

## 1. Intent

A standalone, **public** GitHub repo (`D:\Github_Clones\RAG-lab`) that teaches Retrieval-Augmented
Generation to **developers who can code but are new to RAG**, through a numbered series of Jupyter notebooks.
The curriculum follows the shape of a reference LangChain lab the owner has locally: load, chunk, embed,
retrieve, LCEL chain, advanced retrievers, RAGAS. **All code, prompts and documents are written fresh**,
and the repo neither includes nor mentions the reference lab. The series is beginner-friendly at the start
and production-grade by the end.

By the last notebook a learner should be able to do all three:

1. **Judge and tune** someone else's RAG system: measure it, find out whether failures come from
   retrieval or generation, and improve it with evidence.
2. **Ship a FastAPI RAG service** they can adapt and deploy.
3. **Build an agent that decides when to search** (agentic RAG) and compare it against a fixed pipeline.

## 2. Goals, non-goals, anti-goals

**Goals**
- Each concept is shown "by hand" first (plain Python + OpenAI SDK), then with LangChain, so the
  framework never looks like magic.
- Every notebook runs on its own from a fresh clone, as long as `.env` has `OPENAI_API_KEY` and the core
  install is done (no "run 01–06 first").
- Improvements are *measured*: evaluation comes before optimisation, and each later technique is scored
  against the same baseline.
- Each notebook includes at least one diagram made with Archify.
- Near-zero running cost, within the budget in §10. Cost is shown in the notebooks, and anything that
  could cost money is called out.

**Non-goals**
- No web UI / Angular frontend.
- No LLM providers other than OpenAI (config stays pluggable, but only OpenAI is tested).
- No fine-tuning or training of embedding models.
- No multi-tenant auth, user management, or billing in the service.
- No automated test suite (owner preference: verify by running the notebooks and the service).
- No code, prompt text, data files, or mention/credit of the reference lab anywhere in the repo.

**Anti-goals (failure even if it "works")**
- Notebooks that silently depend on an earlier notebook having been run.
- A later notebook changes shared code so that an earlier notebook's results quietly change or break.
- Surprise API spend (e.g. an unbounded RAGAS run or a public endpoint burning the owner's key).
- Teaching deprecated APIs without saying so.
- Decorative diagrams that don't explain the mechanism.
- Notebook text that claims an improvement which the committed numbers don't show.

## 3. Constraints

- Python ≥ 3.11 (target 3.12; owner machine has 3.13 and `uv`). Windows-first dev box. Paths and
  commands must also work on macOS/Linux.
- The learner supplies `OPENAI_API_KEY` via `.env` (git-ignored; `.env.example` committed).
- Current library majors as of 2026-11-24 (pin compatible ranges in `pyproject.toml`; the builder
  re-checks at build time):
  langchain 1.5, langchain-openai 1.8, langchain-chroma 1.2, langchain-text-splitters 1.2,
  langchain-community 0.4, langchain-classic 1.1, langgraph 1.2, chromadb 1.5, ragas 0.4,
  openai 3.x, fastapi 0.165, sentence-transformers 6.x, rank-bm25 0.2, langchain-postgres 0.0.19, pypdf 7.
- In LangChain 1.x, MultiQueryRetriever, ContextualCompressionRetriever and SelfQueryRetriever live in
  `langchain-classic`. Use them, but label them as legacy in the notebook. Write the RAGAS code against
  0.4 (`EvaluationDataset` / sample objects).
- Cost defaults: `text-embedding-3-small` ($0.02 / 1M tokens). Chat model from env `RAGLAB_CHAT_MODEL`,
  default **`gpt-6-luna` with `reasoning_effort="none"`** ($0.10 in / $0.50 out per 1M tokens, current
  and not deprecated, per OpenAI's pricing page checked 2026-11-24). On a grounding/refusal/injection smoke test it
  gave the same answers as `gpt-5.4-mini` ($0.75 / $4.50) at about 1/8 of the cost. `gpt-5.4-mini` is the
  documented "faster" alternative. Luna is a reasoning model, so code does not pass `temperature`, and
  Chat Completions tool calling requires effort `none`.
- Hosting (notebook 11): Google Cloud Run (min 0 / max 1 instance, budget alert) + Neon free-tier
  Postgres with pgvector. Deployment is an optional final section. Everything also runs locally.

## 4. Approaches considered

**Library approach** — chose **plain Python first, then LangChain**.
- *Plain → LangChain (chosen):* notebooks 01–03 use only the OpenAI SDK, numpy and the raw chromadb client.
  From 04 on, LangChain takes over. Learners see what the framework does for them.
- *LangChain throughout:* closest to the reference lab, but it hides the mechanics and starts learners on legacy APIs.
- *LlamaIndex:* RAG-native, but equally abstract and further from the curriculum's shape.

**Code-sharing approach** — chose **a shared `raglab/` package that grows over the series**.
- *Growing shared package (chosen):* each notebook teaches a concept inline. The polished version then
  "graduates" into `raglab/`, and later notebooks import it. Least duplication, and it reads like a real codebase.
  The main risk (later changes breaking earlier notebooks) is handled by the stability rule in §6.2.
- *Per-notebook folder with a frozen copy of its helpers:* safest isolation, but duplicated code.
- *Everything inline:* fully standalone, but long copy-pasted cells bury the new material in each notebook.

**Corpus** — chose **our own fictional corpus**. Reusing the reference lab's files isn't allowed in a
public repo. Real public docs don't work either: the model already knows them, so you can't tell
retrieval from memory.

## 5. Repository layout

```
RAG-lab/
  README.md                 # what this is, setup, notebook map with the 3 parts, cost note
  pyproject.toml            # uv-managed; extras: [rerank], [prod], [agent]
  .env.example              # OPENAI_API_KEY, RAGLAB_CHAT_MODEL, DATABASE_URL (optional), RAGLAB_STORE
  .gitignore                # .env, .venv, .cache/, chroma dirs, __pycache__
  data/
    corpus/                 # the fictional documents (see §7)
    eval/                   # evaluation question sets (JSON)
    _build/                 # scripts + sources that generate the PDF / CSV, so the corpus is reproducible
  raglab/                   # shared package (see §6)
  notebooks/
    00_setup_and_costs.ipynb
    01_rag_from_scratch.ipynb
    ...
    12_agentic_rag.ipynb
  results/                  # saved scored runs (JSON/CSV) from the owner's full run, see §10
  diagrams/
    <name>.json             # Archify source (typed JSON)
    <name>.html             # Archify interactive output
    <name>.png              # static capture embedded in notebooks
  service/                  # FastAPI app (notebook 11), Dockerfile, deploy notes
  docs/superpowers/specs/   # this document
```

## 6. The `raglab` package

### 6.1 Modules (each lands in the notebook named)

| Module | Introduced | Contents |
|---|---|---|
| `config.py` | 00 | Settings from env/.env (paths, model names, k defaults, store choice); per-hotel metadata table. Single source of truth. |
| `llm.py` | 00 | OpenAI client; `chat()`, `embed()`; **on-disk embedding cache** keyed by (model, text-hash) under `.cache/`; usage + cost tracker (`cost.report()`), plus a per-notebook budget guard that warns before an expensive call. |
| `display.py` | 00 | Pretty-print chunks/sources in notebooks; `show_diagram(name)` (embeds PNG + link to local HTML). |
| `scratch.py` | 01 | Minimal hand-rolled RAG: cosine similarity, top-k, prompt stuffing. |
| `loaders.py` | 02 / 04 | Hand-written PDF/MD/HTML/CSV loaders (02); LangChain loader equivalents (04). Always sets `source_file` metadata. |
| `chunking.py` | 02 / 04 | Fixed, recursive, Markdown-header and HTML-header chunking (by hand in 02, LangChain splitters in 04); metadata enrichment from the hotel table; optional "context prefix" (hotel name + doc type prepended to chunk text). |
| `store_chroma.py` | 03 / 04 | Raw chromadb usage (03); LangChain `Chroma` create/load/reset (04). |
| `chains.py` | 04 | `format_docs`, prompt templates (basic → grounded with explicit "not in the documents" fallback), LCEL `build_rag_chain`. |
| `evaluation.py` | 05 | Eval-set loading; hand-computed hit rate / MRR / recall@k; RAGAS 0.4 wrapper (faithfulness, answer relevancy, context precision, context recall) with a `max_questions` cap and a cost estimate printed before running; save/load scored runs to `results/`. |
| `retrievers.py` | 06 / 07 | MultiQuery, Compression, SelfQuery (`langchain-classic`, labelled legacy); BM25; hybrid via reciprocal rank fusion; cross-encoder rerank (local model, `[rerank]` extra). Single `get_retriever(mode, **kw)` factory. |
| `experiments.py` | 08 | Small grid runner over (chunk size, k, retriever, prompt) → DataFrame of scores, latency and cost. |
| `indexing.py` / `store_pg.py` | 09 | Content-hash–based incremental indexing (add/update/delete), dedupe, doc version metadata (on Chroma); pgvector store (langchain-postgres) behind the same interface. |
| `answering.py` | 10 | Structured answers with citations (Pydantic schema), refusal handling, injection-resistant prompt, streaming. |
| `agent.py` | 12 | Plain tool-calling loop (OpenAI SDK) and a LangGraph agent; tools: `search_docs`, `lookup_rates` (CSV). |
| `bootstrap.py` | 03+ | `ensure_index(profile)` builds (or reuses) the index a notebook needs. This is what makes every notebook standalone. |

### 6.2 Stability rule (mitigates the growing-package risk)
- Once a notebook ships a public function, its **signature and default behaviour never change**.
  Improvements arrive as new functions or as **opt-in** keyword arguments whose defaults keep the old behaviour.
- `bootstrap.ensure_index(profile=...)` takes named, frozen profiles (e.g. `"nb04_baseline"`,
  `"nb07_hybrid"`, `"nb08_broken"`). Each profile maps to its own Chroma collection/dir, so later
  notebooks never overwrite an earlier notebook's index.
- Every notebook ends with a short "Added to raglab" cell listing what graduated into the package.

### 6.3 Self-containment pattern
Every notebook opens with the same 3 setup cells:
1. Locate repo root and add it to `sys.path` (so `import raglab` works without installing). Load `.env`, then fail early
   with a friendly message if the key is missing.
2. `ensure_index(...)` for whatever this notebook needs (cheap: embeddings are cached on disk; first
   full build costs well under $0.01).
3. `cost.reset()`. At the end, `cost.report()` shows what the notebook spent.

## 7. Corpus (fictional, written fresh for this repo)

**Token Hotels & Resorts**, a fictional hotel group of about 8 properties in different cities and continents (owner's choice of name; a web search found no real chain by that name). Hotels suit
metadata filtering (city, continent, stars, price, pool, spa) and lend themselves to plausible
policy, dining and FAQ documents. The README and every corpus document carry a short "fictional company, no affiliation
with any real business" note. All facts must stay consistent across files.

| File | Purpose / what it teaches |
|---|---|
| `hotel_profiles.pdf` | Property guide with **tables** (amenities, check-in/out, price); PDF parsing pain. Generated from `_build/` sources. |
| `dining_guide.md` | Header-structured Markdown → header-aware chunking. |
| `faq.html` | HTML with h1/h2 → HTML header splitting. |
| `cancellation_policy_2024.md` + `cancellation_policy_2025.md` | **Contradicting versions** → version metadata, recency handling (09). |
| `room_rates.csv` | Structured data → when *not* to use vector search; agent tool (12). |
| `loyalty_program.md` + a near-duplicate page | Deduplication (09). |
| `guest_reviews.md` (or similar) | Contains an embedded **prompt-injection** line → defence demo (10). |

**Eval set** (`data/eval/questions.json`, about 40 items; schema: id, category, question,
ground_truth, source_documents, hallucination_prone). Categories: factual, comparative, multi_hop,
metadata_filter, **unanswerable**, **conflicting_versions**, **no_search_needed** (for the agent).
A 5-question `quick` subset keeps RAGAS runs cheap by default.

## 8. Curriculum

Each notebook follows the same shape: *Objectives → Diagram → Concept (short) → Build it → Observe →
Your turn (exercises, solutions in collapsed `<details>`) → Key takeaways → Added to raglab → Cost*.

| # | Notebook | Core content |
|---|---|---|
| **Part 1 — Foundations (no frameworks)** |||
| 00 | Setup & costs | env, key check, first chat + embed call, cost meter, how the series and `raglab` are organised |
| 01 | RAG from scratch | ~50 lines: embed a handful of paragraphs, cosine similarity in numpy, stuff prompt, answer. Same question with vs. without retrieval (hallucination on fictional facts). |
| 02 | Loading & chunking by hand | read PDF/MD/HTML yourself; fixed vs. recursive vs. header-aware chunking; chunk-size histogram; why overlap; metadata |
| 03 | Vector search for real | raw chromadb: collections, metadata filters, persistence, top-k, MMR; peek at distances |
| **Part 2 — Judge & tune (LangChain)** |||
| 04 | The pipeline in LangChain | loaders → splitters → Chroma → retriever → LCEL chain; side-by-side map to 01–03; basic → grounded prompt |
| 05 | Evaluate before you optimise | build baseline; hit rate/MRR by hand; RAGAS 4 metrics; read the scores; where does each failure come from? |
| 06 | Advanced retrievers | MultiQuery, Compression, SelfQuery (legacy-labelled), each scored vs. baseline |
| 07 | Hybrid search & reranking | BM25, RRF fusion, local cross-encoder rerank; scored |
| 08 | Tuning lab + diagnose a broken RAG | (a) experiment grid; failure triage (retrieval vs. generation); latency/cost/quality trade-offs; **"reviewing someone else's RAG" checklist**. (b) **Diagnose a broken RAG** (see §11a). |
| **Part 3 — Production** |||
| 09 | Production indexing | incremental re-indexing via hashes; dedupe; contradicting versions (on Chroma); optional pgvector section on Neon / local Docker Postgres |
| 10 | Answers you can trust | citations with structured output, refusals on unanswerable, prompt-injection defence, streaming |
| 11 | FastAPI service | `service/`: `/ask` (streaming), `/ingest`, `/health`; API-key header + simple rate limit; structured logs with latency, tokens and cost per request; Dockerfile; run locally; optional Cloud Run deploy with caps + budget alert |
| 12 | Agentic RAG | plain tool-calling loop ("decide when to search"); LangGraph version with query rewrite + self-check; agent vs. fixed pipeline on the eval set (incl. `no_search_needed`) |

## 9. Diagrams (Archify)

- Install Archify as a **global Claude Code skill** by cloning `github.com/tt-a1i/archify` (Apache-2.0)
  and copying its `archify/` folder to `~/.claude/skills/archify`. Inspect it rather than piping
  `npx skills add`. Set `ARCHIFY_UPDATE_CHECK_DISABLED=1`. Node 22 is present.
- Flow per diagram: write the typed JSON IR → `archify validate` → `archify deliver` → HTML in `diagrams/`.
- Notebooks can't run Archify's interactive HTML (GitHub and VS Code strip the JS), so for each diagram we
  capture a **PNG with a headless browser** (Playwright). Each notebook displays it with a
  `raglab.display.show_diagram()` code cell (see §11b). Under the image goes an "Open interactive version" link,
  a **relative local link** to `diagrams/<name>.html` that works in a clone (no web hosting).
- At least one diagram per notebook, showing the mechanism taught (e.g. 01 data flow, 05 eval loop,
  07 hybrid fusion, 11 service architecture, 12 agent decision loop).

## 10. Cost, evidence & safety

**Budget (owner-approved):**

- A learner's default run of any one notebook costs **≈ $0.05 or less**. Defaults: cheap hand-computed
  retrieval metrics on all ~40 questions, RAGAS on the 5-question `quick` subset, and small grids.
- The owner's **full run**, which produces the committed outputs, costs **≈ $2 in total** for all
  notebooks, using the full eval set.
- The scored results from the full run are saved to `results/` and loaded in the notebooks, so learners
  can see the evidence without paying. Re-running at full size is opt-in (`FULL_RUN = True`), with a
  printed cost estimate first.
- Notebook text describes whatever the committed full-run numbers actually show. If a technique doesn't
  help on this corpus, the notebook says so and explains why.

**Safety:**

- Embedding cache means re-running notebooks costs almost nothing after the first build.
- The service requires an API key header and has a rate limit, because a public endpoint would spend the owner's OpenAI credits.
- Cloud Run: min 0 / max 1 instance; budget alert documented. Neon free tier. Reranker runs locally (no paid API).
- The `[rerank]` extra pulls in PyTorch (large download, CPU wheels). The README says so up front.

## 11. Verification

No automated tests (owner preference). Done means:
- every notebook runs top-to-bottom in a fresh kernel from a fresh clone (`jupyter nbconvert --execute`),
  in any order;
- the service starts locally, and `/ask` answers with citations;
- notebooks are committed **with outputs**, so they read well on GitHub (no secrets in outputs);
- the full run stays within the §10 budget.

## 11a. Calls made after the builder check (owner may override)

- **Storage across 09 / 11.** Notebook 09 teaches incremental indexing, hashing, dedupe and versioning on
  **Chroma**, so it runs with just the OpenAI key. Its pgvector section runs only when `DATABASE_URL`
  is set (Neon free tier, or local Docker Postgres). Otherwise it prints how to enable it and skips.
  The service selects its store with `RAGLAB_STORE=chroma|pgvector`. Local runs default to Chroma, with
  the index built at startup from `data/corpus` and the embedding cache. The **Cloud Run deploy uses
  pgvector on Neon**, because Cloud Run's disk doesn't persist `/ingest` writes.
- **Extras.** Notebook 07 needs the `[rerank]` extra. Its first cell checks for it and explains the
  install, and the hybrid-search half runs without it. "Any order, key only" applies to the core install.
- **A system to diagnose.** Notebook 08 gains a second half, "Diagnose a broken RAG". A
  `bootstrap` profile `nb08_broken` plants faults: oversized chunks without overlap, k=1, a prompt that
  invites guessing, and a stale index missing one document. Learners use the 05 eval tools and the 08
  checklist to find and fix each fault, so outcome 1 (judging someone *else's* system) is practised
  and not only described.

## 11b. Colab / Kaggle readiness (owner request; badges added later)

Each notebook will later get "Open in Colab" and "Open in Kaggle" badges. Those are just links to the public
GitHub notebook URL. Two things are designed in now so that adding the badges later is trivial:
- **Setup cell 1** detects Colab (`google.colab` importable) or Kaggle (`KAGGLE_KERNEL_RUN_TYPE` in env). When it
  finds one, it `git clone`s `RAGLAB_REPO_URL` (set once the repo is public) and `pip install`s the core package.
  It reads `OPENAI_API_KEY` from Colab Secrets (`google.colab.userdata`) or Kaggle Secrets (`UserSecretsClient`).
  Locally it uses `.env`.
- **Diagrams are shown by a `show_diagram()` code cell**, not a relative markdown image. Its output embeds the PNG,
  so the diagram renders on GitHub, in Colab and in Kaggle alike. Relative image paths in markdown don't resolve
  there. In Colab/Kaggle, the interactive HTML can be shown inline with `show_diagram(name, interactive=True)`.

## 12. Decisions left to the builder

- Exact document text (consistent facts across files; eval ground truth must match the corpus).
- Exact reranker model (small CPU-friendly cross-encoder).
- Choice of PDF generator for the corpus build script.
- Exercise content and the number of exercises per notebook.
- Whether notebook 09's optional pgvector section defaults to Neon or local Docker Postgres (both documented).
