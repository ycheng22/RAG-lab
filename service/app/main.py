"""Token Hotels RAG service.

    uv run uvicorn service.app.main:app --reload        (from the repo root)

Endpoints:
    GET  /health   no auth  -> {"status": "ok", "store": "chroma", "chunks": 109}
    POST /ask      X-API-Key -> an Answer with citations (+ which citations verified),
                               or a text/event-stream of tokens when "stream": true
    POST /ingest   X-API-Key -> upload a .md/.html/.pdf/.csv file; it is screened for prompt
                               injection (warnings are returned, not blocked) and indexed

Configuration (environment or .env): OPENAI_API_KEY, RAGLAB_SERVICE_API_KEY (required unless
RAGLAB_DEV=1), RAGLAB_STORE (chroma | pgvector), DATABASE_URL (pgvector), and optionally
RAGLAB_SERVICE_CORPUS_DIR (default data/corpus) and RAGLAB_SERVICE_COLLECTION (default "token").
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from raglab import answering, bootstrap, indexing, loaders
from raglab.config import get_settings
from raglab.store_pg import get_store
from service.app.auth import check_startup_config, require_api_key
from service.app.logging_mw import JsonLogMiddleware, log_json

ALLOWED_SUFFIXES = {".md", ".html", ".pdf", ".csv"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
CHUNKER = bootstrap.profile_chunker("nb04_baseline")  # the baseline chunking: 800 / 150, with prefix


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)
    stream: bool = False


class AskResponse(answering.Answer):
    verified: list[bool] = Field(description="One flag per citation: is the quote really in that document?")


class Service:
    """Everything the endpoints share, created once at startup."""

    def __init__(self) -> None:
        settings = get_settings()
        self.kind = settings.store
        self.collection = os.environ.get("RAGLAB_SERVICE_COLLECTION", "token")
        self.corpus_dir = Path(os.environ.get("RAGLAB_SERVICE_CORPUS_DIR") or settings.corpus_dir)
        self.manifest = settings.cache_dir / f"service_manifest_{self.kind}_{self.collection}.json"
        self.store = get_store(self.kind, self.collection)
        self.sync_lock = threading.Lock()  # one sync at a time

    def sync(self) -> indexing.SyncReport:
        with self.sync_lock:
            docs = indexing.tag_versions(loaders.load_corpus_lc(corpus_dir=self.corpus_dir))
            return indexing.sync_index(self.store, docs, self.manifest, chunker=CHUNKER)

    def chunk_count(self) -> int:
        if not self.manifest.exists():
            return 0
        return sum(len(v["chunk_ids"]) for v in json.loads(self.manifest.read_text(encoding="utf-8")).values())


service: Service | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    check_startup_config()
    service = Service()
    report = service.sync()  # cheap after the first run: unchanged files are skipped
    log_json(event="startup", store=service.kind, collection=service.collection, sync=asdict(report))
    yield


app = FastAPI(title="Token Hotels RAG", version="1.0", lifespan=lifespan)
app.add_middleware(JsonLogMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "store": service.kind, "chunks": service.chunk_count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _key: str = Depends(require_api_key)):
    retriever = service.store.as_retriever(search_kwargs={"k": req.k, "filter": {"is_latest": True}})
    if req.stream:
        def events():
            for piece in answering.stream_answer(req.question, retriever):
                if piece:
                    yield f"data: {json.dumps(piece)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    answer, docs = answering.answer_with_citations(req.question, retriever)
    return AskResponse(**answer.model_dump(), verified=answering.verify_citations(answer, docs))


@app.post("/ingest")
def ingest(file: UploadFile = File(...), _key: str = Depends(require_api_key)) -> dict:
    name = Path(file.filename or "").name  # never trust a client path
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_SUFFIXES)}")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")

    service.corpus_dir.mkdir(parents=True, exist_ok=True)
    (service.corpus_dir / name).write_bytes(content)
    text = "\n".join(d.page_content for d in loaders.load_corpus_lc({name}, corpus_dir=service.corpus_dir))
    warnings = answering.screen_injection(text)
    report = service.sync()
    return {"file": name, "report": asdict(report), "warnings": warnings}
