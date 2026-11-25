# Token Hotels RAG service. Build from the REPO ROOT:
#   docker build -f service/Dockerfile -t token-rag .
#   docker run --rm -p 8080:8080 -e OPENAI_API_KEY=... -e RAGLAB_SERVICE_API_KEY=... token-rag
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

# Dependencies first (cached between builds). No [rerank] extra: it would add PyTorch (GBs).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra prod --no-install-project

# The app, the corpus it serves, and the facts file that chunk metadata comes from.
COPY raglab/ raglab/
COPY service/ service/
COPY data/corpus/ data/corpus/
COPY data/_build/facts.yaml data/_build/facts.yaml
RUN uv sync --frozen --extra prod

# Run as an unprivileged user; .cache holds the embedding cache, Chroma files and the manifest.
RUN useradd --create-home app && mkdir -p .cache && chown -R app /app/.cache /app/data/corpus
USER app

EXPOSE 8080
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn service.app.main:app --host 0.0.0.0 --port ${PORT}"]
