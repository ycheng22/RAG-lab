# Token Hotels RAG service

The `raglab` pipeline behind a small HTTP API: cited answers, streaming, document upload, API-key
auth, a rate limit and one JSON log line per request (with its OpenAI cost). Notebook 11 walks
through it.

| Endpoint | Auth | What it does |
|---|---|---|
| `GET /health` | none | `{"status": "ok", "store": "chroma", "chunks": 109}` |
| `POST /ask` | `X-API-Key` | `{"question": "...", "k": 4, "stream": false}` → answer, `found`, citations and which citations verified. `"stream": true` returns `text/event-stream` tokens. |
| `POST /ingest` | `X-API-Key` | Multipart `file` (.md, .html, .pdf, .csv, max 5 MB). Screened for prompt injection (warnings are returned, not blocked), then indexed incrementally. |

Errors: 401 (missing or wrong key), 415 (unsupported upload), 422 (invalid body, e.g. a question
over 1,000 characters), 429 (more than 30 requests per minute for one key).

## Run locally

From the repo root, with `OPENAI_API_KEY` and `RAGLAB_SERVICE_API_KEY` in `.env`:

```bash
uv sync --extra prod
uv run uvicorn service.app.main:app --reload
curl -X POST localhost:8000/ask -H "X-API-Key: $RAGLAB_SERVICE_API_KEY" \
     -H "Content-Type: application/json" -d '{"question": "When is check-in at Token Kyoto?"}'
```

Without `RAGLAB_SERVICE_API_KEY` the app refuses to start, unless you set `RAGLAB_DEV=1` (no auth;
your own machine only).

| Setting | Default | Meaning |
|---|---|---|
| `RAGLAB_STORE` | `chroma` | `chroma` (local files) or `pgvector` (needs `DATABASE_URL`) |
| `RAGLAB_SERVICE_CORPUS_DIR` | `data/corpus` | Folder that is indexed at startup and where uploads are saved |
| `RAGLAB_SERVICE_COLLECTION` | `token` | Collection / table name in the vector store |

## Run in Docker

```bash
docker build -f service/Dockerfile -t token-rag .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=sk-... -e RAGLAB_SERVICE_API_KEY=change-me token-rag
curl localhost:8080/health
```

The image has no `[rerank]` extra (PyTorch would add gigabytes).

## Deploy to Google Cloud Run (with Neon Postgres)

This is a guide. Nothing in this repo deploys automatically.

### Costs: read this first

- **OpenAI usage is the real risk.** Every `/ask` costs money (about $0.0001 with `gpt-6-luna`).
  The API key and the 30 requests/minute limit exist so that a leaked URL can't run up a bill.
  Also set a monthly **usage limit** in your OpenAI account.
- **Cloud Run:** with `--min-instances 0` you pay nothing while idle, and the monthly free tier
  covers light use. `--max-instances 1` caps the worst case (and keeps the in-memory rate limit
  correct).
- **Neon:** the free tier includes pgvector and is plenty for this corpus.
- **Artifact Registry** stores the image: 0.5 GB/month is free, and this image is about 3.3 GB
  (the series' dependencies include RAGAS and the notebook tooling), so expect roughly $0.30 per
  month. Delete old images; trimming the service's dependencies is the next optimisation.
- **Secret Manager:** the first 6 active secret versions are free; this uses 3.

### 1. Database (Neon free tier)

1. Create a project at [neon.tech](https://neon.tech).
2. In its SQL editor run `CREATE EXTENSION IF NOT EXISTS vector;`
3. Copy the connection string (`postgresql://...?sslmode=require`).

### 2. Secrets

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com
printf '%s' "sk-..."              | gcloud secrets create openai-api-key --data-file=-
printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create raglab-service-api-key --data-file=-
printf '%s' "postgresql://..."    | gcloud secrets create database-url --data-file=-
```

Give the Cloud Run service account the `roles/secretmanager.secretAccessor` role on these secrets.

### 3. Build and push the image

The Dockerfile lives in `service/`, so build with an explicit file (Cloud Run's `--source` expects
a Dockerfile at the root):

```bash
REGION=europe-west1   # pick yours
gcloud artifacts repositories create rag --repository-format=docker --location=$REGION
docker build -f service/Dockerfile -t $REGION-docker.pkg.dev/$PROJECT_ID/rag/token-rag .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/rag/token-rag
```

### 4. Deploy

```bash
gcloud run deploy token-rag \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/rag/token-rag \
  --region $REGION --allow-unauthenticated \
  --min-instances 0 --max-instances 1 --concurrency 10 --memory 1Gi \
  --set-env-vars RAGLAB_STORE=pgvector \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,RAGLAB_SERVICE_API_KEY=raglab-service-api-key:latest,DATABASE_URL=database-url:latest
```

`--allow-unauthenticated` lets requests reach the container; the `X-API-Key` check is what protects
the API. For a private service, drop it and put Cloud Run's IAM or an API gateway in front.

### 5. Budget alert

```bash
gcloud billing budgets create --billing-account=$BILLING_ACCOUNT --display-name="token-rag" \
  --budget-amount=5USD --threshold-rule=percent=0.5 --threshold-rule=percent=1.0
```

A budget **alerts**, it doesn't stop spending. The OpenAI usage limit and `--max-instances 1` are
the hard caps.

### Things to know in production

- The container's disk is temporary. At every cold start the service re-syncs the corpus baked into
  the image; with pgvector the chunks already exist and are simply re-written (about $0.0003 of
  embeddings, because the embedding cache is temporary too). Files uploaded with `/ingest` stay in
  pgvector, but the uploaded file itself is lost on restart: for durable uploads, save them to a
  Cloud Storage bucket instead.
- The rate limit is in memory, per instance. With more than one instance, move it to Redis or an API gateway.
- Logs are one JSON object per line on stdout, so Cloud Logging parses them automatically.
  Filter on `jsonPayload.cost_usd` to watch spend per request.
