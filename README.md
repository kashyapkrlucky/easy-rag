# EasyRAG

A self-hosted, local-first Retrieval-Augmented Generation system for your Markdown
documents. It runs entirely on your machine — Python, FastAPI, PostgreSQL + pgvector and
Ollama — with no external SaaS.

> **Status: Phase 22 — API key authentication.** Optional API key, CORS and request size limits on top of full pipeline observability. Measured baseline: 77% of answerable questions answered correctly with citations, 88% of unanswerable ones refused (`easyrag eval`). Prompt injection is not yet defended.
> See [ROADMAP.md](ROADMAP.md) for progress.

## What it will do

1. Put Markdown files in `./docs`
2. Run `docker compose up`
3. Documents are indexed automatically and kept in sync as they change
4. Ask questions:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I deploy the application?"}'
```

```json
{
  "answer": "Build the image and run `docker compose up -d` [1].",
  "grounded": true,
  "sources": [{ "document": "deployment.md", "section": "Docker Deployment" }]
}
```

Answers are grounded strictly in your documents. When the documents don't contain the
answer, EasyRAG says it doesn't know instead of guessing.

## Principles

- **Local-first:** no data leaves your machine.
- **Grounded:** answers cite the document sections they came from, and citations are validated.
- **Incremental:** content hashing means only changed sections are re-embedded.
- **Understandable:** the RAG pipeline is plain, tested Python. There's no LangChain.
- **Measurable:** retrieval and answer quality are evaluated, not assumed.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): components, data flow, data model and design decisions
- [ROADMAP.md](ROADMAP.md): the phased development plan

## Running with Docker

```bash
cp .env.example .env                                # then set POSTGRES_PASSWORD in .env
docker compose up --build -d
curl http://localhost:8000/health/live              # {"status":"ok"}
curl http://localhost:8000/health/ready             # {"status":"ready","checks":{...all "ok"}}
docker compose down                                 # add -v to also delete the database volume
```

Database migrations are applied automatically each time the app container starts.
On first start, `ollama-pull` downloads the embedding model (`nomic-embed-text:v1.5`,
~274 MB) and the chat model (`qwen2.5:1.5b-instruct-q4_K_M`, ~1 GB). Until they're
present, readiness reports them as `"missing"`.

**Memory:** the defaults fit a 4 GB Docker VM (Docker Desktop's default on an 8 GB Mac).
Larger chat models answer better but need more memory: CPU inference needs roughly 1.6×
the model's size. See `.env.example` for alternatives.

**macOS:** Docker can't use the Apple GPU, so the bundled Ollama runs on CPU (and its image
is ~4 GB). If you have [Ollama](https://ollama.com) installed natively, it's much faster:
run `ollama pull nomic-embed-text:v1.5` and set
`EASYRAG_OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.

### Indexing your documents

Put Markdown files in `./docs` (or point `EASYRAG_DOCS_HOST_DIR` in `.env` at another
folder). **The app indexes them automatically when it starts**, in the background. If the
models are still downloading, it retries until they're ready.

Indexing is incremental:
- Unchanged files are skipped.
- Only changed sections are re-embedded.
- Deleted files are removed from the index.
- Files that can't be read right now keep their last indexed version.
- If the docs folder is ever *empty* (for example a wrong mount), the index is **not**
  wiped. The status report flags it instead.

**Changes are picked up automatically.** The app watches the docs folder, and adding,
editing or deleting a Markdown file triggers re-indexing within a second or two. Bursts of
changes (a `git pull`, a bulk copy) are grouped into one run. As a safety net, a full
incremental check also runs every 5 minutes (`EASYRAG_WATCH_RESYNC_SECONDS`). If your
docs live on a filesystem that doesn't deliver change events (some network shares), set
`EASYRAG_WATCH_FORCE_POLLING=true`.

You can also re-index on demand:

```bash
curl -X POST localhost:8000/api/v1/index            # 202 started (409 if already running)
curl localhost:8000/api/v1/index/status             # state, last report, document/chunk counts
curl localhost:8000/api/v1/documents                # what is indexed
```

Or from the command line (it waits if the API is indexing at the same moment):

```bash
docker compose run --rm app easyrag index
```

If you change the embedding model or its prefixes, the existing index is refused (vectors
from different models can't be mixed). Rebuild it with:

```bash
docker compose run --rm app easyrag index --rebuild
```

CLI exit codes: `0` success, `1` some documents failed, `2` configuration problem,
`3` embedding model unavailable. Set `EASYRAG_INDEX_ON_STARTUP=false` to index only on demand.

### Asking questions

```bash
curl -s localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"question": "How do I deploy the application with Docker?"}'
```

```json
{
  "answer": "The support response time is one business day [1].",
  "status": "answered",
  "grounded": true,
  "truncated": false,
  "sources": [
    {"document": "support.md", "title": "Support", "section": "Contact",
     "heading_path": ["Support", "Contact"], "anchor": "contact",
     "lines": [5, 5], "score": 0.71, "chunk_id": "…"}
  ]
}
```

- `[k]` in the answer refers to `sources[k-1]`. `sources` lists only the sections the answer
  is based on.
- `status` is `answered`, or `insufficient_evidence` when your documents don't answer the
  question. The answer is then `I don't know based on the provided documents.` and
  `sources` is empty.
- **Answers are verified, not just requested.** When the model doesn't cite, EasyRAG checks
  that the answer's content, including every number, comes from the sources it was shown.
  If nothing supports the answer, you get the "I don't know" reply instead. Set
  `EASYRAG_RAG_UNGROUNDED_POLICY=flag` to receive such answers marked `"grounded": false`.
- With the default small model, answers take roughly 5–35 s on CPU.
- **Known limitation:** a small model can still be manipulated by instructions written
  *inside* a document (prompt injection). Only index documents you trust.

### Searching

`POST /api/v1/retrieve` returns the most similar sections, best first. It does
retrieval only, without generation, which is useful for checking what the system would
use as evidence:

```bash
curl -s localhost:8000/api/v1/retrieve -H 'Content-Type: application/json' \
  -d '{"query": "How do I deploy the application?", "top_k": 3}'
```

```json
{
  "query": "How do I deploy the application?",
  "embedding_model": "ollama/nomic-embed-text:v1.5+p:eb869adb",
  "results": [
    {
      "document": "guides/deployment.md", "title": "Deployment Guide",
      "section": "Docker Deployment", "heading_path": ["Deployment", "Docker Deployment"],
      "anchor": "docker-deployment", "lines": [10, 12], "score": 0.742,
      "chunk_id": "…", "content": "1. Copy `.env.example` …"
    }
  ]
}
```

`top_k` (1–50, default `EASYRAG_RETRIEVAL_TOP_K`) and `min_score` (cosine similarity,
default off) are optional.

Retrieval is **hybrid** by default (`EASYRAG_RETRIEVAL_MODE`). Semantic (vector) search
finds paraphrases, keyword search finds exact terms such as `EASYRAG_WATCH_FORCE_POLLING`,
`v1.5` or an error code, and the two rankings are merged. Pass `"mode": "vector"`,
`"keyword"` or `"hybrid"` to compare them. Each result shows `vector_score` and
`keyword_score`. `score` is the ranking score of the mode used, so only compare it within
one mode.

**Optional reranking.** `EASYRAG_RERANK_ENABLED=true` adds a small cross-encoder that
re-scores the best 20 candidates and keeps the top 5. It runs on CPU, adds roughly 1 s per
question, and downloads about 80 MB once (no telemetry). It's off by default because it
showed no measurable gain over hybrid retrieval on the project's test questions. Try it on
your own documents with `"rerank": true` on `/api/v1/retrieve`. If the model isn't ready,
answers still work, just without reranking. Errors always have the shape
`{"error": {"code": "...", "message": "..."}}`.

The API is published on `127.0.0.1` only. Set `EASYRAG_PORT` in `.env` to change the host port.
PostgreSQL is not published to the host.

## Security

By default the API has no authentication and is reachable only from your own machine.
**Before you make it reachable from other machines, set an API key**, and put a reverse
proxy with TLS in front of it (the key travels in a request header):

```bash
echo "EASYRAG_API_KEY=$(openssl rand -hex 32)" >> .env
docker compose up -d
```

Then send the key with every request, as either header:

```bash
curl -s localhost:8000/api/v1/chat -H "Authorization: Bearer $EASYRAG_API_KEY" \
  -H 'Content-Type: application/json' -d '{"question": "..."}'
curl -s localhost:8000/api/v1/documents -H "X-API-Key: $EASYRAG_API_KEY"
```

- **Everything requires the key**, including `/metrics`. The exceptions are the health
  probes (`/health/live`, `/health/ready`) and the API docs (`/docs`, `/openapi.json`).
  Requests without a valid key get `401 unauthorized`.
- **Prometheus:** set `authorization: {credentials: <key>}` in the scrape config.
- **Request bodies over 64 KiB are rejected** with `413` (`EASYRAG_MAX_REQUEST_BYTES`).
- **Browser apps on other sites** need their origin listed in
  `EASYRAG_CORS_ALLOW_ORIGINS` (comma-separated). CORS is off by default.
- Without a key, the app logs a warning at startup.

## Development

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 if needed).

```bash
uv sync                                             # create .venv and install dependencies
cp .env.example .env                                # optional: local configuration
uv run uvicorn --factory easyrag.main:create_app --reload
curl http://localhost:8000/health/live              # {"status":"ok"}
```

The app needs a database even outside Docker. Start one and point the app at it:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db
export EASYRAG_DATABASE_URL=postgresql+psycopg://easyrag:<password>@127.0.0.1:5432/easyrag
```

Checks (all must pass before committing):

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest                                       # unit tests; integration tests skip
```

Integration tests run against the development database when `EASYRAG_TEST_DATABASE_URL` is set,
and end-to-end tests against a real Ollama when `EASYRAG_TEST_OLLAMA_URL` is set:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db ollama ollama-pull
EASYRAG_TEST_DATABASE_URL=postgresql+psycopg://easyrag:<password>@127.0.0.1:5432/easyrag \
EASYRAG_TEST_OLLAMA_URL=http://127.0.0.1:11434 uv run pytest
```

Database migrations (with `EASYRAG_DATABASE_URL` set):

```bash
uv run python -m easyrag.storage.migrate                              # upgrade to head
uv run alembic revision --autogenerate --rev-id 0002 -m "describe change"
```

New migrations go in `src/easyrag/storage/migrations/versions/`. Review autogenerated
migrations before committing; the integration suite fails if models and migrations differ.

Configuration is read from `EASYRAG_*` environment variables. See [.env.example](.env.example).

## Logs and metrics

- **Logs** are one JSON object per line. Set `EASYRAG_LOG_FORMAT=text` for readable
  development logs.
- **Every response carries an `X-Request-ID` header**, and every log line written while
  handling that request includes the same `request_id`. Send your own `X-Request-ID` to
  correlate with other systems.
- **Questions and document text are never logged at INFO.**

```bash
docker compose logs app | grep '"request_id": "<id>"'   # everything one request did
curl -s localhost:8000/metrics                           # Prometheus format
```

Metrics cover:
- HTTP requests (by route and status), latency, and API errors (by code);
- indexing runs, document and chunk changes, and index size;
- answers by outcome, time per stage (retrieval, generation), tokens, prompt size, and the
  best retrieval score per question;
- every call to Ollama, by endpoint and outcome (including timeouts).

To see why one answer came out as it did, add `"debug": true`. The response then includes
a `trace` with the retrieved chunk ids and scores, which chunks were given to the model and
cited, token counts and timings. It never includes document text.

```bash
curl -s localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"question": "What port do agents use?", "debug": true}' | jq .trace
```

## Evaluating quality

`easyrag eval` measures retrieval and answer quality on a labelled dataset. It indexes the
dataset into a **throwaway database** on your PostgreSQL server, so it never touches your
index. The project ships `eval/datasets/nimbus`: documentation for a *fictional* backup
tool, so a correct answer can only come from retrieval, never from the model's prior
knowledge. It has 30 questions: factual, paraphrased, identifier-style, unanswerable, and
two with a planted prompt injection.

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db ollama ollama-pull
export EASYRAG_DATABASE_URL=postgresql+psycopg://easyrag:<password>@127.0.0.1:5432/easyrag
export EASYRAG_OLLAMA_BASE_URL=http://127.0.0.1:11434
uv run easyrag eval eval/datasets/nimbus --retrieval-only   # seconds: compares 6 retrieval configs
uv run easyrag eval eval/datasets/nimbus                    # + answers with the real model (slow on CPU)
```

The report (JSON, in `eval/reports/`) covers:
- retrieval hit@1, hit@k and MRR for vector, keyword and hybrid, each with and without
  reranking;
- answer correctness (key facts present), citation precision, refusal accuracy on
  unanswerable questions, and resistance to prompt injection;
- latency p50/p95;
- a suggested similarity threshold, and a snapshot of the configuration that produced it.

To evaluate your own documents, copy the dataset layout: `corpus/` plus `questions.yaml`.

## Note on `./docs`

`./docs` is **your document corpus**. It is mounted into the container and is git-ignored.
Project documentation lives in the root-level Markdown files instead.
