# EasyRAG

A self-hosted, local-first Retrieval-Augmented Generation system for your Markdown
documents. It runs entirely on your machine — Python, FastAPI, PostgreSQL + pgvector and
Ollama — with no external SaaS.

> **Status: Phase 13 — RAG pipeline.** Ask questions with `POST /api/v1/chat`. Citations and grounding checks arrive in the next phase.
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
folder), then run:

```bash
docker compose run --rm app easyrag index
```

Re-running is safe: every document is replaced atomically. If you change the embedding
model or its prefixes, the existing index is refused (vectors from different models can't
be mixed). Rebuild it with:

```bash
docker compose run --rm app easyrag index --rebuild
```

Exit codes: `0` success, `1` some documents failed, `2` configuration problem,
`3` embedding model unavailable. Automatic and incremental indexing arrive in a later phase.

### Asking questions

```bash
curl -s localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"question": "How do I deploy the application with Docker?"}'
```

```json
{
  "answer": "1. Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD`. 2. Put your Markdown files in `./docs`. 3. Run `docker compose up -d` …",
  "truncated": false,
  "sources": [
    {"document": "guides/deployment.md", "title": "Deployment Guide", "section": "Docker Deployment",
     "heading_path": ["Deployment", "Docker Deployment"], "anchor": "docker-deployment",
     "lines": [10, 12], "score": 0.742, "chunk_id": "…"}
  ]
}
```

The model is instructed to answer only from your documents and to reply
`I don't know based on the provided documents.` otherwise. With the default small model,
answers take roughly 10–35 s on CPU. **Current limitation:** a small model can still
ignore these instructions, for example by following instructions written inside a
document. Grounding checks and citations arrive in the next phase.

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
default off) are optional. Errors always have the shape
`{"error": {"code": "...", "message": "..."}}`.

The API is published on `127.0.0.1` only. Set `EASYRAG_PORT` in `.env` to change the host port.
PostgreSQL is not published to the host.

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

## Note on `./docs`

`./docs` is **your document corpus**. It is mounted into the container and is git-ignored.
Project documentation lives in the root-level Markdown files instead.
