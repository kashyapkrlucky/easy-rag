# EasyRAG

A self-hosted, local-first Retrieval-Augmented Generation system for your Markdown
documents. It runs entirely on your machine — Python, FastAPI, PostgreSQL + pgvector and
Ollama — with no external SaaS.

> **Status: Phase 1 — minimal FastAPI service.** Only a liveness endpoint exists so far.
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

## Development

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 if needed).

```bash
uv sync                                             # create .venv and install dependencies
cp .env.example .env                                # optional: local configuration
uv run uvicorn --factory easyrag.main:create_app --reload
curl http://localhost:8000/health/live              # {"status":"ok"}
```

Checks (all must pass before committing):

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Configuration is read from `EASYRAG_*` environment variables. See [.env.example](.env.example).

## Note on `./docs`

`./docs` is **your document corpus**. It is mounted into the container and is git-ignored.
Project documentation lives in the root-level Markdown files instead.
