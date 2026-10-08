# EasyRAG Development Roadmap

EasyRAG is built in small, reviewable phases. Each phase produces roughly **one logical
commit**, and every phase must:

- implement only that phase's objective. Future phases are not started early.
- leave the repository runnable, with `pytest` and `ruff` passing.
- include tests appropriate to the phase.
- update `README.md` / `ARCHITECTURE.md` when behaviour or decisions change.

Decision references (D1–D24) point to [ARCHITECTURE.md §9](ARCHITECTURE.md#9-major-design-decisions).

---

## Milestones

| Milestone | Phases | User-visible result |
|---|---|---|
| **M1 Foundation** | 0–4 | `docker compose up` gives a healthy API backed by a migrated Postgres + pgvector |
| **M2 Indexing** | 5–9 | `easyrag index` turns `./docs` into stored chunks + embeddings |
| **M3 Answering** | 10–14 | `POST /api/v1/chat` returns grounded answers with citations |
| **M4 Living index** | 15–16 | Documents are indexed on startup and re-indexed when files change |
| **M5 Quality** | 17–19 | Hybrid retrieval, reranking, and measurable retrieval/answer quality |
| **M6 Operability** | 20–22 | Logs, metrics, traces, and authentication |
| **M7 Product** | 23–25 | Web UI, one-command deployment, docs and benchmarks |

---

## Phases

### Phase 0 — Architecture and repository design
- **Objective:** agree on architecture, boundaries, decisions and the plan. No code.
- **Files:** `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `.gitignore`
- **Tests:** none (no code)
- **Status:** ✅ done
- **Commit:** `docs: add architecture and development roadmap`

### Phase 1 — Minimal Python/FastAPI application
- **Objective:** an installable package with an app factory, typed settings and a liveness endpoint.
- **Files:** `pyproject.toml`, `uv.lock`, `.python-version`, `.env.example`, `src/easyrag/{__init__,main,config}.py`,
  `src/easyrag/api/routes/health.py`, `src/easyrag/observability/logging.py`, `tests/unit/test_health.py`, `tests/unit/test_config.py`
- **Status:** ✅ done
- **Tests:** `GET /health/live` → 200; settings load from env and reject invalid values.
- **Checks:** `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`
- **Not yet:** Docker, DB, anything RAG.
- **Commit:** `feat: add minimal FastAPI service`

### Phase 2 — Dockerize the application
- **Objective:** a reproducible container image and a compose file running only the app.
- **Files:** `Dockerfile` (multi-stage, non-root user, uv), `.dockerignore`, `docker-compose.yml`
- **Tests:** existing tests pass; manual check of `docker compose up` + `curl /health/live`.
- **Status:** ✅ done
- **Commit:** `feat: add Docker runtime`

### Phase 3 — PostgreSQL + pgvector
- **Objective:** add a database service and app connectivity, with readiness reflecting DB health.
- **Files:** `docker-compose.yml` (pgvector/pgvector image, volume, healthcheck), `storage/database.py`,
  `container.py`, `api/dependencies.py`, `api/errors.py`, readiness in `routes/health.py`, integration test scaffold.
- **Tests:** unit test for readiness with a failing DB dependency → 503; integration test pings DB and confirms the `vector` extension is available.
- **Not yet:** tables.
- **Commit:** `feat: add PostgreSQL vector database`

### Phase 4 — Schema and Alembic migrations
- **Objective:** Alembic setup; the `documents`, `chunks` (without the embedding column) and `index_state` tables; `CREATE EXTENSION vector`. (D17)
- **Files:** `alembic.ini`, `migrations/`, `storage/models.py`, container entrypoint runs migrations.
- **Tests:** integration: upgrade → downgrade → upgrade round trip; the schema matches the models (autogenerate produces an empty diff).
- **Commit:** `feat: add database schema and migrations`

### Phase 5 — Markdown document loading
- **Objective:** discover, read, normalise and hash `.md` files; parse front matter and the heading tree into sections. (D6, D8, D22)
- **Files:** `domain/documents.py`, `ingestion/loader.py`, `ingestion/markdown.py`
- **Tests:** fixtures covering nested headings, headings inside code fences, setext headings, front matter, no headings, empty files, CRLF/BOM normalisation, path traversal/symlink escape, stable sort order.
- **Not yet:** chunking, DB writes.
- **Commit:** `feat: add Markdown document loading`

### Phase 6 — Deterministic chunking
- **Objective:** section-bounded, block-aware chunking with overlap, contextual embed text and deterministic IDs. (D7, D9)
- **Files:** `domain/chunks.py`, `ingestion/chunker.py`
- **Tests:** same input → identical chunks/IDs; no chunk crosses a section; code fences and tables stay intact; oversize-block fallback; overlap behaviour; edits to one section change only that section's chunk IDs.
- **Commit:** `feat: add deterministic document chunking`

### Phase 7 — Embedding provider abstraction
- **Objective:** the `EmbeddingProvider` Protocol plus a deterministic fake, and the domain errors.
- **Files:** `embeddings/base.py`, `embeddings/fake.py`
- **Tests:** the fake is deterministic, has the correct dimension, and its vectors are normalised; a conformance test suite usable by every provider.
- **Commit:** `feat: add embedding provider abstraction`

### Phase 8 — Ollama embeddings
- **Objective:** an httpx-based Ollama adapter (batching, timeouts, retries, query/document prefixes); an Ollama compose service with a model-pull step; readiness checks the model. (D3, D19)
- **Files:** `embeddings/ollama.py`, `docker-compose.yml`, `container.py`, README notes on host-Ollama for macOS.
- **Tests:** `respx`-mocked happy path, batching, errors and timeouts; the conformance suite; opt-in e2e against real Ollama.
- **Decides:** the default embedding model + dimension.
- **Commit:** `feat: add Ollama embedding provider`

### Phase 9 — Store chunks and embeddings in pgvector
- **Objective:** a migration adding the `embedding` + `embedding_model` columns and the HNSW index; `VectorStore` + `PgVectorStore`; `DocumentRepository`; a basic full (non-incremental) indexer; `easyrag index` CLI; startup model/dimension guard. (D10 atomicity, D11)
- **Files:** `migrations/versions/…`, `storage/{repositories,vector_store}.py`, `ingestion/indexer.py`, `cli.py`
- **Tests:** integration: indexing a fixture corpus stores the expected rows; re-running yields the same rows; dimension mismatch refuses to start.
- **Commit:** `feat: store document chunks and embeddings`

### Phase 10 — Vector similarity retrieval
- **Objective:** the `Retriever` Protocol + `VectorRetriever`, with configurable top-k, an optional min score and deterministic tie-breaking; a `POST /api/v1/retrieve` debug endpoint.
- **Files:** `retrieval/{base,vector}.py`, `api/schemas.py`, `api/routes/retrieve.py`
- **Tests:** unit with an in-memory store; integration: a known query returns the expected section first; the threshold filters results.
- **Commit:** `feat: add vector similarity retrieval`

### Phase 11 — LLM provider abstraction
- **Objective:** the `LLMProvider` Protocol, `ChatMessage` / `GenerationOptions` / `Generation` (text + token usage + timing), and a scripted fake.
- **Files:** `llm/base.py`, `llm/fake.py`
- **Tests:** fake behaviour; conformance suite.
- **Commit:** `feat: add LLM provider abstraction`

### Phase 12 — Ollama LLM generation
- **Objective:** an Ollama `/api/chat` adapter with explicit `num_ctx`, temperature, seed, timeouts and token-usage extraction; readiness checks the chat model. (D14)
- **Files:** `llm/ollama.py`, `docker-compose.yml` (model pull), `container.py`
- **Tests:** `respx` mocks for options passthrough, usage parsing and error mapping; opt-in e2e.
- **Decides:** the default chat model.
- **Commit:** `feat: add Ollama LLM provider`

### Phase 13 — RAG orchestration pipeline
- **Objective:** a `RagService` (retrieve → prompt → generate), a versioned grounded prompt with a token budget, `POST /api/v1/chat` returning the answer + retrieved sources, and a `RagTrace` data shape. Grounding instructions are part of the baseline prompt. (D12.2)
- **Files:** `domain/answers.py`, `rag/{prompt,service}.py`, `api/routes/chat.py`, a boundary (import) test
- **Tests:** unit with fakes: prompt snapshot test; context respects the budget; errors map to 503.
- **Commit:** `feat: add RAG pipeline`

### Phase 14 — Grounded answers and citations
- **Objective:** `[n]` citation markers, a parser/validator, cited-only sources, the `grounded` flag, the evidence gate that skips the LLM, and an exact refusal sentence. (D12, D13)
- **Files:** `rag/citations.py`, `rag/prompt.py`, `rag/service.py`, schemas
- **Tests:** invalid or out-of-range markers dropped; duplicate markers; refusal detection; the gate bypasses the LLM (the fake asserts it was not called).
- **Commit:** `feat: add grounded answers with source citations`

### Phase 15 — Automatic, incremental indexing
- **Objective:** hash + fingerprint diffing, embedding reuse, deletion of removed documents, an advisory lock, indexing on startup as a background task, and the index status/documents/trigger endpoints. (D8–D10)
- **Files:** `ingestion/indexer.py`, `main.py` lifespan, `api/routes/index.py`
- **Tests:** integration: an unchanged second run makes 0 embed calls; editing one section re-embeds only the changed chunks; delete/rename; a fingerprint change triggers a full re-index; concurrent runs serialise.
- **Commit:** `feat: add incremental document indexing`

### Phase 16 — Filesystem change detection
- **Objective:** a `watchfiles` watcher with debounce, a polling fallback and periodic re-sync, all feeding the same indexer. (D20)
- **Files:** `ingestion/watcher.py`, lifespan wiring, config
- **Tests:** create/modify/delete triggers a sync (temporary directory); debounce coalesces bursts; clean shutdown.
- **Commit:** `feat: add filesystem change detection`

### Phase 17 — Hybrid retrieval
- **Objective:** a generated `tsvector` column + GIN index (migration), `KeywordRetriever`, and `HybridRetriever` with Reciprocal Rank Fusion; the retrieval mode is configurable. (D15)
- **Tests:** an exact-identifier query (e.g. an env var name) is found by keyword search where vector search misses it; RRF ordering is deterministic.
- **Commit:** `feat: add hybrid vector and keyword retrieval`

### Phase 18 — Reranking
- **Objective:** the `Reranker` Protocol, a no-op default and one real implementation (chosen per D16), retrieving candidates with `top_k` and keeping the best `top_n`.
- **Tests:** unit on ordering and truncation; e2e opt-in.
- **Commit:** `feat: add reranking stage`

### Phase 19 — Evaluation
- **Objective:** a dataset format (question, expected document/section, answerable flag, key facts), metrics (recall@k, MRR, citation precision/recall, refusal accuracy, latency p50/p95) and an `easyrag eval` CLI writing a JSON report; calibrate the default similarity threshold.
- **Files:** `evaluation/`, `eval/datasets/`
- **Commit:** `feat: add RAG evaluation framework`

### Phase 20 — Structured logging and metrics
- **Objective:** JSON logs, request ID middleware, a Prometheus `/metrics` endpoint, and indexing counters.
- **Commit:** `feat: add structured logging and metrics`

### Phase 21 — Latency/token/retrieval observability
- **Objective:** export `RagTrace` data: per-stage histograms, token counters, retrieval score distributions; optional `debug` trace in the chat response.
- **Commit:** `feat: add RAG pipeline observability`

### Phase 22 — Authentication and self-hosting config
- **Objective:** optional API key (constant-time compare), CORS config, request size limits, safe-defaults review.
- **Commit:** `feat: add API key authentication`

### Phase 23 — Minimal web UI
- **Objective:** static HTML + vanilla JS chat page with clickable sources, served by FastAPI. (D24)
- **Commit:** `feat: add minimal web UI`

### Phase 24 — End-user Docker Compose
- **Objective:** one-command startup, automatic model pull, a host-Ollama profile, sensible resource limits, `.env.example` polish, and upgrade notes.
- **Commit:** `feat: streamline Docker Compose deployment`

### Phase 25 — Documentation, examples, benchmarks, hardening
- **Objective:** an `examples/docs` corpus, a benchmark report from the eval suite, a troubleshooting guide, and a security and failure-mode review.
- **Commit:** `docs: add examples, benchmarks and operations guide`

---

## Suggested adjustments to the original roadmap

1. **Phase 4 omits the embedding column**, and Phase 9 adds it in its own migration.
   The vector dimension isn't known until a model is chosen (Phase 8), and this keeps
   Phase 4 from guessing the future.
2. **Phase 10 adds a retrieval-only endpoint.** This makes retrieval testable and
   debuggable before any LLM exists, and evaluation (Phase 19) reuses it.
3. **Basic grounding instructions arrive in Phase 13, not 14.** The requirement that the
   LLM is told not to invent information should hold from the first time an LLM answers
   a question. Phase 14 adds citations, validation and the evidence gate.
