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
  `docker-compose.dev.yml`, `container.py`, `api/dependencies.py`, readiness in `routes/health.py`, integration test scaffold.
- **Tests:** unit test for readiness with a failing DB dependency → 503; integration test pings DB and confirms the `vector` extension is available.
- **Not yet:** tables; `CREATE EXTENSION` (Phase 4); `api/errors.py` (moved to Phase 10, the first endpoint that calls a provider).
- **Status:** ✅ done
- **Commit:** `feat: add PostgreSQL vector database`

### Phase 4 — Schema and Alembic migrations
- **Objective:** Alembic setup; the `documents`, `chunks` (without the embedding column) and `index_state` tables; `CREATE EXTENSION vector`. (D17)
- **Files:** `alembic.ini`, `docker/entrypoint.sh`, `storage/{models,migrate}.py`, `storage/migrations/` (in-package), schema check in readiness.
- **Tests:** integration (fresh database per test): upgrade → downgrade → upgrade round trip; idempotent upgrade; concurrent migration processes; the schema matches the models (autogenerate produces an empty diff); constraints and cascade; readiness requires the current schema. Unit: model metadata, single migration head, readiness with schema outdated or timing out.
- **Status:** ✅ done
- **Commit:** `feat: add database schema and migrations`

### Phase 5 — Markdown document loading
- **Objective:** discover, read, normalise and hash `.md` files; parse front matter and the heading tree into sections. (D6, D8, D22)
- **Files:** `domain/documents.py`, `ingestion/loader.py`, `ingestion/markdown.py`
- **Tests:** fixtures covering nested headings, headings inside code fences, setext headings, front matter, no headings, empty files, CRLF/BOM normalisation, path traversal/symlink escape, stable sort order.
- **Not yet:** chunking, DB writes, `EASYRAG_DOCS_DIR` + the `./docs` mount (Phase 9, when indexing first reads the directory).
- **Status:** ✅ done
- **Commit:** `feat: add Markdown document loading`

### Phase 6 — Deterministic chunking
- **Objective:** section-bounded, block-aware chunking with overlap, contextual embed text and deterministic IDs. (D7, D9)
- **Files:** `domain/chunks.py`, `tokens.py`, `ingestion/chunker.py`; `Section.body_start_line` added to the Phase 5 parser.
- **Tests:** same input → identical chunks/IDs; no chunk crosses a section; code fences and tables stay intact; oversize-block fallback; overlap behaviour; edits to one section change only that section's chunk IDs.
- **Not yet:** `EASYRAG_CHUNK_*` settings (Phase 9, where the indexer first constructs a chunker).
- **Status:** ✅ done
- **Commit:** `feat: add deterministic document chunking`

### Phase 7 — Embedding provider abstraction
- **Objective:** the `EmbeddingProvider` Protocol plus a deterministic fake, and the domain errors.
- **Files:** `embeddings/base.py`, `embeddings/fake.py`, `tests/contracts/embedding_provider.py`
- **Tests:** a reusable contract suite (shape, unit norm, order, batching invariance, determinism, empty input, Unicode), checked against deliberately broken providers; fake-specific behaviour (lexical similarity, stability across instances, call log); validation helpers.
- **Not yet:** settings and container wiring (Phase 8, with the first real provider).
- **Status:** ✅ done
- **Commit:** `feat: add embedding provider abstraction`

### Phase 8 — Ollama embeddings
- **Objective:** an httpx-based Ollama adapter (batching, timeouts, retries, query/document prefixes); an Ollama compose service with a model-pull step; readiness checks the model. (D3, D19)
- **Files:** `embeddings/ollama.py`, `health.py` (readiness registry), `config.py`, `container.py`, `main.py` (`container_factory`), compose (`ollama`, `ollama-pull`), `tests/e2e/`, `tests/unit/test_dependencies.py`.
- **Tests:** an in-process fake Ollama (`httpx2.MockTransport`): request shape and prefixes, batching, validation, retry/no-retry matrix, readiness states, and the contract suite. Readiness registry (concurrency, timeouts, no credential leaks). Packaging guard. Opt-in e2e: contract + semantic sanity against real Ollama.
- **Not yet:** `api/errors.py` (Phase 10, the first endpoint that calls a provider).
- **Status:** ✅ done
- **Decides:** the default embedding model + dimension.
- **Commit:** `feat: add Ollama embedding provider`

### Phase 9 — Store chunks and embeddings in pgvector
- **Objective:** a migration adding the `embedding` + `embedding_model` columns; the dimension-specific HNSW index; an `IndexStore` port + `PostgresIndexStore`; a basic full (non-incremental) indexer; `easyrag index [--rebuild]` CLI; `EASYRAG_DOCS_DIR` + read-only `./docs` mount in compose; model/dimension guard. (D10 atomicity, D11)
- **Files:** `storage/migrations/versions/0002_chunk_embeddings.py`, `storage/index_store.py`, `ingestion/indexer.py`, `cli.py`; `PARSER_VERSION` / `CHUNKER_VERSION`; prefix-aware `model_id`.
- **Tests:** unit, with an in-memory store and the fake embedder: what is stored, idempotency, incompatible-index refusal, rebuild, document-level failure vs. provider outage, skipped files, fingerprint sensitivity. Integration, with real Postgres: rows and vectors (float32 tolerance), identical re-runs, replacement without orphans, model/dimension guard and readiness, rebuild switching the vector index, `EXPLAIN` showing the HNSW index in use.
- **Not yet (Phase 15):** skipping unchanged documents, embedding reuse (and its `(content_hash, embedding_model)` index), deleting removed files, the advisory lock, indexing on startup.
- **Status:** ✅ done
- **Commit:** `feat: store document chunks and embeddings`

### Phase 10 — Vector similarity retrieval
- **Objective:** the `Retriever` Protocol + `VectorRetriever`, with configurable top-k, an optional min score and deterministic tie-breaking; a `POST /api/v1/retrieve` debug endpoint.
- **Files:** `retrieval/{base,vector}.py`, `storage/vector_store.py`, `domain/errors.py`, `RetrievedChunk`, `api/{schemas,errors}.py`, `api/routes/retrieve.py`, retrieval settings.
- **Tests:** unit: retriever (query embedding, model/dimension passthrough, overrides, validation); API (response shape, validation, every error mapping, no DSN leak on database errors, generic 500). Integration, with real pgvector: best section first, cosine scores in order, top-k, threshold applied after top-k, tie-breaking by chunk ID, empty index, cross-model refusal, `EXPLAIN` of the production query uses HNSW.
- **Status:** ✅ done
- **Commit:** `feat: add vector similarity retrieval`

### Phase 11 — LLM provider abstraction
- **Objective:** the `LLMProvider` Protocol (with `context_window`), `ChatMessage` / `GenerationOptions` / `Generation` (text, `finish_reason`, optional token usage, timing), and a scriptable fake.
- **Files:** `llm/base.py`, `llm/fake.py`, `tests/contracts/llm_provider.py`
- **Tests:** a contract suite (well-formed result, determinism, `length` on output limit, Unicode, invalid conversations), shown to catch three deliberately broken providers. Fake behaviour: script order, scripted exceptions, responder, call log, truncation, stop sequences, token estimates. Option validation.
- **Not yet:** settings and container wiring (Phase 12, with the Ollama adapter).
- **Status:** ✅ done
- **Commit:** `feat: add LLM provider abstraction`

### Phase 12 — Ollama LLM generation
- **Objective:** an Ollama `/api/chat` adapter with explicit `num_ctx`, temperature, seed, timeouts and token-usage extraction; readiness checks the chat model. (D14)
- **Files:** `llm/ollama.py`, `ollama.py` (shared HTTP plumbing, extracted from the embedding adapter), `config.py`, `container.py`, `docker-compose.yml` (pulls both models), `.env.example`, e2e tests.
- **Tests:** fake Ollama chat (`httpx2.MockTransport`): exact request (`num_ctx` always), option mapping, prompt-overflow refusal, response mapping, `done_reason`, invalid responses, error/retry matrix, readiness states, and the contract suite. The embedding adapter's 34 unchanged tests guard the refactor. E2e with the real model: contract + "answers from supplied context".
- **Decided:** default chat model `qwen2.5:1.5b-instruct-q4_K_M` (D28; the 4B candidate was OOM-killed in a 4 GB Docker VM).
- **Not yet (Phase 13):** generation settings (`temperature`, `seed`, `max_output_tokens`) and the HTTP mapping of LLM errors.
- **Status:** ✅ done
- **Commit:** `feat: add Ollama LLM provider`

### Phase 13 — RAG orchestration pipeline
- **Objective:** a `RagService` (retrieve → prompt → generate), a versioned grounded prompt with a token budget, `POST /api/v1/chat` returning the answer + retrieved sources, and a `RagTrace` data shape. Grounding instructions are part of the baseline prompt. (D12.2)
- **Files:** `domain/answers.py`, `rag/{prompt,service}.py`, `api/routes/chat.py`, `api/{schemas,errors}.py` (LLM errors → 503/502), generation and evidence-budget settings, `tests/unit/test_architecture.py`.
- **Tests:** prompt (version fingerprint guard, exact snapshot, rules restated after sources, budget in rank order with skipping, window-limited budget, delimiter and attribute escaping); service with fakes (options passthrough, sources = what the model saw, truncation flag, trace, no content in INFO logs, error propagation); chat API (shape, validation, error mapping); layer-boundary test; integration over real pgvector with a fake model reporting the sources it received. Guards shown to catch a forbidden import and an unversioned prompt edit.
- **Live result:** prompt v2 (D29) fixed out-of-scope answering; prompt injection still succeeds with the 1.5B model (known limitation, see D29).
- **Status:** ✅ done
- **Commit:** `feat: add RAG pipeline`

### Phase 14 — Grounded answers and citations
- **Objective:** `[n]` citation markers, a parser/validator, cited-only sources, the `grounded` flag, the evidence gate that skips the LLM, and an exact refusal sentence. (D12, D13)
- **Files:** `rag/citations.py`, `rag/attribution.py` (server-side support check, added after measurement), `rag/prompt.py` (v4: citation rule + worked examples + `prompt_fingerprint()`), `rag/service.py`, `domain/answers.py` (`Outcome`, status), schemas, `EASYRAG_RAG_UNGROUNDED_POLICY`.
- **Tests:** citation parsing (forms, renumbering, invalid/duplicate markers, code like `items[0]`); refusal detection (variants, buried refusals); attribution (copied, multi-source, unsupported, coverage threshold, numbers must match, stopwords); service (every outcome, gate bypasses the LLM, both policies, explicit over inferred); API (`status`/`grounded`); integration (empty index refuses without calling the model); fingerprint guard mutation-tested on an example edit.
- **Done from Phase 13's carry-over:** refusals return no sources; uncited answers are verified or refused.
- **Live result (D30):** with the 1.5B model, 6/6 answerable questions answered with correct sources (via inference where the model didn't cite); unanswerable questions refused. Prompt injection remains a known limitation.
- **Status:** ✅ done
- **Commit:** `feat: add grounded answers with source citations`

### Phase 15 — Automatic, incremental indexing
- **Objective:** hash + fingerprint diffing, embedding reuse by **embedding-input hash** (corrected from `content_hash`, see D9), deletion of removed documents (never of files the loader merely *skipped*, never everything because the docs directory came back empty), an advisory lock, indexing on startup as a background task with retries, and the index status/documents/trigger endpoints. (D8–D10, D31)
- **Files:** `ingestion/{indexer,coordinator}.py`, `storage/index_store.py`, migration `0003_embedding_input_hash.py`, `api/routes/index.py`, `main.py` lifespan, `cli.py` report, `EASYRAG_INDEX_ON_STARTUP`.
- **Tests:** unit (in-memory store): unchanged run makes 0 embed calls and 0 writes; one edited section re-embeds 1 chunk; heading change gets a new vector; delete; rename reuses vectors; fingerprint change re-indexes all; skipped file kept; empty-directory guard; rebuild; failures keep the previous version; lock released on failure. Coordinator: backoff schedule, stop on incompatible index, 409 while running, cancellation. API: status/trigger/documents. Integration (real Postgres): zero rewrites on unchanged runs, reuse via input hash, cascade deletion, **concurrent runs serialise**, read models.
- **Status:** ✅ done
- **Commit:** `feat: add incremental document indexing`

### Phase 16 — Filesystem change detection
- **Builds on:** `IndexingCoordinator` (D31). The watcher should *request* a run, coalescing with one already in progress, rather than call the indexer directly.
- **Objective:** a `watchfiles` watcher with debounce, a polling fallback and periodic re-sync, all feeding the same indexer. (D20)
- **Files:** `easyrag/watcher.py` (originally `ingestion/watcher.py`; moved in Phase 18), lifespan wiring, config
- **Tests:** create/modify/delete triggers a sync (temporary directory); debounce coalesces bursts; clean shutdown.
- **Done:** the watcher, coalescing `IndexingCoordinator.request_run()` with `pending` status, watcher state in `/index/status`, `watchfiles` declared as a runtime dependency, settings. Tests: watcher against real temporary directories in **both native-event and polling modes** (create/modify/delete, burst debounce, filtered files, catch-up request, periodic resync, late-appearing directory, prompt clean shutdown; run 6× to check for flakiness); coordinator coalescing (10 requests → 1 follow-up, follow-up after startup retries, no loop on failure). Live: host edits re-indexed in about 1 s through the macOS bind mount; 12-file burst → 1 run.
- **Status:** ✅ done
- **Commit:** `feat: add filesystem change detection`

### Phase 17 — Hybrid retrieval
- **Objective:** a generated `tsvector` column + GIN index (migration), `KeywordRetriever`, and `HybridRetriever` with Reciprocal Rank Fusion; the retrieval mode is configurable. (D15)
- **Tests:** an exact-identifier query (e.g. an env var name) is found by keyword search where vector search misses it; RRF ordering is deterministic.
- **Done (D32):** migration 0004 (`search_vector` with simple + english lexemes, GIN, backfill); `stopwords.py` shared with attribution; `retrieval/{keyword,hybrid}.py`; `storage/keyword_store.py` (BM25-IDF coverage scoring); `EASYRAG_RETRIEVAL_MODE` and per-request `mode`; `vector_score`/`keyword_score` on results. Tests: term extraction; RRF arithmetic, tie-breaking, candidate widening, `min_score` scope; real-Postgres keyword search (identifier phrases, **IDF beats frequency**, mutation-checked; natural questions aren't AND queries; GIN index used; backfill on migration); hybrid over real stores. Live: 12-query comparison chose hybrid (MRR 0.86, hit@3 1.00).
- **Status:** ✅ done
- **Commit:** `feat: add hybrid vector and keyword retrieval`

### Phase 18 — Reranking
- **Objective:** the `Reranker` Protocol, a no-op default and one real implementation (chosen per D16), retrieving candidates with `top_k` and keeping the best `top_n`.
- **Tests:** unit on ordering and truncation; e2e opt-in.
- **Done (D33):** `retrieval/rerank.py` (`Reranker` port, `RerankingRetriever` with graceful fallback, `NoOpReranker`); `rerankers/fastembed.py` (ONNX cross-encoder, background load, worker-thread inference, readiness); settings and per-request `rerank`; Docker `models` volume, `HF_HOME`, telemetry off (Hugging Face and onnxruntime); `fastembed` + `onnxruntime` declared (the packaging guard caught the latter); watcher moved to `easyrag/watcher.py`. Tests: wrapper (candidate widening, top-k, fallback, validation); adapter with a fake encoder (sigmoid scores, ordering, ties, header text, load states, inference failure); opt-in e2e with the real model. Live: works in Docker on the read-only filesystem after the `HF_HOME` fix; measured no gain over hybrid on the saturated test set, so **off by default**.
- **Status:** ✅ done
- **Commit:** `feat: add reranking stage`

### Phase 19 — Evaluation
- **Objective:** a dataset format (question, expected document/section, answerable flag, key facts), metrics (recall@k, MRR, citation precision/recall, refusal accuracy, latency p50/p95) and an `easyrag eval` CLI writing a JSON report; calibrate the default similarity threshold.
- **Files:** `evaluation/`, `eval/datasets/`
- **Must include (from Phases 13–18):** re-check the default retrieval mode, RRF and **reranking on/off** on a clean, non-saturated corpus (D32, D33); faithfulness beyond lexical support; injection cases; outcome and attribution rates (explicit vs inferred); calibration of the attribution thresholds (D30) alongside the similarity threshold; a corpus free of evaluation notes.
- **Done (D34):** `evaluation/{dataset,metrics,runner,report}.py`; `easyrag eval DATASET [--retrieval-only] [--output] [--keep-database]`; the fictional `eval/datasets/nimbus` dataset (7 documents, 30 questions, labels validated against the corpus); isolated throwaway database per run. Tests: dataset validation (including the shipped dataset), metrics, answer scoring (correctness, injection leaks, refusals), error isolation; integration: isolated database created/migrated/dropped (also on failure), retrieval evaluation on the Nimbus corpus. Live baseline: answerable 77.3% correct, refusal 87.5%, injection 0%, p50 3.0 s. Hybrid default, rerank off and threshold off were all confirmed by data.
- **Status:** ✅ done
- **Commit:** `feat: add RAG evaluation framework`

### Phase 20 — Structured logging and metrics
- **Objective:** JSON logs, request ID middleware, a Prometheus `/metrics` endpoint, and indexing counters.
- **Done:** `observability/{context,logging,middleware,metrics}.py`, `/metrics` route, error counting, coordinator observers with `index_run_id`, `EASYRAG_LOG_FORMAT`, uvicorn `--no-access-log`; `prometheus-client` and `starlette` declared (the packaging guard caught the direct `starlette` import). Tests: JSON formatter (fields, extras, odd values, exceptions, context IDs, text mode); request ID generation/acceptance/rejection and propagation into pipeline logs; probes at DEBUG; metrics by route template (no raw-path labels); error counting; index metrics and independent registries; coordinator observers, per-run IDs and failure isolation. Live in Docker: every container log line is JSON; one chat request's 4 log lines share its ID; indexing metrics match the corpus (7 documents, 22 chunks).
- **Status:** ✅ done
- **Commit:** `feat: add structured logging and metrics`

### Phase 21 — Latency/token/retrieval observability
- **Objective:** export `RagTrace` data: per-stage histograms, token counters, retrieval score distributions; optional `debug` trace in the chat response.
- **Done:** `RagService(on_answer=...)` observer wired to `Metrics.record_answer` (answers by outcome, stage durations, tokens, prompt-size and chunk-count histograms, top score by retrieval config, attribution, invalid citations, truncation); `observability/transport.py` times every Ollama call by endpoint and outcome; `RagTrace.reranked`; structured `rag` fields on the trace log line; `"debug": true` returns `trace` from `/api/v1/chat` (D35). Tests: metric recording (incl. RRF-scale buckets, no-evidence and estimate fallback), observer called and isolated, reranked flag, structured log without question text, transport outcomes (status, connect error, timeout, `other`), trace omitted by default and returned on request without text; two mutation checks. Live Docker check still to run (see the phase notes).
- **Status:** ✅ done (live check pending)
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
- **Carried from Phase 19 (D34):** verify *explicit* citations too (numbers and support), since a wrong number with a wrong citation passed; treat prompt injection as a corpus-level risk (it hijacked a non-injection question); re-run `easyrag eval` and compare with the D34 baseline after every change.
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
