# EasyRAG Architecture

> Status: **Phase 0 — design only.** Nothing described here is implemented yet.
> This document is the reference that every later phase is built against. When a
> phase changes a decision recorded here, it updates this document in the same commit.

---

## 1. Purpose and scope

EasyRAG is a self-hosted, local-first Retrieval-Augmented Generation (RAG) system.

```
put Markdown in ./docs  →  docker compose up  →  POST /api/v1/chat  →  grounded answer + citations
```

### Goals

1. Index a folder of Markdown documents automatically, incrementally and idempotently.
2. Answer questions **only** from that content, and say "I don't know" when the evidence is insufficient.
3. Return citations that point to the exact document and section used.
4. Run entirely on the user's machine: no SaaS, no API keys for third-party services.
5. Keep the pipeline understandable: every stage is our own code and can be tested, measured and replaced.

### Non-goals (for now)

- Multi-tenant or multi-user permission models
- Formats other than Markdown (the loader seam allows them later)
- Conversation memory / multi-turn chat
- Agents, tool calling, LangChain / LlamaIndex
- Microservices, message queues, Kubernetes
- Response streaming (may come after Phase 25)

---

## 2. System context

```
                ┌───────────────────────────── docker compose ─────────────────────────────┐
                │                                                                           │
  ./docs  ──────┼──► (bind mount, read-only)                                                │
  (*.md)        │          │                                                                │
                │   ┌──────▼──────────────────────┐         ┌───────────────────────────┐   │
  HTTP client ──┼──►│  easyrag (FastAPI + indexer)│────────►│ PostgreSQL 18 + pgvector  │   │
  / Web UI      │   │  single Python process      │  SQL    │ documents, chunks,        │   │
                │   └──────────────┬──────────────┘         │ embeddings, index state   │   │
                │                  │ HTTP                    └───────────────────────────┘   │
                │   ┌──────────────▼──────────────┐                                         │
                │   │ Ollama                      │  ◄── or the host's Ollama               │
                │   │ embedding model + chat model│      (recommended on macOS for GPU)     │
                │   └─────────────────────────────┘                                         │
                └───────────────────────────────────────────────────────────────────────────┘
```

Three runtime containers: **app**, **postgres**, **ollama**. That is the whole
infrastructure footprint for the lifetime of the project.

---

## 3. Layered architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Interface layer         api/ (FastAPI routes, schemas)   cli.py   web/   │
├──────────────────────────────────────────────────────────────────────────┤
│ Application services    rag/RagService        ingestion/IndexingService  │
│ (orchestration)         evaluation/EvalRunner                            │
├──────────────────────────────────────────────────────────────────────────┤
│ Pipeline components     loader · markdown parser · chunker · retriever   │
│                         reranker · prompt builder · citation parser      │
├──────────────────────────────────────────────────────────────────────────┤
│ Ports (Protocols)       DocumentLoader  EmbeddingProvider  VectorStore   │
│                         Retriever  Reranker  LLMProvider                 │
├──────────────────────────────────────────────────────────────────────────┤
│ Adapters                FilesystemMarkdownLoader  OllamaEmbeddingProvider│
│                         PgVectorStore  OllamaLLMProvider                 │
├──────────────────────────────────────────────────────────────────────────┤
│ Foundation              domain/ (pure types)  config  observability      │
│                         storage/ (SQLAlchemy engine, ORM, migrations)    │
└──────────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** arrows point downward only.

- `domain/` imports nothing from the project.
- Pipeline components depend on `domain/` and on Protocols, never on adapters.
- Only the composition root (`container.py`) knows which adapter implements which port.
- The words "Ollama", "httpx" and "pgvector" appear only inside their adapter modules
  (and in config/compose). `rag/` and `ingestion/` must not import them. Phase 13+
  adds an import-linter style test to enforce this.

---

## 4. Component boundaries

| Concern | Module | Responsibility | Must NOT |
|---|---|---|---|
| Configuration | `config.py` | Typed settings from env vars (`EASYRAG_*`), validation, defaults | Read env vars anywhere else |
| Domain model | `domain/` | Immutable dataclasses: `SourceDocument`, `ParsedDocument`, `Section`, `Chunk`, `RetrievedChunk`, `Citation`, `Answer` | Import SQLAlchemy, FastAPI, httpx |
| Document ingestion | `ingestion/loader.py` | Discover `*.md` files under the docs root, read them, normalise text, compute content hash | Parse Markdown |
| Markdown parsing | `ingestion/markdown.py` | Front matter + heading tree → ordered `Section`s with heading path, anchor, line range | Decide chunk sizes |
| Chunking | `ingestion/chunker.py` | Deterministic split of sections into `Chunk`s with stable IDs | Call any model |
| Indexing | `ingestion/indexer.py` | Diff filesystem vs DB, embed new/changed chunks, upsert, delete removed docs | Know about HTTP |
| Embeddings | `embeddings/` | `EmbeddingProvider` port + Ollama adapter (+ deterministic fake for tests) | Know about chunks or the DB |
| Vector storage | `storage/` | Engine/session, ORM models, `VectorStore` port + pgvector adapter, document repository | Contain ranking logic beyond the SQL query |
| Retrieval | `retrieval/` | `Retriever` port; vector, keyword, hybrid (RRF) retrievers; threshold filtering | Build prompts |
| Reranking | `retrieval/rerank.py` | `Reranker` port; no-op default; real reranker in Phase 18 | — |
| Prompt construction | `rag/prompt.py` | Versioned grounded prompt template, numbered context blocks, context budget | Call the LLM |
| LLM generation | `llm/` | `LLMProvider` port + Ollama adapter (+ scripted fake for tests) | Know about retrieval |
| RAG orchestration | `rag/service.py` | question → retrieve → gate → prompt → generate → parse citations → `Answer` + trace | Know about FastAPI |
| Citations | `rag/citations.py` | Parse `[n]` markers, validate against context, map to sources | — |
| API | `api/` | HTTP schemas, routes, error mapping, dependency wiring | Contain business logic |
| Observability | `observability/` | Logging setup, request IDs, metrics, per-stage timing | — |
| Evaluation | `evaluation/` | Dataset format, retrieval + answer metrics, CLI runner | Run inside the API process |

---

## 5. Repository structure (target)

The `(Pn)` tag shows the phase that introduces each file. Nothing is created before its phase.

```
easy-rag/
├── README.md                         (P0)  user-facing overview & quickstart
├── ARCHITECTURE.md                   (P0)  this document
├── ROADMAP.md                        (P0)  phase plan & definition of done
├── .gitignore                        (P0)
├── pyproject.toml                    (P1)  deps, ruff, mypy, pytest config
├── uv.lock                           (P1)
├── .env.example                      (P1)  every supported EASYRAG_* variable
├── Dockerfile                        (P2)
├── .dockerignore                     (P2)
├── docker-compose.yml                (P2, extended P3/P8/P24)
├── alembic.ini                       (P4)  developer Alembic CLI only
├── docker/entrypoint.sh              (P4)  migrate, then exec the server
├── docs/                             user corpus — git-ignored, mounted read-only
├── examples/docs/                    (P25) sample corpus for demos & tests
├── eval/datasets/                    (P19) question/answer/source datasets
├── src/easyrag/
│   ├── __init__.py                   (P1)
│   ├── main.py                       (P1)  app factory + lifespan
│   ├── config.py                     (P1)  Settings (pydantic-settings)
│   ├── container.py                  (P3)  composition root: builds adapters/services
│   ├── cli.py                        (P9)  `easyrag index | eval | ...` (argparse)
│   ├── domain/
│   │   ├── documents.py              (P5)  SourceDocument, ParsedDocument, Section
│   │   ├── chunks.py                 (P6)  Chunk, RetrievedChunk
│   │   └── answers.py                (P13) Answer, Citation, RagTrace
│   ├── api/
│   │   ├── dependencies.py           (P3)
│   │   ├── errors.py                 (P8)  domain exception → HTTP mapping
│   │   ├── schemas.py                (P10)
│   │   └── routes/
│   │       ├── health.py             (P1, readiness in P3)
│   │       ├── retrieve.py           (P10) debug retrieval endpoint
│   │       ├── chat.py               (P13)
│   │       └── index.py              (P15) status / trigger reindex
│   ├── ingestion/
│   │   ├── loader.py                 (P5)
│   │   ├── markdown.py               (P5)
│   │   ├── chunker.py                (P6)
│   │   ├── indexer.py                (P9, incremental in P15)
│   │   └── watcher.py                (P16)
│   ├── embeddings/
│   │   ├── base.py                   (P7)  EmbeddingProvider Protocol
│   │   ├── fake.py                   (P7)  deterministic test provider
│   │   └── ollama.py                 (P8)
│   ├── storage/
│   │   ├── database.py               (P3)  engine, session factory, ping
│   │   ├── models.py                 (P4)  ORM tables (source of truth for the schema)
│   │   ├── migrate.py                (P4)  upgrade_to_head, schema_status
│   │   ├── migrations/               (P4)  Alembic env + versions, shipped in the package
│   │   ├── repositories.py           (P9)  DocumentRepository
│   │   └── vector_store.py           (P9)  VectorStore Protocol + PgVectorStore
│   ├── retrieval/
│   │   ├── base.py                   (P10) Retriever Protocol
│   │   ├── vector.py                 (P10)
│   │   ├── keyword.py                (P17)
│   │   ├── hybrid.py                 (P17) reciprocal rank fusion
│   │   └── rerank.py                 (P18)
│   ├── llm/
│   │   ├── base.py                   (P11) LLMProvider Protocol
│   │   ├── fake.py                   (P11)
│   │   └── ollama.py                 (P12)
│   ├── rag/
│   │   ├── prompt.py                 (P13)
│   │   ├── service.py                (P13)
│   │   └── citations.py              (P14)
│   ├── observability/
│   │   ├── logging.py                (P1 basic, structured in P20)
│   │   └── metrics.py                (P20)
│   ├── evaluation/                   (P19)
│   │   ├── dataset.py
│   │   ├── metrics.py
│   │   └── runner.py
│   └── web/                          (P23) static HTML/JS served by FastAPI
└── tests/
    ├── conftest.py
    ├── unit/                         no network, no DB — fast, always run
    ├── integration/                  real Postgres+pgvector (marker: integration)
    └── e2e/                          real Ollama (marker: e2e, opt-in)
```

Note: `./docs` is reserved for the **user's corpus**, so project documentation lives in
root-level Markdown files rather than in `docs/`.

---

## 6. Data flow

### 6.1 Ingestion (write path)

```
./docs/**/*.md
     │  FilesystemMarkdownLoader.load() → LoadResult { documents, skipped }
     ▼
SourceDocument { source_path (relative, POSIX, NFC), text (normalised), content_hash, size_bytes }
     │  compare (content_hash, pipeline_fingerprint) with documents table
     ├── unchanged ─────────────────────────────────────────────► skip
     ▼  new / changed
MarkdownParser.parse()
     ▼
ParsedDocument { title, front_matter, sections[ Section{heading_path, anchor, level, text, line range} ] }
     │  Chunker.chunk()
     ▼
Chunk { id, document_path, ordinal, heading_path, anchor, text, embed_text, content_hash, line range }
     │  reuse existing vector if (content_hash, embedding_model) already stored
     │  otherwise EmbeddingProvider.embed_documents(batch of embed_text)
     ▼
one DB transaction per document:
  upsert documents row  ·  replace its chunks (+ vectors)  ·  update hash & fingerprint
     ▼
after all upserts: delete documents whose files no longer exist (chunks cascade)
```

### 6.2 Query (read path)

```
POST /api/v1/chat { question }
     ▼
RagService.answer(question)
     │ 1. EmbeddingProvider.embed_query(question)
     │ 2. Retriever.retrieve(query, top_k)            vector (P10) → hybrid (P17)
     │ 3. Reranker.rerank(...)                        no-op until P18
     │ 4. evidence gate: no chunk ≥ threshold ──────► "I don't know" (LLM not called)
     │ 5. PromptBuilder.build(question, chunks)       numbered [1]..[n] context, token budget
     │ 6. LLMProvider.generate(prompt)                temperature 0, fixed seed
     │ 7. CitationParser.parse(answer, chunks)        keep only valid [n] markers
     ▼
Answer { text, sources[ cited chunks only ], grounded: bool, trace{ timings, tokens, retrieved ids } }
     ▼
HTTP 200 { answer, sources: [{document, section, anchor, lines, score}] }
```

---

## 7. Ports (interfaces)

Interfaces are `typing.Protocol`s (structural typing: adapters and fakes don't inherit,
mypy checks conformance). Signatures below are **design sketches**, finalised in their phase.

```python
class DocumentLoader(Protocol):
    def load(self) -> LoadResult: ...  # documents + skipped files (with reasons)


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...  # stored with every vector
    @property
    def dimension(self) -> int: ...
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    async def replace_document_chunks(
        self, document_id: UUID, chunks: Sequence[EmbeddedChunk]
    ) -> None: ...
    async def search(self, vector: Sequence[float], top_k: int) -> list[RetrievedChunk]: ...


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerationOptions
    ) -> Generation: ...
```

### Why these, and not more

| Abstraction | Value it provides |
|---|---|
| `EmbeddingProvider` | Swap model/runtime; deterministic fake makes ingestion + retrieval testable without Ollama. **Separate `embed_documents` / `embed_query`** because many embedding models need different prefixes for queries vs documents. |
| `LLMProvider` | Swap model/runtime (Ollama today; any OpenAI-compatible server later); scripted fake makes RAG orchestration testable. |
| `VectorStore` | Isolates pgvector SQL from retrieval logic; in-memory fake for unit tests. |
| `Retriever` | Real polymorphism: vector, keyword, hybrid all exist eventually. |
| `Reranker` | Optional stage with a no-op default. |
| `DocumentLoader` | Seam for future formats / sources; also lets tests feed documents from memory. |

**Deliberately not abstracted:** the Markdown parser, chunker, prompt builder and citation
parser (one implementation each — plain classes with config); the database itself
(SQLAlchemy already is the abstraction); settings. A Protocol is added only when a second
implementation or a test fake genuinely needs it.

Honest limitation: `VectorStore` and `DocumentRepository` share a Postgres transaction so a
document's re-index is atomic. Swapping Postgres out would therefore mean re-implementing
both; that is an accepted trade-off for transactional consistency.

---

## 8. Data model

Introduced in Phase 4 (migration `0001`); the `embedding` column and vector index arrive in
Phase 9; the full-text column in Phase 17. Hashes are stored as `VARCHAR(64)` with CHECK
constraints enforcing lowercase SHA-256 hex. Constraints follow a naming convention
(`pk_`, `uq_`, `fk_`, `ck_`, `ix_` prefixes) so migrations can refer to them deterministically.

```
documents
  id                    UUID  PK
  source_path           TEXT  UNIQUE NOT NULL     -- relative POSIX path, e.g. "guides/deployment.md"
  title                 TEXT  NOT NULL            -- front matter title → first H1 → filename
  content_hash          CHAR(64) NOT NULL         -- sha256 of normalised text
  pipeline_fingerprint  CHAR(64) NOT NULL         -- hash of parser/chunker version+params+embedding model
  front_matter          JSONB NOT NULL DEFAULT '{}'
  size_bytes            INTEGER NOT NULL
  indexed_at            TIMESTAMPTZ NOT NULL DEFAULT now()

chunks
  id                    CHAR(32) PK               -- deterministic, see §9.3
  document_id           UUID FK → documents ON DELETE CASCADE
  ordinal               INTEGER NOT NULL          -- position within document
  heading_path          TEXT[] NOT NULL           -- ["Deployment", "Docker Deployment"]
  section_anchor        TEXT NOT NULL             -- "docker-deployment"
  content               TEXT NOT NULL             -- raw chunk text shown to the LLM
  content_hash          CHAR(64) NOT NULL         -- indexed (P4)
  start_line, end_line  INTEGER NOT NULL
  token_estimate        INTEGER NOT NULL
  embedding             VECTOR(d)                 -- (P9) d = configured dimension
  embedding_model       TEXT                      -- (P9)
  search_vector         TSVECTOR GENERATED        -- (P17)
  UNIQUE (document_id, ordinal)
  INDEX hnsw (embedding vector_cosine_ops)        -- (P9)
  INDEX (content_hash, embedding_model)           -- (P9) embedding reuse lookup
  INDEX gin (search_vector)                       -- (P17)

index_state                                        -- key/value
  key TEXT PK, value TEXT, updated_at TIMESTAMPTZ  -- embedding_model, embedding_dimension, ...
```

---

## 9. Major design decisions

Each decision is numbered so later phases and commits can reference it (e.g. "per D9").

**D1 — Modular monolith, one Python process.** API, indexer and watcher run in one
process (indexer as a background task). Separate services would add deployment and
consistency cost with no benefit at this scale. The indexer is still a separate
*module* with no HTTP knowledge, so it can also run via CLI.

**D2 — Async request path, Psycopg 3 for everything.** FastAPI + `async` SQLAlchemy +
`httpx.AsyncClient`: the request path is entirely I/O-bound (DB + model HTTP calls).
Psycopg 3 (`postgresql+psycopg`) supports both sync (Alembic) and async (app) with one
driver. The `pgvector` Python package supplies the SQLAlchemy `Vector` type.

**D3 — Talk to Ollama over plain HTTP (httpx), not the Ollama SDK.** We need two
endpoints (`/api/embed`, `/api/chat`). Owning the client gives explicit timeouts,
retries, error mapping and easy mocking (`respx`). It also keeps an OpenAI-compatible
adapter trivial to add later.

**D4 — Configuration via pydantic-settings.** All settings use the `EASYRAG_` prefix,
are validated at startup (fail fast with a clear message) and are injected — no module
reads `os.environ` directly. Secrets (DB password, later the API key) only come from env.

**D5 — Dependency injection without a framework.** `container.py` builds adapters and
services once in the FastAPI lifespan and stores them on `app.state`; routes receive them
through small `Depends()` functions. Tests override those functions or build a container
with fakes.

**D6 — Markdown parsing with `markdown-it-py`.** A CommonMark-compliant token stream
with source line maps. This handles the cases regex parsers get wrong: `#` inside fenced
code, setext headings, front matter. Front matter is parsed with PyYAML.
*Implemented in Phase 5:*
- Only **top-level** headings delimit sections; headings inside block quotes or lists are
  body text. Skipped levels (H1 → H3) still produce a correct heading path.
- Section text excludes the heading line, trimmed of blank lines; `start_line`/`end_line`
  are 1-based and refer to the original file (heading line included). Text before the
  first heading is a level-0 "preamble" section with an empty heading path. Headings with
  no body produce no section but remain in their children's heading paths.
- Anchors are GitHub-style slugs, de-duplicated per document (`setup`, `setup-1`, …).
- Title: front matter `title` → first H1 → humanised filename.
- Front matter must be a YAML mapping (`yaml.safe_load`); anything else is ignored with
  a warning. Values are made JSON-compatible (dates → ISO strings) for the JSONB column.
- GFM tables and strikethrough are enabled on top of CommonMark.

**D7 — Heading-aware, structure-preserving chunking.**
- Split by sections first (a chunk never spans two sections).
- Within a section, pack whole blocks (paragraph, list, table, code fence) up to
  `EASYRAG_CHUNK_MAX_TOKENS` with `EASYRAG_CHUNK_OVERLAP_TOKENS` overlap.
- Never split inside a code fence or table unless a single block alone exceeds the limit;
  then fall back to line splitting.
- Token counts are a deterministic estimate (no model-specific tokenizer dependency).
  Chunk limits sit well below the embedding model's context window, so the estimate
  doesn't need to be exact.
- **Contextual header:** the text we *embed* is
  `"{document title} > {heading path}\n\n{content}"`, while `content` (raw text) is what
  the LLM sees and what we store. Section context improves retrieval a lot.

**D8 — Content hashing and pipeline fingerprint.** Text is normalised first: UTF-8, BOM
stripped, `\r\n` → `\n`. A document is re-indexed when its `content_hash` changes **or**
when the `pipeline_fingerprint` changes (parser/chunker version, chunk params, embedding
model). A content hash alone would silently keep stale chunks after a config change.

**D9 — Deterministic chunk IDs.**
`chunk.id = sha256(source_path ‖ heading_path ‖ ordinal_in_section ‖ content_hash)[:32]`.
The same input and config always produce the same IDs, which makes indexing reproducible
and diffs meaningful. Separately, **embeddings are reused by `(content_hash,
embedding_model)`**, so editing one paragraph doesn't re-embed the whole document.
Evaluation datasets reference *document + section* rather than chunk IDs, because
chunk IDs legitimately change when chunking config changes.

**D10 — Idempotent, atomic, incremental indexing.**
- Each document is replaced in a single transaction: no half-indexed documents.
- Upserts run before deletes, so a renamed file can reuse its old embeddings.
- A Postgres advisory lock guarantees a single writer (startup sync, watcher and the
  manual trigger can't race).
- Running the indexer twice with no file changes performs zero embedding calls and zero writes.

**D11 — Fixed embedding dimension per deployment.** A pgvector HNSW index needs a fixed
dimension. The dimension comes from config at migration time, and `index_state` records
`embedding_model` + `dimension`. At startup the app checks them against config and
**refuses to start on mismatch**, with instructions to rebuild the index — never silently
mixing vector spaces. Cosine distance; `score = 1 − distance`.

**D12 — Two-layer grounding.**
1. *Retrieval gate (deterministic):* if retrieval returns nothing above the similarity
   threshold, answer "I don't know" **without calling the LLM**.
2. *Prompt contract:* a system prompt says to answer only from the numbered context,
   cite with `[n]`, treat context as data rather than instructions, and reply with an
   exact refusal sentence when the context is insufficient.

The threshold is **disabled by default** until Phase 19 calibrates it, because similarity
scales differ between embedding models.

**D13 — Citations via inline markers, validated server-side.** Small local models produce
`[1]`-style markers far more reliably than strict JSON. The server parses markers,
discards out-of-range indices, and returns only *cited* chunks as `sources` (not
everything retrieved). An answer with no valid citation that is also not a refusal is
returned with `grounded: false`. Prompt templates are versioned (`PROMPT_VERSION`) so
evaluation results can be attributed.

**D14 — Determinism where possible.** Generation uses `temperature=0` and a fixed `seed`.
Retrieval ties are broken by chunk ID, and discovery order is sorted by path.
**`num_ctx` is always set explicitly:** Ollama silently truncates prompts beyond its
default context window, which would quietly drop retrieved evidence. The prompt builder
enforces a token budget below `num_ctx`.

**D15 — Hybrid retrieval stays inside Postgres.** Full-text search (`tsvector`,
`websearch_to_tsquery`) is fused with vector results by Reciprocal Rank Fusion. No
Elasticsearch or OpenSearch.

**D16 — Reranker is an optional port with a no-op default.** Candidates are evaluated
in Phase 18: an ONNX cross-encoder (light, no PyTorch) vs. an LLM-based reranker via the
existing `LLMProvider`. We will not ship a multi-gigabyte PyTorch image.

**D17 — Migrations run before the server starts.** The container entrypoint
(`docker/entrypoint.sh`) runs `python -m easyrag.storage.migrate` (Alembic `upgrade head`) and
then `exec`s the server, so uvicorn stays PID 1. A migration failure exits the container
instead of serving an old schema. The schema is owned only by Alembic — never `create_all()`.
*Refined in Phase 4:*
- Migrations live **inside the package** (`easyrag/storage/migrations`) because the runtime
  image contains only the installed package; `alembic.ini` exists only for developer commands.
- `env.py` holds a Postgres advisory lock for the duration of a migration run, so containers
  starting simultaneously are serialised (verified by a multi-process integration test).
- Readiness reports `schema: outdated` when the database revision differs from the code's head.
- An integration test runs Alembic autogenerate against a migrated database and requires an
  empty diff, so ORM models and migrations cannot drift apart.

**D18 — Health semantics.** `/health/live` = the process is up. `/health/ready` = DB
reachable and migrated, model provider reachable, configured models present. Indexing
progress is reported separately (`/api/v1/index/status`) rather than making the app
"not ready" for minutes on a large first index.
Readiness returns `{"status": "ready"|"not_ready", "checks": {"<name>": "ok"|"unavailable"}}`
with 200 or 503. Every check has a timeout (`EASYRAG_HEALTH_CHECK_TIMEOUT_SECONDS`), and
failures are logged by exception type only (messages can contain connection details). The
app starts even when dependencies are down, so liveness stays green and the container
isn't restart-looped while Postgres boots.

**D19 — Ollama placement is configurable.** Compose ships an `ollama` service, but
`EASYRAG_OLLAMA_BASE_URL` can point at the host (`http://host.docker.internal:11434`).
This matters on macOS, where Docker containers cannot use the Apple GPU and in-container
inference is CPU-only and slow.

**D20 — File watching with a polling fallback.** Phase 16 uses `watchfiles` with
debouncing. Bind-mounted volumes on Docker Desktop don't always deliver native file
events reliably, so polling can be forced via config. A periodic full re-sync is the
safety net, and because indexing is idempotent (D10), it's cheap.

**D21 — Error model.** Adapters raise domain exceptions (`EmbeddingProviderError`,
`LLMProviderUnavailable`, `DocumentParseError`, ...). The API maps them to HTTP status codes
(503 when a dependency is unavailable, 422 for validation errors) with a stable error body.
A single malformed document is logged and skipped; it never aborts the whole index run.

**D22 — Documents are trusted-ish input.** The corpus is the user's own content, but
context is still wrapped in clear delimiters, and the prompt states that context text is
not instructions. File reads are confined to the docs root: symlinks resolved and checked,
`..` rejected.
*Implemented in Phase 5:* symlinked files are followed only when they resolve inside the
root; symlinked directories are never traversed. Hidden files and directories are ignored.
Files over 5 MiB, unreadable files and invalid UTF-8 are **reported** in
`LoadResult.skipped` (not silently dropped), so incremental indexing (Phase 15) can tell a
deleted file from a temporarily unreadable one.

**D23 — Tooling.** Python 3.13, `uv` for dependency management + lockfile (`uv_build`
backend, `src/` layout), `ruff` (lint + format), `mypy --strict` with the pydantic plugin,
and `pytest`. These are dev tools, not runtime frameworks. CLI uses stdlib `argparse`.
The app is created by a factory (`uvicorn --factory easyrag.main:create_app`), never at
import time, so importing a module never reads the environment.

*Revised in Phase 1:* Python 3.12 → 3.13 (mature, wheels available for everything
planned). Starlette 1.7's `TestClient` now uses `httpx2` instead of `httpx`, so `httpx2` is
the dev dependency. The HTTP client for the Ollama adapters (D3) will be chosen to
match in Phase 8, along with its mocking tool.

**D24 — Web UI without a JS build chain.** Phase 23 serves static HTML + vanilla JS from
FastAPI. Node, bundlers and SPA frameworks are not needed for a chat box and a source list.

**D25 — Hardened, minimal container (Phase 2).** A multi-stage build installs locked,
non-dev dependencies into `/app/.venv`; the runtime stage contains only that venv (no uv,
no compilers, no tests). The container runs as UID 10001 with a read-only root filesystem,
all capabilities dropped and `no-new-privileges`. uvicorn is PID 1 (exec-form `CMD`), so
SIGTERM shuts down gracefully. Compose publishes the API on `127.0.0.1` only until
authentication exists (Phase 22). Later phases needing writable paths add explicit volumes.

**D26 — Database runtime and credentials (Phase 3).** `pgvector/pgvector:0.8.7-pg18-trixie`
(PostgreSQL 18, pinned pgvector). `POSTGRES_PASSWORD` has no default: Compose refuses to start
without it, and builds `EASYRAG_DATABASE_URL` from the `POSTGRES_*` variables. The app treats
the URL as a `SecretStr`. The database is never published by the main compose file;
`docker-compose.dev.yml` publishes it on `127.0.0.1` for integration tests and `psql`.
Integration tests run only when `EASYRAG_TEST_DATABASE_URL` is set.

---

## 10. API surface (target)

| Method & path | Phase | Purpose |
|---|---|---|
| `GET /health/live` | 1 | Liveness |
| `GET /health/ready` | 3 (extended 8, 12) | Readiness: DB, migrations, model provider |
| `POST /api/v1/retrieve` | 10 | Retrieval only (debugging, evaluation) |
| `POST /api/v1/chat` | 13 | Question → grounded answer + sources |
| `GET /api/v1/documents` | 15 | Indexed documents with hash & chunk count |
| `GET /api/v1/index/status` | 15 | Last run, counts, errors, in-progress flag |
| `POST /api/v1/index` | 15 | Trigger a sync |
| `GET /metrics` | 20 | Prometheus metrics |

Chat contract (target):

```json
// request
{ "question": "How do I deploy the application?", "top_k": 5 }

// response
{
  "answer": "Build the image and run `docker compose up -d` [1]. ...",
  "grounded": true,
  "sources": [
    {
      "document": "deployment.md",
      "section": "Docker Deployment",
      "heading_path": ["Deployment", "Docker Deployment"],
      "anchor": "docker-deployment",
      "lines": [12, 40],
      "score": 0.83
    }
  ]
}
```

---

## 11. Configuration (target)

| Variable | Default | Phase |
|---|---|---|
| `EASYRAG_LOG_LEVEL` | `INFO` | 1 |
| `EASYRAG_DATABASE_URL` | — (required) | 3 |
| `EASYRAG_DOCS_DIR` | `/data/docs` (container) | 9 |
| `EASYRAG_CHUNK_MAX_TOKENS` / `_OVERLAP_TOKENS` | `400` / `50` | 6 |
| `EASYRAG_OLLAMA_BASE_URL` | `http://ollama:11434` | 8 |
| `EASYRAG_EMBEDDING_MODEL` / `_DIMENSION` | chosen in P8 | 8 |
| `EASYRAG_RETRIEVAL_TOP_K` | `5` | 10 |
| `EASYRAG_RETRIEVAL_MIN_SCORE` | unset (gate off) | 10 / 14 |
| `EASYRAG_LLM_MODEL`, `_NUM_CTX`, `_TEMPERATURE`, `_SEED` | chosen in P12 / 8192 / 0 / 42 | 12 |
| `EASYRAG_INDEX_ON_STARTUP` | `true` | 15 |
| `EASYRAG_WATCH_ENABLED` / `_FORCE_POLLING` | `true` / `false` | 16 |
| `EASYRAG_API_KEY` | unset (auth off) | 22 |

The exact model defaults are chosen and justified in Phases 8 and 12.

---

## 12. Testing strategy

| Layer | Scope | Dependencies | When it runs |
|---|---|---|---|
| Unit | Parser, chunker, hashing, prompt builder, citation parser, RAG service with fakes, API with overridden deps | None | Always; must be fast (< 5 s) |
| Integration | Migrations, PgVectorStore, indexer against a real DB, readiness | Postgres + pgvector (compose) | `pytest -m integration`; skipped when no test DB is configured |
| Adapter contract | Ollama adapters against recorded HTTP responses | `respx` mocks | Always |
| End-to-end | Full stack with real models | Compose + Ollama | `pytest -m e2e`, opt-in |
| Evaluation | Retrieval recall@k, MRR, citation precision, refusal accuracy, latency | Full stack | `easyrag eval`; a tracked result, not a pass/fail unit test |

The deterministic `FakeEmbeddingProvider` (hash-seeded vectors) and the scripted
`FakeLLMProvider` are what let the whole pipeline run in unit tests without models.

---

## 13. Observability (target)

- Structured JSON logs with `request_id` and `index_run_id`. Content is never logged at
  INFO; questions are logged only at DEBUG.
- `RagTrace` per request: stage timings (embed, retrieve, rerank, generate), prompt and
  completion token counts (taken from Ollama's `prompt_eval_count` / `eval_count`),
  retrieved chunk IDs + scores, prompt version, and model IDs.
- Prometheus metrics: request latency histograms per stage, indexing counters
  (documents scanned/changed/deleted, chunks embedded/reused), and provider error counters.

---

## 14. Open questions (resolved in the named phase)

| Question | Phase |
|---|---|
| Default embedding model and dimension (quality vs. CPU speed) | 8 |
| Default chat model (instruction-following + citation reliability at small sizes) | 12 |
| Similarity threshold default per model | 19 |
| Reranker implementation (ONNX cross-encoder vs. LLM rerank) | 18 |
| API key optional vs. required when binding to non-localhost | 22 |
