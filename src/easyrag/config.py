"""Application configuration.

All settings are read from environment variables prefixed with ``EASYRAG_`` (and,
for local development, from a ``.env`` file). This is the only module that reads
the environment; everything else receives a ``Settings`` instance.
"""

import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import AfterValidator, AnyHttpUrl, BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _upper(value: object) -> object:
    return value.upper() if isinstance(value, str) else value


def _blank_to_none(value: object) -> object:
    # `EASYRAG_API_KEY=` (as in a copied .env.example) means "not set".
    return None if isinstance(value, str) and not value.strip() else value


def _split_commas(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def with_password(url: str, password: str) -> str:
    """``url`` with ``password`` inserted (URL-quoted, so any characters are allowed)."""
    parts = urlsplit(url)
    if parts.password is not None:
        raise ValueError("the database URL already contains a password; remove one of them")
    if parts.username is None or parts.hostname is None:
        raise ValueError("the database URL needs a user and host to add a password to")
    netloc = f"{parts.username}:{quote(password, safe='')}@{parts.netloc.rpartition('@')[2]}"
    return urlunsplit(parts._replace(netloc=netloc))


_ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?|\*")


def _valid_origins(origins: list[str]) -> list[str]:
    for origin in origins:
        if not _ORIGIN.fullmatch(origin):
            raise ValueError(
                f"invalid CORS origin {origin!r}: use scheme://host[:port] (no path) or *"
            )
    return origins


class Settings(BaseSettings):
    """Validated, immutable application settings."""

    model_config = SettingsConfigDict(
        env_prefix="EASYRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    log_level: Annotated[LogLevel, BeforeValidator(_upper)] = "INFO"
    # json: one JSON object per line (for log collectors); text: human-readable.
    log_format: Literal["json", "text"] = "json"

    # SQLAlchemy URL, e.g. postgresql+psycopg://user:password@host:5432/dbname
    # Secret because it carries the password; never log it.
    database_url: SecretStr
    # Read the password from this file instead (Docker Compose, Docker secrets); the URL
    # must then have no password. Surrounding whitespace is ignored.
    database_password_file: Path | None = None

    # Upper bound for each dependency check performed by /health/ready.
    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    # --- HTTP API ------------------------------------------------------------------------
    # When set, every endpoint except the health probes and the API schema requires it, as
    # "Authorization: Bearer <key>" or "X-API-Key: <key>". Unset, the API is
    # unauthenticated: keep it on localhost (the Docker Compose default).
    api_key: Annotated[SecretStr | None, BeforeValidator(_blank_to_none)] = Field(
        default=None, min_length=16
    )
    # Browser origins allowed to call the API from another site (comma-separated, e.g.
    # "https://wiki.example.com"); empty disables CORS. Not needed for same-origin use.
    cors_allow_origins: Annotated[
        list[str], NoDecode, BeforeValidator(_split_commas), AfterValidator(_valid_origins)
    ] = []
    # Larger request bodies are rejected with 413. Questions are at most 2000 characters.
    max_request_bytes: int = Field(default=65536, ge=1024, le=10 * 1024 * 1024)
    # Serve the chat page at / (static files, no build step). Off for API-only deployments.
    ui_enabled: bool = True

    # --- Documents and chunking ------------------------------------------------------
    # Directory of Markdown files to index. Docker Compose mounts it at /data/docs.
    docs_dir: Path = Path("docs")
    # Chunk budget in estimated tokens, including the contextual header (ARCHITECTURE D7).
    chunk_max_tokens: int = Field(default=400, ge=32, le=8192)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    # Index the docs directory in the background when the API starts (retrying until
    # Ollama and the database are ready).
    index_on_startup: bool = True
    # Re-index automatically when Markdown files change (ARCHITECTURE D20).
    watch_enabled: bool = True
    # Poll instead of relying on native events (bind mounts / network filesystems that
    # do not deliver them).
    watch_force_polling: bool = False
    # Changes within this window are grouped into one re-index request.
    watch_debounce_ms: int = Field(default=1000, ge=50, le=60000)
    # Safety-net re-index request even without events; 0 disables it.
    watch_resync_seconds: float = Field(default=300.0, ge=0, le=86400)

    # --- Retrieval ---------------------------------------------------------------------
    # vector: dense similarity; keyword: full-text with IDF weighting; hybrid: both, fused
    # with Reciprocal Rank Fusion (ARCHITECTURE D32).
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    # Minimum cosine similarity for a chunk to be returned. Off by default: useful values
    # are model-specific and must be calibrated on your documents (ARCHITECTURE D12).
    retrieval_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    # --- Reranking ----------------------------------------------------------------------
    # A cross-encoder reorders the best `rerank_candidates` retrieval results and keeps
    # `retrieval_top_k` (ARCHITECTURE D33). The model is downloaded once into
    # `model_cache_dir`, then used offline.
    rerank_enabled: bool = False
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", min_length=1)
    rerank_candidates: int = Field(default=20, ge=1, le=50)
    model_cache_dir: Path = Path(".cache/easyrag/models")

    # --- Ollama ------------------------------------------------------------------------
    # Docker Compose sets http://ollama:11434; the default suits `uv run` with a host Ollama.
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    # Generous by default: the first request may have to load a model, and CPU-only
    # generation (e.g. Docker on macOS) of a few hundred tokens can take minutes.
    ollama_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)
    # Pull missing embedding and chat models through Ollama's API in the background at
    # startup (works with the bundled or a host Ollama). Off for curated/air-gapped hosts.
    ollama_pull_models: bool = True

    # --- Chat model ----------------------------------------------------------------------
    # Pin a tag. Default sized for small machines: ~1 GB, fits a 4 GB Docker VM alongside the
    # embedding model and PostgreSQL. Larger models answer better but need more memory.
    llm_model: str = Field(default="qwen2.5:1.5b-instruct-q4_K_M", min_length=1)
    # Tokens per request (prompt + answer), always sent to Ollama as num_ctx.
    llm_context_window: int = Field(default=8192, ge=512, le=262144)
    # Deterministic decoding by default (ARCHITECTURE D14).
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_seed: int | None = 42
    llm_max_output_tokens: int = Field(default=512, ge=16, le=8192)

    # --- RAG -----------------------------------------------------------------------------
    # Upper bound on retrieved evidence in the prompt, in estimated tokens. Prompt size
    # dominates latency on CPU (~30 s for a 3k-token prompt measured in Docker on macOS).
    rag_max_context_tokens: int = Field(default=2000, ge=200, le=200000)
    # A reply citing no source: "refuse" replaces it with the refusal sentence,
    # "flag" returns it marked as ungrounded (ARCHITECTURE D30).
    rag_ungrounded_policy: Literal["refuse", "flag"] = "refuse"
    # Instruction-like text in documents (prompt injection): "redact" removes it from the
    # prompt (users still see the original sources); "off" for a corpus *about* prompts,
    # whose examples would otherwise be removed (ARCHITECTURE D39).
    rag_injection_filter: Literal["redact", "off"] = "redact"
    # Questions answered at the same time; more get 503 "busy" with Retry-After instead
    # of queueing behind slow CPU generation until they time out.
    chat_max_concurrent: int = Field(default=4, ge=1, le=256)

    # --- Embeddings --------------------------------------------------------------------
    # Pin a tag, not "latest": a silently updated model would change every vector.
    embedding_model: str = Field(default="nomic-embed-text:v1.5", min_length=1)
    # Must equal the model's output size. At most 2000: pgvector's HNSW index limit.
    embedding_dimension: int = Field(default=768, ge=2, le=2000)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    # Model-specific instructions prepended to queries and documents (nomic-embed-text
    # expects these). Change them together with the model.
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "

    @model_validator(mode="before")
    @classmethod
    def _password_from_file(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not data.get("database_password_file"):
            return data
        url = data.get("database_url")
        if url is None:
            return data  # reported as a missing database_url
        path = Path(data["database_password_file"])
        try:
            password = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read database_password_file {path}: {exc.strerror}") from None
        if not password:
            raise ValueError(f"database_password_file {path} is empty")
        raw = url.get_secret_value() if isinstance(url, SecretStr) else str(url)
        return {**data, "database_url": with_password(raw, password)}

    @model_validator(mode="after")
    def _overlap_fits_chunk(self) -> "Settings":
        if self.chunk_overlap_tokens > self.chunk_max_tokens // 2:
            raise ValueError("chunk_overlap_tokens must be at most half of chunk_max_tokens")
        return self

    @model_validator(mode="after")
    def _answer_fits_context_window(self) -> "Settings":
        if self.llm_max_output_tokens > self.llm_context_window // 2:
            raise ValueError("llm_max_output_tokens must be at most half of llm_context_window")
        return self
