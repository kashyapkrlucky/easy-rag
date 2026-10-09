"""Application configuration.

All settings are read from environment variables prefixed with ``EASYRAG_`` (and,
for local development, from a ``.env`` file). This is the only module that reads
the environment; everything else receives a ``Settings`` instance.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _upper(value: object) -> object:
    return value.upper() if isinstance(value, str) else value


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

    # SQLAlchemy URL, e.g. postgresql+psycopg://user:password@host:5432/dbname
    # Secret because it carries the password; never log it.
    database_url: SecretStr

    # Upper bound for each dependency check performed by /health/ready.
    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    # --- Documents and chunking ------------------------------------------------------
    # Directory of Markdown files to index. Docker Compose mounts it at /data/docs.
    docs_dir: Path = Path("docs")
    # Chunk budget in estimated tokens, including the contextual header (ARCHITECTURE D7).
    chunk_max_tokens: int = Field(default=400, ge=32, le=8192)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    # --- Retrieval ---------------------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    # Minimum cosine similarity for a chunk to be returned. Off by default: useful values
    # are model-specific and must be calibrated on your documents (ARCHITECTURE D12).
    retrieval_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    # --- Ollama ------------------------------------------------------------------------
    # Docker Compose sets http://ollama:11434; the default suits `uv run` with a host Ollama.
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    # Generous by default: the first request may have to load a model, and CPU-only
    # generation (e.g. Docker on macOS) of a few hundred tokens can take minutes.
    ollama_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)

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
