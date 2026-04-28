"""
Central configuration for LexUK.

Three responsibilities, deliberately co-located so a future contributor
(or your viva examiner) can find every knob in one place:

    1. Environment variables loaded from `.env` via pydantic-settings.
    2. Filesystem paths derived from the project root (no hard-coding).
    3. Pipeline tunables (chunk size, retrieval k, rerank k, ...).

Import pattern (used throughout the codebase):

    from src.config import settings, paths, RAG, AGENT
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# 1. Environment-driven settings
# ---------------------------------------------------------------------------
# pydantic-settings reads `.env` automatically and validates types. If a
# required field is missing, the program fails fast at import time rather
# than mysteriously breaking deep inside an API call.
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Runtime settings sourced from environment variables / `.env`."""

    # Required secret - no default, so absence raises ValidationError early.
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Model defaults: gpt-4o-mini for chat, text-embedding-3-large for vectors.
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-large")

    # Logging verbosity for the whole project.
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # OPENAI_API_KEY == openai_api_key
        extra="ignore",         # Tolerate stray vars in .env
    )


# Single global instance. Import this; do not re-instantiate.
settings = Settings()


# ---------------------------------------------------------------------------
# 2. Filesystem paths
# ---------------------------------------------------------------------------
# All paths are derived from this file's location, so the project is
# portable across machines without hard-coded "C:/shared/..." strings.
# ---------------------------------------------------------------------------

class Paths:
    """Project filesystem layout. Resolved relative to `chatbot/` root."""

    # `__file__` is .../chatbot/src/config.py - go up two levels for root.
    ROOT: Path = Path(__file__).resolve().parent.parent

    # --- Data ---
    DATA: Path = ROOT / "data"
    STATUTES_MD: Path = DATA / "markdown_12_statutes"
    CASELAW_MD: Path = DATA / "markdown_100_caselaws"   # parked for Phase 2
    CHROMA_DB: Path = DATA / "chroma_db"
    CHUNKS_CACHE: Path = DATA / "chunks_cache"          # JSONL of chunked output

    # --- Eval ---
    EVAL: Path = ROOT / "eval"
    GOLD_QUESTIONS: Path = EVAL / "gold_questions.jsonl"
    EVAL_RESULTS: Path = EVAL / "results"
    EVAL_PLOTS: Path = EVAL / "plots"

    @classmethod
    def ensure(cls) -> None:
        """Create directories that should exist at runtime. Idempotent."""
        for directory in (cls.CHROMA_DB, cls.CHUNKS_CACHE,
                          cls.EVAL_RESULTS, cls.EVAL_PLOTS):
            directory.mkdir(parents=True, exist_ok=True)


paths = Paths()


# ---------------------------------------------------------------------------
# 3. RAG pipeline tunables
# ---------------------------------------------------------------------------
# Constants you will sweep during the ablation study. Centralising them
# means the eval harness can override them programmatically without
# editing feature code.
# ---------------------------------------------------------------------------

class RAG:
    """Tunables for chunking, retrieval, reranking, and generation."""

    # --- Chunking ---
    AVERAGE_CHUNK_SIZE_CHARS: int = 1_500   # smaller = finer-grained chunks
    CHUNK_OVERLAP_TARGET_PCT: int = 25
    MAX_PASSAGE_CHARS: int = 50_000         # we use gpt-4.1-nano for the
                                            # chunker, which has a 32K output
                                            # token cap (2x gpt-4o-mini), so
                                            # passages can be larger and the
                                            # whole job runs in fewer API
                                            # calls.

    # --- Embeddings ---
    EMBEDDING_BATCH_SIZE: int = 100         # texts per embedding API call

    # --- Retrieval ---
    RETRIEVAL_K: int = 15                   # candidates returned by Chroma
    RERANK_K: int = 8                       # candidates kept after rerank

    # --- Generation ---
    GENERATION_TEMPERATURE: float = 0.0     # legal answers must be deterministic

    # --- Vector store ---
    COLLECTION_NAME: str = "uk_tenancy_statutes"


# ---------------------------------------------------------------------------
# 4. Agent layer tunables
# ---------------------------------------------------------------------------

class AGENT:
    """Tunables for the Planner agent and its tool-use loop."""

    # Hard cap on planner iterations - protects against runaway tool loops.
    MAX_TOOL_ITERATIONS: int = 6

    # Lower temperature than generation: planner decisions should be stable.
    PLANNER_TEMPERATURE: float = 0.0