"""
Batched embedding helper for the RAG pipeline.

The OpenAI Embeddings API accepts up to a few thousand inputs per call,
but each individual input must be under the embedding model's token
limit (8,192 tokens for text-embedding-3-large).

The vector store calls `_split_oversized_chunks` BEFORE embedding to
ensure no single chunk exceeds the limit (so no content is lost).
This module keeps a defensive truncation as a last-resort safety net
in case anything slips through - it should rarely fire.

Public surface:

    embed_chunks(chunks) -> list[list[float]]
        Embed a list of ChunkRecord objects, one vector per chunk,
        same order as input.

The text that gets embedded for each chunk is `chunk.to_embed_text()`,
which concatenates headline + summary + original_text (mixes query-style language with legal language).
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from src.config import RAG
from src.ingest.chunker import ChunkRecord
from src.llm_client import embed

log = logging.getLogger(__name__)


# Conservative character budget for one embedding input. The OpenAI
# embedding endpoint caps inputs at 8,192 tokens. At ~3.5 chars/token
# for legal English, 24,000 chars leaves a safe margin. Any chunk
# longer than this is truncated (lossy but safe; better than failing
# the entire batch).
_MAX_EMBEDDING_CHARS: int = 24_000


def _truncate(text: str) -> str:
    """Truncate `text` to the embedding model's safe character budget."""
    if len(text) <= _MAX_EMBEDDING_CHARS:
        return text
    return text[:_MAX_EMBEDDING_CHARS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[ChunkRecord]) -> list[list[float]]:
    """
    Embed a list of chunks in batches and return their vectors in order.

    Args:
        chunks: ChunkRecord objects produced by the chunker.

    Returns:
        List of float vectors, one per input chunk, in the same order.
    """
    if not chunks:
        return []

    vectors: list[list[float]] = []
    batch_size = RAG.EMBEDDING_BATCH_SIZE
    truncated_count = 0

    # tqdm progress bar over batches, not individual chunks, so the bar
    # advances visibly even when each batch is large.
    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="embedding chunks",
        unit="batch",
    ):
        end = start + batch_size
        batch = chunks[start:end]

        texts: list[str] = []
        for chunk in batch:
            raw = chunk.to_embed_text()
            truncated = _truncate(raw)
            if len(truncated) < len(raw):
                truncated_count += 1
                log.warning(
                    "truncated chunk %s for embedding (%d -> %d chars)",
                    chunk.chunk_id, len(raw), len(truncated),
                )
            texts.append(truncated)

        batch_vectors = embed(texts)
        vectors.extend(batch_vectors)

    log.info(
        "embedded %d chunks (%d truncated to fit model limit)",
        len(vectors), truncated_count,
    )
    return vectors
