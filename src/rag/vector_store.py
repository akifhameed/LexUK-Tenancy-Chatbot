"""
ChromaDB-backed vector store for the LexUK chatbot.

Two query paths are exposed:

    query(text, k)                 - dense similarity search
    query_with_filter(text, where) - dense search restricted by a Chroma
                                      `where` clause (used by the retriever
                                      to do exact-match provision lookups)

Index payload now includes provision_id / provision_kind / provision_number
so the retriever can filter on them directly.

Persistence: Chroma stores files under `data/chroma_db/`; created on first
use, reused on subsequent runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from tqdm import tqdm

from src.config import RAG, paths
from src.ingest.chunker import ChunkRecord
from src.llm_client import embed
from src.rag.embeddings import embed_chunks

log = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE: int = 500
_MAX_EMBEDDING_CHARS: int = 24_000


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalResult:
    """One chunk returned from a similarity search, with full provenance."""

    chunk_id: str
    headline: str
    summary: str
    original_text: str
    source_file: str
    source_title: str
    source_year: int
    source_url: str
    distance: float
    provision_id: str = ""
    provision_kind: str = ""
    provision_number: str = ""


# ---------------------------------------------------------------------------
# Internal Chroma helpers
# ---------------------------------------------------------------------------

def _client() -> chromadb.PersistentClient:
    paths.ensure()
    return chromadb.PersistentClient(path=str(paths.CHROMA_DB))


def _get_or_create() -> Collection:
    return _client().get_or_create_collection(
        name=RAG.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> Collection:
    """Drop and recreate the collection."""
    client = _client()
    try:
        client.delete_collection(RAG.COLLECTION_NAME)
        log.warning("deleted existing collection '%s'", RAG.COLLECTION_NAME)
    except Exception:
        pass
    return _get_or_create()


# ---------------------------------------------------------------------------
# Oversized-chunk splitting (defensive fallback)
# ---------------------------------------------------------------------------

def _split_oversized_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Split any chunk whose embed text exceeds the embedding model limit."""
    result: list[ChunkRecord] = []
    split_count = 0

    for chunk in chunks:
        embed_text = chunk.to_embed_text()
        if len(embed_text) <= _MAX_EMBEDDING_CHARS:
            result.append(chunk)
            continue

        overhead = len(embed_text) - len(chunk.original_text)
        per_sub = max(1, _MAX_EMBEDDING_CHARS - overhead - 200)

        original = chunk.original_text
        segments = [
            original[i:i + per_sub]
            for i in range(0, len(original), per_sub)
        ]

        for sub_idx, segment in enumerate(segments):
            suffix = chr(ord("a") + sub_idx) if sub_idx < 26 else f"x{sub_idx}"
            result.append(replace(
                chunk,
                chunk_id=f"{chunk.chunk_id}-{suffix}",
                original_text=segment,
            ))

        split_count += 1
        log.warning(
            "split oversized chunk %s into %d sub-chunks (was %d chars)",
            chunk.chunk_id, len(segments), len(original),
        )

    if split_count:
        log.info(
            "split %d oversized chunks; total after splitting: %d",
            split_count, len(result),
        )
    return result


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def index_chunks(chunks: list[ChunkRecord], *, fresh: bool = False) -> int:
    """Embed and store a list of chunks in Chroma. Returns final count."""
    if not chunks:
        log.warning("index_chunks called with empty list")
        return 0

    collection = reset_collection() if fresh else _get_or_create()

    chunks = _split_oversized_chunks(chunks)
    vectors = embed_chunks(chunks)

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.to_embed_text() for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "source_title": chunk.source_title,
            "source_year": chunk.source_year,
            "source_url": chunk.source_url,
            "headline": chunk.headline,
            "summary": chunk.summary,
            "original_text": chunk.original_text,
            "provision_id": chunk.provision_id,
            "provision_kind": chunk.provision_kind,
            "provision_number": chunk.provision_number,
        }
        for chunk in chunks
    ]

    for start in tqdm(
        range(0, len(chunks), _UPSERT_BATCH_SIZE),
        desc="indexing chunks",
        unit="batch",
    ):
        end = start + _UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end],
        )

    final_count = collection.count()
    log.info(
        "collection '%s' now contains %d chunks",
        RAG.COLLECTION_NAME, final_count,
    )
    return final_count


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _result_from_meta(meta: dict[str, Any], distance: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=str(meta.get("chunk_id", "")),
        headline=str(meta.get("headline", "")),
        summary=str(meta.get("summary", "")),
        original_text=str(meta.get("original_text", "")),
        source_file=str(meta.get("source_file", "")),
        source_title=str(meta.get("source_title", "")),
        source_year=int(meta.get("source_year", 0) or 0),
        source_url=str(meta.get("source_url", "")),
        distance=float(distance),
        provision_id=str(meta.get("provision_id", "")),
        provision_kind=str(meta.get("provision_kind", "")),
        provision_number=str(meta.get("provision_number", "")),
    )


def query(text: str, k: int = RAG.RETRIEVAL_K) -> list[RetrievalResult]:
    """Embed `text` and return the top-k most similar chunks."""
    if not text.strip():
        return []
    collection = _get_or_create()
    vector = embed([text])[0]
    raw = collection.query(query_embeddings=[vector], n_results=k)

    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]
    return [_result_from_meta(m, d) for m, d in zip(metadatas, distances)]


def query_with_filter(
    text: str,
    *,
    where: dict[str, Any],
    k: int = 5,
) -> list[RetrievalResult]:
    """
    Dense query restricted by a Chroma `where` clause.

    Used for exact-match provision lookups, e.g.
        where={"$and": [{"source_file": "housing_act_1988"},
                        {"provision_id": "s_21"}]}
    """
    if not text.strip():
        return []
    collection = _get_or_create()
    vector = embed([text])[0]

    try:
        raw = collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=where,
        )
    except Exception as exc:
        log.warning("query_with_filter failed (%s); falling back to dense", exc)
        return []

    metadatas = raw["metadatas"][0] if raw.get("metadatas") else []
    distances = raw["distances"][0] if raw.get("distances") else []
    if not metadatas:
        return []
    return [_result_from_meta(m, d) for m, d in zip(metadatas, distances)]
