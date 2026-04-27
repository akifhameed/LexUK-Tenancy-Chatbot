"""
End-to-end RAG pipeline: rewrite -> retrieve -> rerank -> generate.

This module is the single public entry point for the RAG system. The
agent layer (Batch 4) will wrap `answer_question` as the body of the
`search_statutes` tool.

Public surface:

    answer_question(question, *, history=None) -> RagResponse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import RAG
from src.rag.generator import generate_answer
from src.rag.reranker import rerank
from src.rag.retriever import retrieve
from src.rag.vector_store import RetrievalResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public response type
# ---------------------------------------------------------------------------
# Carries everything the trace panel and the eval harness need without
# requiring further DB lookups.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RagResponse:
    """End-to-end RAG result with full provenance."""

    answer: str
    chunks_used: list[RetrievalResult]   # the chunks actually shown to the LLM
    rewritten_query: str                 # for the trace panel
    best_distance: float                 # best dense-retrieval distance
                                         # (from before reranking; used as the
                                         # refusal signal in the agent layer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> RagResponse:
    """Run the full RAG pipeline for one question."""
    # 1. Retrieve - dual search + exact-match metadata filter.
    candidates, rewritten, exact_ids = retrieve(question, history=history)

    if not candidates:
        return RagResponse(
            answer="The provided sources do not cover this question.",
            chunks_used=[],
            rewritten_query=rewritten,
            best_distance=1.0,
        )

    # 2. Capture the best distance BEFORE reranking - this is the signal
    #    the agent layer will use to decide whether to refuse the query.
    best_distance = candidates[0].distance

    # 3. Rerank the dense candidates - but exact-match chunks are forced
    #    into the final context regardless of what the reranker decides.
    #    This protects against the failure mode where the LLM reranker
    #    deprioritises the canonical provision chunk in favour of more
    #    'topical' but less specific chunks.
    reranked = rerank(question, candidates, top_n=RAG.RERANK_K)

    final_context = _force_exact_matches(
        candidates=candidates,
        reranked=reranked,
        exact_ids=exact_ids,
        top_n=RAG.RERANK_K,
    )

    # 4. Generate - LLM writes the answer using the final context.
    answer = generate_answer(question, final_context, history=history)

    return RagResponse(
        answer=answer,
        chunks_used=final_context,
        rewritten_query=rewritten,
        best_distance=best_distance,
    )


def _force_exact_matches(
    *,
    candidates: list[RetrievalResult],
    reranked: list[RetrievalResult],
    exact_ids: set[str],
    top_n: int,
) -> list[RetrievalResult]:
    """
    Build the generator's context with exact-match chunks pinned to the top.

    This addresses a known failure mode: when the user names a specific
    section (e.g. "section 11 of the Landlord and Tenant Act 1985"), the
    canonical chunk is found via exact metadata filter, but the LLM
    reranker sometimes prefers other 'topical' chunks and pushes the
    exact match out of the top-N before the generator sees it.

    By construction, exact-match chunks are the most relevant possible
    answer; we ALWAYS keep them in the final context.
    """
    # Find the actual exact-match chunk records (preserving order from
    # the merged candidate list, which already has them at the front).
    exact_chunks = [c for c in candidates if c.chunk_id in exact_ids]

    final: list[RetrievalResult] = list(exact_chunks)
    seen = {c.chunk_id for c in final}

    for chunk in reranked:
        if len(final) >= top_n:
            break
        if chunk.chunk_id in seen:
            continue
        final.append(chunk)
        seen.add(chunk.chunk_id)

    return final[:top_n]