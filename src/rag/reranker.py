"""
LLM-based reranker.

Dense retrieval gives us K candidates ranked by cosine distance, but
vector similarity is only a proxy for relevance. An LLM that reads
each candidate fully picks up nuance the embedding model misses.

Mechanism:

    1. We build a numbered list of candidates, each shown with its
       headline and summary (full original_text would bloat the prompt).
    2. Pydantic-typed structured output forces the model to return only
       a list of indices in relevance order.
    3. We reorder the candidate list by that ordering and keep the top N.

Public surface:

    rerank(question, candidates, *, top_n=RERANK_K) -> list[RetrievalResult]
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.config import RAG
from src.llm_client import structured_completion
from src.rag.vector_store import RetrievalResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Structured-output schema (Pydantic)
# ---------------------------------------------------------------------------

class RankOrder(BaseModel):
    """The model's ordering of candidate indices, most relevant first."""

    order: list[int] = Field(
        description="Candidate indices in order of relevance to the question. "
                    "Most relevant first. Use the same indices shown in the "
                    "input."
    )


# ---------------------------------------------------------------------------
# 2. Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a re-ranker for a UK tenancy-law retrieval system. "
    "Given a user question and a numbered list of candidate chunks, "
    "rank the candidates by relevance to the question, most relevant "
    "first. Consider both literal keyword overlap and semantic alignment. "
    "Reply ONLY with the ordered list of indices."
)


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------

def rerank(
    question: str,
    candidates: list[RetrievalResult],
    *,
    top_n: int = RAG.RERANK_K,
) -> list[RetrievalResult]:
    """
    Rerank candidates by LLM-judged relevance and return the top N.

    If the model returns a malformed ordering (missing indices, duplicates),
    we fall back to the original order for the missing positions. This
    prevents an unlucky LLM call from losing chunks.

    Args:
        question:   The user's original question.
        candidates: List of RetrievalResult (typically 10-20 items).
        top_n:      Number of best candidates to keep.

    Returns:
        The top_n candidates, reordered by LLM-judged relevance.
    """
    if len(candidates) <= 1:
        return candidates[:top_n]

    # Build a compact numbered listing - headline + summary only.
    candidate_blocks = [
        f"[{index}] {candidate.headline}\n{candidate.summary}"
        for index, candidate in enumerate(candidates)
    ]
    user_prompt = (
        f"Question: {question}\n\n"
        f"Candidates:\n\n"
        + "\n\n".join(candidate_blocks)
    )

    response: RankOrder = structured_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=RankOrder,
    )

    # Defensive index validation - drop out-of-range, dedupe.
    seen: set[int] = set()
    valid_order: list[int] = []
    for index in response.order:
        if 0 <= index < len(candidates) and index not in seen:
            valid_order.append(index)
            seen.add(index)

    # If the model dropped any indices, append them at the end so we
    # never lose a candidate.
    for index in range(len(candidates)):
        if index not in seen:
            valid_order.append(index)

    reordered = [candidates[i] for i in valid_order]

    # Deterministic provision chunks injected by the retriever are exact-text
    # safety candidates. Keep a few of the highest-confidence ones even if the
    # LLM reranker under-ranks them; this prevents named-section questions from
    # losing the very provision the user asked about.
    protected = [
        candidate
        for candidate in candidates
        if "::provision::" in candidate.chunk_id and candidate.distance <= 0.02
    ][:3]
    if protected:
        protected_ids = {candidate.chunk_id for candidate in protected}
        reordered = protected + [
            candidate
            for candidate in reordered
            if candidate.chunk_id not in protected_ids
        ]

    log.info("reranked %d -> top %d", len(candidates), top_n)
    return reordered[:top_n]
