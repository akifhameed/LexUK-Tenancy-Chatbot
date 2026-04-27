"""
Pure-function metrics for RAG evaluation.

All metrics are deterministic - no LLM calls, no randomness, fully
reproducible. The LLM-as-judge metrics live in `eval.llm_judge`.

Formulas mirror Ed Donner's Week 5 Day 4 evaluation module:

    MRR      - mean reciprocal rank (binary keyword presence)
    NDCG     - normalised discounted cumulative gain at k
    coverage - fraction of keywords present anywhere in retrieved chunks

Plus two project-specific metrics:

    citation_validity_rate - novel: fraction of citations that the
                              CitationValidator marked as valid.
    expected_acts_recall   - fraction of gold-expected Acts that appear
                              in the retrieved chunk set.
"""

from __future__ import annotations

import math

from src.agents.citation_validator import CitationCheck
from src.rag.vector_store import RetrievalResult


# ---------------------------------------------------------------------------
# Internal helper - flatten a chunk's text to one searchable blob
# ---------------------------------------------------------------------------

def _chunk_blob(doc: RetrievalResult) -> str:
    """Concatenated lower-cased text used by all keyword metrics."""
    return f"{doc.headline} {doc.summary} {doc.original_text}".lower()


# ---------------------------------------------------------------------------
# Per-keyword retrieval metrics
# ---------------------------------------------------------------------------

def calculate_mrr(keyword: str, retrieved_docs: list[RetrievalResult]) -> float:
    """Reciprocal rank of the first chunk containing `keyword`. 0 if none."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in _chunk_blob(doc):
            return 1.0 / rank
    return 0.0


def calculate_ndcg(
    keyword: str,
    retrieved_docs: list[RetrievalResult],
    k: int = 10,
) -> float:
    """NDCG@k with binary relevance for one keyword."""
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in _chunk_blob(doc) else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(sorted(relevances, reverse=True))
    )
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-question aggregate metrics (average across the question's keywords)
# ---------------------------------------------------------------------------

def average_mrr(
    keywords: list[str],
    retrieved_docs: list[RetrievalResult],
) -> float:
    """MRR averaged across all keywords for one question."""
    if not keywords:
        return 0.0
    return sum(calculate_mrr(kw, retrieved_docs) for kw in keywords) / len(keywords)


def average_ndcg(
    keywords: list[str],
    retrieved_docs: list[RetrievalResult],
    k: int = 10,
) -> float:
    """NDCG averaged across all keywords for one question."""
    if not keywords:
        return 0.0
    return sum(calculate_ndcg(kw, retrieved_docs, k) for kw in keywords) / len(keywords)


def keyword_coverage(
    keywords: list[str],
    retrieved_docs: list[RetrievalResult],
) -> float:
    """Fraction of keywords appearing in at least one retrieved chunk."""
    if not keywords:
        return 1.0   # vacuous - nothing to find
    found = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if any(kw_lower in _chunk_blob(doc) for doc in retrieved_docs):
            found += 1
    return found / len(keywords)


def expected_acts_recall(
    expected_acts: list[str],
    retrieved_docs: list[RetrievalResult],
) -> float:
    """Fraction of expected source Acts that show up in retrieval."""
    if not expected_acts:
        return 1.0
    expected_lower = {act.lower() for act in expected_acts}
    retrieved_files = {doc.source_file.lower() for doc in retrieved_docs}
    found = sum(1 for act in expected_lower if act in retrieved_files)
    return found / len(expected_lower)


# ---------------------------------------------------------------------------
# Citation validity (novel metric)
# ---------------------------------------------------------------------------

def citation_validity_rate(checks: list[CitationCheck]) -> float:
    """Fraction of citations marked valid by CitationValidator."""
    if not checks:
        return 1.0   # no citations made = no false ones
    return sum(1 for c in checks if c.valid) / len(checks)