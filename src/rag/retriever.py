"""
Provision-aware retriever.

Strategy:

    1. Detect a specific provision id (s_21, sch_2_ground_8, ...) and an Act
       in the user's question via simple regex.
    2. If both are detected, run an EXACT-MATCH Chroma query restricted by
       metadata. This usually returns the canonical chunk in 1 hop.
    3. Always also run dense retrieval with the original question and the
       LLM-rewritten query.
    4. Merge: exact-match results first (in their natural distance order),
       then dense results, deduped by chunk_id.

Step 2 is the difference-maker for legal QA: when a user asks about
"section 21 of the Housing Act 1988", the right chunk is found in one
Chroma round-trip without depending on dense similarity at all.

Public surface:

    retrieve(question, *, history=None, k=RETRIEVAL_K)
        -> tuple[list[RetrievalResult], str]
"""

from __future__ import annotations

import logging
import re

from src.config import RAG
from src.rag.query_rewriter import rewrite_query
from src.rag.vector_store import (
    RetrievalResult,
    query as chroma_query,
    query_with_filter,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex - detect provision references in the user's question
# ---------------------------------------------------------------------------

_SECTION_PATTERN = re.compile(
    r"\b(?:section|s\.?)\s*(\d+[A-Za-z]*)\b",
    re.IGNORECASE,
)
_SCHEDULE_PATTERN = re.compile(
    r"\b(?:schedule|sch\.?)\s*(\d+)"
    r"(?:[,\s]+(?:para(?:graph)?\.?\s*)?(\d+))?\b",
    re.IGNORECASE,
)
_GROUND_PATTERN = re.compile(r"\bground\s+(\d+[A-Za-z]*)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Act triggers - if any of these appear in the query, we'll filter by that Act
# ---------------------------------------------------------------------------

_ACT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "commonhold_and_leasehold_reform_act_2002":
        ("commonhold and leasehold reform", "clra"),
    "deregulation_act_2015":
        ("deregulation act 2015",),
    "homelessness_reduction_act_2017":
        ("homelessness reduction", "hra 2017"),
    "housing_act_1988":
        ("housing act 1988", "ha 1988"),
    "housing_act_2004":
        ("housing act 2004", "ha 2004", "tenancy deposit", "hmo"),
    "immigration_act_2014":
        ("immigration act 2014", "right to rent"),
    "landlord_and_tenant_act_1985":
        ("landlord and tenant act 1985", "lta 1985"),
    "landlord_and_tenant_act_1987":
        ("landlord and tenant act 1987", "lta 1987"),
    "protection_from_eviction_act_1977":
        ("protection from eviction", "pfea"),
    "rent_act_1977":
        ("rent act 1977",),
    "renters_rights_act_2025":
        ("renters' rights act", "renters rights act", "rra 2025"),
    "tenant_fees_act_2019":
        ("tenant fees act", "tfa 2019", "holding deposit"),
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_act(query: str) -> str | None:
    """Pick the Act with the most trigger hits in the query."""
    q = query.lower()
    best: str | None = None
    best_score = 0
    for source_file, triggers in _ACT_TRIGGERS.items():
        score = sum(1 for t in triggers if t in q)
        if score > best_score:
            best, best_score = source_file, score
    return best


def _detect_provision_id(query: str, act: str | None) -> str | None:
    """Build a canonical provision id from regex matches in the query."""
    # 1) Ground inside a schedule.
    g = _GROUND_PATTERN.search(query)
    if g:
        sched = _SCHEDULE_PATTERN.search(query)
        # Default to schedule 2 for Housing Act 1988 grounds (the typical case).
        sch_num = sched.group(1) if sched else (
            "2" if act == "housing_act_1988" else ""
        )
        if sch_num:
            return f"sch_{sch_num.lower()}_ground_{g.group(1).lower()}"

    # 2) Schedule paragraph or schedule heading.
    s = _SCHEDULE_PATTERN.search(query)
    if s:
        sch = s.group(1).lower()
        para = s.group(2)
        if para:
            return f"sch_{sch}_para_{para.lower()}"
        return f"sch_{sch}"

    # 3) Section.
    sec = _SECTION_PATTERN.search(query)
    if sec:
        return f"s_{sec.group(1).lower()}"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    k: int = RAG.RETRIEVAL_K,
) -> tuple[list[RetrievalResult], str, set[str]]:
    """Run exact-match + dual-query dense retrieval.

    Returns:
        (merged_chunks, rewritten_query, exact_match_chunk_ids)

        merged_chunks - all candidates with exact-matches first.
        rewritten_query - LLM-rewritten version (for the trace panel).
        exact_match_chunk_ids - the set of chunk_ids that were retrieved
            via exact metadata filter. The pipeline uses this to force
            them into the generator's context regardless of reranking.
    """
    rewritten = rewrite_query(question, history=history)
    act = _detect_act(question)
    provision_id = _detect_provision_id(question, act)

    # 1. Exact-match path: only fires when we have both an Act and a provision.
    exact: list[RetrievalResult] = []
    if act and provision_id:
        exact = query_with_filter(
            question,
            where={"$and": [
                {"source_file": act},
                {"provision_id": provision_id},
            ]},
            k=3,
        )
        if exact:
            log.info(
                "exact-match hit: %s::%s (%d chunks)",
                act, provision_id, len(exact),
            )

    exact_ids = {chunk.chunk_id for chunk in exact}

    # 2. Dense retrieval (always - acts as fallback / supplements exact).
    dense_orig = chroma_query(question, k=k)
    dense_rew = chroma_query(rewritten, k=k) if rewritten and rewritten != question else []

    # 3. Merge: exact first (already best), then dense by distance, dedup by chunk_id.
    merged: list[RetrievalResult] = []
    seen: set[str] = set()
    for chunk in exact:
        if chunk.chunk_id not in seen:
            merged.append(chunk)
            seen.add(chunk.chunk_id)
    for chunk in sorted(dense_orig + dense_rew, key=lambda c: c.distance):
        if chunk.chunk_id not in seen:
            merged.append(chunk)
            seen.add(chunk.chunk_id)

    log.info(
        "retrieved %d unique candidates (exact=%d, dense_orig=%d, "
        "dense_rew=%d) act=%s provision=%s best_distance=%.4f",
        len(merged), len(exact), len(dense_orig), len(dense_rew),
        act or "-", provision_id or "-",
        merged[0].distance if merged else 1.0,
    )
    return merged, rewritten, exact_ids
