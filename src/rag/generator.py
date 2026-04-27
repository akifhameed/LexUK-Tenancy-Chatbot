"""
Answer generator with mandatory citations.

Now that chunks are 1-to-1 with provisions and carry explicit provision
metadata, the generator's job is straightforward: read the structured
context, answer the question, cite the Act + Provision exactly as shown.

The system prompt no longer needs the elaborate "no inferred citations"
language because the structure is in the data: each context block shows
Act + Provision + Headline + Text, and the model just copies them into
the citation tag.

Public surface:

    generate_answer(question, chunks, *, history=None) -> str
"""

from __future__ import annotations

import logging

from src.config import RAG
from src.llm_client import chat_completion
from src.rag.vector_store import RetrievalResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are LexUK, a UK tenancy-law information assistant. "
    "Answer the user's question using ONLY the context below.\n\n"

    "Each source has these fields:\n"
    "  Act       - the Act of Parliament (use this in the citation)\n"
    "  Provision - the section, schedule paragraph, or ground "
    "(use this in the citation)\n"
    "  Headline  - a short heading\n"
    "  Text      - the verbatim statutory text\n\n"

    "RULES:\n"
    "1. Cite every legal claim inline as [Act, Provision] - for example "
    "[Housing Act 1988, s.21] or [Tenant Fees Act 2019, Sch.1 para.3]. "
    "Take the Act and Provision verbatim from the source headers; never "
    "invent a section number.\n"
    "2. Answer only what the user asked. Do NOT add unrelated provisions "
    "or 'see also' material.\n"
    "3. If the sources do not cover the question at all, say: 'The "
    "provided sources do not cover this question.' Do NOT fabricate.\n"
    "4. Off-domain questions (medical, criminal, financial advice, "
    "general immigration) - decline politely and suggest NHS 111, a "
    "solicitor, or Citizens Advice.\n"
    "5. Be concise: 2-5 sentences for most answers.\n"
    "6. End every substantive answer with: 'This is information about UK "
    "statutes, not legal advice. For your specific situation consult a "
    "solicitor or Citizens Advice.'"
)


# ---------------------------------------------------------------------------
# Provision-label rendering (best-effort fall back if metadata is missing)
# ---------------------------------------------------------------------------

def _provision_label(chunk: RetrievalResult) -> str:
    """Return a short citation-friendly provision label for the source header."""
    # Prefer the structured metadata field set by the chunker.
    if chunk.provision_kind and chunk.provision_number:
        kind = chunk.provision_kind
        num = chunk.provision_number
        if kind == "section":
            return f"s.{num}"
        if kind == "schedule":
            return f"Sch.{num.upper()}"
        if kind == "schedule_para":
            # Recover schedule number from the canonical provision_id.
            sch = ""
            if chunk.provision_id.startswith("sch_") and "_para_" in chunk.provision_id:
                sch = chunk.provision_id.removeprefix("sch_").split("_para_", 1)[0]
            return f"Sch.{sch.upper()} para.{num}" if sch else f"para.{num}"
        if kind == "ground":
            sch = ""
            if chunk.provision_id.startswith("sch_") and "_ground_" in chunk.provision_id:
                sch = chunk.provision_id.removeprefix("sch_").split("_ground_", 1)[0]
            return f"Sch.{sch.upper()} Ground {num}" if sch else f"Ground {num}"

    # Fallback: try to peel a label from the chunk_id.
    cid = chunk.chunk_id
    if "::s_" in cid:
        return f"s.{cid.rsplit('::s_', 1)[1]}"
    if "::sch_" in cid:
        tail = cid.rsplit("::", 1)[1]
        if "_ground_" in tail:
            sch, gnd = tail.removeprefix("sch_").split("_ground_", 1)
            return f"Sch.{sch.upper()} Ground {gnd}"
        if "_para_" in tail:
            sch, para = tail.removeprefix("sch_").split("_para_", 1)
            return f"Sch.{sch.upper()} para.{para}"
        return f"Sch.{tail.removeprefix('sch_').upper()}"
    return ""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(chunks: list[RetrievalResult]) -> str:
    """Render retrieved chunks as a structured context block."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        label = _provision_label(chunk)
        provision_line = f"Provision: {label}\n" if label else ""
        blocks.append(
            f"=== Source [{index}] ===\n"
            f"Act: {chunk.source_title}\n"
            f"{provision_line}"
            f"Headline: {chunk.headline}\n"
            f"Text:\n{chunk.original_text}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    chunks: list[RetrievalResult],
    *,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generate an answer grounded in the retrieved chunks."""
    if not chunks:
        return (
            "The provided sources do not cover this question. "
            "I can only answer about UK tenancy law - please rephrase "
            "or ask about a residential-tenancy topic."
        )

    context_block = _format_context(chunks)
    full_system_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context (your only source of truth):\n\n"
        f"{context_block}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": full_system_prompt}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = chat_completion(
        messages=messages,
        temperature=RAG.GENERATION_TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()

    log.info("generated answer (%d chars) from %d chunks", len(answer), len(chunks))
    return answer
