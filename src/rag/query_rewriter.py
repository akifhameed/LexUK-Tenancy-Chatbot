"""
Query rewriter for the RAG pipeline.

User questions are messy ("my mate's landlord is being a dick about
the deposit, what can he do?"); legal text is formal ("section 214 of
the Housing Act 2004"). The rewriter bridges the gap by asking GPT to
produce a short, search-optimised version of the user's question.

It also resolves references using recent conversation history, so a
follow-up like "and how much can I claim?" gets rewritten to a query
that mentions the original subject ("tenancy deposit compensation").

Public surface:

    rewrite_query(question, history=None) -> str
"""

from __future__ import annotations

import logging

from src.llm_client import chat_completion

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# Kept short and direct. The model's only job is to produce the rewritten
# query - no commentary, no quoting, no punctuation beyond what the query
# itself needs.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a query rewriter for a UK tenancy-law retrieval system. "
    "Convert the user's question into a short, specific search query "
    "(typically 6-12 words) suitable for embedding-based retrieval over "
    "UK statutes. Use any conversation history to resolve pronouns or "
    "implicit subjects. Use formal legal terms where you can (for "
    "example 'section 21 notice', 'assured shorthold tenancy', 'tenancy "
    "deposit protection'). "
    "\n\n"
    "Reply ONLY with the rewritten query - no preamble, no quotes, no "
    "trailing punctuation."
)


# ---------------------------------------------------------------------------
# Helper - format conversation history for the prompt
# ---------------------------------------------------------------------------
# We only feed the last 3 exchanges (6 messages). Older context rarely
# helps and bloats the prompt.
# ---------------------------------------------------------------------------

def _format_history(history: list[dict[str, str]] | None) -> str:
    """Return a short text rendering of the last few conversation turns."""
    if not history:
        return ""
    recent = history[-6:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Rewrite a user question into a retrieval-optimised query.

    Args:
        question: Raw user input.
        history:  Optional list of prior {role, content} messages.

    Returns:
        The rewritten query as a single line of plain text.
    """
    user_prompt = (
        f"{_format_history(history)}"
        f"User's current question: {question}\n\n"
        f"Rewritten query:"
    )

    response = chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    rewritten = response.choices[0].message.content.strip()
    log.debug("rewrote: %r -> %r", question, rewritten)
    return rewritten