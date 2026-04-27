"""
RefusalAgent - returns templated refusals for off-domain queries.

The Planner calls `refuse(reason=...)` whenever it decides a question is
out of LexUK's scope. Templated rather than LLM-generated, so the
refusal text is always identical, well-tested, and free of cost.
"""

from __future__ import annotations

from src.agents.base import Agent


# ---------------------------------------------------------------------------
# Per-reason refusal texts
# ---------------------------------------------------------------------------
# Each refusal: acknowledges scope, redirects to a real service. Written
# to feel helpful rather than dismissive.
# ---------------------------------------------------------------------------

_REFUSAL_TEMPLATES: dict[str, str] = {
    "medical": (
        "I can't help with medical questions. I'm a UK tenancy-law "
        "assistant. For health advice please contact a pharmacist, your "
        "GP, or call NHS 111."
    ),
    "criminal": (
        "I can't help with criminal-law questions. I'm a UK tenancy-law "
        "assistant. For criminal-law issues please contact the police "
        "(101 for non-emergency) or a criminal-law solicitor."
    ),
    "financial": (
        "I can't help with financial advice. I'm a UK tenancy-law "
        "assistant. For financial guidance please contact a regulated "
        "adviser or visit MoneyHelper (moneyhelper.org.uk)."
    ),
    "immigration": (
        "I can only answer immigration questions in the narrow context of "
        "Right-to-Rent checks under the Immigration Act 2014. For wider "
        "immigration matters please consult an immigration solicitor or "
        "the official Home Office guidance at gov.uk."
    ),
    "small_talk": (
        "Hello! I'm LexUK, an assistant for UK tenancy law. Ask me "
        "anything about residential tenancies, landlords, deposits, "
        "repairs, eviction, or related topics."
    ),
    "off_topic": (
        "I can only help with UK tenancy law - questions about "
        "residential tenancies, landlords, tenants, deposits, eviction, "
        "repairs, and related statutes. Could you rephrase as a tenancy "
        "question?"
    ),
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RefusalAgent(Agent):
    """Returns a templated refusal message for a given reason."""

    name = "Refuser"
    colour = Agent.RED

    def refuse(self, reason: str) -> str:
        """Return the templated refusal for `reason`. Falls back to off_topic."""
        message = _REFUSAL_TEMPLATES.get(reason, _REFUSAL_TEMPLATES["off_topic"])
        self.announce(f"refused (reason={reason})")
        return message