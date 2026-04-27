"""
StatuteAgent - wraps the RAG pipeline as a callable tool.

The Planner invokes this agent's `search()` method whenever it decides
the user's question is a genuine UK tenancy query. Internally it just
forwards to `src.rag.pipeline.answer_question`, but it exposes the
result through an Agent identity so the trace panel can attribute the
work correctly.
"""

from __future__ import annotations

from src.agents.base import Agent
from src.rag.pipeline import RagResponse, answer_question


class StatuteAgent(Agent):
    """RAG-backed agent that answers UK tenancy questions from statutes."""

    name = "Statute"
    colour = Agent.BLUE

    def search(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RagResponse:
        """Run the full RAG pipeline for `question` and return the response."""
        self.announce(f"searching statutes for: {question!r}")
        response = answer_question(question, history=history)
        self.announce(
            f"returned answer ({len(response.answer)} chars; "
            f"best distance {response.best_distance:.4f}; "
            f"{len(response.chunks_used)} chunks used)"
        )
        return response