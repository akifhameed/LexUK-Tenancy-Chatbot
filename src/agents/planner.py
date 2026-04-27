"""
PlannerAgent - orchestrates tools using OpenAI function calling.

This is the one true agent in the system: its `run()` method is a loop
that asks GPT what to do next, dispatches the chosen tool, feeds the
result back, and continues until GPT returns a non-tool-call message.

Public surface:

    PlannerAgent()
    .run(question, history=None) -> AgentTrace

The returned `AgentTrace` carries everything the UI / eval harness
needs: the final answer, the chunks used, the citation checks, and a
log of tool calls in order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import Agent
from src.agents.citation_validator import CitationCheck, CitationValidator
from src.agents.refusal_agent import RefusalAgent
from src.agents.statute_agent import StatuteAgent
from src.agents.tools import ALL_TOOLS
from src.config import AGENT
from src.llm_client import chat_completion
from src.rag.vector_store import RetrievalResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------
# Numbered rules. GPT follows numbered rules more reliably than prose.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Planner for LexUK, an assistant for UK tenancy law.

You orchestrate the following tools:
  search_statutes(question)     - Look up an answer in the UK tenancy statutes corpus.
  validate_citations(answer)    - Verify that an answer's citations point to real sections.
  refuse(reason)                - Refuse off-domain queries (medical, criminal, financial,
                                   immigration outside Right-to-Rent, small_talk, off_topic).
  clarify(follow_up_question)   - Ask the user a follow-up if the question is ambiguous.

Workflow rules:
  1. If the question is clearly off-domain, call `refuse` with the appropriate reason.
     Do NOT call search_statutes for off-domain queries.
  2. If the question is genuine UK tenancy law:
       a. Call `search_statutes` with the user's question.
       b. Then call `validate_citations` on the answer text.
       c. Produce a final user-facing message that contains the answer.
          If any citations were marked invalid, prepend a single sentence:
          "Note: one or more citations could not be verified."
  3. If the question is ambiguous (no jurisdiction / no facts / extremely vague),
     call `clarify` with a specific short follow-up.
  4. Be efficient: one search_statutes call per question, then one validate_citations.
  5. NEVER call `refuse` for a question that mentions UK statutes, tenancies,
     landlords, deposits, evictions, harassment, repairs, possession, leasehold,
     commonhold, or related residential-property topics. If `search_statutes`
     returns a 'no info' response for such a question, produce a final
     user-facing message that relays it - do NOT call `refuse`. Refuse is
     reserved for clearly off-domain queries only (medical, criminal, financial
     advice, generic chitchat).
"""


# ---------------------------------------------------------------------------
# Trace dataclass - everything the UI / eval needs
# ---------------------------------------------------------------------------

@dataclass
class AgentTrace:
    """Captures the full execution trace of one planner run."""

    iterations: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    raw_answer: str = ""
    chunks_used: list[RetrievalResult] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    refused: bool = False
    rewritten_query: str = ""
    best_distance: float = 1.0


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

# Tools that should terminate the loop the moment they fire - their output
# is meant to BE the final user message, not material the planner reasons
# further about.
_TERMINAL_TOOLS: set[str] = {"refuse", "clarify", "validate_citations"}


class PlannerAgent(Agent):
    """The orchestrator. One instance per chat session."""

    name = "Planner"
    colour = Agent.GREEN

    def __init__(self) -> None:
        super().__init__()
        self.statute = StatuteAgent()
        self.validator = CitationValidator()
        self.refuser = RefusalAgent()

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> AgentTrace:
        """Execute the tool-use loop for one user question."""
        self.announce(f"received question: {question!r}")
        trace = AgentTrace()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        for iteration in range(AGENT.MAX_TOOL_ITERATIONS):
            trace.iterations = iteration + 1

            response = chat_completion(
                messages=messages,
                tools=ALL_TOOLS,
                temperature=AGENT.PLANNER_TEMPERATURE,
            )
            choice = response.choices[0]
            message = choice.message

            # Append the assistant's response to the running conversation,
            # whether or not it contained tool calls. The OpenAI SDK returns
            # a Pydantic-style object; we serialise it back to dict form.
            messages.append(_serialise_assistant_message(message))

            if choice.finish_reason != "tool_calls":
                # GPT produced a plain answer - we're done.
                trace.final_answer = message.content or ""
                self.announce("finished")
                return trace

            # Otherwise dispatch each tool call in order.
            terminate = False
            for tool_call in message.tool_calls or []:
                if self._dispatch(tool_call, messages, trace, history):
                    terminate = True
            if terminate:
                return trace

        # Iteration cap reached - graceful fallback.
        log.warning("planner hit iteration cap (%d)", AGENT.MAX_TOOL_ITERATIONS)
        trace.final_answer = (
            "I had trouble producing an answer for that question. "
            "Please try rephrasing it."
        )
        return trace

    # -----------------------------------------------------------------------
    # Tool dispatch - returns True if the loop should terminate immediately.
    # -----------------------------------------------------------------------

    def _dispatch(
        self,
        tool_call: Any,
        messages: list[dict[str, Any]],
        trace: AgentTrace,
        history: list[dict[str, str]] | None,
    ) -> bool:
        """Run one tool call. Returns True if it's a terminal tool."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        self.announce(f"tool_call: {name}({args})")
        trace.tool_calls.append({"name": name, "args": args})

        if name == "search_statutes":
            content = self._do_search(args, trace, history)
        elif name == "validate_citations":
            content = self._do_validate(args, trace)
        elif name == "refuse":
            content = self._do_refuse(args, trace)
        elif name == "clarify":
            content = self._do_clarify(args, trace)
        else:
            content = f"unknown tool: {name}"
            self.log.error(content)

        # Append the tool result so GPT can read it on the next iteration.
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": content,
        })

        return name in _TERMINAL_TOOLS

    # -----------------------------------------------------------------------
    # Per-tool handlers
    # -----------------------------------------------------------------------

    def _do_search(
        self,
        args: dict[str, Any],
        trace: AgentTrace,
        history: list[dict[str, str]] | None,
    ) -> str:
        question = args.get("question", "")
        rag_response = self.statute.search(question, history=history)

        # Capture for the trace and the eval harness.
        trace.raw_answer = rag_response.answer
        trace.chunks_used = list(rag_response.chunks_used)
        trace.rewritten_query = rag_response.rewritten_query
        trace.best_distance = rag_response.best_distance

        return json.dumps({
            "answer": rag_response.answer,
            "chunks_used": [
                {
                    "source": chunk.source_title,
                    "headline": chunk.headline,
                }
                for chunk in rag_response.chunks_used
            ],
        })

    def _do_validate(self, args: dict[str, Any], trace: AgentTrace) -> str:
        answer = args.get("answer", "") or trace.raw_answer
        checks = self.validator.validate(answer)
        trace.citations = checks
        trace.final_answer = answer
        if any(not check.valid for check in checks):
            trace.final_answer = (
                "Note: one or more citations could not be verified.\n\n"
                f"{answer}"
            )
        return json.dumps({
            "checks": [
                {"raw": c.raw, "valid": c.valid, "matched_act": c.matched_act}
                for c in checks
            ],
            "valid_count": sum(1 for c in checks if c.valid),
            "total_count": len(checks),
        })

    def _do_refuse(self, args: dict[str, Any], trace: AgentTrace) -> str:
        message = self.refuser.refuse(args.get("reason", "off_topic"))
        trace.refused = True
        trace.final_answer = message
        return message

    def _do_clarify(self, args: dict[str, Any], trace: AgentTrace) -> str:
        follow_up = args.get(
            "follow_up_question",
            "Could you give me a few more details so I can answer accurately?",
        )
        trace.final_answer = follow_up
        return follow_up


# ---------------------------------------------------------------------------
# Helper - serialise an OpenAI assistant message back to dict form so we
# can append it to the running messages list.
# ---------------------------------------------------------------------------

def _serialise_assistant_message(message: Any) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to the dict the API expects."""
    serialised: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        serialised["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return serialised
