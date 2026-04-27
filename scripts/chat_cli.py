"""
Interactive command-line chat with the LexUK system.

Usage:

    python -m scripts.chat_cli                # plain RAG (Batch 3b)
    python -m scripts.chat_cli --agent        # agentic RAG (Batch 4)

In-chat commands:

    /trace      toggle the retrieval / agent trace panel
    /reset      clear conversation history
    /quit       exit

The --agent flag swaps the plain pipeline for the Planner agent, which
adds tool routing (refuse / clarify) and citation validation on top.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.agents.planner import AgentTrace, PlannerAgent
from src.logging_setup import configure_logging
from src.rag.pipeline import RagResponse, answer_question


# ANSI palette for terminal output.
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_GREY = "\033[90m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# A tiny adapter so both modes return the same shape to the print loop.
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    answer: str
    rewritten_query: str
    best_distance: float
    chunks: list                # list[RetrievalResult]
    citations: list             # list[CitationCheck], empty for plain mode
    tool_calls: list            # list[dict],          empty for plain mode
    refused: bool = False


def _from_rag(response: RagResponse) -> TurnResult:
    return TurnResult(
        answer=response.answer,
        rewritten_query=response.rewritten_query,
        best_distance=response.best_distance,
        chunks=list(response.chunks_used),
        citations=[],
        tool_calls=[],
    )


def _from_agent(trace: AgentTrace) -> TurnResult:
    return TurnResult(
        answer=trace.final_answer,
        rewritten_query=trace.rewritten_query,
        best_distance=trace.best_distance,
        chunks=list(trace.chunks_used),
        citations=list(trace.citations),
        tool_calls=list(trace.tool_calls),
        refused=trace.refused,
    )


# ---------------------------------------------------------------------------
# Trace renderer - one block per turn.
# ---------------------------------------------------------------------------

def _print_trace(result: TurnResult, agent_mode: bool) -> None:
    print(f"{_GREY}+- trace -{_RESET}")
    if agent_mode and result.tool_calls:
        print(f"{_GREY}| tool calls:{_RESET}")
        for index, call in enumerate(result.tool_calls, start=1):
            print(f"{_GREY}|   [{index}] {call['name']}({call['args']}){_RESET}")
    if result.rewritten_query:
        print(f"{_GREY}| rewritten query: {result.rewritten_query}{_RESET}")
    print(f"{_GREY}| best distance:   {result.best_distance:.4f}{_RESET}")
    if result.chunks:
        print(f"{_GREY}| chunks used:{_RESET}")
        for index, chunk in enumerate(result.chunks, start=1):
            print(
                f"{_GREY}|   [{index}] {chunk.source_title} "
                f"({chunk.source_year}) -> {chunk.headline}{_RESET}"
            )
    if result.citations:
        valid = sum(1 for c in result.citations if c.valid)
        total = len(result.citations)
        colour = _GREEN if valid == total else _RED
        print(f"{_GREY}| citations: {colour}{valid}/{total} valid{_RESET}")
        for check in result.citations:
            mark = "OK" if check.valid else "X "
            print(f"{_GREY}|   {mark}  {check.raw}{_RESET}")
    if result.refused:
        print(f"{_GREY}| refused: yes{_RESET}")
    print(f"{_GREY}+-{_RESET}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LexUK chat CLI.")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the Planner agent (Batch 4) instead of plain RAG (Batch 3b).",
    )
    args = parser.parse_args()

    configure_logging()

    mode_label = "agentic" if args.agent else "plain"
    print()
    print(f"{_BOLD}LexUK - UK Tenancy Law Assistant ({mode_label} mode){_RESET}")
    print(f"{_GREY}Commands: /trace (toggle), /reset (clear), /quit{_RESET}")
    print()

    history: list[dict[str, str]] = []
    show_trace = True
    planner = PlannerAgent() if args.agent else None

    while True:
        try:
            question = input(f"{_CYAN}You: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question == "/quit":
            break
        if question == "/trace":
            show_trace = not show_trace
            print(f"{_GREY}trace: {'on' if show_trace else 'off'}{_RESET}")
            continue
        if question == "/reset":
            history.clear()
            print(f"{_GREY}history cleared{_RESET}")
            continue

        if planner is not None:
            trace = planner.run(question, history=history)
            result = _from_agent(trace)
        else:
            response = answer_question(question, history=history)
            result = _from_rag(response)

        if show_trace:
            _print_trace(result, agent_mode=args.agent)

        print()
        print(f"{_GREEN}LexUK:{_RESET} {result.answer}")
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()