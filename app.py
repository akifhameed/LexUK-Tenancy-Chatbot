"""
Gradio UI for the LexUK chatbot.

Three-panel layout:

    +----------------------------+--------------------------+
    | Chat (chat history)        | Agent / retrieval trace  |
    | + textbox                  | (live, per-question)     |
    +----------------------------+--------------------------+

Usage (from chatbot/ folder):

    python app.py

Then open http://127.0.0.1:7860 in a browser.

The toggle at the top selects between:
    * Plain RAG  - rewrite -> retrieve -> rerank -> generate
    * Agentic    - Planner with tool-use loop on top of plain RAG

Both modes drive the same underlying corpus and produce the same trace
shape, so the UI code is shared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gradio as gr

from src.agents.planner import AgentTrace, PlannerAgent
from src.logging_setup import configure_logging
from src.rag.pipeline import RagResponse, answer_question
from src.rag.vector_store import RetrievalResult


configure_logging()
log = logging.getLogger(__name__)


# Global Planner instance (one per process).
_planner: PlannerAgent | None = None


def _get_planner() -> PlannerAgent:
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner


# ---------------------------------------------------------------------------
# Trace rendering helpers
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    answer: str
    rewritten_query: str
    chunks: list[RetrievalResult]
    citations: list                # CitationCheck list (agent only)
    tool_calls: list[dict]
    refused: bool


def _from_rag(response: RagResponse) -> TurnResult:
    return TurnResult(
        answer=response.answer,
        rewritten_query=response.rewritten_query,
        chunks=list(response.chunks_used),
        citations=[],
        tool_calls=[],
        refused=False,
    )


def _from_agent(trace: AgentTrace) -> TurnResult:
    return TurnResult(
        answer=trace.final_answer,
        rewritten_query=trace.rewritten_query,
        chunks=list(trace.chunks_used),
        citations=list(trace.citations),
        tool_calls=list(trace.tool_calls),
        refused=trace.refused,
    )


def _render_trace(result: TurnResult, mode: str) -> str:
    """Build the Markdown shown in the trace panel."""
    parts: list[str] = []
    parts.append(f"### Trace - {mode} mode")

    if mode == "Agentic" and result.tool_calls:
        parts.append("**Tool calls:**")
        for index, call in enumerate(result.tool_calls, start=1):
            args_render = ", ".join(f"{k}={v!r}" for k, v in call.get("args", {}).items())
            parts.append(f"{index}. `{call['name']}({args_render})`")

    if result.rewritten_query:
        parts.append(f"**Rewritten query:** _{result.rewritten_query}_")

    if result.chunks:
        parts.append(f"**Chunks used ({len(result.chunks)}):**")
        for index, chunk in enumerate(result.chunks, start=1):
            url = chunk.source_url or ""
            url_md = f" [link]({url})" if url else ""
            parts.append(
                f"{index}. **{chunk.source_title}** -- {chunk.headline}"
                f" (distance {chunk.distance:.3f}){url_md}"
            )

    if result.citations:
        valid = sum(1 for c in result.citations if c.valid)
        total = len(result.citations)
        parts.append(f"**Citations:** {valid}/{total} valid")
        for c in result.citations:
            mark = "OK" if c.valid else "X"
            parts.append(f"- `{mark}` {c.raw}")

    if result.refused:
        parts.append("**Refused:** yes (off-domain)")

    if not result.chunks and not result.tool_calls:
        parts.append("_(no retrieval - direct answer or refusal)_")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(
    message: str,
    history: list[dict[str, str]],
    mode: str,
) -> tuple[str, list[dict[str, str]], str]:
    """Handle one user turn. Returns (clear_textbox, new_history, trace_md).

    `history` is in Gradio 6 messages format: a list of
    {"role": "user"|"assistant", "content": "..."} dicts. This is the
    same shape the OpenAI Chat Completions API expects, so we pass it
    straight through to the planner / RAG pipeline.
    """
    if not message or not message.strip():
        return "", history, ""

    history = history or []

    if mode == "Agentic":
        planner = _get_planner()
        trace = planner.run(message, history=history)
        result = _from_agent(trace)
    else:
        response = answer_question(message, history=history)
        result = _from_rag(response)

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
    trace_md = _render_trace(result, mode)
    return "", new_history, trace_md


def clear_all() -> tuple[list, str]:
    return [], "_(history cleared)_"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_INTRO_MD = """
# LexUK - UK Tenancy Law Assistant

Ask questions about residential tenancies grounded in 12 UK statutes
(Housing Act 1988, Housing Act 2004, Tenant Fees Act 2019, Renters' Rights
Act 2025, and others). Citations link to the relevant section on
[legislation.gov.uk](https://www.legislation.gov.uk/).

**Modes**
- _Plain_: rewrite -> retrieve -> rerank -> generate
- _Agentic_: planner with tool-use (search, validate citations, refuse, clarify)

This is information about UK statutes, not legal advice.
"""


def build_demo() -> gr.Blocks:
    # `theme` belongs on Blocks() in Gradio 5.x. (In 6.x it's on launch();
    # we target 5.x because HF Spaces auto-installs 5.49.)
    with gr.Blocks(title="LexUK", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_INTRO_MD)

        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=["Plain", "Agentic"],
                    value="Agentic",
                    label="Mode",
                    interactive=True,
                )
                # type="messages" matches the OpenAI-style dict format
                # we use in `respond()`; silences a Gradio deprecation
                # warning about defaulting to the legacy "tuples" format.
                chatbot = gr.Chatbot(height=520, type="messages")
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="e.g. What is a section 21 notice?",
                        scale=4,
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear conversation")

            with gr.Column(scale=1):
                gr.Markdown("## Trace")
                trace_panel = gr.Markdown("_(Trace will appear after your first question.)_")

        # Wire events.
        msg.submit(respond, [msg, chatbot, mode], [msg, chatbot, trace_panel])
        send_btn.click(respond, [msg, chatbot, mode], [msg, chatbot, trace_panel])
        clear_btn.click(clear_all, None, [chatbot, trace_panel])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # Note: launch() in Gradio 5.x does NOT accept `theme` (that's on
    # Blocks). Also do NOT use share=True or auth=() on Hugging Face
    # Spaces - they break the Space's iframe routing.
    demo.launch(
        server_name="0.0.0.0",   # 0.0.0.0 so HF Space's container exposes us
        server_port=7860,
        show_error=True,
    )
