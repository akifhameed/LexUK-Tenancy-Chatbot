"""
Gradio UI for the LexUK chatbot.

Two top-level tabs:

    * Chat       - public; Plain / Agentic chatbot with live trace panel.
    * Evaluation - hidden until the operator unlocks it with a password
                   (set as the EVAL_PASSWORD secret on Hugging Face).

The Evaluation tab runs the same supervised metrics as the offline
`run_eval.py` driver, but in-process, with progress feedback and rich
visualisations rendered via matplotlib.

Local usage (from chatbot/ folder):

    python app.py           -> http://127.0.0.1:7860

On Hugging Face Spaces:

    Set EVAL_PASSWORD as a Space Secret (Settings -> Variables and secrets).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import gradio as gr
import pandas as pd

from eval.auth import verify_password
from eval.dataset import TestQuestion, load_gold_questions
from eval.live_eval import (
    EvalRunResult,
    manual_row_to_test_question,
    MAX_QUESTIONS_PER_RUN,
    run_live_eval,
)
from eval.visualizations import (
    fig_category_breakdown,
    fig_citation_health,
    fig_judge_score_heatmap,
    fig_latency_histogram,
    fig_metric_comparison,
    summary_dataframe,
)
from src.agents.planner import AgentTrace, PlannerAgent
from src.ingest.corpus_loader import STATUTE_METADATA
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
# CHAT TAB - existing logic, unchanged
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


def respond(
    message: str,
    history: list[dict[str, str]],
    mode: str,
) -> tuple[str, list[dict[str, str]], str]:
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
# EVALUATION TAB - login, question source, run, dashboards
# ---------------------------------------------------------------------------

# Categories the manual-entry dropdown offers. Match the gold-CSV values.
_CATEGORY_CHOICES: list[str] = [
    "direct_fact", "numerical", "spanning", "comparative",
    "temporal", "procedural", "refusal", "clarification",
]
_DIFFICULTY_CHOICES: list[str] = ["easy", "medium", "hard"]
_KNOWN_ACT_FILES: list[str] = sorted(STATUTE_METADATA.keys())

# Manual-entry table column schema.
_MANUAL_COLUMNS: list[str] = [
    "question_id",
    "question",
    "reference_answer",
    "keywords",
    "expected_citations",
    "expected_acts",
    "category",
    "difficulty",
    "out_of_scope",
]


def _empty_manual_table() -> pd.DataFrame:
    """Start the manual-entry Dataframe with one example row pre-filled."""
    return pd.DataFrame([{
        "question_id": "MQ001",
        "question": "What is the maximum holding deposit a landlord may take?",
        "reference_answer": (
            "Under Schedule 1, paragraph 3 of the Tenant Fees Act 2019, the "
            "maximum holding deposit is one week's rent."
        ),
        "keywords": "holding deposit; one week; Tenant Fees Act 2019",
        "expected_citations": "Tenant Fees Act 2019, Sch.1 para.3",
        "expected_acts": "tenant_fees_act_2019",
        "category": "numerical",
        "difficulty": "easy",
        "out_of_scope": False,
    }], columns=_MANUAL_COLUMNS)


def _load_built_in_count() -> int:
    """Count of questions in the bundled gold dataset."""
    try:
        return len(load_gold_questions())
    except Exception as exc:
        log.warning("could not load gold dataset: %s", exc)
        return 0


def _login(submitted_password: str) -> tuple[Any, Any, str]:
    """
    Verify the password and toggle the Evaluation tab visibility.

    Returns:
        (state_update, eval_tab_update, status_md)
    """
    if verify_password(submitted_password):
        return (
            True,
            gr.update(visible=True),
            "Logged in. Open the **Evaluation** tab.",
        )
    return (
        False,
        gr.update(visible=False),
        "Wrong password.",
    )


def _logout() -> tuple[Any, Any, str]:
    return False, gr.update(visible=False), "Logged out."


def _parse_uploaded_csv(filepath: str | None) -> list[TestQuestion]:
    """Read a user-uploaded CSV file using the gold-CSV schema."""
    import csv
    if not filepath:
        return []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        questions: list[TestQuestion] = []
        for index, row in enumerate(reader, start=1):
            questions.append(manual_row_to_test_question(index, row))
    return questions


def _gather_questions(
    source: str,
    uploaded_file: str | None,
    manual_df: pd.DataFrame,
) -> tuple[list[TestQuestion], str]:
    """
    Build the in-memory question list from the chosen source.

    Returns: (questions, status_message). On failure, returns ([], message).
    """
    try:
        if source == "Built-in gold set":
            qs = load_gold_questions()
            return qs, f"Loaded {len(qs)} built-in questions."

        if source == "Upload CSV":
            if not uploaded_file:
                return [], "No file uploaded."
            qs = _parse_uploaded_csv(uploaded_file)
            return qs, f"Parsed {len(qs)} questions from upload."

        # Manual entry
        if manual_df is None or manual_df.empty:
            return [], "Manual table is empty - add at least one row."
        qs: list[TestQuestion] = []
        for index, row in manual_df.iterrows():
            row_dict = row.to_dict()
            # Skip blank rows (no question text)
            if not str(row_dict.get("question", "")).strip():
                continue
            qs.append(manual_row_to_test_question(index, row_dict))
        if not qs:
            return [], "No filled rows found in the manual table."
        return qs, f"Parsed {len(qs)} manual questions."

    except Exception as exc:
        log.exception("question parsing failed")
        return [], f"Error: {exc}"


def _modes_from_radio(modes_choice: str) -> list[str]:
    """Map the radio label to internal mode names."""
    return {
        "Plain only": ["plain"],
        "Agent only": ["agent"],
        "Both (compare)": ["plain", "agent"],
    }.get(modes_choice, ["plain"])


def _run_evaluation(
    is_evaluator: bool,
    source: str,
    uploaded_file: str | None,
    manual_df: pd.DataFrame,
    modes_choice: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, Any, Any, Any, Any, Any, Any, Any]:
    """
    Top-level Gradio handler for the Run button.

    Returns updates for: status_md, summary_table, metric_fig,
    heatmap_fig, category_fig, latency_fig, citation_fig, raw_rows.
    """
    if not is_evaluator:
        return ("Not authorised.",) + (gr.update(),) * 7

    # Step 1: gather questions
    questions, status = _gather_questions(source, uploaded_file, manual_df)
    if not questions:
        return (status,) + (gr.update(),) * 7

    if len(questions) > MAX_QUESTIONS_PER_RUN:
        return (
            f"Too many questions ({len(questions)}); cap is "
            f"{MAX_QUESTIONS_PER_RUN}.",
        ) + (gr.update(),) * 7

    # Step 2: modes
    modes = _modes_from_radio(modes_choice)

    # Step 3: run with progress reporting via gr.Progress
    progress(0.0, desc="starting...")

    def _cb(fraction: float, status_text: str) -> None:
        progress(fraction, desc=status_text)

    try:
        result: EvalRunResult = run_live_eval(
            questions=questions,
            modes=modes,
            progress_cb=_cb,
        )
    except Exception as exc:
        log.exception("live eval failed")
        return (f"Eval failed: {exc}",) + (gr.update(),) * 7

    # Step 4: build dashboard outputs
    status_md = (
        f"Run finished in {result.duration_seconds:.1f} s "
        f"({len(result.questions)} questions x {len(result.modes)} mode(s))."
    )

    summary_df = summary_dataframe(result)

    metric_fig = fig_metric_comparison(result)
    heatmap_fig = fig_judge_score_heatmap(result)
    category_fig = fig_category_breakdown(result)
    latency_fig = fig_latency_histogram(result)
    citation_fig = fig_citation_health(result)

    # Per-question raw rows (concatenate across modes for table view)
    raw_rows: list[dict[str, Any]] = []
    for mode, rows in result.rows_by_mode.items():
        for row in rows:
            display = {
                "mode": mode,
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "refused": row.get("refused"),
                "n_chunks": row.get("n_chunks"),
                "mrr": row.get("mrr"),
                "ndcg": row.get("ndcg"),
                "kw_cov": row.get("keyword_coverage"),
                "cit_valid": row.get("citation_validity"),
                "acc": row.get("accuracy"),
                "comp": row.get("completeness"),
                "rel": row.get("relevance"),
                "ms": row.get("latency_ms"),
            }
            raw_rows.append(display)
    raw_df = pd.DataFrame(raw_rows)

    return (
        status_md,
        summary_df,
        metric_fig,
        heatmap_fig,
        category_fig,
        latency_fig,
        citation_fig,
        raw_df,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_INTRO_MD = """
# LexUK - UK Tenancy Law Assistant

Ask questions about residential tenancies grounded in 12 UK statutes
(Housing Act 1988, Housing Act 2004, Tenant Fees Act 2019, Renters' Rights
Act 2025, and others). Citations link to the relevant section on
[legislation.gov.uk](https://www.legislation.gov.uk/).

This is information about UK statutes, not legal advice.
"""

_EVAL_INTRO_MD = """
## Live Evaluation Dashboard

Run the supervised evaluation suite against any question set you choose,
in either Plain RAG, Agentic RAG, or both (for comparison). Six charts
plus a summary table render automatically when the run finishes.
"""

_MANUAL_INPUTS_HELP_MD = """
### How to fill the manual table

Each row is one evaluation question. **The question and reference_answer
columns are required**; everything else has sensible defaults but improves
the metrics. Lists use `; ` as the separator inside a cell.

| Column | Required? | What to enter | Example |
|---|---|---|---|
| `question_id` | Optional | Stable identifier; auto-generated if blank | `MQ002` |
| `question` | **Yes** | The user-style question | *"What is the maximum holding deposit a landlord may take?"* |
| `reference_answer` | **Yes** | The gold-standard answer the LLM judge compares against | *"Under Schedule 1, paragraph 3 of the Tenant Fees Act 2019, the maximum holding deposit is one week's rent."* |
| `keywords` | Recommended | Semicolon-separated keywords that should appear in retrieved chunks (used for MRR / NDCG / coverage). Leave blank for refusal questions. | `holding deposit; one week; Tenant Fees Act 2019` |
| `expected_citations` | Recommended | Semicolon-separated citations the system should produce, in `Act, s.X` format | `Tenant Fees Act 2019, Sch.1 para.3` |
| `expected_acts` | Recommended | Semicolon-separated source-file names (no `.md`). Used for expected-acts-recall. | `tenant_fees_act_2019` |
| `category` | Optional | One of `direct_fact`, `numerical`, `spanning`, `comparative`, `temporal`, `procedural`, `refusal`, `clarification` | `numerical` |
| `difficulty` | Optional | One of `easy`, `medium`, `hard` | `easy` |
| `out_of_scope` | **Yes for refusal Qs** | Tick (TRUE) only if the system *should* refuse this question | `False` |

**Valid `expected_acts` values** (the 12 corpus files):
`""" + "`, `".join(_KNOWN_ACT_FILES) + """`
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LexUK", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_INTRO_MD)

        # ------------------------------------------------------------
        # Login section (always visible at the top)
        # ------------------------------------------------------------
        is_evaluator = gr.State(False)

        with gr.Accordion("Evaluator login (operators only)", open=False):
            gr.Markdown(
                "Authorised users can unlock the **Evaluation** tab "
                "to run benchmarks live."
            )
            with gr.Row():
                pw_input = gr.Textbox(
                    label="Password", type="password",
                    placeholder="EVAL_PASSWORD", scale=3,
                )
                login_btn = gr.Button("Unlock", variant="primary", scale=1)
                logout_btn = gr.Button("Lock", scale=1)
            login_status = gr.Markdown("Locked.")

        # ------------------------------------------------------------
        # Tabs
        # ------------------------------------------------------------
        with gr.Tabs() as tabs:
            # ============= CHAT TAB ============================================
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        mode_radio = gr.Radio(
                            choices=["Plain", "Agentic"],
                            value="Agentic",
                            label="Mode",
                            interactive=True,
                        )
                        chatbot = gr.Chatbot(height=520, type="messages")
                        with gr.Row():
                            msg = gr.Textbox(
                                label="",
                                placeholder="e.g. What is a section 21 notice?",
                                scale=4, lines=2,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear conversation")

                    with gr.Column(scale=1):
                        gr.Markdown("## Trace")
                        trace_panel = gr.Markdown(
                            "_(Trace will appear after your first question.)_"
                        )

                msg.submit(respond, [msg, chatbot, mode_radio],
                           [msg, chatbot, trace_panel])
                send_btn.click(respond, [msg, chatbot, mode_radio],
                               [msg, chatbot, trace_panel])
                clear_btn.click(clear_all, None, [chatbot, trace_panel])

            # ============= EVALUATION TAB (hidden until login) ===============
            with gr.Tab("Evaluation", visible=False) as eval_tab:
                gr.Markdown(_EVAL_INTRO_MD)

                # --- Step 1: question source ---
                with gr.Group():
                    gr.Markdown("### Step 1 - Pick the questions")
                    source_radio = gr.Radio(
                        choices=["Built-in gold set", "Upload CSV", "Manual entry"],
                        value="Built-in gold set",
                        label="Question source",
                        interactive=True,
                    )

                    builtin_info = gr.Markdown(
                        f"Bundled gold dataset has **{_load_built_in_count()}** "
                        f"questions ready to run.",
                        visible=True,
                    )

                    upload_file = gr.File(
                        label="Upload a CSV (same columns as the bundled gold CSV)",
                        file_types=[".csv"], visible=False,
                    )

                    with gr.Group(visible=False) as manual_group:
                        gr.Markdown(_MANUAL_INPUTS_HELP_MD)
                        manual_table = gr.Dataframe(
                            value=_empty_manual_table(),
                            headers=_MANUAL_COLUMNS,
                            datatype=[
                                "str", "str", "str", "str", "str",
                                "str", "str", "str", "bool",
                            ],
                            row_count=(1, "dynamic"),
                            col_count=(len(_MANUAL_COLUMNS), "fixed"),
                            interactive=True,
                            label="Manual question table (rows can be added / edited)",
                        )

                # Toggle which input UI is visible based on the radio
                def _switch_source(choice: str):
                    return (
                        gr.update(visible=choice == "Built-in gold set"),
                        gr.update(visible=choice == "Upload CSV"),
                        gr.update(visible=choice == "Manual entry"),
                    )

                source_radio.change(
                    _switch_source,
                    [source_radio],
                    [builtin_info, upload_file, manual_group],
                )

                # --- Step 2: mode selection ---
                with gr.Group():
                    gr.Markdown("### Step 2 - Pick the system(s) to evaluate")
                    modes_radio = gr.Radio(
                        choices=["Plain only", "Agent only", "Both (compare)"],
                        value="Both (compare)",
                        label="Mode",
                        interactive=True,
                    )

                # --- Step 3: run ---
                with gr.Group():
                    gr.Markdown("### Step 3 - Run")
                    run_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
                    eval_status = gr.Markdown("_(no run yet)_")

                # --- Step 4: dashboards ---
                gr.Markdown("### Results")
                with gr.Tabs():
                    with gr.Tab("Summary table"):
                        summary_table = gr.Dataframe(
                            label="Aggregate metrics",
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Metric comparison"):
                        metric_fig = gr.Plot(label="Retrieval / generation / judge")
                    with gr.Tab("Judge heatmap"):
                        heatmap_fig = gr.Plot(label="Per-question judge scores")
                    with gr.Tab("Category breakdown"):
                        category_fig = gr.Plot(label="Mean accuracy by category")
                    with gr.Tab("Latency"):
                        latency_fig = gr.Plot(label="Latency distribution")
                    with gr.Tab("Citation health"):
                        citation_fig = gr.Plot(label="Valid vs invalid citations per Q")
                    with gr.Tab("Raw rows"):
                        raw_table = gr.Dataframe(
                            label="Per-question raw metric rows",
                            interactive=False,
                            wrap=True,
                        )

                run_btn.click(
                    _run_evaluation,
                    [is_evaluator, source_radio, upload_file, manual_table, modes_radio],
                    [eval_status, summary_table, metric_fig, heatmap_fig,
                     category_fig, latency_fig, citation_fig, raw_table],
                )

        # ----------------------------------------------------------------
        # Auth wiring
        # ----------------------------------------------------------------
        login_btn.click(_login, [pw_input], [is_evaluator, eval_tab, login_status])
        pw_input.submit(_login, [pw_input], [is_evaluator, eval_tab, login_status])
        logout_btn.click(_logout, None, [is_evaluator, eval_tab, login_status])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
