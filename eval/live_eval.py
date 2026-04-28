"""
In-process evaluation runner for the Gradio Evaluation tab.

This is a thin wrapper over `eval.run_eval._eval_one_question` that:

    * Accepts an in-memory list of TestQuestion (no file I/O for input).
    * Accepts a list of modes ("plain", "agent", or both).
    * Reports progress via a callback (so the Gradio UI can show a bar).
    * Returns an `EvalRunResult` dict containing per-question rows and
      aggregate summaries, ready for the visualisation module.

The CLI driver (`eval.run_eval`) still exists for reproducible offline
benchmarking; this module only adds a UI-friendly entry point.

Public surface:

    run_live_eval(
        questions, modes, progress_cb, max_questions, max_modes
    ) -> EvalRunResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from eval.dataset import TestQuestion
from eval.run_eval import _eval_one_question
from src.agents.citation_validator import CitationValidator
from src.agents.planner import PlannerAgent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety caps - protect the API budget from runaway eval runs
# ---------------------------------------------------------------------------

MAX_QUESTIONS_PER_RUN: int = 50
MAX_MODES_PER_RUN: int = 2

ALLOWED_MODES: tuple[str, ...] = ("plain", "agent")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EvalRunResult:
    """The complete output of one live evaluation run."""

    # Echo back what was run
    questions: list[TestQuestion] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)

    # Per-question metric rows, keyed by mode
    rows_by_mode: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Aggregate summaries, keyed by mode
    summary_by_mode: dict[str, dict[str, float | int]] = field(default_factory=dict)

    # Run-level metadata
    duration_seconds: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Aggregation helpers (kept private; exposed only via run_live_eval)
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Compute the same aggregate fields as `run_eval._print_summary`.

    None values (judge skipped on refusals) are excluded from averages
    so they don't pull the means down.
    """
    successful = [r for r in rows if "error" not in r]
    n = len(successful)
    if n == 0:
        return {}

    def avg(field_name: str, *, skip_none: bool = False) -> float:
        values = [r.get(field_name) for r in successful]
        if skip_none:
            values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else 0.0

    def frac_true(field_name: str) -> float:
        values = [r.get(field_name) for r in successful]
        return (sum(1 for v in values if v) / len(values)) if values else 0.0

    return {
        "n_questions": n,
        # Retrieval
        "mrr": avg("mrr"),
        "ndcg": avg("ndcg"),
        "keyword_coverage": avg("keyword_coverage"),
        "expected_acts_recall": avg("expected_acts_recall"),
        # Generation
        "citation_validity": avg("citation_validity"),
        "refusal_accuracy": frac_true("refused_correctly"),
        # LLM judge (skip None: refusal rows aren't judged)
        "judge_accuracy": avg("accuracy", skip_none=True),
        "judge_completeness": avg("completeness", skip_none=True),
        "judge_relevance": avg("relevance", skip_none=True),
        # Performance
        "latency_ms_avg": avg("latency_ms"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_live_eval(
    questions: list[TestQuestion],
    modes: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
) -> EvalRunResult:
    """
    Run the supplied questions through the chosen mode(s) and return all
    metrics, rows, and timing.

    Args:
        questions:    The list of TestQuestion to evaluate.
        modes:        Which configurations to run - any subset of
                      ("plain", "agent").
        progress_cb:  Optional callback (fraction in [0,1], status text)
                      invoked after every (mode, question) pair completes.
                      Used by Gradio's gr.Progress.

    Returns:
        An EvalRunResult with per-mode rows and aggregate summaries.

    Raises:
        ValueError: if `questions` or `modes` are empty / over the cap.
    """
    if not questions:
        raise ValueError("No questions supplied.")
    if len(questions) > MAX_QUESTIONS_PER_RUN:
        raise ValueError(
            f"Too many questions ({len(questions)}); "
            f"cap is {MAX_QUESTIONS_PER_RUN}."
        )
    if not modes:
        raise ValueError("No modes selected (need at least 'plain' or 'agent').")
    bad = [m for m in modes if m not in ALLOWED_MODES]
    if bad:
        raise ValueError(f"Unknown mode(s): {bad}")
    if len(modes) > MAX_MODES_PER_RUN:
        raise ValueError("Too many modes (max 2).")

    # Set up shared per-mode state once - cheaper than per-question.
    citator = CitationValidator()
    planner = PlannerAgent() if "agent" in modes else None

    rows_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    total_steps = len(questions) * len(modes)
    step = 0
    started = time.time()

    for q_idx, question in enumerate(questions, start=1):
        for mode in modes:
            step += 1
            status = f"[{step}/{total_steps}] {question.question_id} - {mode}"
            log.info(status)
            if progress_cb is not None:
                progress_cb(step / total_steps, status)

            try:
                row = _eval_one_question(
                    question=question,
                    config=mode,
                    planner=planner,
                    citator=citator,
                )
            except Exception as exc:
                log.error("question %s (%s) failed: %s",
                          question.question_id, mode, exc)
                row = {
                    "question_id": question.question_id,
                    "category": question.category,
                    "difficulty": question.difficulty,
                    "out_of_scope": question.out_of_scope,
                    "config": mode,
                    "error": str(exc),
                }
            rows_by_mode[mode].append(row)

    duration = time.time() - started

    summary_by_mode = {
        mode: _aggregate(rows_by_mode[mode]) for mode in modes
    }

    return EvalRunResult(
        questions=list(questions),
        modes=list(modes),
        rows_by_mode=rows_by_mode,
        summary_by_mode=summary_by_mode,
        duration_seconds=duration,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# Manual-entry helper - validate a Gradio Dataframe row into a TestQuestion
# ---------------------------------------------------------------------------

def manual_row_to_test_question(
    row_idx: int, row: dict[str, Any]
) -> TestQuestion:
    """
    Turn one row from a manual-entry Dataframe into a validated TestQuestion.

    Splits semicolon-delimited list cells, normalises types, raises a
    descriptive ValueError if anything is missing or malformed.
    """
    def _split(value: Any) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in str(value).split(";") if item.strip()]

    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().upper() in {"TRUE", "1", "YES", "Y"}

    try:
        return TestQuestion(
            question_id=str(row.get("question_id") or f"manual_{row_idx:03d}"),
            question=str(row["question"]).strip(),
            category=str(row.get("category", "direct_fact")).strip(),
            keywords=_split(row.get("keywords", "")),
            reference_answer=str(row.get("reference_answer", "")).strip(),
            expected_citations=_split(row.get("expected_citations", "")),
            expected_acts=_split(row.get("expected_acts", "")),
            difficulty=str(row.get("difficulty", "medium")).strip(),
            out_of_scope=_to_bool(row.get("out_of_scope", False)),
        )
    except KeyError as exc:
        raise ValueError(
            f"Row {row_idx}: missing required field {exc}"
        ) from exc
