"""
End-to-end evaluation runner.

Usage (from the chatbot/ folder):

    # Smoke test - 5 questions, ~30 seconds, ~£0.05
    python -m eval.run_eval --config plain --limit 5
    python -m eval.run_eval --config agent --limit 5

    # Full run - all 100 questions, ~10-20 minutes, ~£3-5 per config
    python -m eval.run_eval --config plain
    python -m eval.run_eval --config agent

Output:
    eval/results/{config}_{timestamp}.jsonl  - one row per question
    A summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any

from tqdm import tqdm

from eval.dataset import TestQuestion, load_gold_questions
from eval.llm_judge import judge_answer
from eval.metrics import (
    average_mrr,
    average_ndcg,
    citation_validity_rate,
    expected_acts_recall,
    keyword_coverage,
)
from src.agents.citation_validator import CitationValidator
from src.agents.planner import PlannerAgent
from src.config import paths
from src.logging_setup import configure_logging
from src.rag.pipeline import answer_question

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic refusal detector
# ---------------------------------------------------------------------------
# We detect "did the system refuse?" by looking for any of a small set of
# stable substrings that the refusal templates and the generator's
# "no context" fallback always include. Heuristic, but deterministic and
# good enough for refusal-accuracy scoring.
# ---------------------------------------------------------------------------

_REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't help",
    "i can only",
    "i'm a uk tenancy",
    "consult a pharmacist",
    "consult a financial",
    "nhs 111",
    "the provided sources do not cover",
    "couldn't be verified",
    "i'd need a bit more",
)


def _detect_refusal(answer: str) -> bool:
    """True if the answer looks like a refusal or a 'no info' response."""
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def _eval_one_question(
    question: TestQuestion,
    config: str,
    planner: PlannerAgent | None,
    citator: CitationValidator,
) -> dict[str, Any]:
    """Run one question through the system and collect all metrics."""
    start = time.time()

    # 1. Run the system in the requested config.
    if config == "agent":
        assert planner is not None
        trace = planner.run(question.question)
        answer = trace.final_answer
        chunks = list(trace.chunks_used)
        citation_checks = list(trace.citations)
        rewritten = trace.rewritten_query
    else:   # "plain"
        rag_response = answer_question(question.question)
        answer = rag_response.answer
        chunks = list(rag_response.chunks_used)
        citation_checks = citator.validate(answer)
        rewritten = rag_response.rewritten_query

    latency_ms = int((time.time() - start) * 1000)

    # 2. Refusal logic (no LLM needed - heuristic match)
    refused = _detect_refusal(answer)
    refused_correctly = (refused == question.out_of_scope)

    # 3. Retrieval metrics (no LLM)
    mrr = average_mrr(question.keywords, chunks)
    ndcg = average_ndcg(question.keywords, chunks)
    coverage = keyword_coverage(question.keywords, chunks)
    acts_recall = expected_acts_recall(question.expected_acts, chunks)
    citation_rate = citation_validity_rate(citation_checks)

    # 4. LLM-as-judge - skip for refusal questions and refused outputs.
    if question.out_of_scope or refused:
        accuracy = completeness = relevance = None
        judge_feedback = None
    else:
        try:
            verdict = judge_answer(
                question.question, answer, question.reference_answer
            )
            accuracy = verdict.accuracy
            completeness = verdict.completeness
            relevance = verdict.relevance
            judge_feedback = verdict.feedback
        except Exception as exc:
            log.warning("judge failed for %s: %s", question.question_id, exc)
            accuracy = completeness = relevance = None
            judge_feedback = f"judge error: {exc}"

    return {
        "question_id": question.question_id,
        "category": question.category,
        "difficulty": question.difficulty,
        "out_of_scope": question.out_of_scope,
        "config": config,
        "system_answer": answer,
        "reference_answer": question.reference_answer,
        "rewritten_query": rewritten,
        "refused": refused,
        "refused_correctly": refused_correctly,
        "mrr": mrr,
        "ndcg": ndcg,
        "keyword_coverage": coverage,
        "expected_acts_recall": acts_recall,
        "citation_validity": citation_rate,
        "n_citations": len(citation_checks),
        "accuracy": accuracy,
        "completeness": completeness,
        "relevance": relevance,
        "judge_feedback": judge_feedback,
        "latency_ms": latency_ms,
        "n_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict[str, Any]], config: str) -> None:
    """Print an aggregate summary table to stdout."""
    successful = [r for r in results if "error" not in r]
    if not successful:
        print("(no successful evaluations)")
        return

    n = len(successful)

    def avg(field: str, *, skip_none: bool = False) -> float:
        values = [r.get(field) for r in successful]
        if skip_none:
            values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else 0.0

    def frac_true(field: str) -> float:
        values = [r.get(field) for r in successful]
        return (sum(1 for v in values if v) / len(values)) if values else 0.0

    print()
    print("=" * 60)
    print(f"  Summary - {config} mode  ({n} questions)")
    print("=" * 60)
    print(f"  Retrieval")
    print(f"    MRR avg:                   {avg('mrr'):.4f}")
    print(f"    NDCG avg:                  {avg('ndcg'):.4f}")
    print(f"    Keyword coverage avg:      {avg('keyword_coverage'):.4f}")
    print(f"    Expected-acts recall avg:  {avg('expected_acts_recall'):.4f}")
    print(f"  Generation")
    print(f"    Citation validity avg:     {avg('citation_validity'):.4f}")
    print(f"    Refusal accuracy:          {frac_true('refused_correctly'):.4f}")
    print(f"  LLM-as-judge (1-5 scale)")
    print(f"    Accuracy avg:              {avg('accuracy', skip_none=True):.2f}")
    print(f"    Completeness avg:          {avg('completeness', skip_none=True):.2f}")
    print(f"    Relevance avg:             {avg('relevance', skip_none=True):.2f}")
    print(f"  Latency")
    print(f"    Latency avg (ms):          {avg('latency_ms'):.0f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LexUK evaluation.")
    parser.add_argument(
        "--config",
        choices=["plain", "agent"],
        required=True,
        help="System configuration to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of questions (for smoke tests).",
    )
    args = parser.parse_args()

    configure_logging()
    paths.ensure()

    questions = load_gold_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    log.info("evaluating %d questions in '%s' config", len(questions), args.config)

    planner = PlannerAgent() if args.config == "agent" else None
    citator = CitationValidator()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = paths.EVAL_RESULTS / f"{args.config}_{timestamp}.jsonl"

    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as fout:
        for question in tqdm(questions, desc=f"eval[{args.config}]"):
            try:
                result = _eval_one_question(question, args.config, planner, citator)
            except Exception as exc:
                log.error("question %s failed: %s", question.question_id, exc)
                result = {
                    "question_id": question.question_id,
                    "config": args.config,
                    "error": str(exc),
                }
            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print()
    print(f"[OK] wrote {len(results)} results to {output_path}")
    _print_summary(results, args.config)


if __name__ == "__main__":
    main()