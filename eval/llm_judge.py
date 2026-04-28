"""
LLM-as-a-Judge for answer quality.

Scores each generated answer on three dimensions - accuracy, completeness,
and relevance - each from 1 to 5 against a hand-written reference answer.

Bias mitigations (Zheng et al. 2023, Wang et al. 2024):

    1. Self-preference: judge with `gpt-4o`, a tier above the system
       under test (`gpt-4o-mini`).
    2. Position: each answer is scored independently against the
       reference - no pairwise comparison.
    3. Verbosity: the rubric explicitly penalises padding via the
       relevance dimension.
    4. Score compression: the rubric anchors the extremes ("score 1 if
       wrong; score 5 ONLY for perfect").

Public surface:

    judge_answer(question, generated_answer, reference_answer) -> AnswerEval
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.llm_client import structured_completion

log = logging.getLogger(__name__)


# Stronger model than the generator. Justified in the report's Methodology
# section as a self-preference-bias mitigation.
JUDGE_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------

class AnswerEval(BaseModel):
    """LLM-as-judge verdict on one generated answer."""

    feedback: str = Field(description="2-3 sentences explaining the scores.")
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


# ---------------------------------------------------------------------------
# Rubric prompt (extreme-anchored, per the bias-mitigation literature)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert evaluator of generated answers in the UK tenancy-law "
    "domain. You will be given the user's question, a generated answer, and "
    "a reference (gold) answer. Score the generated answer on three "
    "dimensions, each from 1 to 5:\n\n"

    "1. ACCURACY - is it factually correct compared with the reference?\n"
    "   1 = factually wrong (wrong rule, wrong figure, wrong outcome).\n"
    "   3 = factually correct in substance, but cites a different but "
    "related section, paragraph or schedule than the reference.\n"
    "   5 = factually correct AND cites the same section as the reference.\n"
    "   Treat citation-format differences as a 3, not a 1, when the rule "
    "and outcome match the reference.\n\n"

    "2. COMPLETENESS - does it cover all the information from the reference?\n"
    "   1 = missing critical information; 5 = covers everything.\n"
    "   Score 5 ONLY if every fact present in the reference is reflected in "
    "the generated answer.\n\n"

    "3. RELEVANCE - how directly does it answer the question without padding?\n"
    "   1 = off-topic; 5 = directly answers, no extraneous content.\n"
    "   Score 5 ONLY if no irrelevant material is added.\n\n"

    "Then give 2-3 sentences of feedback explaining the scores."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
) -> AnswerEval:
    """Score a generated answer against the reference using the judge model."""
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Generated Answer:\n{generated_answer}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Provide your scores."
    )

    verdict = structured_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        schema=AnswerEval,
        model=JUDGE_MODEL,
        temperature=0.0,
    )
    log.info(
        "judged: acc=%d comp=%d rel=%d",
        verdict.accuracy, verdict.completeness, verdict.relevance,
    )
    return verdict