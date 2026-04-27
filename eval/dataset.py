"""
Pydantic schema and loader for the gold question set.

The schema is the authoritative definition of a TestQuestion - both the
CSV converter and the eval runner go through this class, so any change
to the gold-data format only needs to be made here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.config import paths


class TestQuestion(BaseModel):
    """One row from the gold dataset."""

    question_id: str
    question: str
    category: str
    keywords: list[str] = Field(default_factory=list)
    reference_answer: str
    expected_citations: list[str] = Field(default_factory=list)
    expected_acts: list[str] = Field(default_factory=list)
    difficulty: str
    out_of_scope: bool = False


def load_gold_questions(path: Path | None = None) -> list[TestQuestion]:
    """Load every question from JSONL into validated TestQuestion objects."""
    target = path or paths.GOLD_QUESTIONS
    if not target.exists():
        raise FileNotFoundError(
            f"Gold questions JSONL not found: {target}. "
            f"Run `python -m eval.csv_to_jsonl` first."
        )

    questions: list[TestQuestion] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(TestQuestion.model_validate_json(line))
    return questions