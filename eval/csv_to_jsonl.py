"""
Convert the gold CSV into JSONL.

Usage (from the chatbot/ folder):

    python -m eval.csv_to_jsonl
    python -m eval.csv_to_jsonl --input eval/gold_questions_100.csv

Reads the CSV (semicolon-separated lists for keywords / citations /
acts) and writes one JSON object per line to eval/gold_questions.jsonl.

Robust to Excel's UTF-8-with-BOM encoding (utf-8-sig).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from eval.dataset import TestQuestion
from src.config import paths
from src.logging_setup import configure_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_list_cell(field: str) -> list[str]:
    """Split a `; `-separated CSV cell into a clean list of strings."""
    if not field or not field.strip():
        return []
    return [item.strip() for item in field.split(";") if item.strip()]


def _parse_bool_cell(field: str) -> bool:
    """CSVs store booleans as strings; accept TRUE/FALSE/1/0/YES/NO."""
    return field.strip().upper() in {"TRUE", "1", "YES"}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def csv_to_jsonl(input_path: Path, output_path: Path) -> int:
    """Convert one CSV to JSONL. Returns the number of rows written."""
    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        for row in reader:
            question = TestQuestion(
                question_id=row["question_id"].strip(),
                question=row["question"].strip(),
                category=row["category"].strip(),
                keywords=_split_list_cell(row.get("keywords", "")),
                reference_answer=row["reference_answer"].strip(),
                expected_citations=_split_list_cell(row.get("expected_citations", "")),
                expected_acts=_split_list_cell(row.get("expected_acts", "")),
                difficulty=row["difficulty"].strip(),
                out_of_scope=_parse_bool_cell(row.get("out_of_scope", "FALSE")),
            )
            fout.write(question.model_dump_json() + "\n")
            rows_written += 1
    return rows_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gold CSV to JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=paths.EVAL / "gold_questions_100.csv",
        help="Path to the gold CSV (default: eval/gold_questions_100.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.GOLD_QUESTIONS,
        help="Output JSONL path (default: eval/gold_questions.jsonl).",
    )
    args = parser.parse_args()

    configure_logging()
    paths.ensure()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    rows = csv_to_jsonl(args.input, args.output)
    print(f"[OK] converted {rows} questions to {args.output}")


if __name__ == "__main__":
    main()