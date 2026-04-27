"""
Quick inspector for an eval results JSONL file.

Usage (from chatbot/):

    python -m eval.inspect_results eval\results\plain_<timestamp>.jsonl
    python -m eval.inspect_results eval\results\plain_<timestamp>.jsonl --bad-only

`--bad-only` shows rows with accuracy < 3 OR refusal mismatch only,
so it's easy to find the failing cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _truncate(text: str | None, limit: int = 250) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a results JSONL file.")
    parser.add_argument(
        "--bad-only",
        action="store_true",
        help="Only show rows with accuracy<3 or a refusal mismatch.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    with args.path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    shown = 0
    for row in rows:
        accuracy = row.get("accuracy")
        is_bad = (
            (accuracy is not None and accuracy < 3)
            or row.get("refused_correctly") is False
            or "error" in row
        )
        if args.bad_only and not is_bad:
            continue

        shown += 1
        print("=" * 90)
        print(
            f"[{row['question_id']}] "
            f"category={row.get('category')} "
            f"difficulty={row.get('difficulty')} "
            f"oos={row.get('out_of_scope')}"
        )
        print(f"Q  : {_truncate(row.get('question'), 200)}")
        print(f"Ref: {_truncate(row.get('reference_answer'))}")
        print(f"Sys: {_truncate(row.get('system_answer'))}")
        print(
            f"refused={row.get('refused')}  "
            f"refused_correctly={row.get('refused_correctly')}  "
            f"n_chunks={row.get('n_chunks')}  "
            f"n_citations={row.get('n_citations')}  "
            f"citation_validity={row.get('citation_validity')}"
        )
        scores = [
            f"{name}={row[name]}"
            for name in ("accuracy", "completeness", "relevance")
            if row.get(name) is not None
        ]
        print(f"judge: {' | '.join(scores) if scores else 'skipped'}")
        if row.get("judge_feedback"):
            print(f"feedback: {_truncate(row['judge_feedback'], 400)}")
        if "error" in row:
            print(f"error: {row['error']}")

    print("=" * 90)
    print(f"({shown}/{len(rows)} rows shown)")


if __name__ == "__main__":
    main()