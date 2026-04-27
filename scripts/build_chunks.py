"""
Run the ingestion pipeline end-to-end.

Usage (from the chatbot/ folder):

    python -m scripts.build_chunks            # uses cache where available
    python -m scripts.build_chunks --force    # ignore cache, re-chunk all

Output:

    JSONL files under data/chunks_cache/, one per statute, plus a summary
    table printed to the terminal.
"""

from __future__ import annotations

import argparse
import logging

from src.ingest.chunker import chunk_corpus
from src.ingest.corpus_loader import load_statutes
from src.logging_setup import configure_logging

log = logging.getLogger(__name__)


def main() -> None:
    # Parse CLI args - just one optional flag.
    parser = argparse.ArgumentParser(description="Build chunk cache for LexUK.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached JSONL files and re-chunk every statute.",
    )
    args = parser.parse_args()

    # Wire up logging once for this entrypoint.
    configure_logging()

    # 1. Load statutes from disk.
    documents = load_statutes()

    # 2. Chunk them (with or without cache).
    chunks = chunk_corpus(documents, force_rechunk=args.force)

    # 3. Print a summary table grouped by statute.
    by_source: dict[str, int] = {}
    for record in chunks:
        by_source[record.source_title] = by_source.get(record.source_title, 0) + 1

    print()
    print(f"{'Statute':<50} {'Chunks':>8}")
    print("-" * 60)
    for title, count in sorted(by_source.items()):
        print(f"{title:<50} {count:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<50} {len(chunks):>8}")
    print()


if __name__ == "__main__":
    main()