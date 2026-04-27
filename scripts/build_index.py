"""
Build the vector index from the cached chunks.

Usage (from the chatbot/ folder):

    python -m scripts.build_index             # incremental upsert (idempotent)
    python -m scripts.build_index --fresh     # wipe collection first

Output:

    A populated ChromaDB at data/chroma_db/, plus a final summary line
    showing the total chunk count.
"""

from __future__ import annotations

import argparse

from src.config import RAG
from src.ingest.chunker import load_cached_chunks
from src.logging_setup import configure_logging
from src.rag.vector_store import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index LexUK chunks into Chroma.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe the existing collection before indexing.",
    )
    args = parser.parse_args()

    configure_logging()

    # 1. Load every cached chunk - no LLM calls, just JSONL reads.
    chunks = load_cached_chunks()
    if not chunks:
        raise SystemExit(
            "No chunks found in data/chunks_cache/. "
            "Run `python -m scripts.build_chunks` first."
        )

    # 2. Embed and persist.
    final_count = index_chunks(chunks, fresh=args.fresh)

    print()
    print(f"[OK] indexed {final_count} chunks into Chroma collection "
          f"'{RAG.COLLECTION_NAME}'")
    print(f"[OK] persisted under data/chroma_db/")


if __name__ == "__main__":
    main()