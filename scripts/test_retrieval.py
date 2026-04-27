"""
Quick retrieval smoke test.

Usage:

    python -m scripts.test_retrieval                    # uses default query
    python -m scripts.test_retrieval "your question"    # custom query

Prints the top-5 nearest chunks with distance scores. Useful for sanity-
checking the index without standing up the full RAG pipeline.
"""

from __future__ import annotations

import sys

from src.logging_setup import configure_logging
from src.rag.vector_store import query


_DEFAULT_QUERY = "How much notice is required for a section 21?"


def main() -> None:
    configure_logging()

    user_query = " ".join(sys.argv[1:]).strip() or _DEFAULT_QUERY

    print()
    print(f"Query: {user_query}")
    print("-" * 100)

    results = query(user_query, k=5)

    if not results:
        print("(no results)")
        return

    print(f"{'#':>3}  {'Distance':>9}  {'Source':<35}  Headline")
    print("-" * 100)
    for index, result in enumerate(results, start=1):
        source_label = f"{result.source_title} ({result.source_year})"
        print(f"{index:>3}  {result.distance:>9.4f}  "
              f"{source_label[:33]:<35}  {result.headline}")
    print()


if __name__ == "__main__":
    main()