"""
One-off sanity check for Batch 1.

Run from the chatbot/ folder:

    python -m scripts.smoke_test

Expected output (3 ticks):

    [OK] config loaded - STATUTES_MD = ...\\data\\markdown_12_statutes
    [OK] logging configured
    [OK] OpenAI reachable - answer: pong
"""

from __future__ import annotations

import logging

from src.config import paths, settings
from src.llm_client import chat_completion
from src.logging_setup import configure_logging


def main() -> None:
    # 1. Config loaded?
    paths.ensure()
    print(f"[OK] config loaded - STATUTES_MD = {paths.STATUTES_MD}")

    # 2. Logging works?
    configure_logging()
    log = logging.getLogger("smoke_test")
    log.info("logging configured")
    print("[OK] logging configured")

    # 3. OpenAI reachable?
    response = chat_completion(
        messages=[{"role": "user", "content": "Reply with a single word: pong"}],
        model=settings.llm_model,
    )
    answer = response.choices[0].message.content.strip()
    print(f"[OK] OpenAI reachable - answer: {answer}")


if __name__ == "__main__":
    main()