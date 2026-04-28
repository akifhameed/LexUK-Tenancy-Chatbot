"""
Tiny password gate for the live evaluation tab.

Reads the expected password from the EVAL_PASSWORD environment variable
so it can be set as a Hugging Face Space Secret in production. Falls
back to a default for local development - that default should never be
considered secret.

Public surface:

    verify_password(input_password) -> bool
"""

from __future__ import annotations

import hmac
import os


# Local-dev default. Override on Hugging Face by setting
# EVAL_PASSWORD as a Space Secret.
_DEFAULT_DEV_PASSWORD: str = "lexuk-eval-2026"


def _get_expected_password() -> str:
    """Return the configured expected password (env var or local fallback)."""
    return os.getenv("EVAL_PASSWORD", _DEFAULT_DEV_PASSWORD)


def verify_password(submitted: str | None) -> bool:
    """
    Constant-time check of a submitted password against the configured one.

    Returns False for missing / empty input; True only on exact match.
    Constant-time comparison guards against trivial timing attacks - not
    that anyone will care, but it costs nothing.
    """
    if not submitted:
        return False
    expected = _get_expected_password()
    return hmac.compare_digest(submitted.strip(), expected)
