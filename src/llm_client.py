"""
Shared OpenAI client with automatic retry on transient failures.

Why a wrapper module rather than calling `OpenAI()` everywhere?

    * One place to inject the API key.
    * One place to apply retry policy (rate limits, network blips).
    * One place to swap providers later (e.g. Anthropic, Groq) without
      touching feature code.

Public surface:

    chat_completion(messages, *, model=None, response_format=None, ...)
    structured_completion(messages, *, schema, model=None, ...)
    embed(texts, *, model=None) -> list[list[float]]
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

log = logging.getLogger(__name__)

# Type variable for `structured_completion` — preserves the Pydantic
# subclass passed in so the IDE knows the exact return type.
T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Single shared client. The OpenAI SDK is thread-safe; one instance is fine.
# ---------------------------------------------------------------------------

_client: OpenAI = OpenAI(api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
# Retry only on transient errors (rate limits, connection drops, generic
# API errors). Validation errors and other 4xx codes should fail fast.
#
# Exponential back-off: 1s, 2s, 4s, 8s, capped at 30s, max 5 attempts.
# ---------------------------------------------------------------------------

_RETRY_KWARGS: dict[str, Any] = dict(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)


# ---------------------------------------------------------------------------
# 1. Plain chat completion — used by the generator, query rewriter, planner.
# ---------------------------------------------------------------------------

@retry(**_RETRY_KWARGS)
def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Call OpenAI Chat Completions and return the raw response object.

    Returning the raw response (rather than just the text) lets the caller
    inspect tool_calls, finish_reason, usage, etc. The planner loop in
    particular needs all of these.

    Args:
        messages:    OpenAI-formatted message list.
        model:       Override default model from settings.
        temperature: Sampling temperature. 0.0 for deterministic outputs.
        max_tokens:  Optional response length cap.
        tools:       Optional function-calling tool schemas (Week 8 pattern).

    Returns:
        The full ChatCompletion response object.
    """
    return _client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# 2. Structured completion — Ed Donner's pattern from Week 5 Day 5.
#    Forces the model to return JSON matching a Pydantic schema.
#    Used by the chunker, reranker, and LLM-as-judge.
# ---------------------------------------------------------------------------

@retry(**_RETRY_KWARGS)
def structured_completion(
    messages: list[dict[str, str]],
    *,
    schema: type[T],
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """
    Call Chat Completions with structured-output (JSON-schema) parsing.

    Args:
        messages:    OpenAI-formatted message list.
        schema:      Pydantic model class describing the expected JSON.
        model:       Override default model from settings.
        temperature: Sampling temperature. Default 0.0 for stability.

    Returns:
        An instance of `schema`, validated and parsed.
    """
    response = _client.chat.completions.parse(
        model=model or settings.llm_model,
        messages=messages,
        temperature=temperature,
        response_format=schema,
    )
    return response.choices[0].message.parsed   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3. Embeddings — used by the indexer (offline) and the retriever (online).
#    Batched to amortise HTTP overhead.
# ---------------------------------------------------------------------------

@retry(**_RETRY_KWARGS)
def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """
    Embed a batch of texts and return their vectors in input order.

    Args:
        texts: List of strings to embed. Empty list returns empty list.
        model: Override default embedding model from settings.

    Returns:
        List of float vectors, same length and order as `texts`.
    """
    if not texts:
        return []
    response = _client.embeddings.create(
        model=model or settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]