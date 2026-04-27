"""
Tool schemas the Planner exposes to GPT via function calling.

Each tool is a JSON-schema dict matching OpenAI's tool-use format. The schemas are passed to the chat-completions
API; GPT uses them to decide which tool to call and with what arguments.

The actual Python implementations live in the matching agent modules:

    search_statutes     -> StatuteAgent.search()
    validate_citations  -> CitationValidator.validate()
    refuse              -> RefusalAgent.refuse()
    clarify             -> handled inline in the Planner

Public surface:

    ALL_TOOLS  - list of all tool schemas, ready to pass to chat_completion()
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# 1. search_statutes - the main RAG tool
# ---------------------------------------------------------------------------

_SEARCH_STATUTES: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_statutes",
        "description": (
            "Search the UK tenancy statutes corpus for an answer to the user's "
            "question. Returns a generated answer with inline citations and the "
            "list of source chunks used. Use this for any genuine UK tenancy "
            "question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question, in their own words.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# 2. validate_citations - pure-Python validator (no LLM)
# ---------------------------------------------------------------------------

_VALIDATE_CITATIONS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "validate_citations",
        "description": (
            "Validate that each citation in an answer (e.g. [Housing Act 1988, "
            "s.21]) refers to an Act and section that actually appears in the "
            "corpus. Returns a list of valid/invalid checks. ALWAYS call this "
            "after search_statutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The generated answer text containing citations.",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# 3. refuse - polite refusal for off-domain queries
# ---------------------------------------------------------------------------

_REFUSE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "refuse",
        "description": (
            "Refuse the user's question if it is off-domain. Use this for "
            "medical, criminal, financial, or general-immigration queries, or "
            "for pure small talk. Returns a polite refusal message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["medical", "criminal", "financial",
                             "immigration", "small_talk", "off_topic"],
                    "description": "Short reason category for the refusal.",
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# 4. clarify - ask the user a follow-up
# ---------------------------------------------------------------------------

_CLARIFY: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "clarify",
        "description": (
            "Ask the user a follow-up question if their query is too "
            "ambiguous to answer (e.g. no jurisdiction, no tenancy type, no "
            "facts). The follow-up must be specific and short."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "follow_up_question": {
                    "type": "string",
                    "description": "The clarifying question to ask the user.",
                },
            },
            "required": ["follow_up_question"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Public list (in priority order GPT typically prefers)
# ---------------------------------------------------------------------------

ALL_TOOLS: list[dict[str, Any]] = [
    _SEARCH_STATUTES,
    _VALIDATE_CITATIONS,
    _REFUSE,
    _CLARIFY,
]