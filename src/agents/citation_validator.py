"""
CitationValidator - verifies that citations in answers refer to real
sections in the corpus.

This is the system's deterministic guard against hallucinated citations.
No LLM is used: regex extraction plus a substring check against the
chunk cache is enough to catch the common failure modes (invented
sections, wrong Act, wrong section number).

The "citation validity rate" computed by this module is one of the
novel metrics reported in the Performance section of the project.

Public surface:

    CitationValidator()
    .validate(answer) -> list[CitationCheck]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.agents.base import Agent
from src.ingest.chunker import load_cached_chunks


# ---------------------------------------------------------------------------
# Citation pattern
# ---------------------------------------------------------------------------
# Matches:  [Housing Act 1988, s.21]
#           [Tenant Fees Act 2019, Sch.1 para.3]
#           [Deregulation Act 2015, ss.33-41]
#           [Housing Act 2004, s.214(4)]
# Group 1 = Act name (everything before the comma)
# Group 2 = Section reference (after the comma, up to the closing bracket)
# ---------------------------------------------------------------------------

_CITATION_PATTERN = re.compile(
    r"\[([^,\]]+?),\s*([^\]]+?)\]"
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CitationCheck:
    """One citation extracted from the answer, with validity verdict."""

    raw: str            # the full bracketed citation as it appears in the text
    act: str            # the Act name as cited
    section: str        # the section reference as cited
    valid: bool         # True if the Act and section both appear in the corpus
    matched_act: str    # the Act title from the corpus that we matched against


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CitationValidator(Agent):
    """Regex-driven citation validator. Stateless after init."""

    name = "Citator"
    colour = Agent.YELLOW

    def __init__(self) -> None:
        super().__init__()
        # Build a per-Act concatenated text blob, lower-cased, for fast
        # substring lookups during validation. Loading the cache is cheap
        # (12 JSONL files) so we do it once at construction time.
        self._corpus_by_act: dict[str, str] = {}
        for chunk in load_cached_chunks():
            current = self._corpus_by_act.get(chunk.source_title, "")
            self._corpus_by_act[chunk.source_title] = (
                current + " " + chunk.original_text.lower()
            )

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def validate(self, answer: str) -> list[CitationCheck]:
        """Extract every citation from `answer` and check each one."""
        checks: list[CitationCheck] = []

        for match in _CITATION_PATTERN.finditer(answer):
            act_cited = match.group(1).strip()
            section_cited = match.group(2).strip()

            matched_act = self._match_act(act_cited)
            if matched_act is None:
                checks.append(CitationCheck(
                    raw=match.group(0),
                    act=act_cited,
                    section=section_cited,
                    valid=False,
                    matched_act="",
                ))
                continue

            valid = self._reference_present(matched_act, section_cited)
            checks.append(CitationCheck(
                raw=match.group(0),
                act=act_cited,
                section=section_cited,
                valid=valid,
                matched_act=matched_act,
            ))

        valid_count = sum(1 for c in checks if c.valid)
        self.announce(
            f"validated {len(checks)} citation(s); "
            f"{valid_count}/{len(checks)} valid"
        )
        return checks

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _match_act(self, cited_act: str) -> str | None:
        """Find a corpus Act whose title overlaps with the cited name."""
        cited_lower = cited_act.lower().strip()
        for known in self._corpus_by_act:
            known_lower = known.lower()
            # Either side may include qualifiers like "(as amended)".
            if known_lower in cited_lower or cited_lower in known_lower:
                return known
        return None

    def _reference_present(self, matched_act: str, section_cited: str) -> bool:
        """Check section, schedule paragraph, and ground-style citations."""
        text = self._corpus_by_act[matched_act]
        reference = section_cited.lower()

        schedule_match = re.search(
            r"\b(?:sch(?:edule)?\.?)\s*(\d+[a-z]*)",
            reference,
        )
        paragraph_match = re.search(
            r"\b(?:para(?:graph)?\.?)\s*(\d+[a-z]*)",
            reference,
        )
        ground_match = re.search(r"\bground\s*(\d+[a-z]*)", reference)
        section_match = re.search(
            r"\b(?:s|section|ss)\.?\s*(\d+[a-z]*)",
            reference,
        )

        if schedule_match:
            schedule_num = schedule_match.group(1)
            schedule_phrases = [
                f"schedule {schedule_num}",
                f"sch. {schedule_num}",
                f"sch.{schedule_num}",
            ]
            if not any(phrase in text for phrase in schedule_phrases):
                return False
            if paragraph_match:
                paragraph_num = paragraph_match.group(1)
                paragraph_phrases = [
                    f"para. {paragraph_num}",
                    f"para.{paragraph_num}",
                    f"paragraph {paragraph_num}",
                    f"\n{paragraph_num}\n",
                ]
                return any(phrase in text for phrase in paragraph_phrases)
            return True

        if ground_match:
            ground_num = ground_match.group(1)
            return f"ground {ground_num}" in text

        number_match = section_match or re.search(r"\d+", section_cited)
        if number_match is None:
            tokens = [t for t in re.split(r"\s+", reference) if t]
            return any(t in text for t in tokens)

        section_num = number_match.group(1) if section_match else number_match.group(0)
        candidate_phrases = [
            f"section {section_num}",
            f"s.{section_num}",
            f"s. {section_num}",
            f"\n{section_num}\n",
            f"§{section_num}",
        ]
        return any(phrase in text for phrase in candidate_phrases)

    def _section_present(self, matched_act: str, section_cited: str) -> bool:
        """Heuristic check: does this section reference appear in the Act?"""
        text = self._corpus_by_act[matched_act]

        # Pull the first integer out of the section reference - that's the
        # section number we expect to see somewhere in the corpus.
        number_match = re.search(r"\d+", section_cited)
        if number_match is None:
            # Schedule references etc. - look for any non-stop word from the
            # citation in the corpus.
            tokens = [t for t in re.split(r"\s+", section_cited.lower()) if t]
            return any(t in text for t in tokens)

        section_num = number_match.group(0)
        # Try the common ways UK statute text refers to a section.
        candidate_phrases = [
            f"section {section_num}",
            f"s.{section_num}",
            f"s. {section_num}",
            f"§{section_num}",
        ]
        return any(phrase in text for phrase in candidate_phrases)
