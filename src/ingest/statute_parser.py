"""
Deterministic statute structure parser.

Parses UK legislation Markdown (legislation.gov.uk export) into a list of
provisions. No LLM call - regex + state machine over the line stream.

Handles the two formats produced by the legislation.gov.uk PDF -> Markdown
pipeline:

    Inline format:
        "21 Recovery of possession on expiry of an AST."

    Two-line format (more common):
        "21"
        "Recovery of possession on expiry of an AST."

Aggressively rejects footnote-table content (citation strings like
"para. 10 (with ss. 82(3)...") that would otherwise be matched as
section headings.

Produces canonical provision identifiers used throughout the system:

    s_21              # section 21
    s_189a            # section 189A
    sch_1             # Schedule 1 heading
    sch_1_para_3      # paragraph 3 of Schedule 1
    sch_2_ground_8    # ground 8 of Schedule 2
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provision:
    """One detected statutory provision with its verbatim text."""

    kind: str            # "section" | "schedule" | "schedule_para" | "ground"
    number: str          # canonical lowercase, e.g. "21", "189a", "2", "8"
    title: str           # short heading (after the number)
    text: str            # verbatim provision content (header + body)
    canonical_id: str    # stable id used as chunk_id suffix
    parent_id: str       # for paras / grounds, the parent schedule id


# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

# Schedule heading.
_SCHEDULE_LINE = re.compile(
    r"^SCHEDULE\s+([0-9A-Z]+)(?:\s*[—–\-]\s*(.+))?$",
    re.IGNORECASE,
)

# Possession ground heading inside a schedule.
_GROUND_LINE = re.compile(
    r"^Ground\s+([0-9]+[A-Z]{0,3})\b\.?\s*(.*)$",
    re.IGNORECASE,
)

# Inline numbered heading: "21 Recovery of possession ..."
# Number 1-3 digits + optional suffix letters (e.g. "189A", "11ZA").
_INLINE_NUMBERED = re.compile(
    r"^([0-9]{1,3}[A-Z]{0,3})\s+(.{5,200})$",
)

# Bare number alone on a line.
_BARE_NUMBER = re.compile(r"^([0-9]{1,3}[A-Z]{0,3})$")

# Footnote markers.
_FOOTNOTE_MARKER = re.compile(r"^[FCIM]\d+[A-Z]?$")


# ---------------------------------------------------------------------------
# Noise-line filtering
# ---------------------------------------------------------------------------

_NOISE_PREFIXES: tuple[str, ...] = (
    "## Page",
    "Document Generated",
    "Status:",
    "Changes to legislation",
    "Textual Amendments",
    "Modifications etc",
    "Commencement Information",
    "Marginal Citations",
    "Editorial Information",
    "Subordinate Legislation",
    "View outstanding changes",
    "Annotations",
    "## ",          # other markdown headings
)


def _is_noise(line: str) -> bool:
    """True for blank lines, page furniture, status banners, footnote markers."""
    if not line:
        return True
    for prefix in _NOISE_PREFIXES:
        if line.startswith(prefix):
            return True
    if _FOOTNOTE_MARKER.fullmatch(line):
        return True
    return False


# ---------------------------------------------------------------------------
# Title cleanliness checks - the difference between a real heading and a
# footnote-table entry that happens to start with a number.
# ---------------------------------------------------------------------------

# Anything starting with these abbreviations is a citation / amendment note.
_FOOTNOTE_PREFIXES: tuple[str, ...] = (
    "para.", "para ", "Sch.", "Sch ", "ss.", "ss ",
    "s.", "s ", "art.", "art ", "reg.", "reg ",
    "S.I.", "S. I.", "c.", "c ",
    "(with", "(see", "(but", "(as", "(except", "(subject",
)

# Page-banner / amendment-banner phrases that the legislation.gov.uk Markdown
# repeats on every page. These are NEVER real section / paragraph titles.
_BANNER_TITLE_PREFIXES: tuple[str, ...] = (
    "Status:",
    "Status :",
    "Changes to legislation",
    "Document Generated",
    "Modifications etc",
    "Modifications etc.",
    "Textual Amendments",
    "Commencement Information",
    "Editorial Information",
    "Marginal Citations",
    "Annotations",
    "View outstanding changes",
    "Subordinate Legislation",
    "Continued",
    "PART ",
    "Part ",
    "CHAPTER ",
    "Chapter ",
)


def _looks_like_citation_string(title: str) -> bool:
    """Detect citation/footnote style titles like 'para. 10 (with ss. 82(3)...'."""
    t = title.strip()
    if not t:
        return True

    # Lowercase abbreviation prefixes are dead giveaways.
    for prefix in _FOOTNOTE_PREFIXES:
        if t.startswith(prefix):
            return True

    # Multiple parenthetical citations are dead giveaways too.
    if t.count("(") >= 2 and (" with " in t or "Sch." in t or " ss. " in t):
        return True

    # Lots of digits in the first 30 chars usually means citations.
    head = t[:30]
    digits = sum(1 for c in head if c.isdigit())
    if digits >= 6:
        return True

    return False


def _is_clean_title(title: str) -> bool:
    """Heuristic: this looks like a real section/paragraph heading."""
    t = title.strip().rstrip(".")
    if len(t) < 5 or len(t) > 200:
        return False
    # Must start with an uppercase letter (real titles do).
    if not t[0].isupper():
        return False
    # Reject page-banner / amendment-banner lines outright. These are the
    # killer false positives for the two-line heading detector: a bare page
    # number "11" sits above "Status: This version of this Act ..." and
    # without this guard the parser thinks "Status: ..." is the title of
    # section 11.
    for banner in _BANNER_TITLE_PREFIXES:
        if t.startswith(banner):
            return False
    # Reject when first token is a footnote marker like "F1".
    first_tok = t.split()[0] if t.split() else ""
    if _FOOTNOTE_MARKER.fullmatch(first_tok):
        return False
    if _looks_like_citation_string(t):
        return False
    # Title should be majority alphabetic in its first 50 chars.
    head = t[:50]
    alpha = sum(1 for c in head if c.isalpha() or c == " ")
    return alpha >= max(20, len(head) * 0.65)


def _safe_title(title: str) -> str:
    """Clean up whitespace and trailing punctuation in a heading."""
    t = re.sub(r"\s+", " ", title or "").strip(" .,;:")
    return t[:160]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_provisions(markdown: str) -> list[Provision]:
    """Parse a statute's Markdown into a list of `Provision` records."""
    lines = markdown.splitlines()

    # Pass 1 - collect heading boundaries.
    Boundary = tuple[int, str, str, str, str, str]
    boundaries: list[Boundary] = []
    current_schedule_id: str = ""

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if _is_noise(line) or not line:
            i += 1
            continue

        # 1. Schedule heading.
        m = _SCHEDULE_LINE.match(line)
        if m:
            num = m.group(1).lower()
            title = _safe_title(m.group(2) or "")
            current_schedule_id = f"sch_{num}"
            boundaries.append((i, "schedule", num, title, "", current_schedule_id))
            i += 1
            continue

        # 2. Ground heading (only meaningful inside a schedule).
        if current_schedule_id:
            m = _GROUND_LINE.match(line)
            if m:
                gnum = m.group(1).lower()
                gtitle = _safe_title(m.group(2) or "")
                gid = f"{current_schedule_id}_ground_{gnum}"
                boundaries.append((i, "ground", gnum, gtitle, current_schedule_id, gid))
                i += 1
                continue

        # 3. Inline format: "21 Recovery of possession ..."
        m = _INLINE_NUMBERED.match(line)
        if m:
            num = m.group(1).lower()
            title = m.group(2).strip()
            if len(num) <= 5 and _is_clean_title(title):
                kind, parent, canonical = _classify(num, current_schedule_id)
                boundaries.append((i, kind, num, _safe_title(title), parent, canonical))
                i += 1
                continue

        # 4. Two-line format: bare number on its own, title on next non-blank line.
        m = _BARE_NUMBER.fullmatch(line)
        if m:
            num = m.group(1).lower()
            if len(num) <= 5:
                # Look ahead for the title.
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if _is_clean_title(next_line):
                        kind, parent, canonical = _classify(num, current_schedule_id)
                        boundaries.append(
                            (i, kind, num, _safe_title(next_line), parent, canonical)
                        )
                        i = j + 1
                        continue

        i += 1

    # Pass 2 - slice text between consecutive boundaries.
    provisions: list[Provision] = []
    for j, (line_idx, kind, num, title, parent, canonical) in enumerate(boundaries):
        start = line_idx
        end = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        body_lines = lines[start:end]
        text = "\n".join(body_lines).strip()

        # Drop too-small fragments (less than ~3 short lines of content).
        if len(text) < 80:
            continue

        # Cap at ~18,000 chars (~5K tokens) to stay safely under the
        # embedding model's 8,192-token limit even with overhead.
        if len(text) > 18_000:
            text = text[:18_000].rstrip() + "\n\n[truncated for embedding]"

        provisions.append(Provision(
            kind=kind, number=num, title=title, text=text,
            canonical_id=canonical, parent_id=parent,
        ))

    # Deduplicate by canonical_id (keep first occurrence - the operative one).
    seen: set[str] = set()
    deduped: list[Provision] = []
    for p in provisions:
        if p.canonical_id in seen:
            continue
        seen.add(p.canonical_id)
        deduped.append(p)

    return deduped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(
    num: str,
    current_schedule_id: str,
) -> tuple[str, str, str]:
    """Return (kind, parent_id, canonical_id) given numbered-heading context."""
    if current_schedule_id:
        return (
            "schedule_para",
            current_schedule_id,
            f"{current_schedule_id}_para_{num}",
        )
    return ("section", "", f"s_{num}")
