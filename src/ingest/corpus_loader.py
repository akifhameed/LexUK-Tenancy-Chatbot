"""
Corpus loader for UK tenancy statutes.

Reads every Markdown file in `data/markdown_12_statutes/` and returns a
list of `Document` objects. Each document carries:

    * filename       - source file on disk (used as a stable identifier)
    * title          - human-readable Act name (e.g. "Housing Act 1988")
    * year           - year of enactment (used for filtering / display)
    * source_url     - canonical legislation.gov.uk URL
    * text           - full Markdown content of the file

Why a dedicated metadata table?

    The gov.uk URL needs the Act's *chapter number* (e.g. 1988 c.50),
    which cannot be inferred from the filename. Hard-coding the table
    once, here, is far simpler than parsing the XML again. If you add
    a 13th statute later, add one row to STATUTE_METADATA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import paths

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Statute metadata
# ---------------------------------------------------------------------------
# Filename (stem only, no .md extension) -> metadata.
# `chapter` is the official chapter number used in legislation.gov.uk URLs.
# Source: https://www.legislation.gov.uk/ukpga/{year}/{chapter}
# ---------------------------------------------------------------------------

STATUTE_METADATA: dict[str, dict[str, str | int]] = {
    "rent_act_1977": {
        "title": "Rent Act 1977",
        "year": 1977,
        "chapter": 42,
    },
    "protection_from_eviction_act_1977": {
        "title": "Protection from Eviction Act 1977",
        "year": 1977,
        "chapter": 43,
    },
    "landlord_and_tenant_act_1985": {
        "title": "Landlord and Tenant Act 1985",
        "year": 1985,
        "chapter": 70,
    },
    "landlord_and_tenant_act_1987": {
        "title": "Landlord and Tenant Act 1987",
        "year": 1987,
        "chapter": 31,
    },
    "housing_act_1988": {
        "title": "Housing Act 1988",
        "year": 1988,
        "chapter": 50,
    },
    "commonhold_and_leasehold_reform_act_2002": {
        "title": "Commonhold and Leasehold Reform Act 2002",
        "year": 2002,
        "chapter": 15,
    },
    "housing_act_2004": {
        "title": "Housing Act 2004",
        "year": 2004,
        "chapter": 34,
    },
    "immigration_act_2014": {
        "title": "Immigration Act 2014",
        "year": 2014,
        "chapter": 22,
    },
    "deregulation_act_2015": {
        "title": "Deregulation Act 2015",
        "year": 2015,
        "chapter": 20,
    },
    "homelessness_reduction_act_2017": {
        "title": "Homelessness Reduction Act 2017",
        "year": 2017,
        "chapter": 13,
    },
    "tenant_fees_act_2019": {
        "title": "Tenant Fees Act 2019",
        "year": 2019,
        "chapter": 4,
    },
    "renters_rights_act_2025": {
        "title": "Renters' Rights Act 2025",
        "year": 2025,
        "chapter": 0,   # update once the chapter number is published
    },
}


# ---------------------------------------------------------------------------
# 2. Document model
# ---------------------------------------------------------------------------
# A plain dataclass is sufficient here - no validation needed beyond what
# the loader guarantees. Frozen so a Document can be safely passed around
# without mutation surprises.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """One UK statute loaded from disk."""

    filename: str       # e.g. "housing_act_1988"
    title: str          # e.g. "Housing Act 1988"
    year: int           # e.g. 1988
    source_url: str     # canonical legislation.gov.uk URL
    text: str           # full Markdown content


# ---------------------------------------------------------------------------
# 3. Public loader
# ---------------------------------------------------------------------------

def load_statutes() -> list[Document]:
    """
    Load every Markdown statute and return a list of `Document` objects.

    Order is deterministic (sorted by filename) so downstream caching and
    chunk IDs are stable across runs.

    Raises:
        FileNotFoundError: if the statutes folder is missing.
        KeyError:          if a markdown file has no entry in STATUTE_METADATA.
    """
    folder = paths.STATUTES_MD
    if not folder.exists():
        raise FileNotFoundError(f"Statutes folder not found: {folder}")

    documents: list[Document] = []

    for md_path in sorted(folder.glob("*.md")):
        stem = md_path.stem  # filename without extension

        if stem not in STATUTE_METADATA:
            raise KeyError(
                f"No metadata for '{stem}'. "
                f"Add it to STATUTE_METADATA in corpus_loader.py."
            )

        meta = STATUTE_METADATA[stem]
        text = md_path.read_text(encoding="utf-8")

        documents.append(Document(
            filename=stem,
            title=str(meta["title"]),
            year=int(meta["year"]),
            source_url=_build_source_url(int(meta["year"]), int(meta["chapter"])),
            text=text,
        ))

        log.info("loaded %s (%d chars)", stem, len(text))

    log.info("loaded %d statutes from %s", len(documents), folder)
    return documents


# ---------------------------------------------------------------------------
# 4. Internal helpers
# ---------------------------------------------------------------------------

def _build_source_url(year: int, chapter: int) -> str:
    """Return the canonical legislation.gov.uk URL for a UK Public General Act."""
    if chapter == 0:
        # Chapter not yet published (e.g. very recent Acts) - fall back to
        # the year-level page so links still work.
        return f"https://www.legislation.gov.uk/ukpga/{year}"
    return f"https://www.legislation.gov.uk/ukpga/{year}/{chapter}"