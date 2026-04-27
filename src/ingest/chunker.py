"""
Statute chunker - one ChunkRecord per detected provision.

This is the Phase 2 chunker: deterministic, no LLM call, ~10x faster
and far more accurate for highly structured legal text than the
prior LLM-based chunker. It depends entirely on
`src.ingest.statute_parser.parse_provisions` for boundary detection.

Each ChunkRecord carries provision metadata (provision_id,
provision_kind, provision_number) which the retriever uses for exact
metadata-filter lookups.

Public surface:

    chunk_corpus(documents, *, force_rechunk=False) -> list[ChunkRecord]
    chunk_document(doc) -> list[ChunkRecord]
    load_cached_chunks() -> list[ChunkRecord]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import paths
from src.ingest.corpus_loader import Document
from src.ingest.statute_parser import Provision, parse_provisions

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk record - the canonical data type used everywhere downstream
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkRecord:
    """A retrievable provision chunk with full provenance for citations."""

    chunk_id: str
    source_file: str
    source_title: str
    source_year: int
    source_url: str
    headline: str
    summary: str
    original_text: str
    # Provision metadata (filled by chunker, used by retriever for filtering).
    provision_id: str = ""        # canonical id, e.g. "s_21"
    provision_kind: str = ""      # "section" | "schedule" | "schedule_para" | "ground"
    provision_number: str = ""    # e.g. "21" or "189a"

    def to_embed_text(self) -> str:
        """Concatenated text used as the embedding input."""
        return f"# {self.headline}\n\n{self.summary}\n\n{self.original_text}"


# ---------------------------------------------------------------------------
# Per-provision rendering
# ---------------------------------------------------------------------------

def _humanise_provision_label(provision: Provision) -> str:
    """Render a provision as a human-readable inline label."""
    num = provision.number
    if provision.kind == "section":
        return f"Section {num.upper()}"
    if provision.kind == "schedule":
        return f"Schedule {num.upper()}"
    if provision.kind == "schedule_para":
        sch = provision.parent_id.removeprefix("sch_").upper()
        return f"Schedule {sch} paragraph {num}"
    if provision.kind == "ground":
        sch = provision.parent_id.removeprefix("sch_").upper()
        return f"Schedule {sch} Ground {num}"
    return num


def _build_headline(provision: Provision) -> str:
    """Headline always begins with the canonical provision label."""
    label = _humanise_provision_label(provision)
    title = provision.title.strip(" -—.")
    if title and len(title) > 2:
        return f"{label} - {title}"
    return label


def _build_summary(provision: Provision) -> str:
    """Cheap deterministic summary: title + first-sentence-ish text."""
    body = provision.text
    # Strip the first line if it's just the heading number.
    lines = body.splitlines()
    if lines and lines[0].strip().split()[:1] == [provision.number]:
        lines = lines[1:]
    body_clean = "\n".join(lines).strip()
    head = body_clean.split(". ")[0][:240].strip()
    if not head:
        head = provision.title or _humanise_provision_label(provision)
    label = _humanise_provision_label(provision)
    return f"{label}: {head}"


def _build_chunk_id(doc: Document, provision: Provision) -> str:
    return f"{doc.filename}::{provision.canonical_id}"


def _build_source_url(doc: Document, provision: Provision) -> str:
    """Best-effort deep link into legislation.gov.uk."""
    base = doc.source_url.rstrip("/")
    if provision.kind == "section":
        return f"{base}/section/{provision.number}"
    if provision.kind == "schedule":
        return f"{base}/schedule/{provision.number}"
    if provision.kind == "schedule_para":
        sch = provision.parent_id.removeprefix("sch_")
        return f"{base}/schedule/{sch}/paragraph/{provision.number}"
    if provision.kind == "ground":
        sch = provision.parent_id.removeprefix("sch_")
        return f"{base}/schedule/{sch}"
    return base


# ---------------------------------------------------------------------------
# Public chunking entry points
# ---------------------------------------------------------------------------

def chunk_document(doc: Document) -> list[ChunkRecord]:
    """Parse the document and produce one ChunkRecord per provision."""
    provisions = parse_provisions(doc.text)
    log.info("parsed %d provisions from %s", len(provisions), doc.filename)

    records: list[ChunkRecord] = []
    for provision in provisions:
        records.append(ChunkRecord(
            chunk_id=_build_chunk_id(doc, provision),
            source_file=doc.filename,
            source_title=doc.title,
            source_year=doc.year,
            source_url=_build_source_url(doc, provision),
            headline=_build_headline(provision),
            summary=_build_summary(provision),
            original_text=provision.text,
            provision_id=provision.canonical_id,
            provision_kind=provision.kind,
            provision_number=provision.number,
        ))
    return records


def chunk_corpus(
    documents: list[Document],
    *,
    force_rechunk: bool = False,
) -> list[ChunkRecord]:
    """Chunk every document, with per-Act JSONL caching."""
    paths.ensure()
    all_records: list[ChunkRecord] = []

    for doc in documents:
        cached = [] if force_rechunk else _read_chunks(doc.filename)
        if cached:
            log.info("cache hit: %s (%d chunks)", doc.filename, len(cached))
            all_records.extend(cached)
            continue

        records = chunk_document(doc)
        _write_chunks(records)
        all_records.extend(records)

    log.info("total chunks across corpus: %d", len(all_records))
    return all_records


def load_cached_chunks() -> list[ChunkRecord]:
    """Load every cached chunk from disk without touching the parser."""
    paths.ensure()
    all_records: list[ChunkRecord] = []
    for jsonl in sorted(paths.CHUNKS_CACHE.glob("*.jsonl")):
        all_records.extend(_read_chunks(jsonl.stem))
    log.info("loaded %d cached chunks", len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Per-statute JSONL cache
# ---------------------------------------------------------------------------

def _cache_path(filename: str) -> Path:
    return paths.CHUNKS_CACHE / f"{filename}.jsonl"


def _write_chunks(records: list[ChunkRecord]) -> None:
    if not records:
        return
    target = _cache_path(records[0].source_file)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _read_chunks(filename: str) -> list[ChunkRecord]:
    target = _cache_path(filename)
    if not target.exists():
        return []
    records: list[ChunkRecord] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Backward compatibility - older cache files had no provision fields.
            data.setdefault("provision_id", "")
            data.setdefault("provision_kind", "")
            data.setdefault("provision_number", "")
            records.append(ChunkRecord(**data))
    return records
