"""
db/normalizer.py
----------------
Pure functions for normalizing raw paper dicts into a canonical schema.
No I/O, no DB dependencies — only data transformation.
"""
from __future__ import annotations

import json
import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Collapse whitespace and convert to string."""
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def tokenize(value: Any) -> list[str]:
    """Extract alphanumeric tokens (lowercase) for keyword matching."""
    return _TOKEN_RE.findall(normalize_text(value).lower())


# ---------------------------------------------------------------------------
# Author helpers
# ---------------------------------------------------------------------------

def authors_to_text(authors: Any) -> str:
    """Flatten authors to a single searchable string (semicolon-separated)."""
    if not authors:
        return ""
    if isinstance(authors, str):
        return normalize_text(authors)
    if isinstance(authors, list):
        names: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                full_name = normalize_text(
                    author.get("FullName", author.get("full_name", ""))
                )
                if full_name:
                    names.append(full_name)
                    continue
                given = normalize_text(
                    author.get("GivenName", author.get("given_name", ""))
                )
                family = normalize_text(
                    author.get("FamilyName", author.get("family_name", ""))
                )
                names.append(normalize_text(f"{given} {family}"))
            else:
                names.append(normalize_text(str(author)))
        return "; ".join(name for name in names if name)
    return normalize_text(authors)


def authors_to_json(authors: Any) -> str:
    """Serialize authors to a canonical JSON string."""
    if authors is None:
        return "[]"
    if isinstance(authors, str):
        try:
            parsed = json.loads(authors)
            return json.dumps(parsed, ensure_ascii=True)
        except json.JSONDecodeError:
            return json.dumps([{"FullName": authors}], ensure_ascii=True)
    return json.dumps(authors, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Field coercions
# ---------------------------------------------------------------------------

def parse_json_field(value: str | None, default: Any) -> Any:
    """Safely parse a JSON string; return *default* on failure."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def coerce_year(value: Any) -> int | None:
    """Parse a year value from various formats; return None if unparseable."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d{4}", str(value))
        if match:
            return int(match.group(0))
    return None


def coerce_paper_id(paper: dict[str, Any]) -> str:
    """Derive a stable paper_id from the raw dict."""
    for key in ("paper_id", "id", "id_value", "_id"):
        value = paper.get(key)
        if value not in (None, ""):
            return str(value)
    title = normalize_text(paper.get("title", paper.get("Title", "")))
    collection = (
        normalize_text(paper.get("collection", paper.get("source", "local")))
        or "local"
    )
    return f"{collection}:{title.lower() or 'unknown'}"


# ---------------------------------------------------------------------------
# Main normalization
# ---------------------------------------------------------------------------

def normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw paper dict (any schema variant) into the canonical flat
    representation expected by PaperRepository.
    """
    authors = paper.get("authors", paper.get("Author", paper.get("author", [])))
    return {
        "paper_id": coerce_paper_id(paper),
        "collection": normalize_text(
            paper.get("collection", paper.get("source", "local"))
        ) or "local",
        "title": normalize_text(paper.get("title", paper.get("Title", ""))),
        "abstract": normalize_text(
            paper.get("abstract", paper.get("Abstract", ""))
        ),
        "full_text": normalize_text(
            paper.get("full_text", paper.get("content", ""))
        ),
        "venue": normalize_text(paper.get("venue", paper.get("Venue", ""))),
        "keywords": normalize_text(
            paper.get("keywords", paper.get("Keyword", ""))
        ),
        "authors_json": authors_to_json(authors),
        "authors_text": authors_to_text(authors),
        "year": coerce_year(
            paper.get("year", paper.get("Year", paper.get("PublicationYear")))
        ),
        "doi": normalize_text(paper.get("doi", paper.get("DOI", ""))),
    }
