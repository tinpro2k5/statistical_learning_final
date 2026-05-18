"""
retrieval/base.py
-----------------
Retriever Protocol — the contract every retrieval backend must satisfy.

Implementations (FTS5, BM25, vector) all satisfy this interface.
SearchService depends on this type, not on any concrete class.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """
    Fetch candidate papers matching *query* + *keywords*.

    Parameters
    ----------
    query:
        Free-text query (e.g. "text before citation" context).
    keywords:
        Explicit keyword string (comma/semicolon separated).
    limit:
        Maximum number of candidates to return.
    filters:
        Optional equality filters applied before text search.
        Recognised keys: ``collection`` (str), ``year`` (int).

    Returns
    -------
    list[dict]
        Each dict is a paper record with at least ``paper_id``, ``title``,
        ``abstract`` keys (same shape as ``PaperRepository._row_to_paper``).
    """

    def search(
        self,
        query: str,
        keywords: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]: ...
