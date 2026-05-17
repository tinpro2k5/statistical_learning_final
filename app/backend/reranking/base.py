"""
reranking/base.py
-----------------
Reranker Protocol — the contract every reranking backend must satisfy.

Current implementation: BertNSPReranker (bert_nsp.py).
Future: CrossEncoder, ColBERT, etc.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """
    Score a list of candidate papers against a query.

    Parameters
    ----------
    query:
        Free-text query string.
    keywords:
        Explicit keyword string.
    papers:
        List of paper dicts (must have ``title`` and ``abstract``).

    Returns
    -------
    list[float]
        Relevance scores in [0, 1], one per paper, same order as *papers*.
    """

    def batch_score(
        self,
        query: str,
        keywords: str,
        papers: list[dict[str, Any]],
    ) -> list[float]: ...
