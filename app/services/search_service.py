"""
services/search_service.py
--------------------------
SearchService orchestrates the two-stage search pipeline:

    Retriever (FTS5 / BM25 / vector)
        → top-N candidates
    Reranker (CrossEncoder)
        → relevance scores (batch)
    Sort + top-K
        → final results

Both Retriever and Reranker are injected — no concrete types imported here.
"""
from __future__ import annotations

from typing import Any, Iterable

from db.normalizer import authors_to_text, normalize_text
from db.repository import PaperRepository
from retrieval.base import Retriever
from reranking.base import Reranker

# How many FTS candidates to fetch before reranking.
# Enough headroom so reranker can discriminate; small enough to stay fast.
_RETRIEVAL_POOL = 136
_MIN_RETRIEVAL_POOL = 20


class SearchService:
    """
    Orchestrates retrieval + reranking.

    Parameters
    ----------
    repo:
        ``PaperRepository`` for direct paper lookups (get_paper / get_papers).
    retriever:
        Any object satisfying the ``Retriever`` protocol.
    reranker:
        Any object satisfying the ``Reranker`` protocol.
    """

    def __init__(
        self,
        repo: PaperRepository,
        retriever: Retriever,
        reranker: Reranker,
    ) -> None:
        self.repo = repo
        self.retriever = retriever
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _query_token_count(query: str) -> int:
        return len([token for token in query.split() if token])

    def _candidate_pool_size(self, query: str, limit: int) -> int:
        requested_limit = max(int(limit), 1)
        token_count = self._query_token_count(query)

        if token_count <= 2:
            multiplier = 9
        elif token_count <= 5:
            multiplier = 7
        elif token_count <= 10:
            multiplier = 5
        else:
            multiplier = 3

        pool_size = requested_limit * multiplier
        return max(_MIN_RETRIEVAL_POOL, min(_RETRIEVAL_POOL, pool_size))

    def search(
        self,
        query: str = "",
        keywords: str = "",
        limit: int = 20,
        collection: str | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Full pipeline search: retrieve candidates → batch rerank → top-K.

        Returns a list of paper dicts with an added ``score`` key.
        """
        query = normalize_text(query)
        keywords = normalize_text(keywords)
        retrieval_limit = self._candidate_pool_size(query, limit)

        # Stage 1 — retrieve candidate pool
        candidates = self.retriever.search(
            query=query,
            keywords=keywords,
            limit=retrieval_limit,
            filters={"collection": collection, "year": year},
        )

        if not candidates:
            return []

        # Stage 2 — batch rerank
        scores = self.reranker.batch_score(query, keywords, candidates)

        # Stage 3 — sort by score, attach score, return top-K
        ranked = sorted(
            zip(scores, candidates),
            key=lambda pair: (
                -pair[0],
                -(pair[1].get("year") or 0),
                pair[1].get("title", ""),
            ),
        )
        return [
            {**paper, "score": round(score, 6)}
            for score, paper in ranked[:limit]
        ]


    def score_papers(
        self,
        paper_refs: Iterable[dict[str, Any]],
        query: str,
        keywords: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Rerank a pre-selected set of papers (from /doc-search with paper_list).

        Skips the retrieval stage — papers are already known.
        """
        query = normalize_text(query)
        keywords = normalize_text(keywords)

        papers = [p for p in self.repo.get_papers(paper_refs) if p]
        if not papers:
            return []

        scores = self.reranker.batch_score(query, keywords, papers)
        ranked = sorted(
            zip(scores, papers),
            key=lambda pair: (
                -pair[0],
                -(pair[1].get("year") or 0),
                pair[1].get("title", ""),
            ),
        )
        return [
            {**paper, "score": round(score, 6)}
            for score, paper in ranked[:limit]
        ]
