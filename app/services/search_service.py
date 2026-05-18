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
_RETRIEVAL_POOL = 100


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

        # Stage 1 — retrieve candidate pool
        candidates = self.retriever.search(
            query=query,
            keywords=keywords,
            limit=_RETRIEVAL_POOL,
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

    def search_by_title(
        self,
        title: str,
        authors: Any | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Title-oriented search with optional author boosting.

        Uses the same two-stage pipeline; exact-title and author matches
        receive a post-hoc score boost to surface precise hits at the top.
        """
        title = normalize_text(title)
        author_text = normalize_text(authors_to_text(authors))
        combined_keywords = author_text

        results = self.search(
            query=title,
            keywords=combined_keywords,
            limit=max(limit * 3, limit),
        )

        # Post-hoc boosting for exact matches (on top of reranker score)
        boosted: list[dict[str, Any]] = []
        for paper in results:
            score = float(paper.get("score", 0.0))
            paper_title = normalize_text(paper.get("title", "")).lower()
            paper_authors = normalize_text(paper.get("authors_text", "")).lower()

            if title and title.lower() == paper_title:
                score += 8.0
            if author_text and author_text.lower() in paper_authors:
                score += 6.0

            boosted.append({**paper, "score": round(score, 6)})

        boosted.sort(
            key=lambda p: (-p["score"], p.get("title", ""))
        )
        return boosted[:limit]

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
