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

import re
from typing import Any, Iterable

from db.normalizer import authors_to_text, normalize_text
from db.repository import PaperRepository
from retrieval.base import Retriever
from reranking.base import Reranker
from search_terms import acronym_key, normalize_scientific_symbols
from services.paper_metadata_enricher import PaperMetadataEnricher

# How many FTS candidates to fetch before reranking.
# Enough headroom so reranker can discriminate; small enough to stay fast.
_RETRIEVAL_POOL = 136
_MIN_RETRIEVAL_POOL = 20
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "method",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
    "all",
    "analysis",
    "approach",
    "approaches",
    "based",
    "data",
    "review",
    "study",
    "studies",
    "survey",
    "using",
    "via",
}


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
        metadata_enricher: PaperMetadataEnricher | None = None,
    ) -> None:
        self.repo = repo
        self.retriever = retriever
        self.reranker = reranker
        self.metadata_enricher = metadata_enricher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _query_token_count(query: str) -> int:
        return len([token for token in query.split() if token])

    @staticmethod
    def _topic_terms(value: str) -> set[str]:
        text = normalize_scientific_symbols(value).lower()
        return {
            match.group(0)
            for match in _TOKEN_RE.finditer(text)
            if len(match.group(0)) >= 3
            and match.group(0) not in _TOPIC_STOPWORDS
        }

    @staticmethod
    def _paper_topic_text(paper: dict[str, Any]) -> str:
        return " ".join(
            str(paper.get(field, ""))
            for field in ("title", "abstract", "keywords", "primary_category")
        ).lower()

    @staticmethod
    def _word_tokens(value: str) -> list[str]:
        return _TOKEN_RE.findall(normalize_scientific_symbols(value).lower())

    @staticmethod
    def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
        if not needle or len(needle) > len(haystack):
            return False
        width = len(needle)
        return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))

    @staticmethod
    def _term_forms(term: str) -> set[str]:
        forms = {term}
        if term.endswith("s") and len(term) > 4:
            forms.add(term[:-1])
        else:
            forms.add(f"{term}s")
        return forms

    @staticmethod
    def _term_hit(term: str, text: str, tokens: set[str]) -> bool:
        forms = SearchService._term_forms(term)
        return any(form in tokens for form in forms)

    @staticmethod
    def _term_weight(term: str) -> float:
        return float(min(max(len(term), 3), 14))

    @staticmethod
    def _weighted_coverage(terms: set[str], text: str, tokens: set[str]) -> float:
        total = sum(SearchService._term_weight(term) for term in terms)
        if total <= 0:
            return 0.0
        hits = sum(
            SearchService._term_weight(term)
            for term in terms
            if SearchService._term_hit(term, text, tokens)
        )
        return min(1.0, hits / total)

    @staticmethod
    def _has_standalone_token(value: str, token: str) -> bool:
        pattern = rf"(?<![A-Za-z0-9-]){re.escape(token)}(?![A-Za-z0-9-])"
        return re.search(pattern, str(value or ""), flags=re.IGNORECASE) is not None

    @staticmethod
    def _title_score(query: str, paper: dict[str, Any]) -> tuple[float, int]:
        """Return a title-match score plus an exact-match sort priority."""
        query_norm = normalize_text(query).lower()
        title_norm = normalize_text(str(paper.get("title", ""))).lower()
        if not query_norm or not title_norm:
            return 0.0, 0

        key = acronym_key(query)
        if key:
            if SearchService._has_standalone_token(str(paper.get("title", "")), key):
                return 0.8, 0
            return 0.0, 0

        if query_norm == title_norm:
            return 1.0, 1

        query_words = SearchService._word_tokens(query_norm)
        title_words = SearchService._word_tokens(title_norm)
        if len(query_words) >= 2 and SearchService._contains_token_sequence(title_words, query_words):
            return 0.8, 0
        if len(query_words) == 1 and query_words[0] in set(title_words):
            return 0.8, 0

        query_terms = SearchService._topic_terms(query)
        if len(query_terms) >= 2:
            title_text = str(paper.get("title", "")).lower()
            title_tokens = SearchService._topic_terms(title_text)
            coverage = SearchService._weighted_coverage(
                query_terms,
                title_text,
                title_tokens,
            )
            if coverage >= 0.85:
                return 0.85, 0
            if coverage >= 0.50:
                return 0.35 + (0.45 * coverage), 0

        return 0.0, 0

    @staticmethod
    def _domain_keyword_score(keywords: str, paper: dict[str, Any]) -> float:
        terms = SearchService._topic_terms(keywords)
        if not terms:
            return 0.0

        paper_text = SearchService._paper_topic_text(paper)
        paper_tokens = SearchService._topic_terms(paper_text)
        return SearchService._weighted_coverage(terms, paper_text, paper_tokens)

    @staticmethod
    def _retrieval_score(index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        return max(0.0, 1.0 - (index / (total - 1)))

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

    def _rank_scored_papers(
        self,
        scores: list[float],
        papers: list[dict[str, Any]],
        query: str,
        keywords: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        ranked = []
        total = len(papers)
        user_keywords = keywords.strip()
        key = acronym_key(query)
        topic_text = (
            user_keywords
            if key and user_keywords
            else " ".join(part for part in (query, user_keywords) if part)
        )
        for index, (model_score, paper) in enumerate(zip(scores, papers)):
            title_score, exact_title_match = self._title_score(query, paper)
            retrieval_score = self._retrieval_score(index, total)
            user_domain_score = self._domain_keyword_score(user_keywords, paper)
            domain_score = self._domain_keyword_score(topic_text, paper)

            if key and user_keywords:
                title_score *= 0.15 + (0.85 * user_domain_score)
            elif user_keywords and title_score < 1.0:
                title_score *= 0.4 + (0.6 * domain_score)
            model_score = float(model_score)
            final_score = (
                0.45 * title_score
                + 0.25 * retrieval_score
                + 0.20 * model_score
                + 0.10 * domain_score
            )
            if exact_title_match and not (key and user_keywords and user_domain_score < 0.20):
                final_score = 1.0
            sort_exact_match = exact_title_match
            if key and user_keywords and user_domain_score < 0.20:
                sort_exact_match = 0

            ranked.append(
                (
                    sort_exact_match,
                    final_score,
                    domain_score,
                    model_score,
                    retrieval_score,
                    paper,
                    title_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                -item[3],
                -item[4],
                -(item[5].get("year") or 0),
                item[5].get("title", ""),
            )
        )
        results = []
        for rank, (
                _,
                final_score,
                domain_score,
                model_score,
                retrieval_score,
                paper,
                title_score,
            ) in enumerate(ranked[:limit], start=1):
            results.append(
                {
                    **paper,
                    "rank": rank,
                    "score": round(min(1.0, final_score), 6),
                    "model_score": round(model_score, 6),
                    "retrieval_score": round(retrieval_score, 6),
                    "domain_score": round(domain_score, 6),
                    "title_score": round(title_score, 6),
                }
            )
        if self.metadata_enricher is not None:
            return self.metadata_enricher.enrich_papers(results)
        return results

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
        return self._rank_scored_papers(scores, candidates, query, keywords, limit)


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
        return self._rank_scored_papers(scores, papers, query, keywords, limit)
