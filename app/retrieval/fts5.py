"""
retrieval/fts5.py
-----------------
FTS5Retriever — SQLite FTS5 implementation of the Retriever protocol.

Schema (papers_fts virtual table) is created on first use together with
INSERT/UPDATE/DELETE triggers that keep the index in sync automatically.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from db.normalizer import parse_json_field
from search_terms import NAME_TO_GREEK, acronym_key, normalize_scientific_symbols

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TITLE_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_BM25_RANK = "bm25(papers_fts, 0.0, 8.0, 2.0, 4.0, 0.5)"

_STOPWORDS = {
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


class FTS5Retriever:
    """
    Full-text search over ``title``, ``abstract``, ``keywords``,
    ``authors_text`` using SQLite FTS5.

    Satisfies the :class:`retrieval.base.Retriever` protocol.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        with self._connect() as conn:
            self._ensure_fts_schema(conn)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # FTS5 schema + triggers
    # ------------------------------------------------------------------

    def _ensure_fts_schema(self, conn: sqlite3.Connection) -> None:
        # Virtual FTS5 table backed by the main papers table.
        # full_text intentionally excluded — keeps index small.
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(
                paper_id  UNINDEXED,
                title,
                abstract,
                keywords,
                authors_text,
                content='papers',
                content_rowid='rowid'
            )
            """
        )

        # Auto-sync triggers so the FTS index stays consistent with papers.
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS papers_fts_ai
            AFTER INSERT ON papers BEGIN
                INSERT INTO papers_fts(rowid, paper_id, title, abstract, keywords, authors_text)
                VALUES (new.rowid, new.paper_id, new.title, new.abstract, new.keywords, new.authors_text);
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS papers_fts_ad
            AFTER DELETE ON papers BEGIN
                INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract, keywords, authors_text)
                VALUES ('delete', old.rowid, old.paper_id, old.title, old.abstract, old.keywords, old.authors_text);
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS papers_fts_au
            AFTER UPDATE ON papers BEGIN
                INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract, keywords, authors_text)
                VALUES ('delete', old.rowid, old.paper_id, old.title, old.abstract, old.keywords, old.authors_text);
                INSERT INTO papers_fts(rowid, paper_id, title, abstract, keywords, authors_text)
                VALUES (new.rowid, new.paper_id, new.title, new.abstract, new.keywords, new.authors_text);
            END
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # FTS5 query builder
    # ------------------------------------------------------------------

    @staticmethod
    def _content_terms(query: str, keywords: str) -> list[str]:
        """Return useful FTS terms after dropping high-frequency stopwords."""
        terms: list[str] = []
        seen: set[str] = set()
        for part in (query, keywords):
            text = normalize_scientific_symbols(part).lower()
            for match in _TOKEN_RE.finditer(text):
                term = match.group(0)
                if len(term) < 2 or term in _STOPWORDS or term in seen:
                    continue
                seen.add(term)
                terms.append(term)
        return terms

    @staticmethod
    def _quote_term(term: str, prefix: bool = True) -> str:
        safe = term.replace('"', '""')
        suffix = "*" if prefix else ""
        return f'"{safe}"{suffix}'

    @staticmethod
    def _term_variants(term: str) -> list[str]:
        variants = [term]
        if term in NAME_TO_GREEK:
            variants.append(NAME_TO_GREEK[term])
        if len(term) <= 4 or not term.endswith("s"):
            return list(dict.fromkeys(variants))
        if term.endswith(("ss", "is", "us", "ics")):
            return list(dict.fromkeys(variants))
        if term.endswith("ies") and len(term) > 5:
            variants.append(f"{term[:-3]}y")
        elif term.endswith(("ches", "shes", "xes", "zes")):
            variants.append(term[:-2])
        else:
            variants.append(term[:-1])
        return list(dict.fromkeys(variants))

    @staticmethod
    def _term_expression(term: str, prefix: bool = True) -> str:
        variants = [
            FTS5Retriever._quote_term(variant, prefix=prefix)
            for variant in FTS5Retriever._term_variants(term)
        ]
        if len(variants) == 1:
            return variants[0]
        return "(" + " OR ".join(variants) + ")"

    @staticmethod
    def _field_term_expression(field: str, term: str, prefix: bool = True) -> str:
        variants = [
            f"{field} : {FTS5Retriever._quote_term(variant, prefix=prefix)}"
            for variant in FTS5Retriever._term_variants(term)
        ]
        if len(variants) == 1:
            return variants[0]
        return "(" + " OR ".join(variants) + ")"

    @staticmethod
    def _build_fts_query(
        query: str,
        keywords: str,
        operator: str = "AND",
        exact_terms: bool = False,
    ) -> str:
        """
        Combine query + keywords into an FTS5 MATCH expression.

        Useful terms become prefix terms (``token*``).  Stopwords are removed
        because terms such as "is" and "all" dominate BM25 and pollute the
        candidate pool before reranking.
        """
        terms = FTS5Retriever._content_terms(query, keywords)
        if not terms:
            return ""
        joiner = f" {operator.upper()} "
        return joiner.join(
            FTS5Retriever._term_expression(term, prefix=not exact_terms)
            for term in terms
        )

    @staticmethod
    def _build_fts_queries(query: str, keywords: str) -> list[str]:
        key = acronym_key(query)
        queries: list[str] = []

        if key:
            exact_acronym_query = FTS5Retriever._quote_term(key.lower(), prefix=False)
            keyword_query = FTS5Retriever._build_fts_query(
                keywords,
                "",
                operator="AND",
            )
            keyword_or_query = FTS5Retriever._build_fts_query(
                keywords,
                "",
                operator="OR",
            )
            if keyword_query:
                queries.append(f"{exact_acronym_query} AND {keyword_query}")
                queries.append(keyword_query)
            if keyword_or_query:
                queries.append(keyword_or_query)

            queries.append(exact_acronym_query)
        else:
            queries.extend(
                [
                    FTS5Retriever._build_fts_query(query, keywords, operator="AND"),
                    FTS5Retriever._build_fts_query(query, keywords, operator="OR"),
                ]
            )

        return [q for i, q in enumerate(queries) if q and q not in queries[:i]]

    @staticmethod
    def _clean_phrase(value: str) -> str:
        return " ".join(normalize_scientific_symbols(value).strip().split())

    @staticmethod
    def _title_words(value: str) -> str:
        text = normalize_scientific_symbols(value).lower()
        return " ".join(_TITLE_WORD_RE.findall(text))

    @staticmethod
    def _has_standalone_token(value: str, token: str) -> bool:
        pattern = rf"(?<![A-Za-z0-9-]){re.escape(token)}(?![A-Za-z0-9-])"
        return re.search(pattern, str(value or ""), flags=re.IGNORECASE) is not None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        keywords: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Return up to *limit* papers matching *query*/*keywords*.

        Falls back to a full-table scan (ordered by year DESC) when no
        search terms are provided, so the API always returns *something*.
        """
        collection = filters.get("collection")
        year = filters.get("year")

        fts_queries = self._build_fts_queries(query, keywords)
        has_user_keywords = bool(str(keywords or "").strip())
        acronym_query = bool(acronym_key(query))
        use_acronym_title_first = acronym_query and not has_user_keywords

        with self._connect() as conn:
            rows: list[sqlite3.Row] = []

            # Exact/phrase title matches are strong signals for paper-title
            # queries such as "Attention Is All You Need".
            rows = self._append_unique(
                rows,
                self._title_phrase_search(conn, query, collection, year, limit),
                limit,
            )
            rows = self._append_unique(
                rows,
                self._title_terms_search(conn, query, collection, year, limit),
                limit,
            )
            if use_acronym_title_first:
                acronym_title_limit = min(limit, max(5, limit // 4))
                rows = self._append_unique(
                    rows,
                    self._acronym_title_search(
                        conn,
                        query,
                        collection,
                        year,
                        acronym_title_limit,
                    ),
                    limit,
                )
            for fts_query in fts_queries:
                if len(rows) >= limit:
                    break
                rows = self._append_unique(
                    rows,
                    self._fts_search(conn, fts_query, collection, year, limit),
                    limit,
                )

            if acronym_query and not use_acronym_title_first and len(rows) < limit:
                rows = self._append_unique(
                    rows,
                    self._acronym_title_search(conn, query, collection, year, limit),
                    limit,
                )

            if not rows:
                rows = self._fallback_search(conn, collection, year, limit)

        return [self._row_to_paper(row) for row in rows]

    @staticmethod
    def _append_unique(
        rows: list[sqlite3.Row],
        new_rows: list[sqlite3.Row],
        limit: int,
    ) -> list[sqlite3.Row]:
        seen = {str(row["paper_id"]) for row in rows}
        merged = list(rows)
        for row in new_rows:
            paper_id = str(row["paper_id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            merged.append(row)
            if len(merged) >= limit:
                break
        return merged

    def _title_phrase_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        collection: str | None,
        year: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        phrase = self._clean_phrase(query)
        if len(phrase) < 4:
            return []
        if acronym_key(query):
            return []

        safe_phrase = phrase.replace('"', '""')
        rows = self._fts_search(
            conn,
            f'title : "{safe_phrase}"',
            collection,
            year,
            max(limit * 5, 50),
        )
        phrase_words = self._title_words(phrase)

        def sort_key(row: sqlite3.Row) -> tuple[int, int, int, str]:
            title_words = self._title_words(row["title"])
            if title_words == phrase_words:
                match_rank = 0
            elif phrase_words and phrase_words in title_words:
                match_rank = 1
            else:
                match_rank = 2
            return (
                match_rank,
                len(title_words),
                -(row["year"] or 0),
                str(row["title"] or ""),
            )

        return sorted(rows, key=sort_key)[:limit]

    def _title_terms_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        collection: str | None,
        year: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        if acronym_key(query):
            return []

        terms = self._content_terms(query, "")
        if len(terms) < 2:
            return []

        title_query = " AND ".join(
            self._field_term_expression("title", term)
            for term in terms
        )
        title_limit = min(limit, max(5, limit // 3))
        return self._fts_search(conn, title_query, collection, year, title_limit)

    def _acronym_title_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        collection: str | None,
        year: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        key = acronym_key(query)
        if not key:
            return []

        title_query = self._field_term_expression("title", key.lower(), prefix=False)
        rows = self._fts_search(
            conn,
            title_query,
            collection,
            year,
            max(limit * 5, 25),
        )
        return [
            row for row in rows
            if self._has_standalone_token(row["title"], key)
        ][:limit]

    def _fts_search(
        self,
        conn: sqlite3.Connection,
        fts_query: str,
        collection: str | None,
        year: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        where_clauses = ["papers_fts MATCH ?"]
        params: list[Any] = [fts_query]

        if collection:
            where_clauses.append("p.collection = ?")
            params.append(collection)
        if year is not None:
            where_clauses.append("p.year = ?")
            params.append(year)

        params.append(limit)
        sql = f"""
            SELECT p.*
            FROM papers_fts
            JOIN papers p ON papers_fts.rowid = p.rowid
            WHERE {' AND '.join(where_clauses)}
            ORDER BY {_BM25_RANK}
            LIMIT ?
        """
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Invalid FTS5 syntax — fall back silently
            return []

    def _fallback_search(
        self,
        conn: sqlite3.Connection,
        collection: str | None,
        year: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if collection:
            where_clauses.append("collection = ?")
            params.append(collection)
        if year is not None:
            where_clauses.append("year = ?")
            params.append(year)
        params.append(limit)

        sql = "SELECT * FROM papers"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY COALESCE(year, 0) DESC, title ASC LIMIT ?"
        return conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------
    # Serialization (mirrors PaperRepository._row_to_paper)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> dict[str, Any]:
        links = parse_json_field(row["links"] if "links" in row.keys() else "[]", [])
        if not links:
            doi = str(row["doi"] or "").strip()
            paper_id = str(row["paper_id"] or "").strip()
            if doi:
                links = [f"https://doi.org/{doi}"]
            elif paper_id:
                links = [f"https://arxiv.org/abs/{paper_id}"]
        return {
            "paper_id":    row["paper_id"],
            "collection":  row["collection"],
            "title":       row["title"],
            "abstract":    row["abstract"],
            "full_text":   row["full_text"],
            "venue":       row["venue"],
            "keywords":    row["keywords"],
            "authors":     parse_json_field(row["authors_json"], []),
            "authors_text": row["authors_text"],
            "year":        row["year"],
            "doi":         row["doi"],
            "primary_category": row["primary_category"] if "primary_category" in row.keys() else "",
            "links":       links,
        }
