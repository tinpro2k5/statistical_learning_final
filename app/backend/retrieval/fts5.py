"""
retrieval/fts5.py
-----------------
FTS5Retriever — SQLite FTS5 implementation of the Retriever protocol.

Schema (papers_fts virtual table) is created on first use together with
INSERT/UPDATE/DELETE triggers that keep the index in sync automatically.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from db.normalizer import parse_json_field


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
    def _build_fts_query(query: str, keywords: str) -> str:
        """
        Combine query + keywords into an FTS5 MATCH expression.

        Each whitespace-separated token becomes a prefix term (``token*``)
        so partial words still match.  Terms are OR-joined so recall is high
        before the reranker trims candidates.
        """
        tokens: list[str] = []
        for part in (query, keywords):
            for word in part.split():
                word = word.strip('",;:')
                if word:
                    # Escape FTS5 special chars: " * ^ { }
                    safe = word.replace('"', '""')
                    tokens.append(f'"{safe}"*')

        if not tokens:
            return ""
        # OR join — high recall, reranker handles precision
        return " OR ".join(tokens)

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

        fts_query = self._build_fts_query(query, keywords)

        with self._connect() as conn:
            if fts_query:
                rows = self._fts_search(conn, fts_query, collection, year, limit)
                if not rows:
                    # Graceful fallback when FTS matches nothing
                    rows = self._fallback_search(conn, collection, year, limit)
            else:
                rows = self._fallback_search(conn, collection, year, limit)

        return [self._row_to_paper(row) for row in rows]

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
            JOIN papers p ON papers_fts.paper_id = p.paper_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rank
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
        }
