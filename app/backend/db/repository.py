"""
db/repository.py
----------------
PaperRepository: all SQLite I/O for the papers table.
No search logic here — that lives in retrieval/.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from db.normalizer import normalize_paper, parse_json_field


class PaperRepository:
    """Handles CRUD operations on the *papers* table."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id     TEXT PRIMARY KEY,
                    collection   TEXT NOT NULL,
                    title        TEXT NOT NULL,
                    abstract     TEXT NOT NULL DEFAULT '',
                    full_text    TEXT NOT NULL DEFAULT '',
                    venue        TEXT NOT NULL DEFAULT '',
                    keywords     TEXT NOT NULL DEFAULT '',
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    authors_text TEXT NOT NULL DEFAULT '',
                    year         INTEGER,
                    doi          TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_collection ON papers(collection)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)"
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM papers").fetchone()
            return int(row["total"]) if row else 0

    def get_paper(
        self, paper_id: str, collection: str | None = None
    ) -> dict[str, Any] | None:
        sql = "SELECT * FROM papers WHERE paper_id = ?"
        params: list[Any] = [str(paper_id)]
        if collection:
            sql += " AND collection = ?"
            params.append(collection)
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_paper(row) if row else None

    def get_papers(
        self, paper_refs: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        for ref in paper_refs:
            paper_id = ref.get("id_value", ref.get("paper_id", ref.get("id", "")))
            collection = ref.get("collection")
            paper = self.get_paper(str(paper_id), collection)
            results.append(paper or {})
        return results

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    _UPSERT_SQL = """
        INSERT INTO papers (
            paper_id, collection, title, abstract, full_text, venue, keywords,
            authors_json, authors_text, year, doi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            collection   = excluded.collection,
            title        = excluded.title,
            abstract     = excluded.abstract,
            full_text    = excluded.full_text,
            venue        = excluded.venue,
            keywords     = excluded.keywords,
            authors_json = excluded.authors_json,
            authors_text = excluded.authors_text,
            year         = excluded.year,
            doi          = excluded.doi
    """

    def _upsert_params(self, normalized: dict[str, Any]) -> tuple:
        return (
            normalized["paper_id"],
            normalized["collection"],
            normalized["title"],
            normalized["abstract"],
            normalized["full_text"],
            normalized["venue"],
            normalized["keywords"],
            normalized["authors_json"],
            normalized["authors_text"],
            normalized["year"],
            normalized["doi"],
        )

    def upsert_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_paper(paper)
        with self.connect() as conn:
            conn.execute(self._UPSERT_SQL, self._upsert_params(normalized))
        return normalized

    def bulk_upsert(self, papers: Iterable[dict[str, Any]]) -> int:
        count = 0
        with self.connect() as conn:
            for paper in papers:
                normalized = normalize_paper(paper)
                conn.execute(self._UPSERT_SQL, self._upsert_params(normalized))
                count += 1
        return count

    def bulk_upsert_normalized(self, papers: Iterable[dict[str, Any]]) -> int:
        """Insert already-normalized paper records in a single transaction."""
        count = 0
        with self.connect() as conn:
            for paper in papers:
                conn.execute(self._UPSERT_SQL, self._upsert_params(paper))
                count += 1
        return count

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _row_to_paper(self, row: sqlite3.Row) -> dict[str, Any]:
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
