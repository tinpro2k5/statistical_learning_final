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

from .normalizer import normalize_paper, parse_json_field


class PaperRepository:
    """Handles CRUD operations on the *papers* table."""

    _IN_CLAUSE_LIMIT = 900

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
                    doi          TEXT NOT NULL DEFAULT '',
                    primary_category TEXT NOT NULL DEFAULT '',
                    links        TEXT NOT NULL DEFAULT '[]',
                    citation_count INTEGER,
                    citation_updated_at TEXT NOT NULL DEFAULT '',
                    venue_updated_at TEXT NOT NULL DEFAULT ''
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
            # Ensure backward-compatible schema changes (add columns if table existed)
            rowcols = [r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()]
            if "primary_category" not in rowcols:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN primary_category TEXT NOT NULL DEFAULT ''"
                )
            if "links" not in rowcols:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN links TEXT NOT NULL DEFAULT '[]'"
                )
            if "citation_count" not in rowcols:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN citation_count INTEGER"
                )
            if "citation_updated_at" not in rowcols:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN citation_updated_at TEXT NOT NULL DEFAULT ''"
                )
            if "venue_updated_at" not in rowcols:
                conn.execute(
                    "ALTER TABLE papers ADD COLUMN venue_updated_at TEXT NOT NULL DEFAULT ''"
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
        refs: list[tuple[str, str | None]] = []
        ids_by_collection: dict[str | None, list[str]] = {}
        seen_refs: set[tuple[str, str | None]] = set()

        for ref in paper_refs:
            paper_id = str(ref.get("id_value", ref.get("paper_id", ref.get("id", "")))).strip()
            collection = ref.get("collection") or None
            refs.append((paper_id, collection))
            ref_key = (paper_id, collection)
            if paper_id and ref_key not in seen_refs:
                seen_refs.add(ref_key)
                ids_by_collection.setdefault(collection, []).append(paper_id)

        if not refs:
            return []

        rows_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
        rows_by_id: dict[str, dict[str, Any]] = {}

        with self.connect() as conn:
            for collection, paper_ids in ids_by_collection.items():
                if not paper_ids:
                    continue
                for chunk_start in range(0, len(paper_ids), self._IN_CLAUSE_LIMIT):
                    chunk = paper_ids[chunk_start : chunk_start + self._IN_CLAUSE_LIMIT]
                    placeholders = ", ".join("?" for _ in chunk)
                    sql = f"SELECT * FROM papers WHERE paper_id IN ({placeholders})"
                    params: list[Any] = list(chunk)
                    if collection is not None:
                        sql += " AND collection = ?"
                        params.append(collection)
                    for row in conn.execute(sql, params).fetchall():
                        paper = self._row_to_paper(row)
                        row_key = (str(row["paper_id"]), row["collection"])
                        rows_by_key[row_key] = paper
                        rows_by_id[str(row["paper_id"])] = paper

        results: list[dict[str, Any]] = []
        for paper_id, collection in refs:
            paper = rows_by_key.get((paper_id, collection))
            if paper is None and collection is None:
                paper = rows_by_id.get(paper_id)
            results.append(paper or {})
        return results

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    _UPSERT_SQL = """
        INSERT INTO papers (
            paper_id, collection, title, abstract, full_text, venue, keywords,
            authors_json, authors_text, year, doi, primary_category, links,
            citation_count, citation_updated_at, venue_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            collection   = excluded.collection,
            title        = excluded.title,
            abstract     = excluded.abstract,
            full_text    = excluded.full_text,
            venue        = COALESCE(NULLIF(excluded.venue, ''), papers.venue),
            keywords     = excluded.keywords,
            authors_json = excluded.authors_json,
            authors_text = excluded.authors_text,
            year         = excluded.year,
            doi          = excluded.doi
            , primary_category = excluded.primary_category
            , links = excluded.links
            , citation_count = COALESCE(excluded.citation_count, papers.citation_count)
            , citation_updated_at = COALESCE(NULLIF(excluded.citation_updated_at, ''), papers.citation_updated_at)
            , venue_updated_at = COALESCE(NULLIF(excluded.venue_updated_at, ''), papers.venue_updated_at)
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
            normalized.get("primary_category", ""),
            normalized.get("links", "[]"),
            normalized.get("citation_count"),
            normalized.get("citation_updated_at", ""),
            normalized.get("venue_updated_at", ""),
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

    def update_paper_metadata(
        self,
        paper_id: str,
        *,
        citation_count: int | None = None,
        citation_updated_at: str | None = None,
        venue: str | None = None,
        venue_updated_at: str | None = None,
    ) -> None:
        assignments: list[str] = []
        params: list[Any] = []

        if citation_count is not None:
            assignments.append("citation_count = ?")
            params.append(citation_count)
        if citation_updated_at is not None:
            assignments.append("citation_updated_at = ?")
            params.append(citation_updated_at)
        if venue is not None:
            assignments.append("venue = ?")
            params.append(venue)
        if venue_updated_at is not None:
            assignments.append("venue_updated_at = ?")
            params.append(venue_updated_at)

        if not assignments:
            return

        params.append(paper_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE papers SET {', '.join(assignments)} WHERE paper_id = ?",
                params,
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _row_to_paper(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        links = parse_json_field(row["links"] if "links" in keys else "[]", [])
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
            "primary_category": row["primary_category"] if "primary_category" in keys else "",
            "links":        links,
            "citation_count": row["citation_count"] if "citation_count" in keys else None,
            "citation_updated_at": row["citation_updated_at"] if "citation_updated_at" in keys else "",
            "venue_updated_at": row["venue_updated_at"] if "venue_updated_at" in keys else "",
        }
