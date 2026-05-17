"""
scripts/build_db.py
---------------------
build the SQLite database from a local file.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from db.normalizer import normalize_paper
from db.repository import PaperRepository
from ingestion.loader import iter_papers_from_file

BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_from_file(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream records from a local file (JSON or JSONL)."""
    return iter_papers_from_file(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the local SQLite database."
    )
    parser.add_argument("--db_path",        default="data/papers.sqlite3")
    parser.add_argument("--raw_dir",        default="data/raw")
    parser.add_argument("--processed_dir",  default="data/processed")
    parser.add_argument("--input_file",     default="", help="Path to a local .json or .jsonl file to build from")
    parser.add_argument("--allow_sample_when_no_sources", action="store_true")
    parser.add_argument("--sample_size",    type=int,   default=24)
    parser.add_argument(
        "--max_papers", type=int, default=0,
        help="Cap total papers across all sources (0 = unlimited).",
    )
    # downloader options removed — input is expected to be a local file
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir       = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_file = processed_dir / "papers_normalized.jsonl"
    repo = PaperRepository(args.db_path)
    inserted = 0
    raw_count = 0
    batch: list[dict[str, Any]] = []

    # require explicit input file
    if not args.input_file:
        raise ValueError("No --input_file provided. Provide a local .json/.jsonl to build from.")

    with processed_file.open("w", encoding="utf-8") as fh:
        print(f"Streaming from: {args.input_file}")
        for raw_record in _stream_from_file(args.input_file):
            raw_count += 1
            if args.max_papers > 0 and raw_count > args.max_papers:
                raw_count = args.max_papers
                break
            normalized = normalize_paper(raw_record)
            fh.write(json.dumps(normalized, ensure_ascii=True) + "\n")
            batch.append(normalized)
            if len(batch) >= BATCH_SIZE:
                inserted += repo.bulk_upsert_normalized(batch)
                batch.clear()

        if batch:
            inserted += repo.bulk_upsert_normalized(batch)
            batch.clear()

    print(
        f"\n{'─'*50}\n"
        f"  Raw records   : {raw_count:,}\n"
        f"  Inserted      : {inserted:,}\n"
        f"  Total in DB   : {repo.count():,}\n"
        f"  DB path       : {Path(args.db_path).resolve()}\n"
        f"  Processed file: {processed_file.resolve()}\n"
        f"{'─'*50}"
    )


if __name__ == "__main__":
    main()
