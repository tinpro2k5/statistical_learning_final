"""
ingestion/loader.py
-------------------
Load raw paper records from local files (JSON / JSONL).

No DB or normalisation logic here — returns raw dicts that callers
can pass through db.normalizer.normalize_paper() before inserting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(file_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Read all non-empty lines from a JSONL file into memory."""
    return list(iter_jsonl(file_path))


def iter_json(file_path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Yield papers from a JSON file.

    Accepts two shapes:
    - A top-level list of paper dicts.
    - A dict with a ``"papers"`` key containing the list.
    """
    path = Path(file_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
    if isinstance(data, dict):
        data = data.get("papers", [])
    for record in data:
        yield record


def load_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Read papers from a JSON file into memory."""
    return list(iter_json(file_path))


def iter_papers_from_file(file_path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Dispatch to the correct loader based on file extension.

    Supports ``.json`` and ``.jsonl``.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        yield from iter_json(path)
        return
    raise ValueError(f"Unsupported file type: {path.suffix!r} — use .json or .jsonl")


def load_papers_from_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Dispatch to the correct loader and materialize the result."""
    return list(iter_papers_from_file(file_path))
