"""Shared utilities: seeding, file I/O, config loading, logging."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load a JSONL file and return a list of dicts. Returns [] if not found."""
    path = Path(file_path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(file_path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts to a JSONL file (creates parent dirs as needed)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_json(file_path: str | Path, data: Any) -> None:
    """Write a dict / list to a pretty-printed JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=True)


def load_json(file_path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config loading (YAML / JSON)
# ---------------------------------------------------------------------------

def load_config(file_path: str | Path) -> dict:
    """Load a YAML or JSON config file and return a flat-merged dict."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configs require PyYAML: pip install pyyaml") from exc
            raw = yaml.safe_load(fh) or {}
        elif path.suffix.lower() == ".json":
            raw = json.load(fh)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    return raw


def merge_config_with_cli(config: dict, cli_args: dict) -> dict:
    """
    Merge a config dict with CLI args.
    CLI values override config values only when they differ from the
    argparse default (i.e. when explicitly provided by the user).
    Since we cannot easily detect defaults here, callers should pass
    only explicitly-supplied CLI overrides.
    """
    merged = dict(config)
    merged.update({k: v for k, v in cli_args.items() if v is not None})
    return merged


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def format_keywords(keywords: Any) -> str:
    """Normalise a keywords field to a semicolon-separated string."""
    if keywords is None:
        return ""
    if isinstance(keywords, list):
        return "; ".join(str(k).strip() for k in keywords if str(k).strip())
    return str(keywords).strip()


def build_citation_prompt(row: dict) -> str:
    """Build the source sequence for citation generation."""
    keywords = format_keywords(row.get("keywords", row.get("keyword", "")))
    context = str(row.get("context", row.get("source_context", ""))).strip()
    title = str(row.get("cited_title", row.get("title", ""))).strip()
    abstract = str(row.get("cited_abstract", row.get("abstract", ""))).strip()
    return " ".join(
        filter(None, [
            f"keywords: {keywords}" if keywords else "",
            f"text before citation: {context}" if context else "",
            f"cited title: {title}" if title else "",
            f"cited abstract: {abstract}" if abstract else "",
        ])
    ).strip()


def build_rerank_pair(row: dict) -> tuple[str, str]:
    """Return (query_text, candidate_text) for a reranking row.
    
    Uses generic prompt labels suitable for any dataset
    (SciDocs topic search, SciFact claim verification, etc.).
    """
    keywords = format_keywords(row.get("keywords", row.get("keyword", "")))
    query_text = str(row.get("query", row.get("context", ""))).strip()
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()

    query_parts = []
    if keywords:
        query_parts.append(f"keywords: {keywords}")
    if query_text:
        query_parts.append(f"query: {query_text}")

    candidate_parts = []
    if title:
        candidate_parts.append(f"document title: {title}")
    if abstract:
        candidate_parts.append(f"document abstract: {abstract}")

    return " ".join(query_parts).strip(), " ".join(candidate_parts).strip()


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def split_rows(rows: list, validation_ratio: float = 0.1, seed: int = 42) -> tuple[list, list]:
    """Randomly split rows into (train, validation) by ratio."""
    if not rows:
        return [], []
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - validation_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def split_by_query_id(rows: list, validation_ratio: float = 0.1, seed: int = 42) -> tuple[list, list]:
    """Split rows so that all rows for a given query_id stay in the same split."""
    query_ids = sorted({str(r.get("query_id", r.get("group_id", ""))) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    val_count = max(1, int(len(query_ids) * validation_ratio))
    val_ids = set(query_ids[:val_count])
    train_rows = [r for r in rows if str(r.get("query_id", r.get("group_id", ""))) not in val_ids]
    val_rows = [r for r in rows if str(r.get("query_id", r.get("group_id", ""))) in val_ids]
    return train_rows, val_rows
