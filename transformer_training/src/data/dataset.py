"""
Dataset classes and collator factories for reranking and seq2seq generation tasks.

Supports any JSONL/CSV dataset whose rows contain the expected field names
(configurable via `field_map` in the YAML config).  Adding a new dataset or
task only requires:
  1. Writing the JSONL rows to disk with the expected field names, OR
  2. Providing a `field_map` in the config to alias your field names.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.helpers import format_keywords


# ---------------------------------------------------------------------------
# Field-map helper
# ---------------------------------------------------------------------------

def _get(row: dict, key: str, field_map: dict[str, str], default: Any = "") -> Any:
    """Look up a key using an optional alias from field_map."""
    mapped = field_map.get(key, key)
    # Support comma-separated fallback list: "cited_title,title"
    for candidate in mapped.split(","):
        candidate = candidate.strip()
        if candidate in row:
            return row[candidate]
    return default


# ---------------------------------------------------------------------------
# Reranking Dataset
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    Each row must have:
        query    – the query string  (or alias via field_map)
        title    – document title    (or alias)
        abstract – document abstract (or alias)
        label    – 0 / 1             (or alias)
        query_id – grouping key for MRR (optional, falls back to query text)

    `field_map` keys: query, title, abstract, label, query_id
    """

    def __init__(self, rows: list[dict], field_map: dict[str, str] | None = None):
        self.rows = rows
        self.field_map: dict[str, str] = field_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        keywords = format_keywords(_get(row, "keywords", self.field_map, row.get("keyword", "")))
        query = str(_get(row, "query", self.field_map, row.get("context", ""))).strip()
        title = str(_get(row, "title", self.field_map, "")).strip()
        abstract = str(_get(row, "abstract", self.field_map, "")).strip()
        label = int(_get(row, "label", self.field_map, 0))
        query_id = str(
            _get(row, "query_id", self.field_map,
                 _get(row, "group_id", self.field_map, str(index)))
        )
        query_text = " ".join(
            part for part in (
                f"keywords: {keywords}" if keywords else "",
                f"query: {query}" if query else "",
            )
            if part
        ).strip()
        candidate_text = " ".join(
            part for part in (
                f"document title: {title}" if title else "",
                f"document abstract: {abstract}" if abstract else "",
            )
            if part
        ).strip()
        return {
            "query_text": query_text,
            "candidate_text": candidate_text,
            "label": label,
            "query_id": query_id,
        }


def make_rerank_collate_fn(tokenizer, max_length: int):
    """Return a collate function for RerankDataset batches."""
    def collate(batch: list[dict]) -> dict:
        pairs = [(item["query_text"], item["candidate_text"]) for item in batch]
        encodings = tokenizer(
            pairs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encodings["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.long
        )
        encodings["query_id"] = [item["query_id"] for item in batch]
        return encodings
    return collate


# ---------------------------------------------------------------------------
# Citation-generation Dataset
# ---------------------------------------------------------------------------

class CitationDataset(Dataset):
    """
    Each row must have:
        source fields – keywords, context, cited_title, cited_abstract
                        (built automatically by build_citation_prompt)
        target        – the reference sentence to generate

    `field_map` keys: target
    """

    def __init__(self, rows: list[dict], field_map: dict[str, str] | None = None):
        self.rows = rows
        self.field_map: dict[str, str] = field_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        keywords = format_keywords(_get(row, "keywords", self.field_map, row.get("keyword", "")))
        context = str(_get(row, "context", self.field_map, row.get("source_context", ""))).strip()
        title = str(_get(row, "cited_title", self.field_map, row.get("title", ""))).strip()
        abstract = str(_get(row, "cited_abstract", self.field_map, row.get("abstract", ""))).strip()
        source = " ".join(
            part for part in (
                f"keywords: {keywords}" if keywords else "",
                f"text before citation: {context}" if context else "",
                f"cited title: {title}" if title else "",
                f"cited abstract: {abstract}" if abstract else "",
            )
            if part
        ).strip()
        target = str(
            _get(row, "target", self.field_map,
                 row.get("citation", row.get("reference_sentence", "")))
        ).strip()
        return {"source": source, "target": target}


def make_seq2seq_collate_fn(tokenizer, max_source_length: int, max_target_length: int):
    """Return a collate function for CitationDataset batches."""
    def collate(batch: list[dict]) -> dict:
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        model_inputs = tokenizer(
            sources,
            max_length=max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_tokens = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_tokens["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
    return collate


# ---------------------------------------------------------------------------
# Factory: pick dataset + collate_fn based on task type
# ---------------------------------------------------------------------------

TASK_DATASET_MAP = {
    "reranking": RerankDataset,
    "document_reranking": RerankDataset,
    "citation_generation": CitationDataset,
    "generation": CitationDataset,
}


def build_dataset(task: str, rows: list[dict], field_map: dict | None = None):
    """Return the correct Dataset subclass for the given task name."""
    cls = TASK_DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(
            f"Unknown task '{task}'. Supported: {list(TASK_DATASET_MAP.keys())}"
        )
    return cls(rows, field_map=field_map)


def build_collate_fn(task: str, tokenizer, config: dict):
    """Return the correct collate function for the given task and config."""
    if task in {"reranking", "document_reranking"}:
        return make_rerank_collate_fn(tokenizer, config.get("max_length", 512))
    if task in {"citation_generation", "generation"}:
        return make_seq2seq_collate_fn(
            tokenizer,
            config.get("max_source_length", 512),
            config.get("max_target_length", 96),
        )
    raise ValueError(f"Unknown task '{task}'")
