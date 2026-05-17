"""
reranking/bert_nsp.py
---------------------
BertNSPReranker — BERT Next Sentence Prediction reranker.

Key fix over the monolith: batch_score() runs a single forward pass per
batch (config.batch_size papers) instead of 1 forward pass per paper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BertNSPConfig:
    enabled: bool = False
    kind: str = "bert_nsp"
    model_name_or_path: str = "scieditor/document-reranking-scibert"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16
    query_template: str = "keywords: {keywords} text before citation: {query}"
    candidate_template: str = "title: {title} abstract: {abstract}"


def load_config(config_path: str | Path) -> BertNSPConfig:
    """Load BertNSPConfig from a JSON file (model_config.json)."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to this file's directory and then to cwd
        for candidate in (Path(__file__).resolve().parent.parent / config_path, Path(config_path)):
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    cfg = raw.get("search_model", raw.get("search", {}))
    if not isinstance(cfg, dict):
        cfg = {}

    return BertNSPConfig(
        enabled=bool(cfg.get("enabled", False)),
        kind=str(cfg.get("kind", "bert_nsp")),
        model_name_or_path=str(cfg.get("model_name_or_path", BertNSPConfig.model_name_or_path)),
        device=str(cfg.get("device", "auto")),
        max_length=int(cfg.get("max_length", 512)),
        batch_size=int(cfg.get("batch_size", 16)),
        query_template=str(cfg.get("query_template", BertNSPConfig.query_template)),
        candidate_template=str(cfg.get("candidate_template", BertNSPConfig.candidate_template)),
    )


class BertNSPReranker:
    """
    Reranker based on BERT Next Sentence Prediction.

    Satisfies :class:`reranking.base.Reranker` protocol via ``batch_score()``.

    Parameters
    ----------
    config:
        ``BertNSPConfig`` instance (loaded from model_config.json).
    """

    def __init__(self, config: BertNSPConfig) -> None:
        self.config = config

        try:
            import torch
            from transformers import BertForNextSentencePrediction, BertTokenizerFast
        except Exception as exc:
            raise RuntimeError(
                "BertNSPReranker requires torch and transformers. "
                "Run: pip install torch transformers"
            ) from exc

        if config.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizerFast.from_pretrained(config.model_name_or_path)
        self.model = (
            BertForNextSentencePrediction
            .from_pretrained(config.model_name_or_path)
            .to(self.device)
        )
        self.model.eval()

        # keep torch imported for forward passes
        self._torch = torch

    # ------------------------------------------------------------------
    # Text builders
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    def build_query_text(self, query: str, keywords: str) -> str:
        return self.config.query_template.format(
            query=self._clean(query),
            keywords=self._clean(keywords),
        )

    def build_candidate_text(self, paper: dict[str, Any]) -> str:
        title = self._clean(paper.get("title", paper.get("Title", "")))
        abstract = self._clean(paper.get("abstract", paper.get("Abstract", "")))
        return self.config.candidate_template.format(title=title, abstract=abstract)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def batch_score(
        self,
        query: str,
        keywords: str,
        papers: list[dict[str, Any]],
    ) -> list[float]:
        """
        Score all *papers* against (*query*, *keywords*) in batches.

        Returns a list of floats in [0, 1] — one score per paper,
        same order as *papers*. Single forward pass per batch of
        ``config.batch_size`` papers.
        """
        if not papers:
            return []

        torch = self._torch
        query_text = self.build_query_text(query, keywords)
        pairs = [(query_text, self.build_candidate_text(p)) for p in papers]

        all_scores: list[float] = []
        batch_size = self.config.batch_size

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoding = self.tokenizer(
                batch,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                logits = self.model(**encoding).logits
                # NSP: index 1 = "IsNext" probability → relevance score
                scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            all_scores.extend(scores)

        return all_scores

    def score_paper(self, query: str, keywords: str, paper: dict[str, Any]) -> float:
        """Single-paper convenience wrapper (uses batch_score internally)."""
        return self.batch_score(query, keywords, [paper])[0]


def load_reranker(config_path: str | Path) -> BertNSPReranker:
    """Load and validate config, then instantiate BertNSPReranker."""
    config = load_config(config_path)
    if not config.enabled:
        raise RuntimeError(
            f"search_model.enabled is false in {config_path}. "
            "Set it to true to use the reranker."
        )
    if config.kind.lower() != "bert_nsp":
        raise ValueError(f"Unsupported reranker kind: {config.kind!r}")
    return BertNSPReranker(config)
