"""
reranking/cross_encoder.py
--------------------------
Sequence-classification cross-encoder reranker.

This backend matches checkpoints trained by
``transformer_training/document_reranking_train.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CrossEncoderConfig:
    enabled: bool = False
    kind: str = "cross_encoder"
    model_name_or_path: str = "outputs/document_reranking/best_checkpoint"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16
    query_template: str = "keywords: {keywords} text before citation: {query}"
    candidate_template: str = "title: {title} abstract: {abstract}"


class CrossEncoderReranker:
    """
    Reranker based on AutoModelForSequenceClassification.

    Supports both common binary-ranking heads:
    - two logits: softmax(logits)[positive_label_index]
    - one logit: sigmoid(logit)
    """

    def __init__(self, config: CrossEncoderConfig) -> None:
        self.config = config

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires torch and transformers. "
                "Run: pip install torch transformers"
            ) from exc

        if config.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path).to(self.device)
        self.model.eval()
        self._torch = torch

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

    def _scores_from_logits(self, logits):
        torch = self._torch
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits[:, 0]).tolist()
        return torch.softmax(logits, dim=-1)[:, 1].tolist()

    def batch_score(
        self,
        query: str,
        keywords: str,
        papers: list[dict[str, Any]],
    ) -> list[float]:
        if not papers:
            return []

        torch = self._torch
        query_text = self.build_query_text(query, keywords)
        pairs = [(query_text, self.build_candidate_text(paper)) for paper in papers]
        all_scores: list[float] = []

        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoding = self.tokenizer(
                batch,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {key: value.to(self.device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = self.model(**encoding).logits
                all_scores.extend(self._scores_from_logits(logits))

        return all_scores

    def score_paper(self, query: str, keywords: str, paper: dict[str, Any]) -> float:
        return self.batch_score(query, keywords, [paper])[0]
