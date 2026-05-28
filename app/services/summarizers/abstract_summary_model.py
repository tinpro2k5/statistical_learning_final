"""
services/summarizers/abstract_summary_model.py
-----------------------------------------------
Base interface for all summary model wrappers.

Any concrete implementation (LLM, BERT-pipeline, etc.) must subclass
AbstractSummaryModel and implement `generate(text) -> str`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractSummaryModel(ABC):
    """Contract that every summary-model wrapper must satisfy."""

    @abstractmethod
    def generate(self, text: str) -> str:
        """Summarize *text* and return the summary string.

        Args:
            text: Plain-text input to summarize.

        Returns:
            A non-empty summary string, or an empty string on failure.
        """
