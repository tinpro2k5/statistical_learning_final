"""
services/paper_summary_service.py
---------------------------------
Orchestrate paper text fetching and summary generation.

This service owns the paper-specific flow:
fetch article text, prepare it for summarization, call a pluggable
summary-model wrapper, and apply lightweight fallbacks.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Callable
from typing import Any

from db.normalizer import normalize_text
from db.repository import PaperRepository
from services.paper_text_fetcher import PaperTextFetcher
from services.summarizers import AbstractSummaryModel

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_BINARY_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SECTION_PREFIX_RE = re.compile(r"^\s*\d+(\.\d+)*\s+")
_PROOF_HINT_RE = re.compile(
    r"\b(proof|theorem|lemma|corollary|appendix|suppose|therefore|hence|claim|equation)\b",
    re.IGNORECASE,
)
_CONTENT_HINT_RE = re.compile(
    r"\b(abstract|introduction|motivation|method|methods|approach|contribution|result|results|experiment|experiments|conclusion|discussion)\b",
    re.IGNORECASE,
)
_MATH_CHAR_RE = re.compile(r"[\$=\^_\\{}\[\]\(\)]")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "using",
    "our",
    "their",
    "these",
    "those",
}


def _is_likely_pdf_stream_noise(text: str) -> bool:
    lowered = text.lower()
    markers = ("endstream", "endobj", "xref", "startxref", "stream")
    return sum(1 for marker in markers if marker in lowered) >= 2


def _math_density(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    matches = len(_MATH_CHAR_RE.findall(cleaned))
    return matches / max(len(cleaned.split()), 1)


class PaperSummaryService:
    """Fetch full text for a paper and summarize it."""

    def __init__(
        self,
        repo: PaperRepository,
        settings: dict[str, Any] | None = None,
        summary_model: AbstractSummaryModel | None = None,
        summary_model_loader: Callable[[], AbstractSummaryModel | None] | None = None,
        fetcher: PaperTextFetcher | None = None,
    ) -> None:
        self.repo = repo
        self.settings = settings or {}
        self.summary_model = summary_model
        self._summary_model_loader = summary_model_loader
        self._summary_model_loaded = summary_model is not None
        self.fetcher = fetcher or PaperTextFetcher()
        self.max_input_words = int(self.settings.get("max_input_words", 700) or 700)
        self.fallback_sentences = int(self.settings.get("fallback_sentences", 3) or 3)

    def _ensure_summary_model(self) -> AbstractSummaryModel | None:
        if self.summary_model is not None:
            return self.summary_model
        if self._summary_model_loaded:
            return self.summary_model
        self._summary_model_loaded = True
        if self._summary_model_loader is not None:
            self.summary_model = self._summary_model_loader()
        return self.summary_model

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        cleaned = normalize_text(text)
        if not cleaned:
            return []
        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
        return sentences or [cleaned]

    @staticmethod
    def _sentence_tokens(sentence: str) -> list[str]:
        return [
            token.lower()
            for token in _WORD_RE.findall(sentence)
            if len(token) > 2 and token.lower() not in _STOPWORDS
        ]

    @staticmethod
    def _chunk_text(text: str, max_words: int) -> list[str]:
        sentences = PaperSummaryService._split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > max_words:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
            current.append(sentence)
            current_words += sentence_words

        if current:
            chunks.append(" ".join(current).strip())

        return chunks or [normalize_text(text)]

    @staticmethod
    def _is_informative_sentence(sentence: str) -> bool:
        stripped = sentence.strip()
        if not stripped:
            return False
        words = stripped.split()
        if len(words) < 5:
            return False
        if _SECTION_PREFIX_RE.match(stripped):
            return False
        if _PROOF_HINT_RE.search(stripped) and _math_density(stripped) > 0.12:
            return False
        return True

    @classmethod
    def _prepare_text_for_summary(cls, text: str) -> str:
        sentences = cls._split_sentences(text)
        if not sentences:
            return ""

        filtered = [s for s in sentences if cls._is_informative_sentence(s)]
        if len(filtered) >= 6:
            return " ".join(filtered[:120])

        # Fall back to leading content for short/clean texts.
        return " ".join(sentences[:80])

    @staticmethod
    def _chunk_score(chunk: str, index: int, total: int) -> float:
        lowered = chunk.lower()
        score = 0.0
        has_content_hint = bool(_CONTENT_HINT_RE.search(lowered))
        has_proof_hint = bool(_PROOF_HINT_RE.search(lowered))

        # Prefer chunks that look like abstract / intro / conclusion / results.
        if has_content_hint:
            score += 2.5

        # Penalize proof-heavy or appendix-style chunks.
        proof_hits = len(_PROOF_HINT_RE.findall(lowered))
        score -= proof_hits * 2.0
        if has_proof_hint:
            score -= 0.8

        density = _math_density(chunk)
        if density > 0.10:
            score -= (density - 0.10) * 16.0
        elif density > 0.04:
            score -= (density - 0.04) * 6.0

        # Keep early chunks slightly favored only when they look like actual narrative text.
        if index == 0 and has_content_hint and not has_proof_hint:
            score += 1.5
        elif index <= 2 and has_content_hint:
            score += 0.8
        elif index >= max(total - 2, 0):
            score += 0.5

        # Avoid tiny or near-empty chunks.
        word_count = max(len(chunk.split()), 1)
        if word_count < 40:
            score -= 1.0

        return score

    @classmethod
    def _select_summary_chunks(
        cls,
        chunks: list[str],
        max_chunks: int = 4,
        abstract_text: str | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        if len(chunks) <= max_chunks:
            return chunks

        scored = [
            (cls._chunk_score(chunk, index, len(chunks)), index, chunk)
            for index, chunk in enumerate(chunks)
        ]

        # Penalize chunks that mostly repeat the abstract. This keeps the model
        # focused on the body of the paper when full text is available.
        if abstract_text:
            abstract_tokens = set(cls._sentence_tokens(abstract_text))
            if abstract_tokens:
                adjusted: list[tuple[float, int, str]] = []
                for score, index, chunk in scored:
                    chunk_tokens = set(cls._sentence_tokens(chunk))
                    if chunk_tokens:
                        overlap = len(abstract_tokens & chunk_tokens) / float(len(chunk_tokens))
                        if overlap > 0.55:
                            score -= 12.0
                        elif overlap > 0.35:
                            score -= 6.0
                        elif overlap > 0.20:
                            score -= 2.0

                        if index == 0 and overlap > 0.35:
                            score -= 4.0
                    adjusted.append((score, index, chunk))
                scored = adjusted

        # Fill with the highest-scoring chunks while preserving original order.
        selected_indices: set[int] = set()

        for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
            if len(selected_indices) >= max_chunks:
                break
            selected_indices.add(index)

        return [chunk for index, chunk in enumerate(chunks) if index in selected_indices]

    def _extractive_summary(self, text: str, max_sentences: int) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        max_sentences = max(1, min(int(max_sentences or self.fallback_sentences), 5))
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        tokens = [token for sentence in sentences for token in self._sentence_tokens(sentence)]
        if not tokens:
            return " ".join(sentences[:max_sentences])

        frequencies = Counter(tokens)
        total = max(len(sentences), 1)
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = self._sentence_tokens(sentence)
            if not sentence_tokens:
                continue
            tf_score = sum(frequencies[token] for token in sentence_tokens) / len(sentence_tokens)
            # Position bias: first and last sentences carry more weight
            # (intro + conclusion are most informative in academic text)
            position_weight = 1.5 if index == 0 else (1.2 if index >= total - 2 else 1.0)
            score = tf_score * position_weight
            scored.append((score, index, sentence))

        if not scored:
            return " ".join(sentences[:max_sentences])

        chosen = sorted(
            sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences],
            key=lambda item: item[1],
        )
        return " ".join(sentence for _, _, sentence in chosen)

    def _generate_with_model(self, text: str) -> str:
        if self.summary_model is None:
            return ""
        return normalize_text(self.summary_model.generate(text))

    def summarize(self, text: str, max_sentences: int = 3, abstract_text: str | None = None) -> str:
        cleaned = normalize_text(text)
        if not cleaned:
            return ""

        prepared = self._prepare_text_for_summary(cleaned)
        model_input = prepared or cleaned

        summary_model = self._ensure_summary_model()
        if summary_model is None:
            return self._sanitize_summary(
                self._extractive_summary(model_input, max_sentences=max_sentences)
            )

        chunks = self._chunk_text(model_input, self.max_input_words)
        if not chunks:
            return self._sanitize_summary(self._extractive_summary(cleaned, max_sentences=max_sentences))

        chunks = self._select_summary_chunks(chunks, max_chunks=4, abstract_text=abstract_text)

        partials = []
        for chunk in chunks:
            prepared_chunk = self._prepare_text_for_summary(chunk)
            chunk_input = prepared_chunk or chunk
            summary = normalize_text(summary_model.generate(chunk_input))
            if summary:
                partials.append(summary)
        if not partials:
            return self._sanitize_summary(self._extractive_summary(model_input, max_sentences=max_sentences))

        if len(partials) == 1:
            return self._sanitize_summary(partials[0])

        joined = " ".join(partials)
        final_summary = normalize_text(
            summary_model.generate(self._prepare_text_for_summary(joined) or joined)
        )
        fallback = self._extractive_summary(joined, max_sentences=max_sentences)
        return self._sanitize_summary(final_summary or fallback)

    @staticmethod
    def _sanitize_summary(text: str) -> str:
        cleaned = _BINARY_CHAR_RE.sub(" ", normalize_text(text))
        if not cleaned:
            return ""
        if _is_likely_pdf_stream_noise(cleaned):
            return ""
        return cleaned[:1600]

    @staticmethod
    def _looks_low_quality_summary(text: str) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return True
        bad_sections = sum(1 for part in _SENTENCE_SPLIT_RE.split(cleaned) if _SECTION_PREFIX_RE.match(part.strip()))
        if bad_sections >= 3:
            return True
        return False

    def summarize_papers(
        self,
        paper_refs: list[dict[str, Any]],
        max_sentences: int = 3,
    ) -> list[dict[str, Any]]:
        papers = self.repo.get_papers(paper_refs)
        response: list[dict[str, Any]] = []

        for ref, paper in zip(paper_refs, papers):
            paper_id = str(
                paper.get("paper_id")
                or ref.get("id_value", ref.get("paper_id", ref.get("id", "")))
            ).strip()
            collection = paper.get("collection") or ref.get("collection") or "local"
            source_url = self.fetcher.choose_source_url(paper)
            fetched_text, final_url, source_kind = self.fetcher.fetch_full_text_from_url(source_url)
            abstract_text = normalize_text(paper.get("abstract"))
            fallback_text = normalize_text(paper.get("full_text")) or abstract_text
            text_to_summarize = fetched_text or fallback_text
            final_source_kind = source_kind if fetched_text else ("db_full_text" if normalize_text(paper.get("full_text")) else "abstract_fallback")
            summary = self.summarize(
                text_to_summarize,
                max_sentences=max_sentences,
                abstract_text=abstract_text,
            )

            # PDF extraction can still be noisy; if summary quality is poor, fall back to abstract.
            # For the abstract fallback, always use extractive (not LED) — LED is not designed
            # for short inputs like abstracts and produces poor results on them.
            if abstract_text and self._looks_low_quality_summary(summary):
                prepared = self._prepare_text_for_summary(abstract_text)
                summary = self._sanitize_summary(
                    self._extractive_summary(prepared or abstract_text, max_sentences=max_sentences)
                )
                final_source_kind = "abstract_fallback"

            response.append(
                {
                    "paper_id": paper_id,
                    "collection": collection,
                    "source_url": final_url or source_url,
                    "source_kind": final_source_kind,
                    "summary": summary or "Summary unavailable for this paper.",
                }
            )

        return response
