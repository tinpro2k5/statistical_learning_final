"""
services/paper_text_fetcher.py
------------------------------
Fetch full text from paper links and return plain text for downstream
summarization. This class owns the network / HTML extraction concerns.
"""
from __future__ import annotations

from html import unescape
import re
from typing import Any
from urllib.parse import urlparse

import requests

from db.normalizer import normalize_text

_TAG_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>|<[^>]+>")
_BINARY_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class PaperTextFetcher:
    """Resolve paper links and fetch a readable article body."""

    _ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
    _NAV_TAG_RE = re.compile(
        r"(?is)<(nav|header|footer|aside|script|style|noscript|figure|figcaption)"
        r"[^>]*>.*?</\1>"
    )
    _MATH_TAG_RE = re.compile(r'(?is)<math[^>]*alttext="([^"]+)"[^>]*>.*?</math>')
    _MATH_TAG_FALLBACK_RE = re.compile(r'(?is)<math[^>]*>.*?</math>')

    @classmethod
    def looks_like_arxiv_id(cls, paper_id: str) -> bool:
        return bool(cls._ARXIV_ID_RE.match(str(paper_id or "").strip()))

    @staticmethod
    def arxiv_id_from_url(url: str) -> str:
        """Extract arXiv ID from abs/pdf URL, e.g. '2101.00001'."""
        parsed = urlparse(url)
        for segment in ("/abs/", "/pdf/"):
            if segment in parsed.path:
                return parsed.path.split(segment)[-1].rstrip("/").removesuffix(".pdf")
        return ""

    def choose_source_url(self, paper: dict[str, Any]) -> str:
        """Return the best URL to fetch full text from.

        Priority for arXiv papers:
          1. arXiv HTML sentinel (to fetch ar5iv clean HTML)
          2. first arXiv link in the paper record
          3. first available link
        """
        paper_id = str(paper.get("paper_id") or "").strip()
        if self.looks_like_arxiv_id(paper_id):
            return f"__arxiv__:{paper_id}"

        links = paper.get("links") or []

        for link in links:
            url = str(link or "").strip()
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.netloc.endswith("arxiv.org") and ("/abs/" in parsed.path or "/pdf/" in parsed.path):
                arxiv_id = self.arxiv_id_from_url(url)
                if arxiv_id:
                    return f"__arxiv__:{arxiv_id}"

        for link in links:
            url = str(link or "").strip()
            if url:
                return url
        return ""

    @classmethod
    def _extract_text_from_html(cls, html: str) -> str:
        stripped = cls._MATH_TAG_RE.sub(r" \1 ", html)
        stripped = cls._MATH_TAG_FALLBACK_RE.sub(" ", stripped)
        stripped = cls._NAV_TAG_RE.sub(" ", stripped)
        cleaned = _TAG_RE.sub(" ", stripped)
        cleaned = unescape(cleaned)
        cleaned = _BINARY_CHAR_RE.sub(" ", cleaned)
        return normalize_text(cleaned)

    def _fetch_ar5iv(self, arxiv_id: str) -> tuple[str, str, str]:
        headers = {"User-Agent": "litsearch-summary/1.0"}
        ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

        try:
            resp = requests.get(ar5iv_url, timeout=15, headers=headers)
            if resp.ok:
                text = self._extract_text_from_html(resp.text)
                if text and len(text.split()) > 200:
                    return text, ar5iv_url, "html"
        except requests.RequestException:
            pass

        return "", f"https://arxiv.org/abs/{arxiv_id}", "fetch_error"

    def fetch_full_text_from_url(self, url: str) -> tuple[str, str, str]:
        if not url:
            return "", "", "missing"

        if url.startswith("__arxiv__:"):
            arxiv_id = url[len("__arxiv__:"):]
            return self._fetch_ar5iv(arxiv_id)

        headers = {"User-Agent": "litsearch-summary/1.0"}
        try:
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            content_type = (response.headers.get("Content-Type") or "").lower()
            final_url = response.url
            is_pdf = (
                "pdf" in content_type
                or final_url.lower().endswith(".pdf")
                or response.content[:4] == b"%PDF"
            )

            if is_pdf:
                return "", final_url, "pdf_unsupported"

            text = self._extract_text_from_html(response.text)
            if text:
                return text, final_url, "html"
        except requests.RequestException:
            return "", url, "fetch_error"

        return "", url, "empty"

    def fetch_paper_text(self, paper: dict[str, Any]) -> tuple[str, str, str]:
        source_url = self.choose_source_url(paper)
        return self.fetch_full_text_from_url(source_url)
