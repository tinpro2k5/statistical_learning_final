"""
services/paper_metadata_enricher.py
-----------------------------------
Best-effort metadata enrichment for papers shown in search results.

The local arXiv snapshot does not contain citation counts and often lacks
journal/conference names.  This service only enriches the small result list
currently being displayed, so search stays current without backfilling the
multi-million-row database.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

from db.normalizer import normalize_text
from db.repository import PaperRepository


class PaperMetadataEnricher:
    """Attach citation count and venue data from external APIs when possible."""

    def __init__(
        self,
        repo: PaperRepository,
        timeout: float = 3.0,
        cache_ttl_seconds: int = 3600,
        refresh_after_seconds: int = 86400,
    ) -> None:
        self.repo = repo
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self.refresh_after_seconds = refresh_after_seconds
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._session = requests.Session()
        self._s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._s2_disabled_until = 0.0
        self._s2_key_rejected = False
        self._s2_last_request_at = 0.0
        self._s2_min_interval_seconds = float(
            os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", "1.1") or 1.1
        )
        self._mailto = os.getenv("OPENALEX_MAILTO", "").strip()

    def enrich_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        refresh_candidates: list[dict[str, Any]] = []
        for paper in papers:
            needs_citation = self._is_stale(paper.get("citation_updated_at"))
            needs_venue = (
                not normalize_text(paper.get("venue"))
                and self._is_stale(paper.get("venue_updated_at"))
            )
            if needs_citation or needs_venue:
                refresh_candidates.append(paper)

        batch_metadata = self._lookup_semantic_scholar_batch(refresh_candidates)
        for paper in papers:
            merged = dict(paper)
            merged.setdefault("citation_count", None)
            merged.setdefault("venue", "")
            needs_citation = self._is_stale(merged.get("citation_updated_at"))
            needs_venue = (
                not normalize_text(merged.get("venue"))
                and self._is_stale(merged.get("venue_updated_at"))
            )
            if not needs_citation and not needs_venue:
                enriched.append(merged)
                continue

            cache_key = self._cache_key(merged)
            metadata = self._lookup_metadata(
                merged,
                needs_citation,
                needs_venue,
                semantic_scholar_metadata=batch_metadata.get(cache_key),
            )
            if merged.get("citation_count") is None and metadata.get("citation_count") is not None:
                merged["citation_count"] = metadata["citation_count"]
            if not normalize_text(merged.get("venue")) and metadata.get("venue"):
                merged["venue"] = metadata["venue"]
            self._persist_metadata(merged, metadata, needs_citation, needs_venue)
            enriched.append(merged)
        return enriched

    def _lookup_metadata(
        self,
        paper: dict[str, Any],
        needs_citation: bool,
        needs_venue: bool,
        semantic_scholar_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(paper)
        if not cache_key:
            return {}

        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self.cache_ttl_seconds:
            return cached[1]

        if semantic_scholar_metadata is not None:
            metadata = dict(semantic_scholar_metadata)
        else:
            metadata = self._lookup_semantic_scholar(paper)
        semantic_scholar_checked = bool(metadata.get("_checked"))

        # Semantic Scholar is the best source for arXiv citation counts, but
        # DOI-backed journal papers can still be enriched by OpenAlex/Crossref
        # when S2 has no useful record.
        if not self._has_useful_metadata(metadata):
            metadata = self._lookup_openalex_by_doi(paper)
        if not self._has_useful_metadata(metadata):
            metadata = self._lookup_crossref(paper)
        if not self._has_useful_metadata(metadata) and semantic_scholar_checked:
            metadata = {"_checked": True}

        if not needs_citation:
            metadata.pop("citation_count", None)
        if not needs_venue:
            metadata.pop("venue", None)

        self._cache[cache_key] = (now, metadata)
        return metadata

    @staticmethod
    def _has_useful_metadata(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata
            and (
                metadata.get("citation_count") is not None
                or normalize_text(metadata.get("venue"))
            )
        )

    def _persist_metadata(
        self,
        paper: dict[str, Any],
        metadata: dict[str, Any],
        needs_citation: bool,
        needs_venue: bool,
    ) -> None:
        paper_id = normalize_text(paper.get("paper_id"))
        if not paper_id:
            return

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        citation_count = metadata.get("citation_count")
        venue = normalize_text(metadata.get("venue"))
        checked = bool(metadata.get("_checked"))
        if not checked and citation_count is None and not venue:
            return
        self.repo.update_paper_metadata(
            paper_id,
            citation_count=int(citation_count) if citation_count is not None else None,
            citation_updated_at=now if needs_citation and checked else None,
            venue=venue or "",
            venue_updated_at=now if needs_venue and checked else None,
        )
        if needs_citation and checked:
            paper["citation_updated_at"] = now
        if needs_venue and checked:
            paper["venue_updated_at"] = now

    def _is_stale(self, value: Any) -> bool:
        updated_at = normalize_text(value)
        if not updated_at:
            return True
        try:
            parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
        return age.total_seconds() >= self.refresh_after_seconds

    @staticmethod
    def _cache_key(paper: dict[str, Any]) -> str:
        doi = normalize_text(paper.get("doi"))
        if doi:
            return f"doi:{doi.lower()}"
        paper_id = normalize_text(paper.get("paper_id"))
        if paper_id:
            return f"id:{paper_id.lower()}"
        title = normalize_text(paper.get("title"))
        return f"title:{title.lower()}" if title else ""

    @staticmethod
    def _paper_external_id(paper: dict[str, Any]) -> str:
        doi = normalize_text(paper.get("doi"))
        if doi:
            return f"DOI:{doi}"

        paper_id = normalize_text(paper.get("paper_id"))
        if paper_id:
            import re
            s = paper_id.strip()
            if s.lower().startswith("arxiv:"):
                s = s[6:].strip()
            elif ":" in s:
                parts = s.split(":", 1)
                if parts[0].lower() in ("arxiv", "local"):
                    s = parts[1].strip()
            
            modern_match = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s)
            old_match = re.match(r"^[a-zA-Z\-]+(\.[a-zA-Z\-]+)?/\d{7}(v\d+)?$", s)
            if modern_match or old_match:
                return f"ARXIV:{s}"
        return ""

    @staticmethod
    def _metadata_from_semantic_scholar(data: dict[str, Any] | None) -> dict[str, Any]:
        if not data:
            return {"_checked": True}

        venue = normalize_text(data.get("venue"))
        publication_venue = data.get("publicationVenue") or {}
        if not venue and isinstance(publication_venue, dict):
            venue = normalize_text(publication_venue.get("name"))
        journal = data.get("journal") or {}
        if not venue and isinstance(journal, dict):
            venue = normalize_text(journal.get("name"))

        return {
            "_checked": True,
            "citation_count": data.get("citationCount"),
            "venue": venue,
        }

    def _lookup_semantic_scholar_batch(
        self,
        papers: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if time.time() < self._s2_disabled_until:
            return {}

        items: list[tuple[str, str]] = []
        seen_ids: set[str] = set()
        for paper in papers:
            cache_key = self._cache_key(paper)
            external_id = self._paper_external_id(paper)
            if not cache_key or not external_id or external_id in seen_ids:
                continue
            seen_ids.add(external_id)
            items.append((cache_key, external_id))

        if not items:
            return {}

        try:
            self._wait_for_semantic_scholar_slot()
            response = self._session.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={
                    "fields": "citationCount,venue,publicationVenue,journal",
                },
                json={"ids": [external_id for _, external_id in items]},
                headers=self._semantic_scholar_headers(),
                timeout=self.timeout,
            )
            if response.status_code == 403 and self._s2_api_key and not self._s2_key_rejected:
                self._s2_key_rejected = True
                self._wait_for_semantic_scholar_slot()
                response = self._session.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={
                        "fields": "citationCount,venue,publicationVenue,journal",
                    },
                    json={"ids": [external_id for _, external_id in items]},
                    headers=self._semantic_scholar_headers(use_key=False),
                    timeout=self.timeout,
                )
            if response.status_code == 429:
                self._s2_disabled_until = time.time() + 300
                return {}
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError):
            return {}

        if not isinstance(rows, list):
            return {}

        result: dict[str, dict[str, Any]] = {}
        for (cache_key, _), row in zip(items, rows):
            result[cache_key] = self._metadata_from_semantic_scholar(row)
        return result

    def _lookup_semantic_scholar(self, paper: dict[str, Any]) -> dict[str, Any]:
        if time.time() < self._s2_disabled_until:
            return {}

        external_id = self._paper_external_id(paper)
        if not external_id:
            return {}

        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(external_id, safe=':')}"

        try:
            self._wait_for_semantic_scholar_slot()
            response = self._session.get(
                url,
                params={
                    "fields": "citationCount,venue,publicationVenue,journal",
                },
                headers=self._semantic_scholar_headers(),
                timeout=self.timeout,
            )
            if response.status_code == 403 and self._s2_api_key and not self._s2_key_rejected:
                self._s2_key_rejected = True
                self._wait_for_semantic_scholar_slot()
                response = self._session.get(
                    url,
                    params={
                        "fields": "citationCount,venue,publicationVenue,journal",
                    },
                    headers=self._semantic_scholar_headers(use_key=False),
                    timeout=self.timeout,
                )
            if response.status_code == 429:
                self._s2_disabled_until = time.time() + 300
                return {}
            if response.status_code == 404:
                return {"_checked": True}
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return {}

        return self._metadata_from_semantic_scholar(data)

    def _semantic_scholar_headers(self, use_key: bool = True) -> dict[str, str]:
        headers = {"User-Agent": "litsearch-metadata/1.0"}
        if use_key and self._s2_api_key and not self._s2_key_rejected:
            headers["x-api-key"] = self._s2_api_key
        return headers

    def _wait_for_semantic_scholar_slot(self) -> None:
        elapsed = time.time() - self._s2_last_request_at
        wait_seconds = self._s2_min_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._s2_last_request_at = time.time()

    def _lookup_crossref(self, paper: dict[str, Any]) -> dict[str, Any]:
        doi = normalize_text(paper.get("doi"))
        if not doi:
            return {}

        headers = {"User-Agent": "litsearch-metadata/1.0"}
        try:
            response = self._session.get(
                f"https://api.crossref.org/works/{quote(doi, safe='')}",
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return {"_checked": True}
            response.raise_for_status()
            data = response.json().get("message", {})
        except (requests.RequestException, ValueError):
            return {}

        container = data.get("container-title") or []
        event = data.get("event") or {}
        venue = ""
        if isinstance(container, list) and container:
            venue = normalize_text(container[0])
        if not venue and isinstance(event, dict):
            venue = normalize_text(event.get("name"))

        return {
            "_checked": True,
            "citation_count": data.get("is-referenced-by-count"),
            "venue": venue,
        }

    def _lookup_openalex_by_doi(self, paper: dict[str, Any]) -> dict[str, Any]:
        doi = normalize_text(paper.get("doi"))
        if not doi:
            paper_id = normalize_text(paper.get("paper_id"))
            if paper_id:
                import re
                s = paper_id.strip()
                if s.lower().startswith("arxiv:"):
                    s = s[6:].strip()
                elif ":" in s:
                    parts = s.split(":", 1)
                    if parts[0].lower() in ("arxiv", "local"):
                        s = parts[1].strip()
                
                modern_match = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s)
                old_match = re.match(r"^[a-zA-Z\-]+(\.[a-zA-Z\-]+)?/\d{7}(v\d+)?$", s)
                if modern_match or old_match:
                    doi = f"10.48550/arxiv.{s.lower()}"

        if not doi:
            return {}

        params = {"filter": f"doi:{doi}", "per-page": 1}
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            response = self._session.get(
                "https://api.openalex.org/works",
                params=params,
                headers={"User-Agent": "litsearch-metadata/1.0"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError):
            return {}

        if not results:
            return {"_checked": True}

        work = results[0]
        source = ((work.get("primary_location") or {}).get("source") or {})
        return {
            "_checked": True,
            "citation_count": work.get("cited_by_count"),
            "venue": normalize_text(source.get("display_name")),
        }
