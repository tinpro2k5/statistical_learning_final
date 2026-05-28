"""
api/routes/papers.py
--------------------
POST /ml-api/get-papers/v1.0

Fetch one or more papers by ID with optional field projection.
"""
from __future__ import annotations

import re

from flask import Blueprint, jsonify, request

from db.repository import PaperRepository
import requests
from services.paper_summary_service import PaperSummaryService


def _extract_title_and_description(html: str) -> dict:
    title = None
    desc = None
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()
    m2 = re.search(r"<meta\s+name=[\"']description[\"']\s+content=[\"'](.*?)[\"']", html, re.IGNORECASE | re.DOTALL)
    if m2:
        desc = m2.group(1).strip()
    return {"title": title or "", "description": desc or ""}


def make_papers_bp(repo: PaperRepository, summary_service: PaperSummaryService) -> Blueprint:
    bp = Blueprint("papers", __name__)

    @bp.post("/ml-api/get-papers/v1.0")
    def get_papers():
        payload = request.get_json(silent=True) or {}
        paper_list = payload.get("paper_list", [])
        projection: dict = payload.get("projection") or {}

        papers = repo.get_papers(paper_list)

        if projection:
            projected = []
            for paper in papers:
                if not paper:
                    projected.append({})
                    continue
                filtered = {
                    key: paper[key]
                    for key, enabled in projection.items()
                    if enabled and key in paper
                }
                # Always include identity fields
                filtered.setdefault("paper_id", paper.get("paper_id"))
                filtered.setdefault("collection", paper.get("collection"))
                projected.append(filtered)
            papers = projected

        return jsonify({"response": papers})

    @bp.post("/ml-api/summarize-papers/v1.0")
    def summarize_papers():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "bad_request", "detail": "JSON body must be an object"}), 400

        paper_list = payload.get("paper_list", [])
        if not isinstance(paper_list, list):
            return jsonify({"error": "bad_request", "detail": "paper_list must be a list"}), 400
        if any(not isinstance(item, dict) for item in paper_list):
            return jsonify({"error": "bad_request", "detail": "paper_list items must be objects"}), 400

        try:
            max_sentences = int(payload.get("max_sentences", 3) or 3)
        except (TypeError, ValueError):
            return jsonify({"error": "bad_request", "detail": "max_sentences must be an integer"}), 400

        summaries = summary_service.summarize_papers(paper_list, max_sentences=max_sentences)
        return jsonify({"response": summaries, "search_stats": {"nSummaries": len(summaries)}})

    @bp.post("/ml-api/fetch-url/v1.0")
    def fetch_url():
        payload = request.get_json(silent=True) or {}
        url = payload.get("url")
        if not url:
            return jsonify({"error": "no url provided"}), 400
        try:
            headers = {"User-Agent": "litsearch-fetcher/1.0"}
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            text = resp.text
            info = _extract_title_and_description(text)
            snippet = text[:4000]
            return jsonify({"status": "ok", "url": url, "final_url": resp.url, "info": info, "snippet": snippet})
        except requests.RequestException as e:
            return jsonify({"status": "error", "error": str(e)}), 502

    return bp
