"""
api/routes/search.py
--------------------
POST /ml-api/doc-search/v1.0

Two modes:
  • paper_list provided → rerank a pre-selected set (skip retrieval)
  • no paper_list       → full retrieve + rerank pipeline
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.search_service import SearchService


def make_search_bp(service: SearchService) -> Blueprint:
    bp = Blueprint("search", __name__)

    @bp.post("/ml-api/doc-search/v1.0")
    def doc_search():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("ranking_variable", "")).strip()
        keywords = str(payload.get("keywords", "")).strip()
        paper_list = payload.get("paper_list", [])
        n_results = int(payload.get("nResults", 20))

        if paper_list:
            ranked = service.score_papers(paper_list, query, keywords, n_results)
        else:
            ranked = service.search(query=query, keywords=keywords, limit=n_results)

        response = [
            {
                "collection": p.get("collection", "local"),
                "id_field":   "paper_id",
                "id_type":    "str",
                "id_value":   p.get("paper_id", ""),
                "paper_id":   p.get("paper_id", ""),
                "title":      p.get("title", ""),
                "abstract":   p.get("abstract", ""),
                "year":       p.get("year"),
                "venue":      p.get("venue", ""),
                "citation_count": p.get("citation_count"),
                "citation_updated_at": p.get("citation_updated_at", ""),
                "venue_updated_at": p.get("venue_updated_at", ""),
                "rank":       p.get("rank"),
                "links":      p.get("links", []),
                "score":      p.get("score"),
            }
            for p in ranked
        ]
        return jsonify({
            "response": response,
            "search_stats": {"nMatchingDocuments": len(ranked)},
        })

    return bp
