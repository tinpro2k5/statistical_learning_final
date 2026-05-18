"""
api/routes/title_search.py
--------------------------
POST /ml-api/title-generic-search/v1.0

Look up papers by title (+ optional authors) and return match metadata.
Each item in the request's ``titles`` list is resolved independently.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.search_service import SearchService


def make_title_search_bp(service: SearchService) -> Blueprint:
    bp = Blueprint("title_search", __name__)

    @bp.post("/ml-api/title-generic-search/v1.0")
    def title_generic_search():
        payload = request.get_json(silent=True) or {}
        titles = payload.get("titles", [])
        projection: dict = payload.get("projection") or {}

        response = []
        for title_item in titles:
            if isinstance(title_item, dict):
                title = str(title_item.get("Title", title_item.get("title", ""))).strip()
                authors = title_item.get("Author", title_item.get("authors", []))
            else:
                title = str(title_item).strip()
                authors = []

            candidates = service.search_by_title(title, authors=authors, limit=1)

            if candidates:
                paper = candidates[0]
                result: dict = {
                    "found":      True,
                    "collection": paper.get("collection", ""),
                    "id_field":   "paper_id",
                    "id_type":    "str",
                    "id_value":   paper.get("paper_id", ""),
                    "_id":        f"{paper.get('collection', 'local')}_{paper.get('paper_id', '')}",
                }
                if projection:
                    for key, enabled in projection.items():
                        if enabled and key in paper:
                            result[key] = paper[key]
                else:
                    result.update(paper)
            else:
                result = {"found": False}
                if isinstance(title_item, dict):
                    result.update(title_item)
                else:
                    result["Title"] = title

            response.append(result)

        return jsonify({"response": response})

    return bp
