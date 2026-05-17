"""
api/routes/papers.py
--------------------
POST /ml-api/get-papers/v1.0

Fetch one or more papers by ID with optional field projection.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from db.repository import PaperRepository


def make_papers_bp(repo: PaperRepository) -> Blueprint:
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

    return bp
