"""
api/routes/health.py
"""
from __future__ import annotations

from flask import Blueprint, jsonify

from db.repository import PaperRepository


def make_health_bp(repo: PaperRepository) -> Blueprint:
    bp = Blueprint("health", __name__)

    @bp.get("/health")
    def health():
        return jsonify({"status": "ok", "paper_count": repo.count()})

    return bp
