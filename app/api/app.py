"""
api/app.py
----------
Flask application factory.

Usage:
    python -m flask --app api.app run --port 8060
    # or in code:
    from api.app import create_app
    app = create_app("data/papers.sqlite3")
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from db.repository import PaperRepository
from db.normalizer import normalize_text
from reranking import load_reranker
from retrieval.fts5 import FTS5Retriever
from services.search_service import SearchService
from services.paper_summary_service import PaperSummaryService
from services.paper_metadata_enricher import PaperMetadataEnricher
from services.summarizers import build_summary_model

from api.routes.health import make_health_bp
from api.routes.papers import make_papers_bp
from api.routes.search import make_search_bp


def _load_json_config(config_path: str | Path) -> dict:
    try:
        with Path(config_path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def create_app(
    db_path: str | Path = "data/papers.sqlite3",
    config_path: str | Path = "model_config.json",
) -> Flask:
    """
    Build and return the Flask application.

    Dependency wiring happens here — routes receive only what they need.
    Swap FTS5Retriever for BM25Retriever / VectorRetriever by changing
    one line below; nothing else needs to change.
    """
    app = Flask(__name__)
    CORS(app)

    # --- infrastructure ---------------------------------------------------
    repo      = PaperRepository(db_path)
    retriever = FTS5Retriever(db_path)          # ← swap point
    reranker  = load_reranker(config_path)
    config = _load_json_config(config_path)
    summary_settings = config.get("summary_model", {})
    summary_model = build_summary_model(summary_settings, normalize_fn=normalize_text)

    summary_service = PaperSummaryService(repo, summary_settings, summary_model)
    metadata_enricher = PaperMetadataEnricher(repo)

    # --- service layer ----------------------------------------------------
    service = SearchService(repo, retriever, reranker, metadata_enricher)

    # --- routes (factory-function pattern) --------------------------------
    app.register_blueprint(make_health_bp(repo))
    app.register_blueprint(make_papers_bp(repo, summary_service))
    app.register_blueprint(make_search_bp(service))

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/search")
    def search_home():
        return render_template("index.html")

    @app.post("/search")
    def search_page():
        query = str(request.form.get("query", "")).strip()
        keywords = str(request.form.get("keywords", "")).strip()
        limit = int(request.form.get("limit", 10) or 10)
        year_raw = str(request.form.get("year", "")).strip()
        collection = str(request.form.get("collection", "")).strip() or None
        year = int(year_raw) if year_raw else None

        results = service.search(
            query=query,
            keywords=keywords,
            limit=limit,
            collection=collection,
            year=year,
        )
        return render_template(
            "index.html",
            query=query,
            keywords=keywords,
            limit=limit,
            collection=collection or "",
            year=year_raw,
            results=results,
            result_count=len(results),
        )

    # --- generic error handlers -------------------------------------------
    @app.errorhandler(400)
    def bad_request(err):
        return jsonify({"error": "bad_request", "detail": str(err)}), 400

    @app.errorhandler(404)
    def not_found(err):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(500)
    def internal(err):
        return jsonify({"error": "internal_server_error", "detail": str(err)}), 500

    return app
