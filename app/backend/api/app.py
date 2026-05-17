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

from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from db.repository import PaperRepository
from reranking import load_reranker
from retrieval.fts5 import FTS5Retriever
from services.search_service import SearchService

from api.routes.health import make_health_bp
from api.routes.papers import make_papers_bp
from api.routes.search import make_search_bp
from api.routes.title_search import make_title_search_bp


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

    # --- service layer ----------------------------------------------------
    service = SearchService(repo, retriever, reranker)

    # --- routes (factory-function pattern) --------------------------------
    app.register_blueprint(make_health_bp(repo))
    app.register_blueprint(make_papers_bp(repo))
    app.register_blueprint(make_search_bp(service))
    app.register_blueprint(make_title_search_bp(service))

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
