from __future__ import annotations

import json
from pathlib import Path

from reranking.bert_nsp import BertNSPConfig, BertNSPReranker
from reranking.cross_encoder import CrossEncoderConfig, CrossEncoderReranker


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.exists():
        return path

    candidate = Path(__file__).resolve().parent.parent / config_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Config not found: {config_path}")


def _load_search_model_config(config_path: str | Path) -> dict:
    path = _resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    cfg = raw.get("search_model", raw.get("search", {}))
    if not isinstance(cfg, dict):
        return {}
    return cfg


def load_reranker(config_path: str | Path):
    cfg = _load_search_model_config(config_path)
    if not bool(cfg.get("enabled", False)):
        raise RuntimeError(
            f"search_model.enabled is false in {config_path}. "
            "Set it to true to use the reranker."
        )

    kind = str(cfg.get("kind", "bert_nsp")).lower()
    common = {
        "enabled": bool(cfg.get("enabled", False)),
        "kind": kind,
        "model_name_or_path": str(cfg.get("model_name_or_path", "")),
        "device": str(cfg.get("device", "auto")),
        "max_length": int(cfg.get("max_length", 512)),
        "batch_size": int(cfg.get("batch_size", 16)),
        "query_template": str(cfg.get("query_template", CrossEncoderConfig.query_template)),
        "candidate_template": str(cfg.get("candidate_template", CrossEncoderConfig.candidate_template)),
    }

    if kind in {"cross_encoder", "sequence_classification", "seq_cls"}:
        if not common["model_name_or_path"]:
            common["model_name_or_path"] = CrossEncoderConfig.model_name_or_path
        return CrossEncoderReranker(CrossEncoderConfig(**common))

    if kind == "bert_nsp":
        if not common["model_name_or_path"]:
            common["model_name_or_path"] = BertNSPConfig.model_name_or_path
        return BertNSPReranker(BertNSPConfig(**common))

    raise ValueError(f"Unsupported reranker kind: {kind!r}")
