"""
Model builder: load the correct HuggingFace model class based on task name.

Supported tasks
---------------
reranking / document_reranking  →  AutoModelForSequenceClassification
citation_generation / generation →  AutoModelForSeq2SeqLM

To add a new task, add an entry to TASK_MODEL_CLS_MAP and optionally
provide task-specific kwargs in `_model_kwargs`.
"""
from __future__ import annotations


# Mapping: task name → HuggingFace Auto-class name (string to avoid eager imports)
TASK_MODEL_CLS_MAP: dict[str, str] = {
    "reranking": "AutoModelForSequenceClassification",
    "document_reranking": "AutoModelForSequenceClassification",
    "citation_generation": "AutoModelForSeq2SeqLM",
    "generation": "AutoModelForSeq2SeqLM",
}

TASK_TOKENIZER_CLS_MAP: dict[str, str] = {
    "reranking": "AutoTokenizer",
    "document_reranking": "AutoTokenizer",
    "citation_generation": "AutoTokenizer",
    "generation": "AutoTokenizer",
}


def _resolve_cls(module_name: str):
    """Dynamically import a class from transformers."""
    import importlib
    transformers = importlib.import_module("transformers")
    return getattr(transformers, module_name)


def build_tokenizer(task: str, model_name_or_path: str, use_fast: bool = True):
    """Return an initialised tokenizer appropriate for the task."""
    cls_name = TASK_TOKENIZER_CLS_MAP.get(task)
    if cls_name is None:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(TASK_TOKENIZER_CLS_MAP.keys())}")
    cls = _resolve_cls(cls_name)
    return cls.from_pretrained(model_name_or_path, use_fast=use_fast)


def build_model(task: str, model_name_or_path: str, config: dict | None = None):
    """
    Return an initialised model appropriate for the task.

    Extra kwargs that can be set via config:
        num_labels              (reranking, default 2)
        ignore_mismatched_sizes (bool, default True)
    """
    config = config or {}
    cls_name = TASK_MODEL_CLS_MAP.get(task)
    if cls_name is None:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(TASK_MODEL_CLS_MAP.keys())}")
    cls = _resolve_cls(cls_name)

    kwargs: dict = {}
    if cls_name == "AutoModelForSequenceClassification":
        kwargs["num_labels"] = config.get("num_labels", 2)
        if config.get("ignore_mismatched_sizes", True):
            kwargs["ignore_mismatched_sizes"] = True

    # Nâng cao: Hỗ trợ truyền tham số Dropout để chống overfitting
    if "hidden_dropout_prob" in config:
        kwargs["hidden_dropout_prob"] = config["hidden_dropout_prob"]
    if "attention_probs_dropout_prob" in config:
        kwargs["attention_probs_dropout_prob"] = config["attention_probs_dropout_prob"]

    return cls.from_pretrained(model_name_or_path, **kwargs)
