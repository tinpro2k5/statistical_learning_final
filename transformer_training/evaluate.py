"""
evaluate.py – Standalone evaluation script.

Loads a saved checkpoint and evaluates it on any JSONL dataset.

Usage
-----
# Reranking
python evaluate.py \\
    --task document_reranking \\
    --checkpoint outputs/reranking_scifact/best_checkpoint \\
    --data_file ../data/reranking/test.jsonl \\
    --output_file evaluation/reranking_test_metrics.json

# Citation generation
python evaluate.py \\
    --task citation_generation \\
    --checkpoint outputs/citation_generation/best_checkpoint \\
    --data_file ../data/citation_generation/test.jsonl \\
    --output_file evaluation/citation_test_metrics.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CitationDataset,
    RerankDataset,
    make_rerank_collate_fn,
    make_seq2seq_collate_fn,
)
from src.metrics.evaluator import compute_reranking_metrics, compute_rouge_metrics
from src.models.builder import build_tokenizer
from src.utils.helpers import get_logger, load_config, load_jsonl, save_json, set_seed

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Task-specific evaluation implementations
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_reranking(model, tokenizer, rows, device, config) -> dict:
    if not rows:
        logger.warning("No rows to evaluate.")
        return {}
    model.eval()
    loader = DataLoader(
        RerankDataset(rows, field_map=config.get("field_map", {})),
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        collate_fn=make_rerank_collate_fn(tokenizer, config.get("max_length", 512)),
    )
    scored = []
    for batch in loader:
        labels = batch.pop("labels").tolist()
        query_ids = batch.pop("query_id")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu()
        if logits.shape[-1] == 1:
            import torch.nn.functional as F
            probs = torch.sigmoid(logits[:, 0]).tolist()
        else:
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1)[:, 1].tolist()
        for qid, lbl, score in zip(query_ids, labels, probs):
            scored.append({"query_id": qid, "label": lbl, "score": score})

    metrics = compute_reranking_metrics(scored)
    metrics["num_samples"] = len(scored)
    metrics["sample_predictions"] = scored[:10]
    return metrics


@torch.no_grad()
def evaluate_generation(model, tokenizer, rows, device, config) -> dict:
    if not rows:
        logger.warning("No rows to evaluate.")
        return {}
    model.eval()
    loader = DataLoader(
        CitationDataset(rows, field_map=config.get("field_map", {})),
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        collate_fn=make_seq2seq_collate_fn(
            tokenizer,
            config.get("max_source_length", 512),
            config.get("max_target_length", 96),
        ),
    )
    all_preds, all_refs, samples = [], [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=config.get("max_target_length", 96),
            num_beams=config.get("num_beams", 4),
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ref_ids = labels.clone()
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)
        if len(samples) < 10:
            for p, r in zip(preds, refs):
                samples.append({"prediction": p, "reference": r})

    metrics = compute_rouge_metrics(all_preds, all_refs)
    metrics["num_samples"] = len(all_preds)
    metrics["sample_predictions"] = samples
    return metrics


EVAL_FN_MAP = {
    "reranking": evaluate_reranking,
    "document_reranking": evaluate_reranking,
    "citation_generation": evaluate_generation,
    "generation": evaluate_generation,
}

MODEL_CLS_MAP = {
    "reranking": "AutoModelForSequenceClassification",
    "document_reranking": "AutoModelForSequenceClassification",
    "citation_generation": "AutoModelForSeq2SeqLM",
    "generation": "AutoModelForSeq2SeqLM",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a JSONL dataset.")
    parser.add_argument("--config", dest="config_file", default="",
                        help="Optional YAML config file. CLI args override.")
    parser.add_argument("--task", default=None,
                        help="Task name: reranking / citation_generation")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved model checkpoint directory.")
    parser.add_argument("--data_file", default=None,
                        help="Path to evaluation JSONL file.")
    parser.add_argument("--output_file", default="evaluation/metrics.json")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument("--max_target_length", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true", default=None)
    args = parser.parse_args()

    config: dict = {}
    if args.config_file:
        config = load_config(args.config_file)

    # Merge CLI overrides
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config_file" and v is not None}
    config.update(cli_overrides)

    task = config.get("task")
    checkpoint = config.get("checkpoint")
    data_file = config.get("data_file")

    if not task:
        parser.error("--task is required.")
    if not checkpoint:
        parser.error("--checkpoint is required.")
    if not data_file:
        parser.error("--data_file is required.")

    set_seed(config.get("seed", 42))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.get("no_cuda", False) else "cpu"
    )
    logger.info(f"Task: {task} | Checkpoint: {checkpoint} | Device: {device}")

    rows = load_jsonl(data_file)
    logger.info(f"Loaded {len(rows)} rows from {data_file}")

    # Load model
    import importlib
    transformers = importlib.import_module("transformers")
    cls_name = MODEL_CLS_MAP.get(task)
    if cls_name is None:
        raise ValueError(f"Unknown task '{task}'")
    model_cls = getattr(transformers, cls_name)
    tokenizer = build_tokenizer(task, checkpoint)
    model = model_cls.from_pretrained(checkpoint).to(device)

    eval_fn = EVAL_FN_MAP[task]
    metrics = eval_fn(model, tokenizer, rows, device, config)

    output_file = Path(config.get("output_file", "evaluation/metrics.json"))
    save_json(output_file, metrics)
    logger.info(f"Metrics saved to {output_file}")
    logger.info(str(metrics))


if __name__ == "__main__":
    main()
