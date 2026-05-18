from __future__ import annotations

import argparse
import shutil
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from src.data.dataset import RerankDataset
from src.metrics.evaluator import compute_reranking_metrics
from src.models.builder import build_model, build_tokenizer
from src.utils.helpers import (
    get_logger,
    load_config,
    load_jsonl,
    save_json,
    set_seed,
    split_by_query_id,
)

logger = get_logger("train_hf")

# ---------------------------------------------------------------------------
# Task constants (kept identical to train.py for consistency)
# ---------------------------------------------------------------------------

SELECTION_METRIC = {
    "reranking": "ndcg_at_10",
    "document_reranking": "ndcg_at_10",
    "citation_generation": "rougeL",
    "generation": "rougeL",
}

RERANKING_TASKS = {"reranking", "document_reranking"}
GENERATION_TASKS = {"citation_generation", "generation"}


# ---------------------------------------------------------------------------
# Data collator that strips query_id before feeding to model
# (Trainer calls model(**batch), so non-tensor keys must be removed)
# ---------------------------------------------------------------------------

class RerankCollatorForTrainer:
    """
    Wraps make_rerank_collate_fn but keeps query_id separate so
    Trainer can forward only tensor fields to the model.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        pairs = [(item["query_text"], item["candidate_text"]) for item in batch]
        encodings = self.tokenizer(
            pairs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encodings["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.long
        )
        # Store query_id as a plain Python list (not a tensor).
        # Trainer will ignore non-tensor dict values in the batch.
        encodings["query_id"] = [item["query_id"] for item in batch]
        return encodings


# ---------------------------------------------------------------------------
# Custom Trainer subclass: handles query_id passthrough and custom loss
# ---------------------------------------------------------------------------

class RerankTrainer(Trainer):
    """
    Extends Trainer with:
      1. Custom binary/multiclass cross-entropy loss (with optional label smoothing)
      2. Strips query_id from batch before forwarding to model
      3. Stores query_ids so compute_metrics can use them
    """

    def __init__(self, *args, label_smoothing: float = 0.0,
                 query_ids: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        # Populated during prediction; used to reconstruct scored rows
        self._query_ids: list[str] = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn.functional as F

        labels = inputs.pop("labels")
        inputs.pop("query_id", None)   # remove before model forward
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], labels.float())
        else:
            loss = F.cross_entropy(
                logits, labels.long(),
                label_smoothing=self.label_smoothing,
            )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Capture query_ids before popping from inputs
        query_ids = inputs.pop("query_id", [])
        self._query_ids.extend(query_ids)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# ---------------------------------------------------------------------------
# Evaluation helper (runs after Trainer.predict)
# ---------------------------------------------------------------------------

def _score_predictions(logits: np.ndarray, labels: np.ndarray,
                        query_ids: list[str]) -> list[dict]:
    """Convert raw logits to scored rows compatible with compute_reranking_metrics."""
    if logits.shape[-1] == 1:
        scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))   # sigmoid
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = exp[:, 1] / exp.sum(axis=1)            # softmax → class-1

    return [
        {"query_id": qid, "label": int(lbl), "score": float(sc)}
        for qid, lbl, sc in zip(query_ids, labels, scores)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    task = config.get("task", "reranking")
    if task not in RERANKING_TASKS:
        raise NotImplementedError(
            f"train_hf.py currently supports only reranking tasks. Got: '{task}'. "
            "For generation tasks, keep using train.py."
        )

    set_seed(config.get("seed", 42))
    logger.info(f"Task: {task}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_path = config.get("train_file", "")
    val_path   = config.get("validation_file", "")
    test_path  = config.get("test_file", "")

    if not train_path:
        raise ValueError("train_file must be set in config or via CLI.")

    train_rows = load_jsonl(train_path)
    logger.info(f"Loaded {len(train_rows)} training rows from {train_path}")

    if val_path:
        val_rows = load_jsonl(val_path)
    else:
        train_rows, val_rows = split_by_query_id(
            train_rows,
            validation_ratio=config.get("validation_ratio", 0.1),
            seed=config.get("seed", 42),
        )
        logger.info("No validation_file; auto-split from training data.")

    test_rows = load_jsonl(test_path) if test_path else []
    logger.info(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    # ── Model & tokenizer ──────────────────────────────────────────────────
    model_name = config.get("model_name_or_path", "")
    if not model_name:
        raise ValueError("model_name_or_path must be set in config.")

    logger.info(f"Loading tokenizer & model: {model_name}")
    tokenizer = build_tokenizer(task, model_name)
    model     = build_model(task, model_name, config)

    # ── Datasets & collator ────────────────────────────────────────────────
    field_map = config.get("field_map", {})
    train_dataset = RerankDataset(train_rows, field_map=field_map)
    val_dataset   = RerankDataset(val_rows,   field_map=field_map)
    test_dataset  = RerankDataset(test_rows,  field_map=field_map) if test_rows else None
    data_collator = RerankCollatorForTrainer(tokenizer, config.get("max_length", 512))

    # ── Output dir ─────────────────────────────────────────────────────────
    output_dir = Path(config.get("output_dir", "outputs/model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "training_config.json", config)

    # Trainer writes intermediate checkpoints here; we'll rename best later
    hf_ckpt_dir = output_dir / "hf_checkpoints"

    # ── Compute warmup_steps manually to avoid deprecation warning ──────────
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    num_epochs  = int(config.get("num_epochs", 3))
    batch_size  = int(config.get("batch_size", 8))
    # Estimate optimizer steps (Trainer uses same formula internally)
    from math import ceil
    optimizer_steps_per_epoch = ceil(len(train_dataset) / (batch_size * grad_accum))
    total_optimizer_steps = optimizer_steps_per_epoch * num_epochs
    warmup_steps = int(total_optimizer_steps * float(config.get("warmup_ratio", 0.06)))
    logger.info(
        f"optimizer_steps/epoch={optimizer_steps_per_epoch} | "
        f"total={total_optimizer_steps} | warmup={warmup_steps}"
    )

    # ── TrainingArguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(hf_ckpt_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(config.get("eval_batch_size", 16)),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(config.get("learning_rate", 2e-5)),
        warmup_steps=warmup_steps,
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        adam_epsilon=float(config.get("adam_epsilon", 1e-8)),
        seed=int(config.get("seed", 42)),
        label_smoothing_factor=float(config.get("label_smoothing", 0.0)),
        fp16=bool(config.get("fp16", False)),
        # ── Checkpoint strategy ──
        load_best_model_at_end=True,
        metric_for_best_model="eval_ndcg_at_10",
        greater_is_better=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        # ── Logging: nhỏ để thấy loss sớm, override bằng config nếu muốn ──
        logging_steps=int(config.get("logging_steps", 10)),
        logging_first_step=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── compute_metrics (called every eval epoch) ──────────────────────────
    # We pass a mutable container so compute_metrics can access query_ids
    # that were captured during prediction_step.

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # query_ids were stored in trainer._query_ids during prediction_step
        qids = trainer._query_ids[:]
        trainer._query_ids.clear()

        if len(qids) != len(labels):
            raise RuntimeError(
                f"query_id count ({len(qids)}) != label count ({len(labels)}). "
                "This means prediction_step did not capture all query_ids correctly. "
                "NDCG/MRR would be meaningless with mismatched IDs."
            )

        scored = _score_predictions(logits, labels, qids)
        metrics = compute_reranking_metrics(scored)
        return {k: v for k, v in metrics.items()
                if k not in ("num_samples", "sample_predictions", "samples")}

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    train_result = trainer.train()
    logger.info(f"Training result: {train_result}")



    # ── Best checkpoint metadata ───────────────────────────────────────────
    best_src = Path(trainer.state.best_model_checkpoint)
    tokenizer.save_pretrained(str(best_src))   # ensure tokenizer is present

    best_score = trainer.state.best_metric
    selection_key = SELECTION_METRIC[task]
    logger.info(f"Best checkpoint located at → {best_src} ({selection_key}={best_score:.4f})")

    # ── Final test evaluation (on best_checkpoint, not last!) ──────────────
    test_metrics: dict = {}
    if test_dataset is not None:
        logger.info("Running final test evaluation on best_checkpoint …")
        trainer._query_ids.clear()
        # Temporarily disable compute_metrics so it does not clear _query_ids
        # during the predict loop – we read them ourselves right after.
        _saved_compute_metrics = trainer.compute_metrics
        trainer.compute_metrics = None
        pred_output = trainer.predict(test_dataset)
        trainer.compute_metrics = _saved_compute_metrics

        qids = trainer._query_ids[:]
        trainer._query_ids.clear()

        if len(qids) != len(pred_output.label_ids):
            raise RuntimeError(
                f"query_id count ({len(qids)}) != label count ({len(pred_output.label_ids)}) "
                "during test evaluation. NDCG/MRR would be meaningless with mismatched IDs."
            )

        scored = _score_predictions(
            pred_output.predictions,
            pred_output.label_ids,
            qids,
        )
        test_metrics = compute_reranking_metrics(scored)
        test_metrics["num_samples"] = len(scored)
        test_metrics["sample_predictions"] = scored[:10]
        save_json(output_dir / "test_metrics.json", test_metrics)
        logger.info(f"Test metrics: { {k: v for k, v in test_metrics.items() if k not in ('sample_predictions',)} }")

    # ── Build history from Trainer log_history ─────────────────────────────
    history = []
    # Group log entries by completed epoch (eval entries have whole-number epoch,
    # loss entries have fractional epoch → round up to the epoch they belong to).
    epoch_records: dict[int, dict] = {}
    for entry in trainer.state.log_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        # Eval entries are emitted at exactly epoch N.0; train-loss entries are
        # fractional (e.g. 0.97). Use ceil so fractional steps map to the epoch
        # they are part of, matching the eval entry for that epoch.
        ep_key = int(ep) if float(ep) == int(ep) else int(ep) + 1
        rec = epoch_records.setdefault(ep_key, {"epoch": ep_key})
        if "loss" in entry:
            rec["train_loss"] = entry["loss"]
        if "eval_ndcg_at_10" in entry:
            rec["validation"] = {k.replace("eval_", ""): v
                                  for k, v in entry.items()
                                  if k.startswith("eval_")}
    history = sorted(epoch_records.values(), key=lambda r: r["epoch"])

    # ── training_summary.json (identical schema to train.py) ──────────────
    summary = {
        "task": task,
        "model_name_or_path": model_name,
        "best_checkpoint_path": str(best_src),
        f"best_val_{selection_key}": best_score,
        "history": history,
        "test": test_metrics,
        "output_dir": str(output_dir),
        "config": config,
    }
    save_json(output_dir / "training_summary.json", summary)
    logger.info("Training complete.")
    logger.info(f"Best val {selection_key}: {best_score:.4f}")




# ---------------------------------------------------------------------------
# CLI (identical to train.py)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a reranking Transformer using HuggingFace Trainer. "
                    "Specify a YAML config file; CLI args override config values."
    )
    parser.add_argument("--config", dest="config_file", default="",
                        help="Path to YAML config file.")
    parser.add_argument("--task",                  default=None)
    parser.add_argument("--model_name_or_path",    default=None)
    parser.add_argument("--train_file",            default=None)
    parser.add_argument("--validation_file",       default=None)
    parser.add_argument("--test_file",             default=None)
    parser.add_argument("--output_dir",            default=None)
    parser.add_argument("--num_epochs",  type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    config: dict = {}
    if args.config_file:
        config = load_config(args.config_file)
        logger.info(f"Loaded config from {args.config_file}")

    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config_file" and v is not None}
    config.update(cli_overrides)

    if not config.get("task"):
        parser.error("--task must be set in config or via --task CLI argument.")
    if not config.get("train_file"):
        parser.error("--train_file must be set in config or via --train_file CLI argument.")

    train(config)


if __name__ == "__main__":
    main()
