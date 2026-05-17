"""
train.py – Single entry point for all fine-tuning tasks.

Usage examples
--------------
# Reranking with SciBERT on SciFact
python train.py --config configs/reranking_scifact.yaml

# Citation generation with T5
python train.py --config configs/citation_generation.yaml

# Override any config key directly from the CLI
python train.py --config configs/reranking_scifact.yaml --num_epochs 5 --batch_size 16

Supported tasks (set in config YAML):
  reranking / document_reranking  →  cross-encoder sequence classification
  citation_generation / generation →  seq2seq (T5 / BART)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make src importable when running from the transformer_training/ root ──
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import build_collate_fn, build_dataset
from src.metrics.evaluator import compute_reranking_metrics, compute_rouge_metrics
from src.models.builder import build_model, build_tokenizer
from src.utils.helpers import (
    get_logger,
    load_config,
    load_jsonl,
    save_json,
    set_seed,
    split_by_query_id,
    split_rows,
)

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Evaluation helpers (task-specific)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_reranking(model, tokenizer, rows, device, config):
    from src.data.dataset import RerankDataset, make_rerank_collate_fn
    model.eval()
    if not rows:
        return {}
    loader = DataLoader(
        RerankDataset(rows, field_map=config.get("field_map", {})),
        batch_size=int(config.get("eval_batch_size", 16)),
        shuffle=False,
        collate_fn=make_rerank_collate_fn(tokenizer, config.get("max_length", 512)),
    )
    scored: list[dict] = []
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
        for qid, label, score in zip(query_ids, labels, probs):
            scored.append({"query_id": qid, "label": label, "score": score})
    metrics = compute_reranking_metrics(scored)
    metrics["samples"] = scored[:10]
    return metrics


@torch.no_grad()
def _evaluate_generation(model, tokenizer, rows, device, config):
    from src.data.dataset import CitationDataset, make_seq2seq_collate_fn
    model.eval()
    if not rows:
        return {}
    loader = DataLoader(
        CitationDataset(rows, field_map=config.get("field_map", {})),
        batch_size=int(config.get("eval_batch_size", 4)),
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
        refs_ids = labels.clone()
        refs_ids[refs_ids == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(refs_ids, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)
        if len(samples) < 5:
            for p, r in zip(preds, refs):
                samples.append({"prediction": p, "reference": r})
    metrics = compute_rouge_metrics(all_preds, all_refs)
    metrics["samples"] = samples
    return metrics


EVAL_FN_MAP = {
    "reranking": _evaluate_reranking,
    "document_reranking": _evaluate_reranking,
    "citation_generation": _evaluate_generation,
    "generation": _evaluate_generation,
}

SELECTION_METRIC = {
    "reranking": "ndcg_at_10",
    "document_reranking": "ndcg_at_10",
    "citation_generation": "rougeL",
    "generation": "rougeL",
}

LOSS_FN_MAP = {
    "reranking": "reranking",
    "document_reranking": "reranking",
    "citation_generation": "seq2seq",
    "generation": "seq2seq",
}


def _compute_loss(logits_or_outputs, labels, task_type: str, config: dict):
    import torch.nn.functional as F
    if LOSS_FN_MAP.get(task_type) == "seq2seq":
        # seq2seq: loss is directly on the outputs object
        return logits_or_outputs.loss
    # reranking: binary or multiclass cross-entropy
    logits = logits_or_outputs
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits[:, 0], labels.float())
    
    # Nâng cao: Hỗ trợ Label Smoothing để chống Overconfidence
    label_smoothing = config.get("label_smoothing", 0.0)
    return F.cross_entropy(logits, labels.long(), label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    task = config.get("task", "reranking")
    set_seed(config.get("seed", 42))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.get("no_cuda", False) else "cpu"
    )
    logger.info(f"Task: {task} | Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_path = config.get("train_file", "")
    val_path = config.get("validation_file", "")
    test_path = config.get("test_file", "")

    if not train_path:
        raise ValueError("train_file must be set in config or via --train_file CLI argument.")

    train_rows = load_jsonl(train_path)
    logger.info(f"Loaded {len(train_rows)} training rows from {train_path}")

    if val_path:
        val_rows = load_jsonl(val_path)
    else:
        split_fn = split_by_query_id if task in {"reranking", "document_reranking"} else split_rows
        train_rows, val_rows = split_fn(
            train_rows,
            validation_ratio=config.get("validation_ratio", 0.1),
            seed=config.get("seed", 42),
        )
        logger.info("No validation_file provided; auto-split from training data.")

    test_rows = load_jsonl(test_path) if test_path else []
    logger.info(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")

    # ── Build model & tokenizer ────────────────────────────────────────────
    model_name = config.get("model_name_or_path", "")
    if not model_name:
        raise ValueError("model_name_or_path must be set in config.")

    logger.info(f"Loading tokenizer & model: {model_name}")
    tokenizer = build_tokenizer(task, model_name)
    model = build_model(task, model_name, config).to(device)

    # ── Build dataloader ───────────────────────────────────────────────────
    field_map = config.get("field_map", {})
    train_dataset = build_dataset(task, train_rows, field_map)
    collate_fn = build_collate_fn(task, tokenizer, config)
    g = torch.Generator()
    g.manual_seed(config.get("seed", 42))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
    )

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    from transformers import get_linear_schedule_with_warmup
    
    # Nâng cao: Setup Weight Decay (L2 Regularization) chuẩn cho Transformer
    # (Không áp dụng cho bias và LayerNorm)
    weight_decay = float(config.get("weight_decay", 0.01))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    learning_rate = float(config.get("learning_rate", 2e-5))
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = max(len(train_loader) * int(config.get("num_epochs", 3)), 1)
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.06)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Output dir ─────────────────────────────────────────────────────────
    output_dir = Path(config.get("output_dir", "outputs/model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "training_config.json", config)

    # ── Training loop ──────────────────────────────────────────────────────
    eval_fn = EVAL_FN_MAP[task]
    selection_key = SELECTION_METRIC[task]
    best_score = -1.0
    best_dir = output_dir / "best_checkpoint"
    history: list[dict] = []
    grad_accum = config.get("gradient_accumulation_steps", 1)

    for epoch in range(1, config.get("num_epochs", 3) + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")

        for step, batch in enumerate(progress, start=1):
            # seq2seq: all tensors go to device directly
            if LOSS_FN_MAP.get(task) == "seq2seq":
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = _compute_loss(outputs, None, task, config) / grad_accum
            else:
                labels = batch.pop("labels").to(device)
                batch.pop("query_id", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = _compute_loss(outputs.logits, labels, task, config) / grad_accum

            loss.backward()
            running_loss += loss.item()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({"loss": f"{running_loss / step:.4f}"})

        # ── Evaluate ──────────────────────────────────────────────────────
        val_metrics = eval_fn(model, tokenizer, val_rows, device, config)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation": val_metrics,
        }
        history.append(epoch_record)
        score = val_metrics.get(selection_key, 0.0)
        logger.info(f"Epoch {epoch} | loss={epoch_record['train_loss']:.4f} | {selection_key}={score:.4f}")

        if score >= best_score:
            best_score = score
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_json(best_dir / "checkpoint_metadata.json", {
                "task": task,
                "base_model": model_name,
                "checkpoint_role": "best_validation",
                "selection_metric": selection_key,
                "selection_score": score,
                "epoch": epoch,
                "config": config,
                "validation_metrics": val_metrics,
            })
            logger.info(f"  → New best checkpoint saved ({selection_key}={score:.4f})")

    # ── Save last checkpoint ───────────────────────────────────────────────
    last_dir = output_dir / "last_checkpoint"
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    save_json(last_dir / "checkpoint_metadata.json", {
        "task": task,
        "base_model": model_name,
        "checkpoint_role": "last",
        "epochs": config.get("num_epochs", 3),
        "config": config,
    })

    # ── Final test evaluation ──────────────────────────────────────────────
    test_metrics = eval_fn(model, tokenizer, test_rows, device, config) if test_rows else {}

    summary = {
        "task": task,
        "model_name_or_path": model_name,
        f"best_val_{selection_key}": best_score,
        "history": history,
        "test": test_metrics,
        "output_dir": str(output_dir),
        "config": config,
    }
    save_json(output_dir / "training_summary.json", summary)
    logger.info("Training complete.")
    logger.info(f"Best {selection_key}: {best_score:.4f}")
    if test_metrics:
        logger.info(f"Test metrics: {test_metrics}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune any supported Transformer task. "
                    "Specify a YAML config file; CLI args override config values."
    )
    parser.add_argument("--config", dest="config_file", default="",
                        help="Path to YAML config file (required).")
    # Common overrides – all optional (None = use config value)
    parser.add_argument("--task", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load base config from file (if provided)
    config: dict = {}
    if args.config_file:
        config = load_config(args.config_file)
        logger.info(f"Loaded config from {args.config_file}")

    # Override with any explicitly-provided CLI arguments
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
