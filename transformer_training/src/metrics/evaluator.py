"""
Evaluation metrics for reranking and generation tasks.

Reranking  : MRR, NDCG@10, MAP, Hit@K  (primary ranking metrics)
             Accuracy, Precision, Recall, F1 (auxiliary classification metrics)
Generation : ROUGE-1, ROUGE-2, ROUGE-L

Both `compute_reranking_metrics` and `compute_generation_metrics` accept
a list of scored rows so they can be called from:
  - the custom training loop (train.py)
  - the standalone evaluate.py script
  - unit tests
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ranking metric helpers
# ---------------------------------------------------------------------------

def _dcg(ranked_labels: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(
        (2 ** label - 1) / math.log2(i + 2)
        for i, label in enumerate(ranked_labels[:k])
    )


def _ndcg_for_group(items: list[dict], k: int = 10) -> float:
    """NDCG@k for a single query group."""
    ranked = sorted(items, key=lambda r: r["score"], reverse=True)
    ranked_labels = [int(r["label"]) for r in ranked]
    ideal_labels = sorted(ranked_labels, reverse=True)
    dcg  = _dcg(ranked_labels, k)
    idcg = _dcg(ideal_labels, k)
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision_for_group(items: list[dict]) -> float:
    """Average Precision for a single query group."""
    ranked = sorted(items, key=lambda r: r["score"], reverse=True)
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(ranked, 1):
        if int(r["label"]) == 1:
            hits += 1
            precision_sum += hits / i
    total_pos = sum(int(r["label"]) for r in items)
    return precision_sum / total_pos if total_pos > 0 else 0.0


# ---------------------------------------------------------------------------
# Main reranking evaluator
# ---------------------------------------------------------------------------

def compute_reranking_metrics(scored_rows: list[dict]) -> dict[str, Any]:
    """
    Compute ranking and classification metrics from a list of scored rows.

    Each row must have:
        label (int)    – ground truth 0/1
        score (float)  – predicted relevance probability
        query_id (str) – used for per-query ranking metrics

    Primary ranking metrics (for model selection):
        ndcg_at_10, map, mrr, hit_at_1, hit_at_5, hit_at_10

    Auxiliary classification metrics (for debugging):
        accuracy, precision, recall, f1
    """
    if not scored_rows:
        return {k: 0.0 for k in (
            "ndcg_at_10", "map", "mrr",
            "hit_at_1", "hit_at_5", "hit_at_10",
            "accuracy", "precision", "recall", "f1",
        )}

    # ── Classification stats (auxiliary) ──────────────────────────────────
    total = len(scored_rows)
    tp = pp = ap = correct = 0
    grouped: dict[str, list] = defaultdict(list)

    for row in scored_rows:
        pred = 1 if row["score"] >= 0.5 else 0
        label = int(row["label"])
        tp += int(pred == 1 and label == 1)
        pp += int(pred == 1)
        ap += int(label == 1)
        correct += int(pred == label)
        grouped[row["query_id"]].append(row)

    accuracy  = correct / total
    precision = tp / max(pp, 1)
    recall    = tp / max(ap, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)

    # ── Per-query ranking metrics ──────────────────────────────────────────
    rr_list:   list[float] = []
    ndcg_list: list[float] = []
    ap_list:   list[float] = []
    hit1 = hit5 = hit10 = n_groups = 0

    for items in grouped.values():
        ranked = sorted(items, key=lambda r: r["score"], reverse=True)
        pos_ranks = [i for i, r in enumerate(ranked, 1) if int(r["label"]) == 1]
        if not pos_ranks:
            continue
        n_groups += 1
        best = pos_ranks[0]

        rr_list.append(1.0 / best)
        hit1 += int(best <= 1)
        hit5 += int(best <= 5)
        hit10 += int(best <= 10)
        ndcg_list.append(_ndcg_for_group(items, k=10))
        ap_list.append(_average_precision_for_group(items))

    denom = max(n_groups, 1)
    return {
        # ── Primary ranking metrics ──
        "ndcg_at_10": round(sum(ndcg_list) / denom, 4) if ndcg_list else 0.0,
        "map":        round(sum(ap_list)   / denom, 4) if ap_list   else 0.0,
        "mrr":        round(sum(rr_list)   / denom, 4) if rr_list   else 0.0,
        "hit_at_1":  round(hit1  / denom, 4),
        "hit_at_5":  round(hit5  / denom, 4),
        "hit_at_10": round(hit10 / denom, 4),
        # ── Auxiliary classification metrics ──
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ---------------------------------------------------------------------------
# Generation (ROUGE)
# ---------------------------------------------------------------------------

def compute_rouge_metrics(
    predictions: list[str],
    references: list[str],
    rouge_keys: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
) -> dict[str, float]:
    """
    Compute average ROUGE F-measures over prediction/reference pairs.

    Requires:  pip install rouge-score
    """
    from rouge_score import rouge_scorer  # lazy import – optional dependency

    scorer = rouge_scorer.RougeScorer(list(rouge_keys), use_stemmer=True)
    totals: dict[str, float] = defaultdict(float)
    count = max(len(predictions), 1)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_keys:
            totals[key] += scores[key].fmeasure

    return {key: round(totals[key] / count, 4) for key in rouge_keys}


# ---------------------------------------------------------------------------
# Task-level dispatcher
# ---------------------------------------------------------------------------

def get_metric_fn(task: str):
    """
    Return the appropriate metric function for a given task.

    Usage::
        metric_fn = get_metric_fn("reranking")
        metrics = metric_fn(scored_rows)
    """
    if task in {"reranking", "document_reranking"}:
        return compute_reranking_metrics
    if task in {"citation_generation", "generation"}:
        # For generation we return a wrapper that accepts (predictions, references)
        return compute_rouge_metrics
    raise ValueError(f"Unknown task '{task}'")
