"""
Convert local raw datasets into JSONL format
with BM25 hard negative mining.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import get_logger, save_jsonl, split_by_query_id

logger = get_logger("preprocess_data_hard_neg")


# ---------------------------------------------------------------------------
# Common Data Loaders  (identical to preprocess_data.py)
# ---------------------------------------------------------------------------

def _read_jsonl_as_dict(path: Path) -> dict:
    rows = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[str(row["_id"])] = row
    return rows


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            qid = str(row.get("query-id", row.get("query_id", "")))
            cid = str(row.get("corpus-id", row.get("corpus_id", "")))
            score = int(row.get("score", 0))
            if score > 0 and qid and cid:
                qrels.setdefault(qid, set()).add(cid)
    return qrels


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------

def _build_bm25_index(corpus: dict):
    """
    Build a BM25Okapi index over the entire corpus.

    Tokenizes title + abstract with a simple whitespace split (lowercase).
    Returns (bm25_index, corpus_ids) where corpus_ids is a sorted list so
    the mapping index -> corpus_id is deterministic across runs.

    Requires: rank-bm25  (pip install rank-bm25)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is required for hard negative mining. "
            "Install it with:  pip install rank-bm25"
        ) from exc

    corpus_ids = sorted(corpus.keys())   # sorted for reproducibility
    tokenized: list[list[str]] = []
    for cid in corpus_ids:
        doc = corpus[cid]
        text = (
            doc.get("title", "") + " " + doc.get("text", "")
        ).lower().split()
        # BM25Okapi requires at least one token per document
        tokenized.append(text if text else ["[empty]"])

    logger.info(f"Building BM25 index over {len(corpus_ids)} documents ...")
    index = BM25Okapi(tokenized)
    logger.info("BM25 index built.")
    return index, corpus_ids


# ---------------------------------------------------------------------------
# Hard-negative pair builder
# ---------------------------------------------------------------------------

def _make_pairs_with_hard_neg(
    queries: dict,
    corpus: dict,
    qrels: dict[str, set[str]],
    negatives_per_positive: int,
    rng: random.Random,
    bm25_index,
    corpus_ids: list[str],
    n_hard: int = 2,
    bm25_pool_size: int = 20,
) -> list[dict]:
    """
    Build (query, document, label) pairs using a mixed negative strategy:
      - n_hard   : BM25 top-k negatives (high lexical similarity, label=0)
      - remaining: random negatives from the rest of the corpus

    Parameters
    ----------
    bm25_pool_size:
        Number of top BM25 hits to consider as hard negative candidates.
        Hard negatives are sampled from this pool (not taken greedily) to
        add diversity and avoid overly brittle training signal.
    """
    n_random = max(0, negatives_per_positive - n_hard)
    rows: list[dict] = []

    for qid, pos_ids in sorted(qrels.items()):
        query = queries.get(qid)
        if not query:
            continue

        excluded = set(pos_ids)
        query_tokens = query.get("text", "").lower().split()

        # ── Stage 1: BM25 hard negatives ──────────────────────────────
        bm25_scores = bm25_index.get_scores(query_tokens)
        # Rank corpus_ids by descending BM25 score
        ranked_indices = sorted(
            range(len(corpus_ids)), key=lambda i: -bm25_scores[i]
        )
        hard_pool = [
            corpus_ids[i]
            for i in ranked_indices
            if corpus_ids[i] not in excluded
        ][:bm25_pool_size]

        # Sample from top pool to add diversity
        hard_sample = rng.sample(hard_pool, min(n_hard, len(hard_pool)))
        hard_set = set(hard_sample)

        # ── Stage 2: Random negatives from remaining corpus ───────────
        random_pool = [
            cid for cid in corpus_ids
            if cid not in excluded and cid not in hard_set
        ]
        random_sample = rng.sample(random_pool, min(n_random, len(random_pool)))

        # ── Build rows for each positive ──────────────────────────────
        for pos_id in sorted(pos_ids):
            pos_doc = corpus.get(pos_id)
            if not pos_doc:
                continue

            rows.append({
                "query_id": qid,
                "query":    query.get("text", ""),
                "paper_id": pos_id,
                "title":    pos_doc.get("title", ""),
                "abstract": pos_doc.get("text", ""),
                "label":    1,
                "neg_type": "positive",
            })

            for neg_id in hard_sample:
                neg_doc = corpus[neg_id]
                rows.append({
                    "query_id": qid,
                    "query":    query.get("text", ""),
                    "paper_id": neg_id,
                    "title":    neg_doc.get("title", ""),
                    "abstract": neg_doc.get("text", ""),
                    "label":    0,
                    "neg_type": "hard",
                })

            for neg_id in random_sample:
                neg_doc = corpus[neg_id]
                rows.append({
                    "query_id": qid,
                    "query":    query.get("text", ""),
                    "paper_id": neg_id,
                    "title":    neg_doc.get("title", ""),
                    "abstract": neg_doc.get("text", ""),
                    "label":    0,
                    "neg_type": "random",
                })

    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Negative type distribution reporter
# ---------------------------------------------------------------------------

def _log_neg_distribution(rows: list[dict], split_name: str) -> None:
    """Log hard vs. random negative counts for a given split."""
    from collections import Counter
    counts = Counter(r.get("neg_type", "unknown") for r in rows)
    total = len(rows)
    logger.info(
        f"  {split_name}: {total} rows — "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def _build_scifact(args) -> tuple[list, list, list]:
    rng = random.Random(args.seed)
    scifact_dir = Path(args.input_dir)

    if not scifact_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {scifact_dir}. "
            "Please place the dataset manually."
        )

    corpus = _read_jsonl_as_dict(scifact_dir / "corpus.jsonl")
    queries = _read_jsonl_as_dict(scifact_dir / "queries.jsonl")
    train_qrels = _load_qrels(scifact_dir / "qrels" / "train.tsv")
    test_qrels  = _load_qrels(scifact_dir / "qrels" / "test.tsv")

    bm25_index, corpus_ids = _build_bm25_index(corpus)

    train_all = _make_pairs_with_hard_neg(
        queries, corpus, train_qrels,
        args.negatives_per_positive, rng,
        bm25_index, corpus_ids,
        n_hard=args.n_hard,
        bm25_pool_size=args.bm25_pool_size,
    )
    train_rows, val_rows = split_by_query_id(train_all, args.validation_ratio, args.seed)

    # Test set: use ONLY random negatives (n_hard=0) so evaluation is unbiased.
    # Hard negatives are a training-only strategy to make the model more
    # discriminative; mixing them into test would overstate difficulty and
    # make cross-paper comparisons unreliable.
    test_rows = _make_pairs_with_hard_neg(
        queries, corpus, test_qrels,
        args.negatives_per_positive, rng,
        bm25_index, corpus_ids,
        n_hard=0,
        bm25_pool_size=args.bm25_pool_size,
    )

    return train_rows, val_rows, test_rows


def _build_scidocs(args) -> tuple[list, list, list]:
    rng = random.Random(args.seed)
    scidocs_dir = Path(args.input_dir)

    if not scidocs_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {scidocs_dir}. "
            "Please place the dataset manually."
        )

    corpus = _read_jsonl_as_dict(scidocs_dir / "corpus.jsonl")
    queries = _read_jsonl_as_dict(scidocs_dir / "queries.jsonl")

    # SciDocs only has a single qrel file — we must split at the QUERY level
    # *before* building pairs, so that test pairs can be built with n_hard=0
    # (random negatives only) for an unbiased evaluation protocol.
    all_qrels = _load_qrels(scidocs_dir / "qrels" / "test.tsv")

    # ── Split query IDs into train_val / test ────────────────────────────────
    all_query_ids = sorted(all_qrels.keys())
    rng_split = random.Random(args.seed)
    rng_split.shuffle(all_query_ids)

    n_test = max(1, int(len(all_query_ids) * args.test_ratio))
    test_query_ids  = set(all_query_ids[:n_test])
    trainval_query_ids = set(all_query_ids[n_test:])

    # Further split train_val into train / val
    trainval_list = sorted(trainval_query_ids)
    rng_split.shuffle(trainval_list)
    n_val = max(1, int(len(trainval_list) * args.validation_ratio))
    val_query_ids   = set(trainval_list[:n_val])
    train_query_ids = set(trainval_list[n_val:])

    # Build per-split qrel dicts
    train_qrels = {qid: all_qrels[qid] for qid in train_query_ids}
    val_qrels   = {qid: all_qrels[qid] for qid in val_query_ids}
    test_qrels  = {qid: all_qrels[qid] for qid in test_query_ids}

    logger.info(
        f"SciDocs query split → train: {len(train_qrels)} | "
        f"val: {len(val_qrels)} | test: {len(test_qrels)}"
    )

    bm25_index, corpus_ids = _build_bm25_index(corpus)

    # Train + Val: BM25 hard negatives enabled
    train_rows = _make_pairs_with_hard_neg(
        queries, corpus, train_qrels,
        args.negatives_per_positive, rng,
        bm25_index, corpus_ids,
        n_hard=args.n_hard,
        bm25_pool_size=args.bm25_pool_size,
    )
    val_rows = _make_pairs_with_hard_neg(
        queries, corpus, val_qrels,
        args.negatives_per_positive, rng,
        bm25_index, corpus_ids,
        n_hard=args.n_hard,
        bm25_pool_size=args.bm25_pool_size,
    )

    # Test: random negatives ONLY (n_hard=0) → unbiased evaluation
    test_rows = _make_pairs_with_hard_neg(
        queries, corpus, test_qrels,
        args.negatives_per_positive, rng,
        bm25_index, corpus_ids,
        n_hard=0,
        bm25_pool_size=args.bm25_pool_size,
    )

    return train_rows, val_rows, test_rows


# Dataset registry

DATASET_BUILDERS: dict[str, Callable] = {
    "scifact": _build_scifact,
    "scidocs": _build_scidocs,
}

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess BEIR datasets into JSONL format with BM25 hard negative mining. "
            "Produces output compatible with train.py."
        )
    )
    parser.add_argument("--dataset", default="scifact",
                        choices=list(DATASET_BUILDERS.keys()),
                        help="Which dataset to process.")
    parser.add_argument("--input_dir", required=True,
                        help="Path to the local BEIR-formatted dataset directory.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write train/validation/test JSONL files.")
    parser.add_argument("--negatives_per_positive", type=int, default=4,
                        help="Total negative examples per positive (hard + random). Default: 4")
    parser.add_argument("--n_hard", type=int, default=2,
                        help=(
                            "Number of BM25 hard negatives per positive. "
                            "Remaining slots are filled with random negatives. "
                            "Must be <= negatives_per_positive. Default: 2"
                        ))
    parser.add_argument("--bm25_pool_size", type=int, default=20,
                        help="BM25 top-k pool from which hard negatives are sampled. Default: 20")
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                        help="Fraction of training queries to use as validation. Default: 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of queries to use as test (for single-split datasets). Default: 0.1")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    args = parser.parse_args()

    # Validate n_hard <= negatives_per_positive
    if args.n_hard > args.negatives_per_positive:
        parser.error(
            f"--n_hard ({args.n_hard}) cannot exceed "
            f"--negatives_per_positive ({args.negatives_per_positive})."
        )

    logger.info(
        f"Dataset: {args.dataset} | "
        f"negatives_per_positive: {args.negatives_per_positive} "
        f"({args.n_hard} hard + {args.negatives_per_positive - args.n_hard} random)"
    )

    builder = DATASET_BUILDERS[args.dataset]
    train_rows, val_rows, test_rows = builder(args)

    # Log negative type distribution
    _log_neg_distribution(train_rows, "train")
    _log_neg_distribution(val_rows,   "val")
    _log_neg_distribution(test_rows,  "test")

    output_dir = Path(args.output_dir)
    save_jsonl(output_dir / "train.jsonl",      train_rows)
    save_jsonl(output_dir / "validation.jsonl", val_rows)
    save_jsonl(output_dir / "test.jsonl",       test_rows)

    metadata = {
        "dataset":                  args.dataset,
        "negative_strategy":        "hard_neg",
        "negatives_per_positive":   args.negatives_per_positive,
        "n_hard":                   args.n_hard,
        "n_random":                 args.negatives_per_positive - args.n_hard,
        "bm25_pool_size":           args.bm25_pool_size,
        "seed":                     args.seed,
        "counts": {
            "train":      len(train_rows),
            "validation": len(val_rows),
            "test":       len(test_rows),
        },
    }
    save_jsonl(output_dir / "dataset_metadata.jsonl", [metadata])
    logger.info(f"Saved to {output_dir}: {metadata['counts']}")


if __name__ == "__main__":
    main()
