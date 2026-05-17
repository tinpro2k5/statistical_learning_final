"""
preprocess_data.py - Convert local raw datasets into JSONL format.

Currently supported datasets:
  scifact - BEIR SciFact (reranking task)
  scidocs - BEIR SciDocs (reranking task)

Usage
-----
# SciFact reranking (requires data already downloaded in input_dir)
python preprocess_data.py --dataset scifact --input_dir "../data/raw/scifact" --output_dir ../data/scifact

# SciDocs (splits the test set into train/val/test)
python preprocess_data.py --dataset scidocs --input_dir "../data/raw/scidocs" --output_dir ../data/scidocs

Adding a new dataset
--------------------
1.  Write a function  `_build_<dataset>(args) -> tuple[list, list, list]`
    that returns (train_rows, validation_rows, test_rows) as lists of dicts.
2.  Register it in DATASET_BUILDERS.
3.  Done - the main() function handles writing JSONL and metadata.
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

logger = get_logger("preprocess_data")


# Common Data Loaders for BEIR datasets

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


def _make_pairs(queries, corpus, qrels, negatives_per_positive, rng) -> list[dict]:
    corpus_ids = list(corpus.keys())
    rows = []
    for qid, pos_ids in sorted(qrels.items()):
        query = queries.get(qid)
        if not query:
            continue
        excluded = set(pos_ids)
        neg_pool = [cid for cid in corpus_ids if cid not in excluded]
        for pos_id in sorted(pos_ids):
            pos_doc = corpus.get(pos_id)
            if not pos_doc:
                continue
            rows.append({
                "query_id": qid,
                "query": query.get("text", ""),
                "paper_id": pos_id,
                "title": pos_doc.get("title", ""),
                "abstract": pos_doc.get("text", ""),
                "label": 1,
            })
            sample_size = min(negatives_per_positive, len(neg_pool))
            for neg_id in rng.sample(neg_pool, sample_size):
                neg_doc = corpus[neg_id]
                rows.append({
                    "query_id": qid,
                    "query": query.get("text", ""),
                    "paper_id": neg_id,
                    "title": neg_doc.get("title", ""),
                    "abstract": neg_doc.get("text", ""),
                    "label": 0,
                })
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def _build_scifact(args) -> tuple[list, list, list]:
    rng = random.Random(args.seed)
    scifact_dir = Path(args.input_dir)
    
    if not scifact_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {scifact_dir}. Please place the dataset manually.")

    corpus = _read_jsonl_as_dict(scifact_dir / "corpus.jsonl")
    queries = _read_jsonl_as_dict(scifact_dir / "queries.jsonl")
    train_qrels = _load_qrels(scifact_dir / "qrels" / "train.tsv")
    test_qrels = _load_qrels(scifact_dir / "qrels" / "test.tsv")

    train_all = _make_pairs(queries, corpus, train_qrels, args.negatives_per_positive, rng)
    train_rows, val_rows = split_by_query_id(train_all, args.validation_ratio, args.seed)
    test_rows = _make_pairs(queries, corpus, test_qrels, args.negatives_per_positive, rng)

    return train_rows, val_rows, test_rows


def _build_scidocs(args) -> tuple[list, list, list]:
    rng = random.Random(args.seed)
    scidocs_dir = Path(args.input_dir)
    
    if not scidocs_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {scidocs_dir}. Please place the dataset manually.")

    corpus = _read_jsonl_as_dict(scidocs_dir / "corpus.jsonl")
    queries = _read_jsonl_as_dict(scidocs_dir / "queries.jsonl")
    
    # scidocs thường chỉ có tập test, ta load và chia thành 3 phần
    test_qrels = _load_qrels(scidocs_dir / "qrels" / "test.tsv")
    all_rows = _make_pairs(queries, corpus, test_qrels, args.negatives_per_positive, rng)
    
    # Chia test_ratio cho tập test (mặc định args.test_ratio)
    train_val_rows, test_rows = split_by_query_id(all_rows, args.test_ratio, args.seed)
    
    # Tính lại tỷ lệ validation cho tập train_val còn lại
    # Ví dụ nếu test_ratio=0.1, val_ratio=0.1 -> cần lấy 0.1/(1-0.1) = 0.111 của train_val làm val
    val_ratio_adjusted = args.validation_ratio / (1.0 - args.test_ratio + 1e-9)
    train_rows, val_rows = split_by_query_id(train_val_rows, val_ratio_adjusted, args.seed)

    return train_rows, val_rows, test_rows


# Dataset registry

DATASET_BUILDERS: dict[str, Callable] = {
    "scifact": _build_scifact,
    "scidocs": _build_scidocs,
}

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess local NLP datasets into JSONL format."
    )
    parser.add_argument("--dataset", default="scifact",
                        choices=list(DATASET_BUILDERS.keys()),
                        help="Which dataset to process.")
    parser.add_argument("--input_dir", required=True,
                        help="Path to the local directory containing the dataset files.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write train/validation/test JSONL files.")
    parser.add_argument("--negatives_per_positive", type=int, default=4,
                        help="Negative examples per positive.")
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                        help="Fraction of training queries to use as validation.")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of queries to use as test (used if dataset only has one split).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DATASET_BUILDERS[args.dataset]
    logger.info(f"Building dataset: {args.dataset} from {args.input_dir}")
    train_rows, val_rows, test_rows = builder(args)

    output_dir = Path(args.output_dir)
    save_jsonl(output_dir / "train.jsonl", train_rows)
    save_jsonl(output_dir / "validation.jsonl", val_rows)
    save_jsonl(output_dir / "test.jsonl", test_rows)

    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "counts": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
    }
    save_jsonl(output_dir / "dataset_metadata.jsonl", [metadata])
    logger.info(f"Saved to {output_dir}: {metadata['counts']}")


if __name__ == "__main__":
    main()
