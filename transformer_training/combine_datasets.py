"""
combine_datasets.py - Merge multiple preprocessed JSONL datasets into one.

Strategy
--------
  - Each source dataset is split INDEPENDENTLY into train/val/test FIRST
    (via preprocess_data.py) to prevent query-level data leakage.
  - This script then concatenates the splits:
      combined/train.jsonl = scifact/train + scidocs/train  (shuffled)
      combined/val.jsonl   = scifact/val   + scidocs/val    (shuffled)
      combined/test.jsonl  = scifact/test  + scidocs/test   (NOT shuffled,
                             keeps source-dataset blocks for per-dataset eval)

Usage
-----
python combine_datasets.py \\
    --input_dirs "../data/scifact" "../data/scidocs" \\
    --output_dir "../data/combined"

# With custom seed
python combine_datasets.py \\
    --input_dirs "../data/scifact" "../data/scidocs" \\
    --output_dir "../data/combined" \\
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import get_logger, load_jsonl, save_jsonl

logger = get_logger("combine_datasets")

SPLITS = ("train", "validation", "test")


def _tag_source(rows: list[dict], source_name: str) -> list[dict]:
    """Add a 'source_dataset' field to each row for traceability."""
    for row in rows:
        row["source_dataset"] = source_name
    return rows


def combine(input_dirs: list[Path], output_dir: Path, seed: int) -> None:
    rng = random.Random(seed)

    split_data: dict[str, list[dict]] = {s: [] for s in SPLITS}

    for dataset_dir in input_dirs:
        source_name = dataset_dir.name  # e.g. "scifact", "scidocs"
        logger.info(f"Reading dataset: {source_name} from {dataset_dir}")

        for split in SPLITS:
            fpath = dataset_dir / f"{split}.jsonl"
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing split file: {fpath}\n"
                    f"Run preprocess_data.py for '{source_name}' first."
                )
            rows = load_jsonl(fpath)
            rows = _tag_source(rows, source_name)
            split_data[split].extend(rows)
            logger.info(f"  {split:10s}: {len(rows):>6,} rows from {source_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        rows = split_data[split]

        # Shuffle train and val, but keep test un-shuffled so per-dataset
        # evaluation blocks remain contiguous (easier to slice later).
        if split in ("train", "validation"):
            rng.shuffle(rows)

        out_path = output_dir / f"{split}.jsonl"
        save_jsonl(out_path, rows)
        logger.info(f"Saved {split:10s}: {len(rows):>6,} rows → {out_path}")

    # Write metadata
    metadata = {
        "sources": [str(d) for d in input_dirs],
        "seed": seed,
        "counts": {s: len(split_data[s]) for s in SPLITS},
    }
    meta_path = output_dir / "dataset_metadata.jsonl"
    save_jsonl(meta_path, [metadata])
    logger.info(f"Metadata saved → {meta_path}")

    # Summary
    print("\n=== Combined Dataset Summary ===")
    for split in SPLITS:
        print(f"  {split:10s}: {len(split_data[split]):>7,} rows")
    print(f"  Output     : {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple preprocessed JSONL datasets into one."
    )
    parser.add_argument(
        "--input_dirs", nargs="+", required=True,
        help="Paths to preprocessed dataset directories (must each contain train/validation/test.jsonl).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Destination directory for the combined dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling train/val. Default: 42",
    )
    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    output_dir = Path(args.output_dir)

    combine(input_dirs, output_dir, args.seed)


if __name__ == "__main__":
    main()
