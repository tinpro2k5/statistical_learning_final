# Scientific Paper Search with Transformer Reranking

A self-contained project demonstrating **transformer-based document reranking** for scientific paper retrieval. The repository is split into two independent components:

| Component | Purpose |
|-----------|---------|
| `transformer_training/` | Fine-tune & evaluate cross-encoder rerankers  |
| `app/` | Flask demo backend wired to a trained checkpoint and SQLite FTS index |

---

## Repository Structure

```
statistical_learning_final/
├── app/                        # Flask web application
│   ├── api/
│   │   ├── app.py              # Application factory (create_app)
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── papers.py
│   │   │   ├── search.py
│   │   │   └── title_search.py
│   │   └── templates/
│   │       └── index.html      # Demo UI
│   ├── db/                     # SQLite repository layer
│   ├── ingestion/              # Data ingestion helpers
│   ├── reranking/              # Cross-encoder inference wrapper
│   │   ├── __init__.py         # load_reranker() entry point
│   │   ├── base.py
│   │   └── cross_encoder.py
│   ├── retrieval/              # FTS5 retrieval layer
│   │   ├── base.py
│   │   └── fts5.py
│   ├── scripts/
│   │   └── build_db.py         # Offline DB build script
│   ├── services/
│   │   └── search_service.py   # Orchestrates retrieval + reranking
│   ├── model_config.json       # Runtime model configuration
│   └── requirements.txt
│
├── transformer_training/       # Training & evaluation pipeline
│   ├── configs/                # YAML experiment configs
│   │   ├── roberta-base_combined.yaml
│   │   ├── roberta-base_hardx2.yaml
│   │   ├── scibert_combined.yaml
│   │   └── scibert_hardx2.yaml
│   ├── src/
│   │   ├── data/               # Dataset & collator classes
│   │   ├── metrics/            # NDCG, MRR, ROUGE evaluators
│   │   ├── models/             # Model & tokenizer builder
│   │   └── utils/              # Logging, JSONL I/O, helpers
│   ├── preprocess_data.py      # BM25 hard-negative mining
│   ├── combine_datasets.py     # Merge multiple JSONL datasets
│   ├── train.py                # Fine-tuning entry point (HF Trainer)
│   ├── evaluate.py             # Standalone evaluation script
│   └── requirements.txt
│
├── data/                       # Raw & preprocessed datasets (not tracked)
└── docs/                       # Analysis reports
```

---

## What is NOT included

- Raw paper corpora (arXiv JSONL snapshot, SciFact, SciDocs raw files).
- Prebuilt SQLite databases (`data/papers.sqlite3`).
- Trained model checkpoints (`transformer_training/outputs/`).

---

## Part 1 — Transformer Training

### 1.1 Install dependencies

```bash
cd transformer_training
pip install -r requirements.txt
```

> `rank-bm25` is required for hard negative mining.
> `torch>=2.0`, `transformers>=4.40`, `datasets>=2.18` are the core deps.

---

### 1.2 Download raw datasets

Place the BEIR-formatted datasets under `data/raw/`:

```
data/raw/
├── scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── train.tsv
│       └── test.tsv
└── scidocs/
    ├── corpus.jsonl
    ├── queries.jsonl
    └── qrels/
        └── test.tsv
```

- **SciFact**: https://huggingface.co/datasets/BeIR/scifact
- **SciDocs**: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip


---

### 1.3 Preprocess — BM25 Hard Negative Mining

`preprocess_data.py` converts raw BEIR datasets into `(query, document, label)` pairs using a **mixed negative strategy**: augmented with BM25 hard negatives (lexically similar but irrelevant), the rest are random.

> **Test sets always use random negatives only** (`n_hard=0`) to ensure an unbiased, comparable evaluation protocol across experiments.


#### Strategy: `combined` (standard 2 hard + 2 random per positive)

```bash
python preprocess_data.py \
    --dataset scidocs \
    --input_dir  "../data/raw/scidocs" \
    --output_dir "../data/scidocs" \

python preprocess_data.py \
    --dataset scifact \
    --input_dir  "../data/raw/scifact" \
    --output_dir "../data/scifact" \

python combine_datasets.py \
    --input_dirs "../data/scifact" "../data/scidocs" \
    --output_dir "../data/combined"
```
#### Strategy: `hardx2` (4 hard + 4 random per positive)
```bash
cd transformer_training

python preprocess_data.py \
    --dataset scidocs \
    --input_dir  "../data/raw/scidocs" \
    --output_dir "../data/scidocs_hardx2" \
    --negatives_per_positive 8 \
    --n_hard 4

python preprocess_data.py \
    --dataset scifact \
    --input_dir  "../data/raw/scifact" \
    --output_dir "../data/scifact_hardx2" \
    --negatives_per_positive 8 \
    --n_hard 4

python combine_datasets.py \
    --input_dirs "../data/scifact_hardx2" "../data/scidocs_hardx2" \
    --output_dir "../data/combined_hardx2"
```

#### Key CLI arguments for `preprocess_data.py`

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `scifact` | `scifact` or `scidocs` |
| `--input_dir` | — | Path to raw BEIR dataset directory |
| `--output_dir` | — | Where to write `train/validation/test.jsonl` |
| `--negatives_per_positive` | `4` | Total negatives per positive (hard + random) |
| `--n_hard` | `2` | BM25 hard negatives; the rest are random |
| `--bm25_pool_size` | `20` | Top-k BM25 candidates to sample hard negatives from |
| `--validation_ratio` | `0.1` | Fraction of training queries held out for validation |
| `--test_ratio` | `0.1` | Fraction for test (SciDocs only — single-split dataset) |
| `--seed` | `42` | Random seed for reproducibility |

---

### 1.4 Training

Training is driven by a YAML config file. Four configs are provided out of the box:

| Config | Backbone | Data strategy |
|--------|----------|---------------|
| `scibert_combined.yaml` | `allenai/scibert_scivocab_uncased` | Standard random negatives |
| `scibert_hardx2.yaml` | `allenai/scibert_scivocab_uncased` | 4 hard + 4 random negatives |
| `roberta-base_combined.yaml` | `roberta-base` | Standard random negatives |
| `roberta-base_hardx2.yaml` | `roberta-base` | 4 hard + 4 random negatives |

```bash
cd transformer_training

# Example: SciBERT baseline
python train.py --config configs/scibert_combined.yaml

# Example: RoBERTa with hard negatives
python train.py --config configs/roberta-base_hardx2.yaml

```

Any YAML key can be overridden from the CLI:

```bash
python train.py --config configs/roberta-base_hardx2.yaml \
    --num_epochs 5 \
    --learning_rate 1e-5
```

#### Key hyperparameters (from `roberta-base_hardx2.yaml`)

```yaml
model_name_or_path: "roberta-base"
batch_size: 16
gradient_accumulation_steps: 2   # effective batch = 32
learning_rate: 2e-5
num_epochs: 3
fp16: true
warmup_ratio: 0.1
label_smoothing: 0.1
max_length: 512
```

#### Training outputs

Each run saves the following under `outputs/<run_name>/`:

| File | Content |
|------|---------|
| `training_config.json` | Resolved config used for this run |
| `training_summary.json` | Full history (loss, NDCG@10 per epoch) + best val metric |
| `test_metrics.json` | Final test-set metrics (NDCG@10, MRR@10, etc.) |
| `hf_checkpoints/` | HuggingFace Trainer checkpoints (best + last) |

Best model selection criterion: **NDCG@10 on the validation set**.

---

### 1.5 Standalone Evaluation

Evaluate any saved checkpoint against any JSONL file:

```bash
# Reranking (cross-encoder)
python evaluate.py \
    --task document_reranking \
    --checkpoint outputs/roberta-base_hardx2/hf_checkpoints/checkpoint-XXXX \
    --data_file  ../data/combined_hardx2/test.jsonl \
    --output_file evaluation/roberta_hardx2_test.json
```

Reported metrics for reranking: **NDCG@5, NDCG@10, MRR@10**.

---

## Part 2 — Flask Application

### 2.1 Install dependencies

```bash
cd app
pip install -r requirements.txt
```

---

### 2.2 Raw Data — Source & Placement

The application backend is built on top of the **arXiv metadata snapshot** — a large JSONL dump of paper metadata maintained by Kaggle/arXiv.

#### Download

| Source | Format | Size (approx.) |
|--------|--------|----------------|
| [arXiv Metadata (Kaggle)](https://www.kaggle.com/datasets/Cornell-University/arxiv) | `.jsonl` | ~5 GB |

Download the file `arxiv-metadata-oai-snapshot.json` and place it here:

```
app/
└── data/
    └── raw/
        └── arxiv-metadata-oai-snapshot.jsonl   ← rename to .jsonl if needed
```

> A small sample file (`arxiv_sample.jsonl`, ~8 MB) is already present in `data/raw/` and can be used for quick local testing.

#### Expected raw record format

Each line in the JSONL is a JSON object with fields like:

```json
{
  "id": "2101.00001",
  "title": "Example Paper Title",
  "abstract": "The abstract text ...",
  "authors": "Smith, J.; Doe, A.",
  "categories": "cs.LG",
  "update_date": "2021-01-15",
  "versions": [{"version": "v1", "created": "Mon, 4 Jan 2021 ..."}]
}
```

The normalizer (`db/normalizer.py`) handles many field name variants automatically (e.g. `title`/`Title`, `author`/`authors`/`Author`, `year`/`Year`/`PublicationYear`), so the script accepts both arXiv-style and other common schemas.

---

### 2.3 Build the SQLite Database (one-time offline step)

The build pipeline has three implicit stages:

```
Raw JSONL file
    │
    ▼  (1) Stream & sample
    │       iter_papers_from_file() — streams line-by-line, no full load into RAM
    │
    ▼  (2) Normalize
    │       normalize_paper() — maps any raw schema → canonical flat dict
    │       Balanced year sampling: picks papers evenly across publication years
    │         so the DB isn't dominated by recent arXiv submissions
    │
    ▼  (3) Bulk upsert → SQLite
            PaperRepository.bulk_upsert_normalized() — 1 000 records/batch
            Writes FTS5 virtual table for full-text search
            Saves normalized records to data/processed/papers_normalized.jsonl
```

#### Run the build script

```bash
cd app

# Full arXiv snapshot — balanced sample of 360 000 papers
python scripts/build_db.py \
    --input_file    data/raw/arxiv-metadata-oai-snapshot.jsonl \
    --db_path       data/papers.sqlite3 \
    --processed_dir data/processed \
    --max_papers    360000

# Quick test with the bundled sample (~8 MB, no --max_papers cap needed)
python scripts/build_db.py \
    --input_file data/raw/arxiv_sample.jsonl \
    --db_path    data/papers.sqlite3
```

#### CLI arguments for `build_db.py`

| Argument | Default | Description |
|---|---|---|
| `--input_file` | *(required)* | Path to `.json` or `.jsonl` source file |
| `--db_path` | `data/papers.sqlite3` | Output SQLite database path |
| `--raw_dir` | `data/raw` | Directory for raw input files |
| `--processed_dir` | `data/processed` | Where to write `papers_normalized.jsonl` |
| `--max_papers` | `0` (unlimited) | Cap total papers; `0` = ingest everything |

#### Normalized canonical schema

After normalization, every record written to the DB has these fields:

| Field | Description |
|---|---|
| `paper_id` | Stable unique ID (from `id`, `paper_id`, `_id`, or derived from title) |
| `collection` | Source tag (e.g. `"local"`, `"arxiv"`) |
| `title` | Cleaned title string |
| `abstract` | Cleaned abstract text |
| `full_text` | Full body text (if present) |
| `primary_category` | e.g. `cs.LG`, `stat.ML` |
| `authors_text` | Semicolon-separated author names (for FTS search) |
| `authors_json` | JSON array of author dicts |
| `year` | Publication year (integer); inferred from `versions[0].created` for arXiv |
| `venue` | Conference/journal name if available |
| `keywords` | Keyword string |
| `doi` | DOI string |
| `links` | JSON array of URLs |

The SQLite database exposes an **FTS5 virtual table** (`papers_fts`) indexed over `title`, `abstract`, `authors_text`, and `keywords`, which the `FTS5Retriever` queries at runtime.

---

### 2.3 Configure the reranker

Edit `app/model_config.json` to point to your trained checkpoint:

```json
{
  "search_model": {
    "enabled": true,
    "kind": "cross_encoder",
    "model_name_or_path": "../transformer_training/outputs/roberta-base_hardx2/hf_checkpoints/checkpoint-2716",
    "device": "auto",
    "max_length": 512,
    "batch_size": 16,
    "query_template": "keywords: {keywords} text before citation: {query}",
    "candidate_template": "title: {title} abstract: {abstract}"
  }
}
```

`model_name_or_path` accepts either a **local checkpoint folder** or a **Hugging Face model ID** (e.g. `"cross-encoder/ms-marco-MiniLM-L-6-v2"`).

To disable reranking and use FTS5-only retrieval, set `"enabled": false`.

---

### 2.4 Run the development server

```bash
cd app
python -m flask --app api.app run --port 8060
```

The UI is served at `http://localhost:8060`.

---

### 2.5 Application Architecture

```
Request
  │
  ▼
Flask Routes (api/routes/)
  │   ├── GET  /            → index.html
  │   ├── POST /search      → full search (query + keywords + filters)
  │   ├── GET  /api/papers  → paginated paper list
  │   └── GET  /api/health  → DB connectivity check
  ▼
SearchService (services/search_service.py)
  │   ├── FTS5Retriever  → SQLite FTS5 full-text retrieval
  │   └── CrossEncoderReranker → transformer reranking (batched)
  ▼
PaperRepository (db/)
  └── SQLite (data/papers.sqlite3)
```

The service layer is fully modular — `FTS5Retriever` can be swapped for any other retriever by changing one line in `api/app.py`.

---

### 2.6 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Demo search UI |
| `POST` | `/search` | Demo UI full search (query + keywords + filters) |
| `GET` | `/api/papers` | Paginated paper list |
| `POST` | `/ml-api/doc-search/v1.0` | JSON document search (full retrieve + rerank) |
| `GET` | `/api/health` | Health check (DB status) |

---

### 2.7 Demo Queries

Because the transformer models (e.g. `roberta_combined`) were trained on a **combined dataset** of both SciFact and SciDocs, the reranker can handle both claim-based reasoning and title-based citation matching seamlessly. 

You can test both types of queries directly in the **same Query box** on the Demo UI (`http://localhost:8060/`).

#### Type 1: SciFact-style (Claim verification)

Short scientific claims or hypotheses. The model retrieves papers that support or contradict the claim.

| Query (paste into search box) | Keywords (optional) |
|-------------------------------|---------------------|
| Smoking increases the risk of lung cancer through DNA methylation changes | cancer, epigenetics |
| BERT outperforms traditional TF-IDF methods on biomedical text classification | NLP, classification |
| Transformer models require less training data than CNNs for image recognition | deep learning |
| mRNA vaccines produce stronger immune responses than protein subunit vaccines | immunology, COVID |
| Preterm infants show altered white matter diffusion compared to full-term infants | MRI, neuroscience |

#### Type 2: SciDocs-style (Citation / Related papers)

Paper titles used as retrieval queries. The model retrieves related papers 

| Query (paste into search box) | Keywords (optional) |
|-------------------------------|---------------------|
| Attention Is All You Need | transformer, self-attention |
| BERT Pre-training of Deep Bidirectional Transformers for Language Understanding | NLP, pre-training |
| Deep Residual Learning for Image Recognition | ResNet, computer vision |
| Active Metric Learning for Classification of Remotely Sensed Hyperspectral Images | remote sensing |

#### Using filters

The UI also supports additional filters to narrow results:

| Filter | Example value | Effect |
|--------|--------------|--------|
| **Year** | `2021` | Only papers published in 2021 |
| **Limit** | `10` | Number of results to return |
| **Collection** | `local` | Filter by dataset source tag (e.g. `local`, `scidocs`) |

#### Understanding the Score

The `score` displayed next to each paper in the UI is the **predicted probability** (from `0.0` to `1.0`) computed by the Transformer cross-encoder model. 

1. **Retrieval**: FTS5 (SQLite) rapidly fetches a broad pool of candidates (e.g., 100 papers) matching the keywords.
2. **Reranking**: The Transformer reads the full text of the `(query, abstract)` pair.
3. **Scoring**: The model outputs logits, which are converted via **softmax** to a probability representing how confident the model is that the paper is relevant to the query. Papers are then sorted descending by this score.

---
