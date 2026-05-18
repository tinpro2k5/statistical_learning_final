# SciLit Transformer App

A focused standalone project extracted from SciLit for two Transformer-based NLP tasks:

- Citation generation with T5.
- Document reranking with SciBERT.

This folder is intentionally smaller than the original repository. It contains:

- Training and evaluation scripts for both core models.
- Dependency files for Python and frontend packages.

## What is included

- `transformer_training/` for fine-tuning and evaluation.
- `app/api/templates/index.html` for the Flask demo interface.

## What is not included

- Large raw paper corpora.
- Prebuilt SQLite databases.
- Full upstream microservice stack.

You can plug the trained checkpoints back into the original SciLit services if needed.

## Suggested use

1. Put your dataset in JSONL format.
2. Edit a YAML config under `transformer_training/configs/`.
3. Run the train script for the task you want.
4. Run the evaluation script on validation or test data.
5. Use the wrapper UI or wire the checkpoint into the original runtime services.

## Training with YAML configs

Prepare a quick scientific reranking dataset from BEIR SciFact:

```bash
python transformer_training/prepare_scifact_reranking.py --output_dir data/reranking --negatives_per_positive 4
```

Document reranking:

```bash
python transformer_training/document_reranking_train.py --config_file transformer_training/configs/document_reranking.yaml
python transformer_training/document_reranking_evaluate.py --data_file data/reranking/test.jsonl --checkpoint outputs/document_reranking/best_checkpoint
```

The default reranking config fine-tunes `allenai/scibert_scivocab_uncased` as a
binary sequence-classification cross-encoder. For a smaller already-trained
generic reranker, use:

```bash
python transformer_training/document_reranking_train.py --config_file transformer_training/configs/document_reranking_minilm.yaml
```

`app/model_config.json` is the runtime config used by the Flask app.


Citation generation:

```bash
python transformer_training/citation_generation_train.py --config_file transformer_training/configs/citation_generation.yaml
python transformer_training/citation_generation_evaluate.py --data_file data/citation_generation/test.jsonl --checkpoint outputs/citation_generation/best_checkpoint
```

Each training run saves:

- `training_config_resolved.json`
- `training_summary.json`
- `best_checkpoint/checkpoint_metadata.json`
- `last_checkpoint/checkpoint_metadata.json`

## Simplified database layer

```bash
cd statistical_learning_final
python app/scripts/build_db.py --input_file data/raw/arxiv_sample.jsonl --db_path data/papers.sqlite3 --processed_dir data/processed
python -m flask --app statistical_learning_final.app.api.app run --port 8060
```

The offline pipeline is split into three explicit stages:

1. Download raw paper files or URLs.
2. Process and normalize records into a stable schema.
3. Build the SQLite database once, then keep the runtime API read-only.

To make the backend load a Transformer reranker, switch `app/model_config.json` to something like:

```json
{
	"search_model": {
		"enabled": true,
		"kind": "cross_encoder",
		"model_name_or_path": "outputs/document_reranking/best_checkpoint"
	}
}
```

`model_name_or_path` can point to either a local checkpoint folder or a Hugging Face model id.

## Python setup

Install the Python packages from `backend/requirements.txt` and the task-specific extras in `transformer_training/requirements.txt`.

