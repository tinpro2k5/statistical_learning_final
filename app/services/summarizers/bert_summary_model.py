"""
services/summarizers/bert_summary_model.py
-------------------------------------------
Wrapper for encoder-decoder summarization models that are loaded directly
with HuggingFace ``AutoModelForSeq2SeqLM`` (e.g. BART, T5, DistilBART, LED).

Unlike ``LLMSummaryModel`` these models:
- Accept a plain string input (no chat template).
- Are typically much lighter-weight and suitable for CPU inference.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .abstract_summary_model import AbstractSummaryModel


class BertSummaryModel(AbstractSummaryModel):
    """Wraps a HuggingFace seq2seq model for summarization.

    Args:
        model_name_or_path: HuggingFace model ID or local path, e.g.
                            ``"facebook/bart-large-cnn"``.
        device:             ``"cpu"`` or ``"cuda:0"``, etc.
        max_new_tokens:     Maximum length (in tokens) of the generated summary.
        min_new_tokens:     Minimum length (in tokens) of the generated summary.
        normalize_fn:       Optional callable to post-process the decoded string.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        min_new_tokens: int = 10,
        max_input_tokens: int | None = None,
        normalize_fn=None,
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        if torch is not None and device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.max_input_tokens = self._resolve_max_input_tokens(max_input_tokens)
        self._normalize = normalize_fn or (lambda x: x)

    def _resolve_max_input_tokens(self, requested_max: int | None) -> int:
        limits: list[int] = []

        if requested_max is not None and requested_max > 0:
            limits.append(int(requested_max))

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)

        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and 0 < model_limit < 1_000_000:
            limits.append(model_limit)

        encoder_limit = getattr(getattr(self.model, "encoder", None), "max_position_embeddings", None)
        if isinstance(encoder_limit, int) and 0 < encoder_limit < 1_000_000:
            limits.append(encoder_limit)

        return min(limits) if limits else 1024

    # ------------------------------------------------------------------
    # AbstractSummaryModel interface
    # ------------------------------------------------------------------

    def generate(self, text: str) -> str:
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
            if torch is not None and self.device.startswith("cuda"):
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                num_beams=4,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._normalize(summary)
        except Exception as exc:
            print(f"[BertSummaryModel] generation failed: {exc}")
            return ""
