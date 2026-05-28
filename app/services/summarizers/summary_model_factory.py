"""
services/summarizers/summary_model_factory.py
----------------------------------------------
Factory function that reads the ``summary_model`` block from
``model_config.json`` and returns the appropriate
``AbstractSummaryModel`` subclass, or ``None`` when the model is
disabled / misconfigured.

Supported ``kind`` values
--------------------------
``"llm"``
    Loads an ``AutoModelForCausalLM`` (e.g. Llama, Mistral) and wraps
    it in ``LLMSummaryModel``.

``"bert"``
    Loads a HuggingFace seq2seq model (e.g. BART, T5, LED) and wraps
    it in ``BertSummaryModel``.

Example ``model_config.json`` snippet
--------------------------------------
.. code-block:: json

    {
      "summary_model": {
        "enabled": true,
        "kind": "llm",
        "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "device": "auto",
        "load_in_4bit": true,
        "max_new_tokens": 256,
        "do_sample": true,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "system_prompt": "You are a helpful academic assistant...",
        "user_prompt_template": "Summarize this text:\\n\\n{text}"
      }
    }
"""
from __future__ import annotations

from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .abstract_summary_model import AbstractSummaryModel
from .llm_summary_model import LLMSummaryModel
from .bert_summary_model import BertSummaryModel


def _resolve_device(device_setting: Any) -> str:
    """Translate the ``device`` config value to a concrete device string."""
    if isinstance(device_setting, int):
        return f"cuda:{device_setting}" if device_setting >= 0 else "cpu"
    if device_setting == "auto":
        if (
            torch is not None
            and hasattr(torch.cuda, "is_available")
            and torch.cuda.is_available()
        ):
            return "cuda:0"
        return "cpu"
    if isinstance(device_setting, str) and device_setting.startswith("cuda"):
        return device_setting
    return "cpu"


def build_summary_model(
    settings: dict[str, Any],
    normalize_fn: Callable[[str], str] | None = None,
) -> AbstractSummaryModel | None:
    """Construct and return the configured summary model wrapper.

    Args:
        settings:      The ``summary_model`` dict from ``model_config.json``.
        normalize_fn:  Optional text-normalisation callable passed through to
                       the wrapper so it can clean decoded output.

    Returns:
        An ``AbstractSummaryModel`` instance, or ``None`` when the model is
        disabled, not configured, or fails to load.
    """
    if not settings.get("enabled", False):
        return None

    model_name = str(settings.get("model_name_or_path", "")).strip()
    if not model_name:
        return None

    kind = str(settings.get("kind", "llm")).strip().lower()
    target_device = _resolve_device(settings.get("device", "auto"))

    try:
        if kind == "llm":
            return _build_llm(settings, model_name, target_device, normalize_fn)
        if kind == "bert":
            return _build_bert(settings, model_name, target_device, normalize_fn)
        print(
            f"[build_summary_model] Unknown kind '{kind}'. "
            "Supported values: 'llm', 'bert'."
        )
        return None
    except Exception as exc:
        print(f"[build_summary_model] Failed to load '{model_name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Private builders – one per kind
# ---------------------------------------------------------------------------

def _build_llm(
    settings: dict[str, Any],
    model_name: str,
    target_device: str,
    normalize_fn,
) -> LLMSummaryModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    load_in_8bit = bool(settings.get("load_in_8bit", False))
    load_in_4bit = bool(settings.get("load_in_4bit", False))

    if target_device == "cpu":
        device_map: str | dict[str, str] = {"": "cpu"}
    elif target_device.startswith("cuda"):
        device_map = {"": target_device}
    else:
        device_map = "auto"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": device_map,
    }
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit and not load_in_4bit,
            load_in_4bit=load_in_4bit,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return LLMSummaryModel(
        tokenizer=tokenizer,
        model=model,
        system_prompt=settings.get(
            "system_prompt",
            "You are a helpful academic assistant. Summarize the given text concisely.",
        ),
        user_template=settings.get(
            "user_prompt_template",
            "Summarize this text in 7 to 12 sentences:\n\n{text}",
        ),
        do_sample=bool(settings.get("do_sample", True)),
        temperature=float(settings.get("temperature", 0.3) or 0.3),
        top_p=float(settings.get("top_p", 0.9) or 0.9),
        repetition_penalty=float(settings.get("repetition_penalty", 1.05) or 1.05),
        max_new_tokens=int(settings.get("max_new_tokens", 256) or 256),
        normalize_fn=normalize_fn,
    )


def _build_bert(
    settings: dict[str, Any],
    model_name: str,
    target_device: str,
    normalize_fn,
) -> BertSummaryModel:
    return BertSummaryModel(
        model_name_or_path=model_name,
        device=target_device,
        max_new_tokens=int(settings.get("max_new_tokens", 256) or 256),
        min_new_tokens=int(settings.get("min_new_tokens", 10) or 10),
        max_input_tokens=int(settings.get("max_input_tokens", 0) or 0) or None,
        normalize_fn=normalize_fn,
    )