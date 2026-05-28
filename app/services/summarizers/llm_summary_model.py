"""
services/summarizers/llm_summary_model.py
------------------------------------------
Wrapper for causal / generative LLMs (e.g. Llama, Mistral) that are loaded
via HuggingFace ``AutoModelForCausalLM``.

The wrapper formats a chat prompt, runs generation, strips the prompt prefix
from the output, and preserves valid LaTeX while fixing obvious delimiter noise.
"""
from __future__ import annotations

import re

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .abstract_summary_model import AbstractSummaryModel

# ---------------------------------------------------------------------------
# LaTeX / markup cleanup helpers
# ---------------------------------------------------------------------------
_EXCESS_DOLLARS_RE = re.compile(r"\${3,}")
_SPACED_DOLLARS_RE = re.compile(r"\$\s+\$")


def _normalize_latex_markup(text: str) -> str:
    """Keep real math markup, but collapse clearly broken dollar noise.

    The model should be allowed to emit valid math such as ``$x^2$`` or
    ``$$...$$``. This helper only repairs malformed delimiter runs that would
    otherwise break rendering.
    """
    if not text:
        return text

    cleaned = text.replace(r"\$", "$")
    cleaned = _EXCESS_DOLLARS_RE.sub("$$", cleaned)
    cleaned = _SPACED_DOLLARS_RE.sub("$$", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# LLMSummaryModel
# ---------------------------------------------------------------------------

class LLMSummaryModel(AbstractSummaryModel):
    """Wraps a HuggingFace causal LLM for summarization via chat templates.

    Args:
        tokenizer:           A loaded ``AutoTokenizer`` instance.
        model:               A loaded ``AutoModelForCausalLM`` instance (eval mode).
        system_prompt:       System-role message sent before the user turn.
        user_template:       User-role message template; use ``{text}`` as placeholder.
        do_sample:           Whether to use sampling vs. greedy decoding.
        temperature:         Sampling temperature (ignored when ``do_sample=False``).
        top_p:               Nucleus-sampling top-p.
        repetition_penalty:  Token-level repetition penalty.
        max_new_tokens:      Maximum number of tokens to generate.
        normalize_fn:        Optional callable to post-process the decoded string.
    """

    def __init__(
        self,
        tokenizer,
        model,
        system_prompt: str,
        user_template: str,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        normalize_fn=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self._normalize = normalize_fn or (lambda x: x)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.user_template.replace("{text}", text)},
        ]
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback for models without a chat template
        user_content = self.user_template.replace("{text}", text)
        return f"System: {self.system_prompt}\nUser: {user_content}\nAssistant:"

    # ------------------------------------------------------------------
    # AbstractSummaryModel interface
    # ------------------------------------------------------------------

    def generate(self, text: str) -> str:
        try:
            prompt = self._build_prompt(text)
            encoded = self.tokenizer(prompt, return_tensors="pt")
            if torch is not None:
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

            outputs = self.model.generate(
                **encoded,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            input_len = encoded["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            decoded = _normalize_latex_markup(decoded)
            return self._normalize(decoded)
        except Exception as exc:
            print(f"[LLMSummaryModel] generation failed: {exc}")
            return ""
