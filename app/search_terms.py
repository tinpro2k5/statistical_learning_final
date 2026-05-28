from __future__ import annotations

import re


GREEK_TO_NAME: dict[str, str] = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}
NAME_TO_GREEK: dict[str, str] = {name: symbol for symbol, name in GREEK_TO_NAME.items()}


def normalize_scientific_symbols(value: str) -> str:
    text = str(value or "")
    for symbol, name in GREEK_TO_NAME.items():
        text = text.replace(symbol, f" {name} ")
        text = text.replace(symbol.upper(), f" {name} ")
    return text


def acronym_key(query: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(query or "").strip())
    if 2 <= len(compact) <= 6 and compact.isupper():
        return compact
    return ""
