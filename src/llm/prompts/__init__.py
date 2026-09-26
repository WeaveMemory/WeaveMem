"""Prompt text loading.

Prompts are stored as .txt files in this directory. Placeholders use the {{KEY}} form
(to avoid clashing with the {} in JSON examples). render(name, KEY=value) reads the file
and does literal substitution.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def render(name: str, **variables: str) -> str:
    """Read {name}.txt and replace each {{KEY}} with its value."""
    text = (_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
