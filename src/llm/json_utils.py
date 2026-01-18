from __future__ import annotations

"""Utilities for parsing imperfect LLM JSON outputs."""

import json
from typing import Any, Mapping, Sequence


def _strip_code_fence(text: str) -> str:
    # Accept markdown fenced code blocks to keep prompts flexible.
    lines = [line.strip() for line in text.strip().splitlines()]
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _find_json_substring(text: str) -> str:
    # Extract the widest plausible JSON span when extra text is present.
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


def safe_json_loads(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("safe_json_loads expects a non-empty string")
    cleaned = _strip_code_fence(text)
    cleaned = _find_json_substring(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: collapse newlines and retry.
        cleaned = cleaned.replace("\n", " ").strip()
        return json.loads(cleaned)


def coerce_float(value: Any) -> float | None:
    # Normalize strings like "12.5%" or "1,234.5" into floats.
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
            try:
                return float(cleaned) / 100.0
            except ValueError:
                return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def ensure_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping JSON object.")
    return value


def ensure_sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Expected a JSON array.")
    return value
