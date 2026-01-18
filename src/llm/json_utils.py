from __future__ import annotations

"""Utilities for parsing imperfect LLM JSON outputs."""

import ast
import json
import re
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


_EXPRESSION_RE = re.compile(r"(:\s*)([-+0-9. ]+[+-][0-9. ]+)(?=\s*[},])")


def _safe_eval_expression(expr: str) -> float | None:
    # Safely evaluate simple +/- arithmetic without using eval().
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float | None:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            if operand is None:
                return None
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                return None
            return left + right if isinstance(node.op, ast.Add) else left - right
        return None

    return _eval(tree)


def _normalize_numeric_expressions(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        expr = match.group(2).strip()
        value = _safe_eval_expression(expr)
        if value is None:
            return match.group(0)
        if abs(value - round(value)) < 1e-6:
            value_str = str(int(round(value)))
        else:
            value_str = str(value)
        return f"{match.group(1)}{value_str}"

    return _EXPRESSION_RE.sub(_replace, text)


def safe_json_loads(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("safe_json_loads expects a non-empty string")
    cleaned = _strip_code_fence(text)
    cleaned = _find_json_substring(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        normalized = _normalize_numeric_expressions(cleaned)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            # Last resort: collapse newlines and retry (plus expression fix).
            compact = cleaned.replace("\n", " ").strip()
            compact_normalized = _normalize_numeric_expressions(compact)
            for candidate in (compact, compact_normalized):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            try:
                import json5
            except ModuleNotFoundError as exc:
                raise exc
            for candidate in (normalized, compact_normalized, compact):
                try:
                    return json5.loads(candidate)
                except Exception:
                    continue
            raise


def coerce_float(value: Any) -> float | None:
    # Normalize strings like "12.5%" or "1,234.5" into floats.
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"na", "n/a"}:
            return None
        if cleaned in {"-", "\u2014", "\u2013"}:
            return None
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        if neg:
            cleaned = cleaned[1:-1].strip()
        cleaned = cleaned.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
            try:
                val = float(cleaned) / 100.0
                return -val if neg else val
            except ValueError:
                return None
        try:
            val = float(cleaned)
            return -val if neg else val
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
