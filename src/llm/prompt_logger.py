from __future__ import annotations

"""Append prompt/response pairs to a markdown log."""

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


def append_prompt_log(
    path: Path,
    *,
    title: str,
    messages: Sequence[Mapping[str, str]],
    response: str,
) -> None:
    # Write a markdown section per LLM call for traceability.
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    lines = [
        f"## {title}",
        f"- timestamp: {timestamp}",
        "",
        "### Prompt",
    ]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.extend(
            [
                f"#### {role}",
                "```text",
                content,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "### Response",
            "```text",
            response,
            "```",
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
