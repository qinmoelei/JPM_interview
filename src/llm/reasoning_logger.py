from __future__ import annotations

"""Append short LLM reasoning snippets to a plain-text log."""

from datetime import datetime
from pathlib import Path


def append_reasoning_log(path: Path, *, title: str, reasoning: str | None) -> None:
    # Keep a lightweight text log so each LLM call has a traceable rationale.
    if reasoning is None or not str(reasoning).strip():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    header = f"{title} | {timestamp}"
    lines = [
        header,
        "-" * len(header),
        str(reasoning).strip(),
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
