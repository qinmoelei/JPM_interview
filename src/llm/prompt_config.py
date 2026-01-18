from __future__ import annotations

"""Load and render LLM prompt templates from YAML config."""

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

ENV_PROMPT_CONFIG = "JPM_PROMPT_CONFIG"
DEFAULT_PROMPT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "prompt_config.yaml"

_PROMPT_CACHE: Optional[Mapping[str, Mapping[str, str]]] = None


def load_prompt_config(path: Optional[str] = None) -> Mapping[str, Mapping[str, str]]:
    # Cache the YAML config to avoid repeated disk I/O.
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None and path is None:
        return _PROMPT_CACHE
    cfg_path = Path(path) if path else Path(os.getenv(ENV_PROMPT_CONFIG, DEFAULT_PROMPT_CONFIG))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing prompt_config at {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError("prompt_config must be a mapping of prompt sections.")
    if path is None:
        _PROMPT_CACHE = data
    return data


def get_prompt_section(name: str, *, path: Optional[str] = None) -> Mapping[str, str]:
    # Retrieve a single prompt section by name.
    cfg = load_prompt_config(path)
    section = cfg.get(name)
    if not isinstance(section, Mapping):
        raise KeyError(f"Missing prompt section: {name}")
    return section


def render_prompt(section: Mapping[str, str], **kwargs: Any) -> list[dict[str, str]]:
    # Render system + user messages with provided placeholders.
    system = section.get("system", "")
    user_template = section.get("user_template", "")
    user = user_template.format(**kwargs)
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
