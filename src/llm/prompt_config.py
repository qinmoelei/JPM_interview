from __future__ import annotations

"""Load and render LLM prompt templates from YAML config."""

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

ENV_PROMPT_DIR = "JPM_PROMPT_DIR"
ENV_PROMPT_CONFIG = "JPM_PROMPT_CONFIG"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompt"

_PROMPT_CACHE: Optional[Mapping[str, Mapping[str, str]]] = None
_PROMPT_CACHE_PATH: Optional[Path] = None


def _resolve_prompt_path(path: Optional[str]) -> Path:
    # Prefer an explicit path, then a directory env var, then the legacy config env var.
    if path:
        return Path(path)
    env_dir = os.getenv(ENV_PROMPT_DIR)
    if env_dir:
        return Path(env_dir)
    env_file = os.getenv(ENV_PROMPT_CONFIG)
    if env_file:
        return Path(env_file)
    return DEFAULT_PROMPT_DIR


def _load_prompt_dir(path: Path) -> Mapping[str, Mapping[str, str]]:
    # Merge each YAML file in the prompt directory as a named prompt section.
    sections: dict[str, Mapping[str, str]] = {}
    for file_path in sorted(path.glob("*.yaml")):
        data = yaml.safe_load(file_path.read_text()) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"Prompt file {file_path} must be a mapping.")
        sections[file_path.stem] = data
    if not sections:
        raise FileNotFoundError(f"No prompt YAML files found in {path}")
    return sections


def load_prompt_config(path: Optional[str] = None) -> Mapping[str, Mapping[str, str]]:
    # Cache the prompt config to avoid repeated disk I/O.
    global _PROMPT_CACHE, _PROMPT_CACHE_PATH
    cfg_path = _resolve_prompt_path(path)
    if _PROMPT_CACHE is not None and _PROMPT_CACHE_PATH == cfg_path and path is None:
        return _PROMPT_CACHE
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing prompt config at {cfg_path}")
    if cfg_path.is_dir():
        data = _load_prompt_dir(cfg_path)
    else:
        data = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(data, Mapping):
            raise ValueError("prompt_config must be a mapping of prompt sections.")
    if path is None:
        _PROMPT_CACHE = data
        _PROMPT_CACHE_PATH = cfg_path
    return data


def get_prompt_section(name: str, *, path: Optional[str] = None) -> Mapping[str, str]:
    # Retrieve a single prompt section by name.
    cfg = load_prompt_config(path)
    if "system" in cfg and "user_template" in cfg and name not in cfg:
        return cfg  # Treat a single-file prompt as the requested section.
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
