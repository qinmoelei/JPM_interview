from __future__ import annotations
import yaml, os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_ENV = "JPM_CONFIG_PATH"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "config.yaml"

@dataclass
class Config:
    """Strongly-typed view of configs/config.yaml."""
    tickers: list
    macro_tickers: list
    frequency: str
    start_year: int
    end_year: int
    paths: dict
    training: dict

def load_config(path: str) -> Config:
    """Load YAML config and return a dataclass so attributes are IDE-friendly.

    Example:
        >>> cfg = load_config(str(DEFAULT_CONFIG_RELATIVE))  # doctest: +SKIP
        >>> isinstance(cfg.tickers, list)  # doctest: +SKIP
        True
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def ensure_dir(path: str) -> None:
    """Create a directory if needed (recursively)."""
    os.makedirs(path, exist_ok=True)

def get_default_config_path() -> str:
    """Return the config path, preferring `JPM_CONFIG_PATH` over the repo default.

    Example:
        >>> isinstance(get_default_config_path(), str)
        True
    """
    env_path = os.getenv(DEFAULT_CONFIG_ENV)
    if env_path:
        return env_path
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / DEFAULT_CONFIG_RELATIVE)
