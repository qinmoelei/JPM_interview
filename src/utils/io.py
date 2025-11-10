from __future__ import annotations
import yaml, os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_ENV = "JPM_CONFIG_PATH"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "config.yaml"

@dataclass
class Config:
    tickers: list
    macro_tickers: list
    frequency: str
    start_year: int
    end_year: int
    paths: dict
    training: dict

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_default_config_path() -> str:
    """Return the config path, preferring an env override over the repo default."""
    env_path = os.getenv(DEFAULT_CONFIG_ENV)
    if env_path:
        return env_path
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / DEFAULT_CONFIG_RELATIVE)
