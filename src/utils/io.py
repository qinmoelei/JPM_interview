from __future__ import annotations
import yaml, os
from dataclasses import dataclass

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
