from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config, get_default_config_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained checkpoints against the configured dataset.")
    ap.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the YAML config file. Defaults to %(default)s or $JPM_CONFIG_PATH if set.",
    )
    args = ap.parse_args(cli_args)
    cfg = load_config(args.config)
    logger.info("Evaluation script placeholder. After training, load checkpoints, run rolling-origin backtests and report identity violations, RMSE, MAPE.")
if __name__ == "__main__":
    main()
