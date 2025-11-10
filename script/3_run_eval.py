from __future__ import annotations
import argparse, numpy as np
from src.utils.io import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    logger.info("Evaluation script placeholder. After training, load checkpoints, run rolling-origin backtests and report identity violations, RMSE, MAPE.")
if __name__ == "__main__":
    main()
