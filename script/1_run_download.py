from __future__ import annotations
import argparse, os
from src.utils.io import load_config, ensure_dir
from src.utils.logging import get_logger
from src.datahandler.data_download import download_universe

logger = get_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = cfg.paths["raw_dir"]
    ensure_dir(raw_dir)
    download_universe(cfg.tickers, out_dir=raw_dir, frequency=cfg.frequency)
    logger.info("Done.")
if __name__ == "__main__":
    main()
