from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.datahandler.preprocess import run_preprocessing_pipeline
from src.utils.io import get_default_config_path, load_config


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean, impute, normalize and aggregate raw statements into training tensors."
    )
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the YAML config file. Defaults to %(default)s or $JPM_CONFIG_PATH if set.",
    )
    parser.add_argument("--variant", default=None, help="Explicit variant name; defaults to freq_<freq>_<norm>.")
    parser.add_argument("--norm-method", choices=["scale", "zscore"], default="scale", help="Normalization strategy.")
    parser.add_argument("--frequency", choices=["quarterly", "annual"], default=None, help="Override frequency for preprocessing.")
    args = parser.parse_args(cli_args)
    cfg = load_config(args.config)
    frequency = args.frequency or cfg.frequency
    variant = args.variant or f"freq_{frequency}_{args.norm_method}"
    run_preprocessing_pipeline(
        args.config,
        variant=variant,
        norm_method=args.norm_method,
        override_frequency=args.frequency,
    )


if __name__ == "__main__":
    main()
