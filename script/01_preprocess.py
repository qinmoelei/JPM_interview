from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.datahandler.preprocess import run_preprocessing_pipeline
from src.utils.io import get_default_config_path


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build simulation-ready states/drivers from raw statements."
    )
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the YAML config file. Defaults to %(default)s or $JPM_CONFIG_PATH if set.",
    )
    parser.add_argument("--variant", default="simulation", help="Output subdirectory name under data/processed.")
    parser.add_argument(
        "--frequency",
        choices=["quarterly", "annual"],
        default=None,
        help="Override frequency for preprocessing (defaults to config frequency).",
    )
    args = parser.parse_args(cli_args)
    run_preprocessing_pipeline(
        args.config,
        variant=args.variant,
        override_frequency=args.frequency,
    )


if __name__ == "__main__":
    main()
