from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bonus.loan_pricing_pipeline import run_bonus_loan_pricing


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Bonus D: loan pricing pipeline.")
    ap.add_argument("--out-dir", default="results/bonus/loan_pricing", help="Output directory.")
    ap.add_argument("--nrows", type=int, default=5000, help="Rows to load from LendingClub.")
    args = ap.parse_args(cli_args)

    run_bonus_loan_pricing(Path(args.out_dir), nrows=args.nrows)


if __name__ == "__main__":
    main()
