from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bonus.credit_rating_pipeline import run_credit_rating_pipeline


def _parse_tickers(raw: Optional[str]) -> Optional[Sequence[str]]:
    if not raw:
        return None
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Bonus B: credit rating pipeline.")
    ap.add_argument("--proc-dir", default="data/processed/year", help="Processed states directory.")
    ap.add_argument("--raw-dir", default="data/raw", help="Raw statements directory.")
    ap.add_argument("--out-dir", default="results/bonus/credit_rating", help="Output directory.")
    ap.add_argument("--tickers", default=None, help="Comma-separated ticker list.")
    ap.add_argument("--max-tickers", type=int, default=40, help="Max tickers to include.")
    ap.add_argument("--evergrande-pdf", default="data/bonus_pdf/ar2022.pdf", help="Evergrande PDF path.")
    args = ap.parse_args(cli_args)

    run_credit_rating_pipeline(
        Path(args.proc_dir),
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        tickers=_parse_tickers(args.tickers),
        max_tickers=args.max_tickers,
        evergrande_pdf=Path(args.evergrande_pdf),
    )


if __name__ == "__main__":
    main()
