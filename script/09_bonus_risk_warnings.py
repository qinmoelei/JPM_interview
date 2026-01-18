from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bonus.risk_warning_pipeline import run_risk_warning_pipeline


def _parse_paths(raw: Optional[str]) -> Sequence[Path]:
    if not raw:
        return [Path("data/bonus_pdf/ar2022.pdf")]
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Bonus C: risk warning extraction.")
    ap.add_argument("--pdfs", default=None, help="Comma-separated PDF paths.")
    ap.add_argument("--out-dir", default="results/bonus/risk_warnings", help="Output directory.")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K paragraphs.")
    args = ap.parse_args(cli_args)

    run_risk_warning_pipeline(
        _parse_paths(args.pdfs),
        out_dir=Path(args.out_dir),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
