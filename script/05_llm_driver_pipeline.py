from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.driver_pipeline import run_llm_driver_pipeline

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config_stable_top.yaml"


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="LLM driver forecast + ensemble + robustness pipeline.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config path.")
    ap.add_argument("--variant", default=None, help="Processed subdir (year/quarter).")
    ap.add_argument("--frequency", choices=["annual", "quarterly"], default=None, help="Override frequency.")
    ap.add_argument("--tickers", default=None, help="Comma-separated tickers (override config).")
    ap.add_argument("--max-tickers", type=int, default=6, help="Max tickers to use.")
    ap.add_argument("--min-rows", type=int, default=4, help="Min rows for driver history.")
    ap.add_argument("--filter-max-abs", type=float, default=1000.0, help="Skip tickers with extreme drivers.")
    ap.add_argument("--window", type=int, default=3, help="Recent-history window for the LLM prompt.")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    ap.add_argument("--model", default=None, help="Override LLM model.")
    ap.add_argument("--max-calls", type=int, default=20, help="Limit total LLM calls.")
    ap.add_argument("--robust-runs", type=int, default=0, help="Run robustness repeats.")
    ap.add_argument("--robust-temperature", type=float, default=0.1, help="Temperature for robustness runs.")
    ap.add_argument("--cfo-ticker", default=None, help="Ticker for CFO recommendation.")
    ap.add_argument("--baseline-results", default=None, help="Path to Part1 results directory.")
    ap.add_argument("--out-dir", default="results/part2_llm_run", help="Output directory.")
    args = ap.parse_args(cli_args)

    run_llm_driver_pipeline(
        config_path=args.config,
        variant=args.variant,
        frequency=args.frequency,
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        min_rows=args.min_rows,
        filter_max_abs=args.filter_max_abs,
        window=args.window,
        temperature=args.temperature,
        model=args.model,
        max_calls=args.max_calls,
        robust_runs=args.robust_runs,
        robust_temperature=args.robust_temperature,
        cfo_ticker=args.cfo_ticker,
        out_dir=args.out_dir,
        baseline_results_dir=args.baseline_results,
    )


if __name__ == "__main__":
    main()
