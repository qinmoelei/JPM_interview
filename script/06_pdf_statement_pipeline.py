from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.pdf_statement_pipeline import run_pdf_statement_extraction
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

GM_URL = "https://investor.gm.com/static-files/1fff6f59-551f-4fe0-bca9-74bfc9a56aeb"
LVMH_URL = "https://lvmh-com.cdn.prismic.io/lvmh-com/Z5kVBpbqstJ999KR_Financialdocuments-December31%2C2024.pdf"

DEFAULT_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        path.write_bytes(response.read())


def _parse_page_pair(raw: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if raw is None or raw.lower() == "auto":
        return None, None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Pages must be specified as 'income,balance' or 'auto'.")
    return int(parts[0]), int(parts[1])


def _parse_models(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _run_one(
    company: str,
    pdf_path: Path,
    out_dir: Path,
    pages: Tuple[Optional[int], Optional[int]],
    models: Sequence[str],
    runs: int,
    temperature: float,
    max_tokens: int,
) -> None:
    income_page, balance_page = pages
    LOGGER.info("Running %s extraction for %s", company, pdf_path)
    run_pdf_statement_extraction(
        pdf_path,
        company,
        out_dir=out_dir,
        income_page=income_page,
        balance_page=balance_page,
        models=models,
        runs_per_model=runs,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="PDF statement extraction + multi-LLM robustness.")
    ap.add_argument("--out-dir", default="results/part2_llm_run_e2i", help="Output directory.")
    ap.add_argument("--runs", type=int, default=5, help="Runs per model.")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    ap.add_argument("--max-tokens", type=int, default=1200, help="Max tokens per call.")
    ap.add_argument("--models", default=None, help="Comma-separated model list.")
    ap.add_argument("--gm-pdf", default="data/pdf/gm_2023_ar.pdf", help="GM PDF path.")
    ap.add_argument("--lvmh-pdf", default="data/pdf/lvmh_2024.pdf", help="LVMH PDF path.")
    ap.add_argument("--gm-pages", default="auto", help="GM income,balance page numbers or 'auto'.")
    ap.add_argument("--lvmh-pages", default="auto", help="LVMH income,balance page numbers or 'auto'.")
    args = ap.parse_args(cli_args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _parse_models(args.models)

    gm_pdf = Path(args.gm_pdf)
    lvmh_pdf = Path(args.lvmh_pdf)
    _download(GM_URL, gm_pdf)
    _download(LVMH_URL, lvmh_pdf)

    gm_pages = _parse_page_pair(args.gm_pages)
    lvmh_pages = _parse_page_pair(args.lvmh_pages)

    _run_one(
        "General Motors",
        gm_pdf,
        out_dir / "GM_2023",
        gm_pages,
        models,
        args.runs,
        args.temperature,
        args.max_tokens,
    )
    _run_one(
        "LVMH",
        lvmh_pdf,
        out_dir / "LVMH_2024",
        lvmh_pages,
        models,
        args.runs,
        args.temperature,
        args.max_tokens,
    )

    summary = {
        "models": models,
        "runs": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "outputs": {
            "GM_2023": str(out_dir / "GM_2023"),
            "LVMH_2024": str(out_dir / "LVMH_2024"),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
