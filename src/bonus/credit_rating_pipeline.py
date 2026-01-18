from __future__ import annotations

"""Credit rating pipeline for bonus questions (model + Evergrande scoring)."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
from pypdf import PdfReader

from src.llm.credit_rating import build_rating_dataset, train_rating_models
from src.llm.risk_warnings import OPINION_PATTERNS, extract_audit_opinion, rank_risk_paragraphs
from src.llm.shenanigans import run_shenanigans_scan
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def collect_tickers(proc_dir: Path) -> List[str]:
    tickers = {path.name.split("_")[0] for path in proc_dir.glob("*_states.csv")}
    return sorted(tickers)


def summarize_rating_dataset(dataset: pd.DataFrame) -> Dict[str, object]:
    if dataset.empty:
        return {"rows": 0, "tickers": 0, "rating_dist": {}}
    rating_dist = dataset["rating_label"].value_counts().to_dict()
    return {
        "rows": int(dataset.shape[0]),
        "tickers": int(dataset["ticker"].nunique()),
        "rating_dist": {str(k): int(v) for k, v in rating_dist.items()},
    }


def rating_override(opinion: str, flags: Mapping[str, bool]) -> Optional[str]:
    if opinion in {"disclaimer", "adverse"}:
        return "D"
    if flags.get("going_concern"):
        return "CC"
    if opinion == "qualified":
        return "CCC"
    return None


def _read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "").replace("\x00", " ").strip() for page in reader.pages]


def _snippet(text: str, start: int, end: int, width: int = 140) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def find_opinion_evidence(
    pages: Sequence[str],
    patterns: Mapping[str, re.Pattern[str]] = OPINION_PATTERNS,
    *,
    max_hits: int = 8,
) -> List[Dict[str, object]]:
    evidence: List[Dict[str, object]] = []
    for page_idx, text in enumerate(pages, start=1):
        for tag, pattern in patterns.items():
            match = pattern.search(text)
            if not match:
                continue
            evidence.append(
                {
                    "pattern": tag,
                    "page": page_idx,
                    "snippet": _snippet(text, match.start(), match.end()),
                }
            )
            if len(evidence) >= max_hits:
                return evidence
    return evidence


def score_annual_report(pdf_path: Path, company: str) -> Dict[str, object]:
    pages = _read_pdf_pages(pdf_path)
    text = "\n".join(pages)
    opinion = extract_audit_opinion(text)
    override = rating_override(opinion["opinion"], opinion["flags"])
    rating = override or "CCC"
    evidence = find_opinion_evidence(pages)
    top_risks = rank_risk_paragraphs(text, top_k=12)
    return {
        "company": company,
        "pdf": str(pdf_path),
        "rating": rating,
        "audit_opinion": opinion,
        "override": override,
        "evidence": evidence,
        "top_risk_paragraphs": top_risks,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def run_credit_rating_pipeline(
    proc_dir: Path,
    *,
    raw_dir: Optional[Path] = None,
    out_dir: Path,
    tickers: Optional[Sequence[str]] = None,
    max_tickers: Optional[int] = 40,
    evergrande_pdf: Optional[Path] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_list = list(tickers) if tickers is not None else collect_tickers(proc_dir)
    if max_tickers is not None:
        ticker_list = ticker_list[:max_tickers]

    dataset = build_rating_dataset(proc_dir, ticker_list)
    dataset_path = out_dir / "rating_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    summary = summarize_rating_dataset(dataset)
    (out_dir / "rating_dataset_summary.json").write_text(json.dumps(summary, indent=2))

    metrics: Dict[str, object] = {}
    if not dataset.empty:
        try:
            metrics = train_rating_models(dataset)
        except ValueError as exc:
            LOGGER.warning("Rating model training skipped: %s", exc)
            metrics = {"error": str(exc)}
    metrics_out = {k: v for k, v in metrics.items() if k not in {"model", "ordinal_model"}}
    if metrics_out:
        (out_dir / "rating_model_metrics.json").write_text(json.dumps(metrics_out, indent=2))

    shenanigans = []
    if raw_dir is not None and raw_dir.exists():
        shenanigans = run_shenanigans_scan(ticker_list, raw_dir)
        (out_dir / "shenanigans_scan.json").write_text(json.dumps(shenanigans, indent=2))

    evergrande = None
    if evergrande_pdf is not None and evergrande_pdf.exists():
        evergrande = score_annual_report(evergrande_pdf, "China Evergrande Group")
        (out_dir / "evergrande_rating.json").write_text(json.dumps(evergrande, indent=2))

    output = {
        "dataset": str(dataset_path),
        "dataset_summary": summary,
        "tickers": ticker_list,
        "metrics_path": str(out_dir / "rating_model_metrics.json") if metrics_out else None,
        "shenanigans_path": str(out_dir / "shenanigans_scan.json") if shenanigans else None,
        "evergrande_path": str(out_dir / "evergrande_rating.json") if evergrande else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(output, indent=2))
    return output
