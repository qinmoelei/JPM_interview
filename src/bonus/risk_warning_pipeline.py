from __future__ import annotations

"""Risk warning extraction pipeline for bonus questions."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from pypdf import PdfReader

from src.llm.risk_warnings import OPINION_PATTERNS, RISK_KEYWORDS, analyze_pdf
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "").replace("\x00", " ").strip() for page in reader.pages]


def _snippet(text: str, start: int, end: int, width: int = 140) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def find_keyword_evidence(
    pages: Sequence[str],
    keywords: Sequence[str],
    *,
    max_hits: int = 12,
) -> List[Dict[str, object]]:
    evidence: List[Dict[str, object]] = []
    for page_idx, text in enumerate(pages, start=1):
        lowered = text.lower()
        for kw in keywords:
            if kw not in lowered:
                continue
            pos = lowered.find(kw)
            evidence.append(
                {
                    "keyword": kw,
                    "page": page_idx,
                    "snippet": _snippet(text, pos, pos + len(kw)),
                }
            )
            if len(evidence) >= max_hits:
                return evidence
    return evidence


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


def run_risk_warning_pipeline(
    pdf_paths: Sequence[Path],
    *,
    out_dir: Path,
    top_k: int = 20,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, object] = {"files": {}, "timestamp": datetime.utcnow().isoformat() + "Z"}

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            LOGGER.warning("Missing PDF: %s", pdf_path)
            continue
        result = analyze_pdf(pdf_path, top_k=top_k)
        pages = _read_pdf_pages(pdf_path)
        result["opinion_evidence"] = find_opinion_evidence(pages)
        result["keyword_evidence"] = find_keyword_evidence(pages, RISK_KEYWORDS)
        result["keywords"] = list(RISK_KEYWORDS)

        out_path = out_dir / f"{pdf_path.stem}_risk.json"
        out_path.write_text(json.dumps(result, indent=2))
        outputs["files"][pdf_path.stem] = str(out_path)

    (out_dir / "run_summary.json").write_text(json.dumps(outputs, indent=2))
    return outputs
