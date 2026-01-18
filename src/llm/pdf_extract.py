from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pypdf import PdfReader

from src.llm.apiyi import extract_chat_content, load_apiyi_config, request_chat_completion
from src.llm.json_utils import coerce_float, safe_json_loads
from src.llm.ratios import compute_ratios
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


STATEMENT_KEYS = [
    "revenue",
    "cogs",
    "sga",
    "operating_income",
    "net_income",
    "interest_expense",
    "tax_expense",
    "depreciation",
    "amortization",
    "current_assets",
    "inventory",
    "current_liabilities",
    "total_assets",
    "total_equity",
    "total_debt",
]


@dataclass(frozen=True)
class ExtractConfig:
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 900


def _read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.replace("\x00", " ").strip())
    return pages


def _score_page(text: str, keywords: Sequence[str]) -> int:
    t = text.lower()
    hit_count = sum(1 for k in keywords if k in t)
    if hit_count == 0:
        return 0
    digit_count = sum(ch.isdigit() for ch in text)
    numeric_bonus = min(10, digit_count // 200)
    unit_bonus = 2 if "in millions" in t else 0
    return hit_count * 10 + numeric_bonus + unit_bonus


def _find_best_page(pages: Sequence[str], keyword_sets: Sequence[Sequence[str]]) -> Optional[int]:
    best_idx = None
    best_score = 0
    for idx, text in enumerate(pages):
        for keywords in keyword_sets:
            score = _score_page(text, keywords)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def _collect_context(pages: Sequence[str], page_idx: int, window: int = 1) -> str:
    start = max(0, page_idx - window)
    end = min(len(pages), page_idx + window + 1)
    return "\n\n".join(pages[start:end])


def _build_extract_prompt(company: str, statement_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract numeric values from financial statements. "
                "Return ONLY valid JSON with numeric values (no commas). "
                "Use the most recent year column."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company}\n"
                "Extract these line items from the statement text. "
                "Use the most recent year column. If a value is missing, return null. "
                "Values should be numeric, in the units used in the report.\n"
                "Synonyms to map:\n"
                "- revenue: net revenue, total revenue, net sales\n"
                "- cogs: cost of sales, cost of revenue\n"
                "- sga: selling, general and administrative expenses\n"
                "- operating_income: operating profit, income from operations\n"
                "- net_income: profit attributable to owners, net profit\n"
                "- interest_expense: finance costs, interest paid\n"
                "- tax_expense: income tax expense, tax\n"
                "Required keys:\n"
                + ", ".join(STATEMENT_KEYS)
                + "\n\nStatement text:\n"
                + statement_text
            ),
        },
    ]


async def _extract_items(statement_text: str, company: str, cfg: ExtractConfig) -> Dict[str, float | None]:
    messages = _build_extract_prompt(company, statement_text)
    response = await request_chat_completion(
        messages,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    content = extract_chat_content(response)
    raw = safe_json_loads(content)
    if not isinstance(raw, Mapping):
        raise ValueError("LLM extraction must return a JSON object.")
    parsed: Dict[str, float | None] = {}
    for key in STATEMENT_KEYS:
        parsed[key] = coerce_float(raw.get(key))
    return parsed


def extract_from_pdf(
    pdf_path: Path,
    company: str,
    *,
    cfg: Optional[ExtractConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or ExtractConfig()
    pages = _read_pdf_pages(pdf_path)
    income_idx = _find_best_page(
        pages,
        [
            ["consolidated", "statements of income"],
            ["consolidated", "statement of income"],
            ["income statement"],
            ["statements of earnings"],
        ],
    )
    balance_idx = _find_best_page(
        pages,
        [
            ["consolidated", "balance sheets"],
            ["balance sheet"],
            ["statement of financial position"],
        ],
    )
    if income_idx is None or balance_idx is None:
        LOGGER.warning("Could not confidently locate statement pages in %s", pdf_path)
    texts = []
    if income_idx is not None:
        texts.append(_collect_context(pages, income_idx, window=1))
    if balance_idx is not None:
        texts.append(_collect_context(pages, balance_idx, window=1))
    statement_text = "\n\n".join(texts)
    items = asyncio_run_extract(statement_text, company, cfg)
    ratios = compute_ratios(items)
    meta = {
        "pdf": str(pdf_path),
        "income_page": income_idx + 1 if income_idx is not None else None,
        "balance_page": balance_idx + 1 if balance_idx is not None else None,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return {"items": items, "ratios": ratios, "meta": meta}


def asyncio_run_extract(statement_text: str, company: str, cfg: ExtractConfig) -> Dict[str, float | None]:
    import asyncio

    return asyncio.run(_extract_items(statement_text, company, cfg))


def write_report(out_dir: Path, payload: Mapping[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extracted_items.json").write_text(json.dumps(payload["items"], indent=2))
    (out_dir / "ratios.json").write_text(json.dumps(payload["ratios"], indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(payload["meta"], indent=2))
    report_lines = [
        "# PDF Extraction Report",
        "",
        "## Metadata",
        "```json",
        json.dumps(payload["meta"], indent=2),
        "```",
        "",
        "## Extracted Items",
        "```json",
        json.dumps(payload["items"], indent=2),
        "```",
        "",
        "## Ratios",
        "```json",
        json.dumps(payload["ratios"], indent=2),
        "```",
    ]
    (out_dir / "data_report.md").write_text("\n".join(report_lines))
