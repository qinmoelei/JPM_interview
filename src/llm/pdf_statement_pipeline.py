from __future__ import annotations

"""PDF statement extraction with pdfplumber + LLM + sanity checks."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pdfplumber

from src.llm.apiyi import create_async_client, extract_chat_content, load_apiyi_config, request_chat_completion
from src.llm.json_utils import coerce_float, safe_json_loads
from src.llm.prompt_config import get_prompt_section, render_prompt
from src.llm.prompt_logger import append_prompt_log
from src.llm.ratios import compute_ratios
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

INCOME_KEYS = [
    "revenue",
    "cogs",
    "sga",
    "operating_income",
    "net_income",
    "interest_expense",
    "tax_expense",
    "depreciation",
    "amortization",
]

BALANCE_KEYS = [
    "current_assets",
    "inventory",
    "current_liabilities",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "short_term_debt",
    "long_term_debt",
    "total_debt",
]

ALL_KEYS = INCOME_KEYS + BALANCE_KEYS

UNIT_HINTS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
}


@dataclass(frozen=True)
class StatementRunConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 1200


def _extract_text(page) -> str:
    text = page.extract_text() or ""
    return text.replace("\x00", " ").strip()


def _extract_tables(page) -> List[List[List[str]]]:
    tables = page.extract_tables() or []
    cleaned = []
    for table in tables:
        rows = []
        for row in table:
            if row is None:
                continue
            rows.append([(cell or "").strip() for cell in row])
        if rows:
            cleaned.append(rows)
    return cleaned


def _format_tables(tables: Sequence[Sequence[Sequence[str]]]) -> str:
    if not tables:
        return "(no tables extracted)"
    lines: List[str] = []
    for idx, table in enumerate(tables, start=1):
        lines.append(f"[table {idx}]")
        for row in table:
            lines.append(" | ".join(cell for cell in row))
        lines.append("")
    return "\n".join(lines).strip()


def _detect_units(text: str) -> Tuple[Optional[str], Optional[float]]:
    lowered = text.lower()
    for label, scale in UNIT_HINTS.items():
        if f"in {label}" in lowered or f"{label} of" in lowered:
            return label, scale
    return None, None


def _normalize_text(text: str) -> str:
    # Collapse whitespace so multi-word keywords match across line breaks.
    return " ".join(text.lower().split())


def _score_page(text: str, keywords: Sequence[str], line_items: Sequence[str] | None = None) -> int:
    normalized = _normalize_text(text)
    hit_count = sum(1 for key in keywords if key in normalized)
    if hit_count == 0:
        return 0
    line_hits = sum(1 for key in (line_items or []) if key in normalized)
    if line_items and line_hits == 0:
        return 0
    digit_count = sum(ch.isdigit() for ch in text)
    numeric_bonus = min(10, digit_count // 200)
    unit_bonus = 2 if "in millions" in normalized or "in billions" in normalized else 0
    return hit_count * 10 + line_hits * 5 + numeric_bonus + unit_bonus


def _find_best_page(
    pages: Sequence[str],
    keyword_sets: Sequence[Sequence[str]],
    line_items: Sequence[str] | None = None,
) -> Optional[int]:
    best_idx = None
    best_score = 0
    for idx, text in enumerate(pages):
        for keywords in keyword_sets:
            score = _score_page(text, keywords, line_items)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def _auto_detect_pages(pages: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    income_items = [
        "revenue",
        "cost of sales",
        "gross margin",
        "marketing and selling",
        "general and administrative",
        "earnings per share",
    ]
    balance_items = [
        "total assets",
        "total liabilities",
        "total equity",
        "current assets",
        "current liabilities",
        "cash and cash equivalents",
        "inventories",
    ]
    income_idx = _find_best_page(
        pages,
        [
            ["consolidated", "statements of income"],
            ["consolidated", "statement of income"],
            ["income statement"],
            ["statements of earnings"],
            ["statement of earnings"],
        ],
        income_items,
    )
    balance_idx = _find_best_page(
        pages,
        [
            ["consolidated", "balance sheets"],
            ["balance sheet"],
            ["statement of financial position"],
        ],
        balance_items,
    )
    return income_idx, balance_idx


def _build_prompt(
    *,
    company: str,
    page_notes: str,
    income_text: str,
    income_tables: str,
    balance_text: str,
    balance_tables: str,
) -> List[Dict[str, str]]:
    section = get_prompt_section("statement_extract")
    return render_prompt(
        section,
        company=company,
        page_notes=page_notes,
        income_text=income_text,
        income_tables=income_tables,
        balance_text=balance_text,
        balance_tables=balance_tables,
    )


def _parse_items(raw: Mapping[str, Any]) -> Tuple[Dict[str, float | None], Dict[str, Any]]:
    income = raw.get("income_statement")
    balance = raw.get("balance_sheet")
    if not isinstance(income, Mapping):
        income = {}
    if not isinstance(balance, Mapping):
        balance = {}
    items: Dict[str, float | None] = {}
    for key in INCOME_KEYS:
        items[key] = coerce_float(income.get(key))
    for key in BALANCE_KEYS:
        items[key] = coerce_float(balance.get(key))
    if items.get("total_debt") is None:
        short_debt = items.get("short_term_debt") or 0.0
        long_debt = items.get("long_term_debt") or 0.0
        if short_debt or long_debt:
            items["total_debt"] = short_debt + long_debt
    meta = {
        "units": raw.get("units"),
        "currency": raw.get("currency"),
        "reasoning": raw.get("reasoning"),
    }
    return items, meta


def _sanity_checks(items: Mapping[str, float | None]) -> Dict[str, Any]:
    flags: List[str] = []
    checks: Dict[str, Any] = {}
    assets = items.get("total_assets")
    liabilities = items.get("total_liabilities")
    equity = items.get("total_equity")
    if assets is not None and liabilities is not None and equity is not None:
        diff = float(assets - liabilities - equity)
        rel = abs(diff) / max(abs(assets), 1.0)
        checks["assets_vs_liabilities_equity"] = {"diff": diff, "rel": rel}
        if rel > 0.02:
            flags.append("assets_not_equal_liabilities_plus_equity")

    revenue = items.get("revenue")
    cogs = items.get("cogs")
    sga = items.get("sga")
    operating_income = items.get("operating_income")
    if None not in (revenue, cogs, sga, operating_income):
        implied = float(revenue - cogs - sga)
        diff = implied - float(operating_income)
        rel = abs(diff) / max(abs(revenue), 1.0)
        checks["operating_income_check"] = {"diff": diff, "rel": rel}
        if rel > 0.1:
            flags.append("operating_income_mismatch")

    total_debt = items.get("total_debt")
    short_debt = items.get("short_term_debt")
    long_debt = items.get("long_term_debt")
    if total_debt is not None and None not in (short_debt, long_debt):
        diff = float(total_debt - short_debt - long_debt)
        rel = abs(diff) / max(abs(total_debt), 1.0)
        checks["debt_rollup_check"] = {"diff": diff, "rel": rel}
        if rel > 0.02:
            flags.append("total_debt_mismatch")

    if revenue is not None and revenue < 0:
        flags.append("negative_revenue")
    if assets is not None and assets < 0:
        flags.append("negative_assets")
    return {"checks": checks, "flags": flags}


def _aggregate_stats(values: Sequence[float | None]) -> Dict[str, Any]:
    present = [v for v in values if v is not None]
    missing = len(values) - len(present)
    if not present:
        return {"mean": None, "std": None, "cv": None, "missing": missing}
    arr = np.array(present, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    denom = max(abs(mean), 1e-6)
    return {"mean": mean, "std": std, "cv": float(std / denom), "missing": missing}


def _summarize_stability(
    runs: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    cv_threshold: float = 0.1,
    missing_threshold: float = 0.2,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"thresholds": {"cv": cv_threshold, "missing_rate": missing_threshold}, "fields": {}}
    for key in keys:
        values = [run.get("items", {}).get(key) for run in runs]
        stats = _aggregate_stats(values)
        missing_rate = stats["missing"] / max(len(values), 1)
        unstable = False
        if stats["cv"] is not None and stats["cv"] > cv_threshold:
            unstable = True
        if missing_rate > missing_threshold:
            unstable = True
        summary["fields"][key] = {
            "stats": stats,
            "missing_rate": missing_rate,
            "unstable": unstable,
        }
    return summary


def _aggregate_items(runs: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float | None]:
    aggregate: Dict[str, float | None] = {}
    for key in keys:
        values = [run.get("items", {}).get(key) for run in runs if run.get("items", {}).get(key) is not None]
        if not values:
            aggregate[key] = None
            continue
        arr = np.array(values, dtype=float)
        aggregate[key] = float(np.median(arr))
    return aggregate


def _sdk_version() -> Optional[str]:
    try:
        import openai
    except Exception:
        return None
    return getattr(openai, "__version__", None)


async def _run_model(
    messages: Sequence[Mapping[str, str]],
    cfg: StatementRunConfig,
    *,
    runs: int,
    prompt_log_path: Path,
    client,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for run_idx in range(1, runs + 1):
        response = await request_chat_completion(
            messages,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=client,
        )
        content = extract_chat_content(response)
        append_prompt_log(prompt_log_path, title=f"{cfg.model}:run_{run_idx}", messages=messages, response=content)
        raw = safe_json_loads(content)
        if not isinstance(raw, Mapping):
            raise ValueError("LLM response must be a JSON object.")
        items, meta = _parse_items(raw)
        ratios = compute_ratios(items)
        checks = _sanity_checks(items)
        outputs.append(
            {
                "run": run_idx,
                "items": items,
                "ratios": ratios,
                "meta": meta,
                "checks": checks,
                "raw": raw,
            }
        )
    return outputs


async def _run_all_models(
    messages: Sequence[Mapping[str, str]],
    models: Sequence[str],
    *,
    runs_per_model: int,
    temperature: float,
    max_tokens: int,
    out_dir: Path,
) -> Tuple[Dict[str, Any], Mapping[str, Any]]:
    config = load_apiyi_config()
    client = create_async_client(config, max_retries=0)
    model_outputs: Dict[str, Any] = {}
    try:
        for model in models:
            cfg = StatementRunConfig(model=model, temperature=temperature, max_tokens=max_tokens)
            model_dir = out_dir / "models" / model
            model_dir.mkdir(parents=True, exist_ok=True)
            prompt_log_path = model_dir / "llm_prompt_log.md"
            try:
                runs = await _run_model(messages, cfg, runs=runs_per_model, prompt_log_path=prompt_log_path, client=client)
                for run in runs:
                    run_path = model_dir / f"run_{run['run']:02d}.json"
                    run_path.write_text(json.dumps(run, indent=2))
                stability = _summarize_stability(runs, ALL_KEYS)
                aggregate_items = _aggregate_items(runs, ALL_KEYS)
                aggregate_ratios = compute_ratios(aggregate_items)
                answers = {
                    "net_income": aggregate_items.get("net_income"),
                    "cost_to_income": aggregate_ratios.get("cost_to_income"),
                    "quick_ratio": aggregate_ratios.get("quick_ratio"),
                    "debt_to_equity": aggregate_ratios.get("debt_to_equity"),
                    "debt_to_assets": aggregate_ratios.get("debt_to_assets"),
                    "debt_to_capital": aggregate_ratios.get("debt_to_capital"),
                    "debt_to_ebitda": aggregate_ratios.get("debt_to_ebitda"),
                    "interest_coverage": aggregate_ratios.get("interest_coverage"),
                }
                model_summary = {
                    "model": model,
                    "runs": runs_per_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "answers": answers,
                    "aggregate_items": aggregate_items,
                    "aggregate_ratios": aggregate_ratios,
                    "stability": stability,
                    "sdk_version": _sdk_version(),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                (model_dir / "summary.json").write_text(json.dumps(model_summary, indent=2))
                (model_dir / "stability.json").write_text(json.dumps(stability, indent=2))
                model_outputs[model] = model_summary
            except Exception as exc:
                error_summary = {
                    "model": model,
                    "error": str(exc),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                (model_dir / "error.json").write_text(json.dumps(error_summary, indent=2))
                model_outputs[model] = error_summary
    finally:
        await client.close()
    return model_outputs, {"api_base_url": config.base_url}


def run_pdf_statement_extraction(
    pdf_path: Path,
    company: str,
    *,
    out_dir: Path,
    income_page: Optional[int] = None,
    balance_page: Optional[int] = None,
    models: Sequence[str],
    runs_per_model: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_texts = [_extract_text(page) for page in pdf.pages]
    auto_income, auto_balance = _auto_detect_pages(page_texts)
    income_idx = (income_page - 1) if income_page else auto_income
    balance_idx = (balance_page - 1) if balance_page else auto_balance
    if income_idx is None or balance_idx is None:
        raise ValueError("Could not determine income or balance sheet pages.")

    with pdfplumber.open(str(pdf_path)) as pdf:
        income_page_obj = pdf.pages[income_idx]
        balance_page_obj = pdf.pages[balance_idx]
        income_text = _extract_text(income_page_obj)
        balance_text = _extract_text(balance_page_obj)
        income_tables = _format_tables(_extract_tables(income_page_obj))
        balance_tables = _format_tables(_extract_tables(balance_page_obj))

    (out_dir / f"page_{income_idx + 1}_text.txt").write_text(income_text)
    (out_dir / f"page_{income_idx + 1}_tables.txt").write_text(income_tables)
    (out_dir / f"page_{balance_idx + 1}_text.txt").write_text(balance_text)
    (out_dir / f"page_{balance_idx + 1}_tables.txt").write_text(balance_tables)

    unit_label, unit_scale = _detect_units(income_text + "\n" + balance_text)
    page_notes = f"income_page={income_idx + 1}, balance_page={balance_idx + 1}"
    messages = _build_prompt(
        company=company,
        page_notes=page_notes,
        income_text=income_text,
        income_tables=income_tables,
        balance_text=balance_text,
        balance_tables=balance_tables,
    )

    model_outputs, tool_meta = asyncio.run(
        _run_all_models(
            messages,
            models,
            runs_per_model=runs_per_model,
            temperature=temperature,
            max_tokens=max_tokens,
            out_dir=out_dir,
        )
    )

    tools = {
        "pdfplumber": getattr(pdfplumber, "__version__", "unknown"),
        "api_base_url": tool_meta.get("api_base_url"),
        "openai_sdk": _sdk_version(),
    }
    metadata = {
        "company": company,
        "pdf": str(pdf_path),
        "income_page": income_idx + 1,
        "balance_page": balance_idx + 1,
        "unit_hint": unit_label,
        "unit_scale": unit_scale,
        "tools": tools,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"metadata": metadata, "models": model_outputs}
