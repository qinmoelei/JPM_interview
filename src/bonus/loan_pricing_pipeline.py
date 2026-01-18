from __future__ import annotations

"""Loan pricing pipeline wrapper for bonus questions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.llm.loan_pricing import run_loan_pricing_pipeline


def run_bonus_loan_pricing(out_dir: Path, *, nrows: int = 40000) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = run_loan_pricing_pipeline(out_dir, nrows=nrows)
    summary = {
        "out_dir": str(out_dir),
        "nrows": int(nrows),
        "output": output,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
