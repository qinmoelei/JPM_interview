from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A2D_DEFAULT_DIRS = [
    "results/part2_llm_run_a2d_year_top",
    "results/part2_llm_run_a2d_quarter_top",
]
E2I_DEFAULT_DIR = "results/part2_llm_run_e2i"

RATIO_KEYS = [
    "net_income",
    "cost_to_income",
    "quick_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "debt_to_capital",
    "debt_to_ebitda",
    "interest_coverage",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def _safe_stats(values: List[Optional[float]]) -> dict:
    cleaned = [v for v in values if v is not None and not np.isnan(v)]
    if not cleaned:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(cleaned, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def _summarize_a2d(run_dir: Path, out_dir: Path) -> dict:
    run_cfg = _load_json(run_dir / "run_config.json")
    variant = run_cfg.get("variant", run_dir.name)
    baseline = _load_json(run_dir / "baseline_metrics.json")
    llm = _load_json(run_dir / "llm_metrics.json")
    ensemble = _load_json(run_dir / "ensemble_metrics.json")

    rows = []

    for name, metrics in baseline.items():
        row = {
            "variant": variant,
            "model_type": "baseline",
            "model": name,
        }
        for metric_name in ("driver_test", "state_test"):
            for key, val in metrics[metric_name].items():
                row[f"{metric_name}_{key}"] = val
        rows.append(row)

    row = {
        "variant": variant,
        "model_type": "llm",
        "model": "llm",
    }
    for metric_name in ("driver_test", "state_test"):
        for key, val in llm[metric_name].items():
            row[f"{metric_name}_{key}"] = val
    rows.append(row)

    for name, metrics in ensemble.items():
        row = {
            "variant": variant,
            "model_type": "ensemble",
            "model": name,
        }
        for metric_name in ("driver_test", "state_test"):
            for key, val in metrics[metric_name].items():
                row[f"{metric_name}_{key}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["model_type", "model"], kind="mergesort")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"a2d_metrics_{variant}.csv"
    md_path = out_dir / f"a2d_metrics_{variant}.md"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False, floatfmt=".6g")

    state_cols = [
        "variant",
        "model_type",
        "model",
        "state_test_mse",
        "state_test_mae",
        "state_test_rel_l1",
        "state_test_rel_l2",
    ]
    state_df = df[state_cols].copy()
    state_csv = out_dir / f"a2d_state_{variant}.csv"
    state_md = out_dir / f"a2d_state_{variant}.md"
    state_df.to_csv(state_csv, index=False)
    state_df.to_markdown(state_md, index=False, floatfmt=".6g")
    state_tex = out_dir / f"a2d_state_{variant}.tex"
    state_df.to_latex(state_tex, index=False, float_format="%.6g")

    # ranking by state_test_mse (exclude perfect oracle)
    def _rank(df_in: pd.DataFrame, metric: str) -> dict:
        filtered = df_in.copy()
        filtered = filtered[filtered["model"] != "perfect"]
        filtered = filtered[filtered[metric].notna()]
        if filtered.empty:
            return {"metric": metric, "best": None}
        best_idx = filtered[metric].idxmin()
        best_row = filtered.loc[best_idx]
        return {
            "metric": metric,
            "best": {
                "model_type": best_row["model_type"],
                "model": best_row["model"],
                "value": float(best_row[metric]),
            },
        }

    ranking = {
        "variant": variant,
        "rankings": [
            _rank(df, "state_test_mse"),
            _rank(df, "state_test_mae"),
            _rank(df, "state_test_rel_l1"),
            _rank(df, "state_test_rel_l2"),
        ],
    }

    (out_dir / f"a2d_ranking_{variant}.json").write_text(json.dumps(ranking, indent=2))

    # plot state_test_mse
    plot_df = df.copy()
    plot_df = plot_df.sort_values("state_test_mse")
    labels = plot_df["model_type"] + ":" + plot_df["model"]
    values = plot_df["state_test_mse"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = {"baseline": "#4C78A8", "llm": "#F58518", "ensemble": "#54A24B"}
    colors = [color_map.get(t, "#888888") for t in plot_df["model_type"]]
    ax.barh(labels, values, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("state_test MSE (log scale)")
    ax.set_title(f"A2D State Forecast Error ({variant})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig_path = out_dir / f"a2d_state_mse_{variant}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {
        "variant": variant,
        "csv": str(csv_path),
        "md": str(md_path),
        "state_csv": str(state_csv),
        "state_md": str(state_md),
        "state_tex": str(state_tex),
        "plot": str(fig_path),
    }


def _summarize_e2i(e2i_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"companies": {}}

    for company_dir in sorted(p for p in e2i_dir.iterdir() if p.is_dir()):
        if not (company_dir / "metadata.json").exists():
            continue
        metadata = _load_json(company_dir / "metadata.json")
        company_name = metadata.get("company", company_dir.name)
        models_dir = company_dir / "models"
        rows = []

        for model_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
            runs = sorted(model_dir.glob("run_*.json"))
            if not runs:
                continue
            ratio_values: Dict[str, List[Optional[float]]] = {key: [] for key in RATIO_KEYS}
            units = []
            currency = []
            for run_path in runs:
                payload = _load_json(run_path)
                ratios = payload.get("ratios", {})
                meta = payload.get("meta", {})
                units.append(meta.get("units"))
                currency.append(meta.get("currency"))
                for key in RATIO_KEYS:
                    ratio_values[key].append(ratios.get(key))

            row = {
                "company": company_name,
                "model": model_dir.name,
                "n_runs": len(runs),
                "units": next((u for u in units if u), None),
                "currency": next((c for c in currency if c), None),
            }
            for key in RATIO_KEYS:
                stats = _safe_stats(ratio_values[key])
                row[f"{key}_mean"] = stats["mean"]
                row[f"{key}_std"] = stats["std"]
                row[f"{key}_n"] = stats["n"]
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("model", kind="mergesort")

        csv_path = out_dir / f"e2i_{company_dir.name}_ratios.csv"
        md_path = out_dir / f"e2i_{company_dir.name}_ratios.md"
        df.to_csv(csv_path, index=False)
        df.to_markdown(md_path, index=False, floatfmt=".6g")

        compact = pd.DataFrame({
            "company": df["company"],
            "model": df["model"],
            "n_runs": df["n_runs"],
            "net_income": df.apply(lambda r: _format_metric(r["net_income_mean"]) + " +/- " + _format_metric(r["net_income_std"]), axis=1),
            "cost_to_income": df.apply(lambda r: _format_metric(r["cost_to_income_mean"]) + " +/- " + _format_metric(r["cost_to_income_std"]), axis=1),
            "quick_ratio": df.apply(lambda r: _format_metric(r["quick_ratio_mean"]) + " +/- " + _format_metric(r["quick_ratio_std"]), axis=1),
            "debt_to_equity": df.apply(lambda r: _format_metric(r["debt_to_equity_mean"]) + " +/- " + _format_metric(r["debt_to_equity_std"]), axis=1),
            "debt_to_assets": df.apply(lambda r: _format_metric(r["debt_to_assets_mean"]) + " +/- " + _format_metric(r["debt_to_assets_std"]), axis=1),
            "debt_to_capital": df.apply(lambda r: _format_metric(r["debt_to_capital_mean"]) + " +/- " + _format_metric(r["debt_to_capital_std"]), axis=1),
            "debt_to_ebitda": df.apply(lambda r: _format_metric(r["debt_to_ebitda_mean"]) + " +/- " + _format_metric(r["debt_to_ebitda_std"]), axis=1),
            "interest_coverage": df.apply(lambda r: _format_metric(r["interest_coverage_mean"]) + " +/- " + _format_metric(r["interest_coverage_std"]), axis=1),
        })
        compact_md = out_dir / f"e2i_{company_dir.name}_ratios_compact.md"
        compact.to_markdown(compact_md, index=False)
        compact_tex = out_dir / f"e2i_{company_dir.name}_ratios_compact.tex"
        compact_tex_df = compact.copy()
        for col in compact_tex_df.columns:
            if col in ("company", "model", "n_runs"):
                continue
            compact_tex_df[col] = compact_tex_df[col].str.replace("+/-", "\\\\pm", regex=False)
        compact_tex_df.to_latex(compact_tex, index=False, escape=False)

        summary["companies"][company_dir.name] = {
            "company": company_name,
            "units": metadata.get("unit_hint"),
            "currency": None,
            "csv": str(csv_path),
            "md": str(md_path),
            "compact_md": str(compact_md),
            "compact_tex": str(compact_tex),
            "n_models": int(df.shape[0]),
        }

    summary_path = out_dir / "e2i_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Part 2 A2D/E2I outputs.")
    ap.add_argument("--a2d-dirs", nargs="*", default=A2D_DEFAULT_DIRS, help="A2D result directories.")
    ap.add_argument("--e2i-dir", default=E2I_DEFAULT_DIR, help="E2I result directory.")
    ap.add_argument("--out-dir", default="results/part2_summary", help="Output summary directory.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a2d_outputs = []
    for path in args.a2d_dirs:
        run_dir = Path(path)
        if not run_dir.exists():
            continue
        a2d_outputs.append(_summarize_a2d(run_dir, out_dir))

    e2i_summary = None
    e2i_dir = Path(args.e2i_dir)
    if e2i_dir.exists():
        e2i_summary = _summarize_e2i(e2i_dir, out_dir)

    combined = {
        "a2d": a2d_outputs,
        "e2i": e2i_summary,
    }
    (out_dir / "part2_summary.json").write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
