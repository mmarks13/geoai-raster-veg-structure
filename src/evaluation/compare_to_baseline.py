#!/usr/bin/env python3
"""
Head-to-head comparison of model predictions vs. structural (veg_structure)
baseline at forest-plot locations.

Consumes the outputs of `compare_predictions_to_plots.py` for both the model run
and the baseline run, performs paired statistical tests on the shared plot set,
and writes tables, a markdown summary, and figures into
<output-dir>/ (intended to live at <COMPARISON_DIR>/vs_baseline/).

Usage:
    python src/evaluation/compare_to_baseline.py \\
        --model-csv      <COMPARISON_DIR>/comparison_results.csv \\
        --baseline-csv   data/processed/veg_structure_baseline/comparison/baseline_comparison_results.csv \\
        --model-stats    <COMPARISON_DIR>/comparison_stats.json \\
        --baseline-stats data/processed/veg_structure_baseline/comparison/baseline_comparison_stats.json \\
        --output-dir     <COMPARISON_DIR>/vs_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if len(x) < 2:
        return float("nan")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2 * cov / denom)


def _williams_t(r_mf: float, r_bf: float, r_mb: float, n: int) -> Tuple[float, float]:
    """
    Williams' t-test for the difference between two dependent Pearson correlations
    that share a common variable.

    r_mf : corr(model_pred, field)
    r_bf : corr(baseline_pred, field)
    r_mb : corr(model_pred, baseline_pred)
    """
    if n < 4 or any(np.isnan([r_mf, r_bf, r_mb])):
        return float("nan"), float("nan")
    # Determinant of the 3x3 correlation matrix
    det_R = (
        1.0
        - r_mf ** 2
        - r_bf ** 2
        - r_mb ** 2
        + 2.0 * r_mf * r_bf * r_mb
    )
    r_avg = (r_mf + r_bf) / 2.0
    numerator = (r_mf - r_bf) * np.sqrt((n - 1) * (1 + r_mb))
    denom_sq = (
        2.0 * ((n - 1) / (n - 3)) * det_R
        + r_avg ** 2 * (1 - r_mb) ** 3
    )
    if denom_sq <= 0:
        return float("nan"), float("nan")
    t = numerator / np.sqrt(denom_sq)
    # two-sided p-value with n-3 df
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 3))
    return float(t), float(p)


def _bootstrap_delta_corr(
    x_model: np.ndarray,
    x_baseline: np.ndarray,
    y_field: np.ndarray,
    corr_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    n_boot: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap on Δ = corr(model, field) - corr(baseline, field).

    Returns (delta_point, ci_low, ci_high, p_two_sided) using percentile CI and
    the empirical two-sided p-value (fraction of |draws| >= |point|, adjusted
    symmetrically around zero).
    """
    n = len(y_field)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    r_m, _ = corr_fn(x_model, y_field)
    r_b, _ = corr_fn(x_baseline, y_field)
    delta_point = r_m - r_b

    rng = np.random.default_rng(seed)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = idx_all[i]
        xm, xb, yf = x_model[idx], x_baseline[idx], y_field[idx]
        rm, _ = corr_fn(xm, yf)
        rb, _ = corr_fn(xb, yf)
        deltas[i] = rm - rb

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return float(delta_point), float("nan"), float("nan"), float("nan")

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # Two-sided empirical p: center distribution at zero under H0
    centered = deltas - np.mean(deltas)
    p = float(np.mean(np.abs(centered) >= abs(delta_point)))
    return float(delta_point), float(ci_low), float(ci_high), float(p)


def _descriptives(pred: np.ndarray, field: np.ndarray) -> Dict[str, float]:
    r, r_p = _safe_pearson(pred, field)
    rho, rho_p = _safe_spearman(pred, field)
    err = pred - field
    return {
        "pearson_r": r,
        "pearson_p": r_p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "bias": float(np.mean(err)) if len(err) else float("nan"),
        "ccc": _lin_ccc(pred, field),
    }


def _paired_tests(
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    field: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    n = len(field)
    out: Dict[str, float] = {"n": int(n)}

    # Correlations needed for Williams'
    r_mf, _ = _safe_pearson(model_pred, field)
    r_bf, _ = _safe_pearson(baseline_pred, field)
    r_mb, _ = _safe_pearson(model_pred, baseline_pred)
    t, p = _williams_t(r_mf, r_bf, r_mb, n)
    out["williams_t"] = t
    out["williams_p"] = p

    dr, dr_lo, dr_hi, dr_p = _bootstrap_delta_corr(
        model_pred, baseline_pred, field, _safe_pearson, n_boot, seed
    )
    out.update(
        {"delta_r": dr, "delta_r_ci_low": dr_lo, "delta_r_ci_high": dr_hi, "delta_r_boot_p": dr_p}
    )

    drho, drho_lo, drho_hi, drho_p = _bootstrap_delta_corr(
        model_pred, baseline_pred, field, _safe_spearman, n_boot, seed + 1
    )
    out.update(
        {
            "delta_rho": drho,
            "delta_rho_ci_low": drho_lo,
            "delta_rho_ci_high": drho_hi,
            "delta_rho_boot_p": drho_p,
        }
    )

    # Wilcoxon on |errors|
    abs_err_m = np.abs(model_pred - field)
    abs_err_b = np.abs(baseline_pred - field)
    diffs = abs_err_m - abs_err_b
    if n >= 3 and np.any(diffs != 0):
        w_stat, w_p = stats.wilcoxon(abs_err_m, abs_err_b, zero_method="wilcox")
        out["wilcoxon_stat"] = float(w_stat)
        out["wilcoxon_p"] = float(w_p)
    else:
        out["wilcoxon_stat"] = float("nan")
        out["wilcoxon_p"] = float("nan")

    rmse_m = float(np.sqrt(np.mean((model_pred - field) ** 2)))
    rmse_b = float(np.sqrt(np.mean((baseline_pred - field) ** 2)))
    out["skill_score"] = float("nan") if rmse_b == 0 else 1.0 - rmse_m / rmse_b
    return out


# ---------------------------------------------------------------------------
# Data loading / matching
# ---------------------------------------------------------------------------

def _intersect_metrics(model_stats_path: Path, baseline_stats_path: Path) -> List[str]:
    with open(model_stats_path) as f:
        model_stats = json.load(f)
    with open(baseline_stats_path) as f:
        baseline_stats = json.load(f)
    shared = sorted(set(model_stats.keys()) & set(baseline_stats.keys()))
    if not shared:
        raise ValueError(
            f"No shared metrics between {model_stats_path} and {baseline_stats_path}. "
            f"Model metrics: {sorted(model_stats)}, baseline metrics: {sorted(baseline_stats)}."
        )
    return shared


def _load_matched(
    model_csv: Path, baseline_csv: Path
) -> pd.DataFrame:
    model_df = pd.read_csv(model_csv)
    baseline_df = pd.read_csv(baseline_csv)
    merged = model_df.merge(
        baseline_df, on="plot_id", suffixes=("_model", "_baseline"), how="inner"
    )
    mismatches = merged["site_name_model"] != merged["site_name_baseline"]
    if mismatches.any():
        bad = merged.loc[mismatches, ["plot_id", "site_name_model", "site_name_baseline"]]
        raise ValueError(
            "site_name disagreement between model and baseline CSVs on shared plot_ids:\n"
            f"{bad.to_string(index=False)}"
        )
    merged["site_name"] = merged["site_name_model"]
    logger.info(
        f"Matched {len(merged)} plots on plot_id across "
        f"{merged['site_name'].nunique()} sites."
    )
    return merged


def _metric_frame(merged: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = {
        "plot_id": "plot_id",
        "site_name": "site_name",
        f"{metric}_pred_model": "model_pred",
        f"{metric}_pred_baseline": "baseline_pred",
    }
    # field value: prefer model side, fall back to baseline
    field_col_m = f"{metric}_field_model"
    field_col_b = f"{metric}_field_baseline"
    if field_col_m in merged.columns and field_col_b in merged.columns:
        field = merged[field_col_m].combine_first(merged[field_col_b])
    elif field_col_m in merged.columns:
        field = merged[field_col_m]
    elif field_col_b in merged.columns:
        field = merged[field_col_b]
    else:
        raise KeyError(f"{metric}_field column not present in merged dataframe")

    missing = [c for c in cols if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing columns for metric {metric}: {missing}")

    df = merged[list(cols.keys())].rename(columns=cols).copy()
    df["field"] = field
    df = df.dropna(subset=["model_pred", "baseline_pred", "field"]).reset_index(drop=True)
    df["metric"] = metric
    df["model_abs_error"] = (df["model_pred"] - df["field"]).abs()
    df["baseline_abs_error"] = (df["baseline_pred"] - df["field"]).abs()
    return df


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _row_for_group(df: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, float]:
    m = df["model_pred"].to_numpy()
    b = df["baseline_pred"].to_numpy()
    f = df["field"].to_numpy()
    model_desc = _descriptives(m, f)
    baseline_desc = _descriptives(b, f)
    row: Dict[str, float] = {"n": int(len(df))}
    for k, v in model_desc.items():
        row[f"model_{k}"] = v
    for k, v in baseline_desc.items():
        row[f"baseline_{k}"] = v
    row.update(_paired_tests(m, b, f, n_boot, seed))
    return row


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_f(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _overall_md_table(overall: pd.DataFrame) -> str:
    header = (
        "| Metric | n | Pearson r (B → M) | Spearman ρ (B → M) | RMSE (B → M) | CCC (B → M) | "
        "Williams t (p) | Δρ bootstrap [95% CI] (p) | Wilcoxon p | Skill Score |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for _, r in overall.iterrows():
        arrow_r = "↑" if (r["model_pearson_r"] > r["baseline_pearson_r"]) else "↓"
        arrow_rho = "↑" if (r["model_spearman_rho"] > r["baseline_spearman_rho"]) else "↓"
        arrow_rmse = "↓" if (r["model_rmse"] < r["baseline_rmse"]) else "↑"
        arrow_ccc = "↑" if (r["model_ccc"] > r["baseline_ccc"]) else "↓"
        rows.append(
            "| "
            + " | ".join(
                [
                    r["metric"],
                    f"{int(r['n'])}",
                    f"{_fmt_f(r['baseline_pearson_r'])} → {_fmt_f(r['model_pearson_r'])} {arrow_r}",
                    f"{_fmt_f(r['baseline_spearman_rho'])} → {_fmt_f(r['model_spearman_rho'])} {arrow_rho}",
                    f"{_fmt_f(r['baseline_rmse'])} → {_fmt_f(r['model_rmse'])} {arrow_rmse}",
                    f"{_fmt_f(r['baseline_ccc'])} → {_fmt_f(r['model_ccc'])} {arrow_ccc}",
                    f"{_fmt_f(r['williams_t'])} ({_fmt_p(r['williams_p'])})",
                    f"{_fmt_f(r['delta_rho'])} [{_fmt_f(r['delta_rho_ci_low'])}, {_fmt_f(r['delta_rho_ci_high'])}] ({_fmt_p(r['delta_rho_boot_p'])})",
                    _fmt_p(r["wilcoxon_p"]),
                    _fmt_f(r["skill_score"]),
                ]
            )
            + " |"
        )
    return header + "\n".join(rows) + "\n"


def _by_site_md_section(by_site: pd.DataFrame, metric: str) -> str:
    sub = by_site[by_site["metric"] == metric].copy()
    if sub.empty:
        return ""
    header = (
        f"### {metric}\n\n"
        "| Site | n | Pearson r (B → M) | Spearman ρ (B → M) | RMSE (B → M) | "
        "Williams p | Δρ p | Wilcoxon p | Skill Score |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for _, r in sub.iterrows():
        arrow_r = "↑" if (r["model_pearson_r"] > r["baseline_pearson_r"]) else "↓"
        arrow_rho = "↑" if (r["model_spearman_rho"] > r["baseline_spearman_rho"]) else "↓"
        arrow_rmse = "↓" if (r["model_rmse"] < r["baseline_rmse"]) else "↑"
        rows.append(
            "| "
            + " | ".join(
                [
                    r["site_name"],
                    f"{int(r['n'])}",
                    f"{_fmt_f(r['baseline_pearson_r'])} → {_fmt_f(r['model_pearson_r'])} {arrow_r}",
                    f"{_fmt_f(r['baseline_spearman_rho'])} → {_fmt_f(r['model_spearman_rho'])} {arrow_rho}",
                    f"{_fmt_f(r['baseline_rmse'])} → {_fmt_f(r['model_rmse'])} {arrow_rmse}",
                    _fmt_p(r["williams_p"]),
                    _fmt_p(r["delta_rho_boot_p"]),
                    _fmt_p(r["wilcoxon_p"]),
                    _fmt_f(r["skill_score"]),
                ]
            )
            + " |"
        )
    return header + "\n".join(rows) + "\n\n"


def _write_markdown(overall: pd.DataFrame, by_site: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Model vs. Structural Baseline — Forest Plot Comparison\n",
        "Paired comparison of model predictions vs. the veg-structure baseline on the same field plots.\n",
        "Notation: `B → M` = baseline value → model value. ↑/↓ indicates which value is preferable "
        "(↑ for correlations and CCC, ↓ for RMSE).\n",
        "Tests: **Williams' t-test** on dependent Pearson correlations; **Δρ bootstrap** "
        "(10,000 resamples, percentile 95% CI) for Spearman difference; "
        "**Wilcoxon signed-rank** on |errors|; **Skill Score** = 1 − RMSE_model / RMSE_baseline.\n",
        "## Overall\n",
        _overall_md_table(overall),
        "## Per-site\n",
    ]
    for metric in overall["metric"]:
        lines.append(_by_site_md_section(by_site, metric))
    path.write_text("\n".join(lines))
    logger.info(f"Wrote markdown summary: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_MODEL_COLOR = "#1f77b4"
_BASELINE_COLOR = "#d62728"


def _annotate_scatter(ax, pred, field, label: str):
    r, _ = _safe_pearson(pred, field)
    rho, _ = _safe_spearman(pred, field)
    ax.scatter(field, pred, alpha=0.7, s=30, edgecolor="k", linewidth=0.3)
    lo = float(min(np.min(field), np.min(pred)))
    hi = float(max(np.max(field), np.max(pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Field")
    ax.set_ylabel(f"{label} prediction")
    ax.set_title(label)
    return r, rho


def _plot_scatter_pair(df: pd.DataFrame, metric: str, out_path: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    r_b, rho_b = _annotate_scatter(axes[0], df["baseline_pred"].to_numpy(), df["field"].to_numpy(), "Baseline")
    r_m, rho_m = _annotate_scatter(axes[1], df["model_pred"].to_numpy(), df["field"].to_numpy(), "Model")

    def _annot(ax, r, rho, r_other, rho_other):
        r_mark = " ▲" if r >= r_other else ""
        rho_mark = " ▲" if rho >= rho_other else ""
        ax.text(
            0.04, 0.96,
            f"Pearson r = {r:.3f}{r_mark}\nSpearman ρ = {rho:.3f}{rho_mark}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
        )

    _annot(axes[0], r_b, rho_b, r_m, rho_m)
    _annot(axes[1], r_m, rho_m, r_b, rho_b)
    title = f"{metric}: pred vs field (n = {len(df)})"
    if title_suffix:
        title = f"{title_suffix} — {title}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["field"], df["baseline_pred"], color=_BASELINE_COLOR, alpha=0.6, s=30, label="Baseline")
    ax.scatter(df["field"], df["model_pred"], color=_MODEL_COLOR, alpha=0.6, s=30, label="Model")
    lo = float(min(df["field"].min(), df["baseline_pred"].min(), df["model_pred"].min()))
    hi = float(max(df["field"].max(), df["baseline_pred"].max(), df["model_pred"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")
    for col, color in [("baseline_pred", _BASELINE_COLOR), ("model_pred", _MODEL_COLOR)]:
        x, y = df["field"].to_numpy(), df[col].to_numpy()
        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, slope * xs + intercept, color=color, lw=1, alpha=0.8)
    ax.set_xlabel("Field")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{metric}: overlay (n = {len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_error(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    d = df.sort_values("baseline_abs_error", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.18 * len(d))))
    for i, row in d.iterrows():
        color = "#2ca02c" if row["model_abs_error"] < row["baseline_abs_error"] else "#d62728"
        ax.annotate(
            "",
            xy=(row["model_abs_error"], i),
            xytext=(row["baseline_abs_error"], i),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.85},
        )
    ax.scatter(d["baseline_abs_error"], y, color=_BASELINE_COLOR, s=18, label="Baseline", zorder=3)
    ax.scatter(d["model_abs_error"], y, color=_MODEL_COLOR, s=18, label="Model", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["plot_id"].astype(str), fontsize=6)
    ax.set_xlabel("|error|")
    ax.set_ylabel("plot_id (sorted by baseline error)")
    ax.set_title(f"{metric}: paired |error| (green = model better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_error_box(df: pd.DataFrame, metric: str, wilcoxon_p: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df["baseline_abs_error"].to_numpy(), df["model_abs_error"].to_numpy()]
    labels = ["Baseline", "Model"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [_BASELINE_COLOR, _MODEL_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for i, arr in enumerate(data):
        xs = np.full_like(arr, i + 1, dtype=float) + (np.random.default_rng(0).uniform(-0.08, 0.08, size=len(arr)))
        ax.scatter(xs, arr, alpha=0.6, s=14, color="k")
    ax.set_ylabel("|error|")
    ax.set_title(f"{metric}: |error| (Wilcoxon p = {_fmt_p(wilcoxon_p)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_taylor(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Simple Taylor-style polar diagram (angle = arccos(r), radius = σ_pred)."""
    ref_sd = float(np.std(df["field"], ddof=1))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, np.pi / 2)

    # Reference arc (σ = ref_sd)
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(theta, np.full_like(theta, ref_sd), "k--", lw=0.8, alpha=0.6, label="ref σ")

    for site in sorted(df["site_name"].unique()):
        sub = df[df["site_name"] == site]
        if len(sub) < 3:
            continue
        for col, color, marker, label_base in [
            ("baseline_pred", _BASELINE_COLOR, "o", "Baseline"),
            ("model_pred", _MODEL_COLOR, "s", "Model"),
        ]:
            r, _ = _safe_pearson(sub[col].to_numpy(), sub["field"].to_numpy())
            if np.isnan(r):
                continue
            sd = float(np.std(sub[col], ddof=1))
            ax.scatter(np.arccos(np.clip(r, -1, 1)), sd, color=color, marker=marker, s=50, edgecolor="k", linewidth=0.4)
            ax.annotate(site[:3], (np.arccos(np.clip(r, -1, 1)), sd), fontsize=7)

    ax.set_title(f"{metric}: Taylor diagram (reference σ = {ref_sd:.2f})", pad=20)
    # Custom legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BASELINE_COLOR, markersize=8, label="Baseline"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_MODEL_COLOR, markersize=8, label="Model"),
    ]
    ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_site_r_bars(
    by_site: pd.DataFrame, per_metric_frames: Dict[str, pd.DataFrame], out_path: Path,
    n_boot: int, seed: int
) -> None:
    metrics = sorted(by_site["metric"].unique())
    sites = sorted(by_site["site_name"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = by_site[by_site["metric"] == metric].set_index("site_name")
        x = np.arange(len(sites))
        width = 0.35
        model_r: List[float] = []
        base_r: List[float] = []
        model_ci: List[Tuple[float, float]] = []
        base_ci: List[Tuple[float, float]] = []
        mf = per_metric_frames[metric]
        for site in sites:
            if site in sub.index:
                model_r.append(sub.loc[site, "model_pearson_r"])
                base_r.append(sub.loc[site, "baseline_pearson_r"])
            else:
                model_r.append(np.nan)
                base_r.append(np.nan)
            site_frame = mf[mf["site_name"] == site]
            if len(site_frame) >= 4:
                m_r_boot = _bootstrap_single_corr(site_frame["model_pred"].to_numpy(), site_frame["field"].to_numpy(), n_boot, seed)
                b_r_boot = _bootstrap_single_corr(site_frame["baseline_pred"].to_numpy(), site_frame["field"].to_numpy(), n_boot, seed + 1)
                model_ci.append(m_r_boot)
                base_ci.append(b_r_boot)
            else:
                model_ci.append((np.nan, np.nan))
                base_ci.append((np.nan, np.nan))

        def _err(rs, cis):
            rs_a = np.array(rs, dtype=float)
            lo = np.array([c[0] for c in cis], dtype=float)
            hi = np.array([c[1] for c in cis], dtype=float)
            return np.vstack([np.clip(rs_a - lo, 0, None), np.clip(hi - rs_a, 0, None)])

        ax.bar(x - width / 2, base_r, width, yerr=_err(base_r, base_ci), color=_BASELINE_COLOR, alpha=0.8, label="Baseline", capsize=3)
        ax.bar(x + width / 2, model_r, width, yerr=_err(model_r, model_ci), color=_MODEL_COLOR, alpha=0.8, label="Model", capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=30, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("Pearson r")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_single_corr(pred: np.ndarray, field: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    n = len(pred)
    if n < 4:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rs = np.empty(n_boot)
    for i in range(n_boot):
        r, _ = _safe_pearson(pred[idx[i]], field[idx[i]])
        rs[i] = r
    rs = rs[~np.isnan(rs)]
    if len(rs) == 0:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(rs, [2.5, 97.5]))


def _plot_per_site_skill_score(by_site: pd.DataFrame, out_path: Path) -> None:
    metrics = sorted(by_site["metric"].unique())
    sites = sorted(by_site["site_name"].unique())
    x = np.arange(len(sites))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(1.2 * len(sites) + 3, 5))
    for i, metric in enumerate(metrics):
        sub = by_site[by_site["metric"] == metric].set_index("site_name")
        vals = [sub.loc[site, "skill_score"] if site in sub.index else np.nan for site in sites]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width, label=metric, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=30, ha="right")
    ax.set_ylabel("Skill Score (1 − RMSE_m / RMSE_b)")
    ax.set_title("Per-site Skill Score (positive ⇒ model beats baseline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--model-stats", type=Path, required=True)
    parser.add_argument("--baseline-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    metrics = _intersect_metrics(args.model_stats, args.baseline_stats)
    logger.info(f"Shared metrics with stats in both files: {metrics}")

    merged = _load_matched(args.model_csv, args.baseline_csv)

    per_metric_frames: Dict[str, pd.DataFrame] = {}
    matched_rows: List[pd.DataFrame] = []
    overall_rows: List[Dict[str, float]] = []
    by_site_rows: List[Dict[str, float]] = []

    for metric in metrics:
        frame = _metric_frame(merged, metric)
        per_metric_frames[metric] = frame
        matched_rows.append(
            frame[
                ["plot_id", "site_name", "metric", "field", "model_pred", "baseline_pred",
                 "model_abs_error", "baseline_abs_error"]
            ].rename(columns={"field": "field_value"})
        )

        overall = _row_for_group(frame, args.bootstrap_n, args.seed)
        overall["metric"] = metric
        overall_rows.append(overall)

        for site, sub in frame.groupby("site_name"):
            row = _row_for_group(sub, args.bootstrap_n, args.seed)
            row["metric"] = metric
            row["site_name"] = site
            by_site_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    # Put metric first
    overall_df = overall_df[["metric"] + [c for c in overall_df.columns if c != "metric"]]
    by_site_df = pd.DataFrame(by_site_rows)
    by_site_df = by_site_df[
        ["metric", "site_name"] + [c for c in by_site_df.columns if c not in ("metric", "site_name")]
    ].sort_values(["metric", "site_name"]).reset_index(drop=True)
    matched_df = pd.concat(matched_rows, ignore_index=True)

    overall_path = args.output_dir / "vs_baseline_overall.csv"
    by_site_path = args.output_dir / "vs_baseline_by_site.csv"
    matched_path = args.output_dir / "vs_baseline_matched_plots.csv"
    summary_path = args.output_dir / "vs_baseline_summary.md"

    overall_df.to_csv(overall_path, index=False)
    by_site_df.to_csv(by_site_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    logger.info(f"Wrote {overall_path}")
    logger.info(f"Wrote {by_site_path}")
    logger.info(f"Wrote {matched_path}")

    _write_markdown(overall_df, by_site_df, summary_path)

    # Figures
    per_site_scatter_dir = figures_dir / "scatter_by_site"
    per_site_scatter_dir.mkdir(exist_ok=True)
    for metric, frame in per_metric_frames.items():
        _plot_scatter_pair(frame, metric, figures_dir / f"scatter_{metric}.png")
        for site, sub in frame.groupby("site_name"):
            if len(sub) < 2:
                continue
            _plot_scatter_pair(
                sub,
                metric,
                per_site_scatter_dir / f"scatter_{metric}_{site}.png",
                title_suffix=site,
            )
        _plot_overlay(frame, metric, figures_dir / f"overlay_{metric}.png")
        _plot_paired_error(frame, metric, figures_dir / f"paired_error_{metric}.png")
        wpvals = overall_df.loc[overall_df["metric"] == metric, "wilcoxon_p"].iloc[0]
        _plot_error_box(frame, metric, wpvals, figures_dir / f"error_box_{metric}.png")
        _plot_taylor(frame, metric, figures_dir / f"taylor_{metric}.png")

    _plot_per_site_r_bars(by_site_df, per_metric_frames, figures_dir / "per_site_r_bars.png", args.bootstrap_n, args.seed)
    _plot_per_site_skill_score(by_site_df, figures_dir / "per_site_skill_score.png")
    logger.info(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
