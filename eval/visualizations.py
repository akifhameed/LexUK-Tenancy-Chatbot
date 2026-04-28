"""
Matplotlib visualisations for the live evaluation dashboard.

Six chart functions, each takes an `EvalRunResult` and returns a
matplotlib `Figure` ready to plug into Gradio's `gr.Plot`. The summary
table is exposed as a pandas DataFrame for `gr.Dataframe`.

All charts:
    * Use a consistent colour palette (plain = blue, agent = red).
    * Render on a transparent background so they look fine on any theme.
    * Are safe to call even when only one mode was evaluated, or when
      data is sparse.

Public surface:

    summary_dataframe(result) -> pandas.DataFrame
    fig_metric_comparison(result) -> matplotlib.figure.Figure
    fig_judge_score_heatmap(result) -> Figure
    fig_category_breakdown(result) -> Figure
    fig_latency_histogram(result) -> Figure
    fig_citation_health(result) -> Figure
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib

# Use a non-interactive backend so we never try to open a window from a
# server context. Must be set before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from eval.live_eval import EvalRunResult


# ---------------------------------------------------------------------------
# Palette & shared styling
# ---------------------------------------------------------------------------

_MODE_COLOURS: dict[str, str] = {
    "plain": "#1f77b4",   # blue
    "agent": "#d62728",   # red
}
_MODE_LABELS: dict[str, str] = {
    "plain": "Plain RAG",
    "agent": "Agentic RAG",
}


def _new_figure(width: float = 9.0, height: float = 5.0) -> Figure:
    """Create a clean Figure with sane defaults."""
    fig, _ = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0)
    return fig


def _label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def _colour(mode: str) -> str:
    return _MODE_COLOURS.get(mode, "#888888")


# ---------------------------------------------------------------------------
# 1. Summary table - rendered as gr.Dataframe (not a chart)
# ---------------------------------------------------------------------------

_METRIC_DISPLAY: list[tuple[str, str, str]] = [
    # (key, display name, format spec)
    ("n_questions",         "N questions",          "{:.0f}"),
    ("mrr",                 "MRR",                  "{:.3f}"),
    ("ndcg",                "NDCG",                 "{:.3f}"),
    ("keyword_coverage",    "Keyword coverage",     "{:.3f}"),
    ("expected_acts_recall","Expected-acts recall", "{:.3f}"),
    ("citation_validity",   "Citation validity",    "{:.3f}"),
    ("refusal_accuracy",    "Refusal accuracy",     "{:.3f}"),
    ("judge_accuracy",      "Judge accuracy (1-5)", "{:.2f}"),
    ("judge_completeness",  "Judge completeness",   "{:.2f}"),
    ("judge_relevance",     "Judge relevance",      "{:.2f}"),
    ("latency_ms_avg",      "Latency p50 (ms)",     "{:.0f}"),
]


def summary_dataframe(result: EvalRunResult) -> pd.DataFrame:
    """Build the headline metrics table as a pandas DataFrame."""
    rows: list[dict[str, str]] = []
    for key, name, fmt in _METRIC_DISPLAY:
        row: dict[str, str] = {"Metric": name}
        for mode in result.modes:
            value = result.summary_by_mode.get(mode, {}).get(key)
            row[_label(mode)] = fmt.format(value) if value is not None else "-"
        # Delta column when both modes ran
        if "plain" in result.modes and "agent" in result.modes:
            p = result.summary_by_mode.get("plain", {}).get(key)
            a = result.summary_by_mode.get("agent", {}).get(key)
            if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                delta = a - p
                row["Δ (Agent - Plain)"] = f"{delta:+.3f}" if abs(delta) >= 0.01 else "≈ 0"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Bar chart - metric-by-metric comparison across modes
# ---------------------------------------------------------------------------

# Subset of metrics that share the [0, 1] scale or [1, 5] scale so they
# look good on the same axes. Latency and N-questions go on their own.
_BAR_METRICS_NORMALISED: list[tuple[str, str]] = [
    ("mrr",                 "MRR"),
    ("ndcg",                "NDCG"),
    ("keyword_coverage",    "Keyword cov."),
    ("expected_acts_recall","Acts recall"),
    ("citation_validity",   "Citation valid"),
    ("refusal_accuracy",    "Refusal acc."),
]
_BAR_METRICS_JUDGE: list[tuple[str, str]] = [
    ("judge_accuracy",      "Accuracy"),
    ("judge_completeness",  "Completeness"),
    ("judge_relevance",     "Relevance"),
]


def fig_metric_comparison(result: EvalRunResult) -> Figure:
    """
    Two side-by-side bar groups:
      (left)  six retrieval+generation metrics on the [0,1] scale
      (right) three LLM-as-judge metrics on the [1,5] scale
    Each metric shows one bar per mode evaluated.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_alpha(0)

    modes = result.modes
    n_modes = len(modes)

    # --- Left panel: [0, 1] metrics ---
    labels = [name for _, name in _BAR_METRICS_NORMALISED]
    x = np.arange(len(labels))
    bar_width = 0.8 / max(n_modes, 1)

    for i, mode in enumerate(modes):
        values = [
            result.summary_by_mode.get(mode, {}).get(key, 0.0)
            for key, _ in _BAR_METRICS_NORMALISED
        ]
        ax_left.bar(
            x + i * bar_width - (n_modes - 1) * bar_width / 2,
            values, bar_width, label=_label(mode), color=_colour(mode),
        )

    ax_left.set_ylim(0, 1.05)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=30, ha="right")
    ax_left.set_ylabel("Score (0-1)")
    ax_left.set_title("Retrieval + generation metrics")
    ax_left.legend()
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Right panel: [1, 5] judge metrics ---
    j_labels = [name for _, name in _BAR_METRICS_JUDGE]
    jx = np.arange(len(j_labels))

    for i, mode in enumerate(modes):
        values = [
            result.summary_by_mode.get(mode, {}).get(key, 0.0)
            for key, _ in _BAR_METRICS_JUDGE
        ]
        ax_right.bar(
            jx + i * bar_width - (n_modes - 1) * bar_width / 2,
            values, bar_width, label=_label(mode), color=_colour(mode),
        )

    ax_right.set_ylim(0, 5.2)
    ax_right.set_xticks(jx)
    ax_right.set_xticklabels(j_labels, rotation=0, ha="center")
    ax_right.set_ylabel("Score (1-5)")
    ax_right.set_title("LLM-as-judge")
    ax_right.legend()
    ax_right.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Judge-score heatmap (per-question, per-dimension)
# ---------------------------------------------------------------------------

def fig_judge_score_heatmap(result: EvalRunResult) -> Figure:
    """
    For each evaluated mode, render a heatmap with rows = questions and
    columns = (accuracy, completeness, relevance). NaN cells (refusal
    rows where the judge was skipped) shown in grey.
    """
    n_modes = len(result.modes)
    if n_modes == 0:
        return _new_figure()

    fig, axes = plt.subplots(
        1, n_modes,
        figsize=(6.5 * n_modes, max(4.0, 0.35 * len(result.questions) + 1.5)),
        squeeze=False,
    )
    fig.patch.set_alpha(0)

    cols = ["accuracy", "completeness", "relevance"]
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#dddddd")

    for ax_idx, mode in enumerate(result.modes):
        ax = axes[0][ax_idx]
        rows = result.rows_by_mode.get(mode, [])
        # Build a (n_questions, 3) matrix; NaN where judge skipped.
        matrix = np.full((len(rows), len(cols)), np.nan)
        ylabels: list[str] = []
        for i, row in enumerate(rows):
            ylabels.append(row.get("question_id", f"Q{i:03d}"))
            for j, col in enumerate(cols):
                v = row.get(col)
                if v is not None:
                    matrix[i, j] = float(v)

        im = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto", cmap=cmap, vmin=1, vmax=5,
        )
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([c.capitalize() for c in cols])
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(f"{_label(mode)} - per-question judge scores")

        # Cell annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center",
                            color="black", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Category breakdown - mean judge accuracy per category, per mode
# ---------------------------------------------------------------------------

def fig_category_breakdown(result: EvalRunResult) -> Figure:
    """Grouped bar chart: mean judge accuracy by question category, per mode."""
    fig = _new_figure(width=10, height=5)
    ax = fig.axes[0]

    # Collect per-category accuracy means
    means_by_mode: dict[str, dict[str, float]] = {}
    all_categories: set[str] = set()
    for mode in result.modes:
        cat_accs: dict[str, list[float]] = defaultdict(list)
        for row in result.rows_by_mode.get(mode, []):
            v = row.get("accuracy")
            if v is None:
                continue
            cat = row.get("category", "unknown")
            cat_accs[cat].append(float(v))
            all_categories.add(cat)
        means_by_mode[mode] = {
            cat: (sum(vals) / len(vals)) if vals else 0.0
            for cat, vals in cat_accs.items()
        }

    categories = sorted(all_categories)
    x = np.arange(len(categories))
    n_modes = max(len(result.modes), 1)
    bar_width = 0.8 / n_modes

    for i, mode in enumerate(result.modes):
        values = [means_by_mode[mode].get(cat, 0.0) for cat in categories]
        ax.bar(
            x + i * bar_width - (n_modes - 1) * bar_width / 2,
            values, bar_width, label=_label(mode), color=_colour(mode),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Mean judge accuracy (1-5)")
    ax.set_title("Judge accuracy by question category")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Latency histogram (overlay per mode)
# ---------------------------------------------------------------------------

def fig_latency_histogram(result: EvalRunResult) -> Figure:
    """Overlapped histograms of per-question latency, one bin set per mode."""
    fig = _new_figure(width=9, height=5)
    ax = fig.axes[0]

    all_latencies: list[float] = []
    for mode in result.modes:
        all_latencies.extend(
            row["latency_ms"]
            for row in result.rows_by_mode.get(mode, [])
            if "latency_ms" in row
        )

    if not all_latencies:
        ax.text(0.5, 0.5, "No latency data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    lo = min(all_latencies)
    hi = max(all_latencies)
    bins = np.linspace(lo, hi, 12)

    for mode in result.modes:
        latencies = [
            row["latency_ms"] for row in result.rows_by_mode.get(mode, [])
            if "latency_ms" in row
        ]
        if not latencies:
            continue
        ax.hist(
            latencies, bins=bins, alpha=0.6, label=_label(mode),
            color=_colour(mode), edgecolor="black", linewidth=0.5,
        )
        # P50 marker
        p50 = float(np.median(latencies))
        ax.axvline(p50, color=_colour(mode), linestyle="--", alpha=0.8,
                   label=f"{_label(mode)} p50: {p50:.0f} ms")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Number of questions")
    ax.set_title("Per-question latency distribution")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Citation health - stacked bars showing valid vs invalid per question
# ---------------------------------------------------------------------------

def fig_citation_health(result: EvalRunResult) -> Figure:
    """
    For each question (rows) and mode (group), show stacked bars:
    valid citations (green) vs invalid (red). If a question produced no
    citations, the bar is empty (zero height).
    """
    fig = _new_figure(width=11, height=max(4.0, 0.3 * len(result.questions) + 1.5))
    ax = fig.axes[0]

    questions = result.questions
    n_q = len(questions)
    if n_q == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Bars are grouped: per question, one slot per mode.
    n_modes = max(len(result.modes), 1)
    bar_width = 0.8 / n_modes
    y = np.arange(n_q)

    for i, mode in enumerate(result.modes):
        rows = result.rows_by_mode.get(mode, [])
        valid_counts = []
        invalid_counts = []
        for q_idx, _q in enumerate(questions):
            row = rows[q_idx] if q_idx < len(rows) else {}
            n_cit = int(row.get("n_citations", 0) or 0)
            validity = float(row.get("citation_validity", 1.0) or 1.0)
            v = round(n_cit * validity)
            iv = n_cit - v
            valid_counts.append(v)
            invalid_counts.append(iv)

        offset = i * bar_width - (n_modes - 1) * bar_width / 2
        ax.barh(
            y + offset, valid_counts, height=bar_width,
            label=f"{_label(mode)} valid", color="#2ca02c", alpha=0.85,
            edgecolor="black", linewidth=0.4,
        )
        ax.barh(
            y + offset, invalid_counts, height=bar_width,
            left=valid_counts, label=f"{_label(mode)} invalid",
            color="#d62728", alpha=0.85,
            edgecolor="black", linewidth=0.4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([q.question_id for q in questions], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Citation count (valid + invalid)")
    ax.set_title("Citation health per question")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig
