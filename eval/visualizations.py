"""
Interactive Plotly visualisations for the live evaluation dashboard.

Five chart functions, each takes an `EvalRunResult` and returns a Plotly
`Figure` ready to plug into Gradio's `gr.Plot`. The summary table is
exposed as a pandas DataFrame for `gr.Dataframe`.

Plotly is used (instead of matplotlib PNGs) so the charts render as
proper interactive HTML on the page - axes, ticks, hover tooltips and
legends all stay legible and zoomable, with no rasterisation blur.

All charts:
    * Use a consistent colour palette (plain = blue, agent = red).
    * Render on a transparent background so they look fine on any theme.
    * Are safe to call even when only one mode was evaluated, or when
      data is sparse.

Public surface:

    summary_dataframe(result) -> pandas.DataFrame
    fig_metric_comparison(result) -> plotly.graph_objects.Figure
    fig_judge_score_heatmap(result) -> Figure
    fig_category_breakdown(result) -> Figure
    fig_latency_histogram(result) -> Figure
    fig_citation_health(result) -> Figure
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Clean white-canvas styling: solid white background, dark ink text,
# light grey gridlines. No transparency, no decorative tinting -
# we want maximum legibility on the Gradio page.
_INK = "#1F2937"
_GRID = "rgba(60,72,90,0.18)"
_AXIS = "rgba(60,72,90,0.45)"

_LAYOUT_BASE: dict = {
    "paper_bgcolor": "white",
    "plot_bgcolor":  "white",
    "font":          {"size": 15, "color": _INK, "family": "Inter, Calibri, Arial, sans-serif"},
    "margin":        {"l": 70, "r": 40, "t": 70, "b": 80},
    "legend":        {"orientation": "h", "y": -0.18, "x": 0,
                      "font": {"size": 13, "color": _INK}},
    "title":         {"font": {"size": 18, "color": _INK}},
}

_AXIS_DEFAULTS: dict = {
    "showgrid":   True,
    "gridcolor":  _GRID,
    "gridwidth":  1,
    "zeroline":   False,
    "showline":   True,
    "linecolor":  _AXIS,
    "linewidth":  1,
    "ticks":      "outside",
    "tickcolor":  _AXIS,
    "tickfont":   {"size": 13, "color": _INK},
    "title":      {"font": {"size": 14, "color": _INK}},
}


def _label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


def _colour(mode: str) -> str:
    return _MODE_COLOURS.get(mode, "#888888")


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font={"size": 14, "color": "#888"},
    )
    fig.update_layout(**_LAYOUT_BASE, height=300)
    return fig


# ---------------------------------------------------------------------------
# Score helpers - everything on a single 1-5 scale
# ---------------------------------------------------------------------------
#
# We expose four per-mode scores on the same 1-5 scale so they can be
# compared directly:
#
#   - Accuracy / Completeness / Relevance: the three LLM-as-Judge dimensions
#     (averaged across in-scope, judged questions only).
#   - Refusal handling: each question scored 5 if refused_correctly else 1,
#     averaged across ALL questions for that mode. This pulls refusal
#     behaviour into the same scale so a Combined headline is meaningful.
#
# Combined = unweighted mean of all four, computed per question (so a
# refusal correctly handled contributes 5, an over-refusal contributes 1)
# and then averaged across the mode. This removes the selection-bias
# loophole where over-refusing on an in-scope question silently drops it
# from the judge sample.

def _row_judge_score(row: dict) -> float | None:
    """Mean of the three judge dimensions, or None if the judge skipped."""
    parts = [row.get("accuracy"), row.get("completeness"), row.get("relevance")]
    parts = [float(v) for v in parts if v is not None]
    if not parts:
        return None
    return sum(parts) / len(parts)


def _mode_judge_mean(result: EvalRunResult, mode: str) -> float | None:
    """Mean LLM-as-Judge score across all judged questions for one mode."""
    rows = result.rows_by_mode.get(mode, [])
    scores = [s for s in (_row_judge_score(r) for r in rows) if s is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _row_refusal_score(row: dict) -> float:
    """Per-question refusal score on the 1-5 scale: 5 if correct, else 1."""
    return 5.0 if row.get("refused_correctly") else 1.0


def _mode_refusal_score(result: EvalRunResult, mode: str) -> float | None:
    """Mean refusal-handling score (1-5) across ALL questions for one mode."""
    rows = result.rows_by_mode.get(mode, [])
    if not rows:
        return None
    scores = [_row_refusal_score(r) for r in rows]
    return sum(scores) / len(scores)


def _row_combined_score(row: dict) -> float:
    """
    Per-question score on the 1-5 scale:
      - If the judge ran (in-scope, not refused), use the mean of the
        three judge dimensions.
      - Otherwise (refusal target or refused output), use the refusal
        score: 5 if refused_correctly else 1.
    """
    judge = _row_judge_score(row)
    return judge if judge is not None else _row_refusal_score(row)


def _mode_combined_score(result: EvalRunResult, mode: str) -> float | None:
    """Mean combined score across all questions for one mode."""
    rows = result.rows_by_mode.get(mode, [])
    if not rows:
        return None
    scores = [_row_combined_score(r) for r in rows]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 1. Summary table - rendered as gr.Dataframe (not a chart)
# ---------------------------------------------------------------------------

# Standard rows pulled straight from result.summary_by_mode.
# Order: retrieval -> generation -> judge dimensions -> performance.
_METRIC_DISPLAY: list[tuple[str, str, str]] = [
    # (key, display name, format spec)
    ("n_questions",         "N questions",                       "{:.0f}"),
    ("mrr",                 "MRR",                               "{:.3f}"),
    ("ndcg",                "NDCG",                              "{:.3f}"),
    ("keyword_coverage",    "Keyword coverage",                  "{:.3f}"),
    ("expected_acts_recall","Expected-acts recall",              "{:.3f}"),
    ("citation_validity",   "Citation validity",                 "{:.3f}"),
    ("refusal_accuracy",    "Refusal accuracy",                  "{:.3f}"),
    ("judge_accuracy",      "LLM as Judge — Accuracy (1-5)",     "{:.2f}"),
    ("judge_completeness",  "LLM as Judge — Completeness (1-5)", "{:.2f}"),
    ("judge_relevance",     "LLM as Judge — Relevance (1-5)",    "{:.2f}"),
    ("latency_ms_avg",      "Latency p50 (ms)",                  "{:.0f}"),
]


def summary_dataframe(result: EvalRunResult) -> pd.DataFrame:
    """
    Build the headline metrics table as a pandas DataFrame.

    The three LLM-as-Judge dimensions are shown as separate rows so the
    audience can see which dimension is driving the score; a final row
    gives the unweighted mean across all three for a single headline.
    """
    rows: list[dict[str, str]] = []
    for key, name, fmt in _METRIC_DISPLAY:
        row: dict[str, str] = {"Metric": name}
        for mode in result.modes:
            value = result.summary_by_mode.get(mode, {}).get(key)
            row[_label(mode)] = fmt.format(value) if value is not None else "-"
        if "plain" in result.modes and "agent" in result.modes:
            p = result.summary_by_mode.get("plain", {}).get(key)
            a = result.summary_by_mode.get("agent", {}).get(key)
            if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                delta = a - p
                row["Δ (Agent - Plain)"] = (
                    f"{delta:+.3f}" if abs(delta) >= 0.01 else "≈ 0"
                )
        rows.append(row)

    # Refusal handling row (5 = correct, 1 = wrong) on the same 1-5 scale.
    refusal_row: dict[str, str] = {"Metric": "Refusal handling (1-5)"}
    for mode in result.modes:
        v = _mode_refusal_score(result, mode)
        refusal_row[_label(mode)] = f"{v:.2f}" if v is not None else "-"
    if "plain" in result.modes and "agent" in result.modes:
        p = _mode_refusal_score(result, "plain")
        a = _mode_refusal_score(result, "agent")
        if p is not None and a is not None:
            delta = a - p
            refusal_row["Δ (Agent - Plain)"] = (
                f"{delta:+.2f}" if abs(delta) >= 0.01 else "≈ 0"
            )
    rows.append(refusal_row)

    # Combined headline: unweighted mean of all four 1-5 dimensions,
    # computed per question and averaged. Includes refusal correctness
    # so over-refusal can no longer hide behind a smaller judge sample.
    judge_row: dict[str, str] = {"Metric": "LLM as Judge — Combined (1-5)"}
    for mode in result.modes:
        v = _mode_combined_score(result, mode)
        judge_row[_label(mode)] = f"{v:.2f}" if v is not None else "-"
    if "plain" in result.modes and "agent" in result.modes:
        p = _mode_combined_score(result, "plain")
        a = _mode_combined_score(result, "agent")
        if p is not None and a is not None:
            delta = a - p
            judge_row["Δ (Agent - Plain)"] = (
                f"{delta:+.2f}" if abs(delta) >= 0.01 else "≈ 0"
            )
    rows.append(judge_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Bar chart - metric-by-metric comparison across modes
# ---------------------------------------------------------------------------

# Retrieval + generation metrics on the 0-1 scale.
_BAR_METRICS_NORMALISED: list[tuple[str, str]] = [
    ("mrr",                 "MRR"),
    ("ndcg",                "NDCG"),
    ("keyword_coverage",    "Keyword cov."),
    ("expected_acts_recall","Acts recall"),
    ("citation_validity",   "Citation valid"),
    ("refusal_accuracy",    "Refusal acc."),
]

# LLM-as-Judge dimensions on the 1-5 scale.
_BAR_METRICS_JUDGE: list[tuple[str, str]] = [
    ("judge_accuracy",      "Accuracy"),
    ("judge_completeness",  "Completeness"),
    ("judge_relevance",     "Relevance"),
]


def fig_metric_comparison(result: EvalRunResult) -> go.Figure:
    """
    Two side-by-side bar groups:
      (left)  six retrieval + generation metrics on the [0, 1] scale
      (right) the three LLM-as-Judge dimensions on the [1, 5] scale
    Each metric shows one bar per mode evaluated.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("Retrieval + generation metrics",
                        "Quality on the 1-5 scale (judge + refusal)"),
        horizontal_spacing=0.14,
    )

    # --- Left panel: 0-1 metrics ---
    labels = [name for _, name in _BAR_METRICS_NORMALISED]
    for mode in result.modes:
        values = [
            result.summary_by_mode.get(mode, {}).get(key, 0.0)
            for key, _ in _BAR_METRICS_NORMALISED
        ]
        fig.add_trace(
            go.Bar(
                x=labels, y=values,
                name=_label(mode), marker_color=_colour(mode),
                hovertemplate="%{x}: %{y:.3f}<extra>" + _label(mode) + "</extra>",
                legendgroup=mode,
            ),
            row=1, col=1,
        )

    # --- Right panel: three judge dimensions + refusal handling on 1-5 ---
    j_labels = [name for _, name in _BAR_METRICS_JUDGE] + ["Refusal"]
    for mode in result.modes:
        values = [
            result.summary_by_mode.get(mode, {}).get(key, 0.0)
            for key, _ in _BAR_METRICS_JUDGE
        ]
        # Append the refusal score (1-5) for the same mode.
        refusal_v = _mode_refusal_score(result, mode)
        values.append(refusal_v if refusal_v is not None else 0.0)
        fig.add_trace(
            go.Bar(
                x=j_labels, y=values,
                name=_label(mode), marker_color=_colour(mode),
                hovertemplate="%{x}: %{y:.2f}<extra>" + _label(mode) + "</extra>",
                legendgroup=mode, showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_yaxes(_AXIS_DEFAULTS, title_text="Score (0-1)",
                     range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(_AXIS_DEFAULTS, title_text="Score (1-5)",
                     range=[0, 5.2], row=1, col=2)
    fig.update_xaxes(_AXIS_DEFAULTS, tickangle=-25, row=1, col=1)
    fig.update_xaxes(_AXIS_DEFAULTS, tickangle=0, row=1, col=2)

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        bargap=0.25,
        height=600,
        title_text="Metric comparison across modes",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Judge-score heatmap (per-question, per-dimension)
# ---------------------------------------------------------------------------

def fig_judge_score_heatmap(result: EvalRunResult) -> go.Figure:
    """
    For each evaluated mode, render a heatmap with rows = questions and
    columns = (accuracy, completeness, relevance). Refusal rows (where
    the judge was skipped) appear as blank cells.

    Colour scale: red = 1 (worst) -> green = 5 (best). Hover any cell to
    see the question id, dimension and score.
    """
    n_modes = len(result.modes)
    if n_modes == 0:
        return _empty_fig("No data")

    cols = ["accuracy", "completeness", "relevance"]
    cols_display = [c.capitalize() for c in cols]

    fig = make_subplots(
        rows=1, cols=n_modes,
        subplot_titles=[f"{_label(m)}" for m in result.modes],
        horizontal_spacing=0.18,
    )

    # RdYlGn-style colour scale, 1 (red) to 5 (green).
    colorscale = [
        [0.00, "#d73027"],
        [0.25, "#fdae61"],
        [0.50, "#ffffbf"],
        [0.75, "#a6d96a"],
        [1.00, "#1a9850"],
    ]

    for ax_idx, mode in enumerate(result.modes, start=1):
        rows = result.rows_by_mode.get(mode, [])
        matrix = np.full((len(rows), len(cols)), np.nan)
        ylabels: list[str] = []
        for i, row in enumerate(rows):
            ylabels.append(row.get("question_id", f"Q{i:03d}"))
            for j, col in enumerate(cols):
                v = row.get(col)
                if v is not None:
                    matrix[i, j] = float(v)

        text = [
            [f"{int(v)}" if not np.isnan(v) else "" for v in row_vals]
            for row_vals in matrix
        ]

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=cols_display,
                y=ylabels,
                zmin=1, zmax=5,
                colorscale=colorscale,
                text=text,
                texttemplate="%{text}",
                hovertemplate=(
                    "<b>%{y}</b><br>%{x}: %{z:.0f} / 5"
                    "<extra>" + _label(mode) + "</extra>"
                ),
                colorbar={
                    "title": "Score",
                    "tickvals": [1, 2, 3, 4, 5],
                    "len": 0.85,
                    "x": (ax_idx / n_modes) - 0.02,
                } if ax_idx == n_modes else None,
                showscale=(ax_idx == n_modes),
            ),
            row=1, col=ax_idx,
        )

    # Height grows with question count so labels stay legible.
    height = max(520, 32 * max((len(result.rows_by_mode.get(m, []))
                                for m in result.modes), default=10) + 160)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=height,
        title_text="Per-question judge scores (1 = worst, 5 = best)",
    )
    fig.update_yaxes(_AXIS_DEFAULTS, autorange="reversed",
                     showgrid=False)
    fig.update_xaxes(_AXIS_DEFAULTS, showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# 4. Category breakdown - mean LLM-as-Judge score per category, per mode
# ---------------------------------------------------------------------------

def fig_category_breakdown(result: EvalRunResult) -> go.Figure:
    """Grouped bar chart: mean LLM-as-Judge score by category, per mode."""
    means_by_mode: dict[str, dict[str, float]] = {}
    all_categories: set[str] = set()
    for mode in result.modes:
        cat_scores: dict[str, list[float]] = defaultdict(list)
        for row in result.rows_by_mode.get(mode, []):
            v = _row_judge_score(row)
            if v is None:
                continue
            cat = row.get("category", "unknown")
            cat_scores[cat].append(v)
            all_categories.add(cat)
        means_by_mode[mode] = {
            cat: (sum(vals) / len(vals)) if vals else 0.0
            for cat, vals in cat_scores.items()
        }

    categories = sorted(all_categories)
    if not categories:
        return _empty_fig("No judged data")

    fig = go.Figure()
    for mode in result.modes:
        values = [means_by_mode[mode].get(cat, 0.0) for cat in categories]
        fig.add_trace(go.Bar(
            x=categories, y=values,
            name=_label(mode), marker_color=_colour(mode),
            hovertemplate="%{x}: %{y:.2f} / 5<extra>" + _label(mode) + "</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        bargap=0.25,
        height=580,
        title_text="LLM-as-Judge score by question category",
    )
    fig.update_xaxes(_AXIS_DEFAULTS, title_text="Category")
    fig.update_yaxes(_AXIS_DEFAULTS, title_text="Mean score (1-5)",
                     range=[0, 5.2])
    return fig


# ---------------------------------------------------------------------------
# 5. Latency histogram (overlay per mode)
# ---------------------------------------------------------------------------

def fig_latency_histogram(result: EvalRunResult) -> go.Figure:
    """Overlapped histograms of per-question latency, one trace per mode."""
    fig = go.Figure()

    all_latencies: list[float] = []
    for mode in result.modes:
        all_latencies.extend(
            row["latency_ms"]
            for row in result.rows_by_mode.get(mode, [])
            if "latency_ms" in row
        )

    if not all_latencies:
        return _empty_fig("No latency data")

    for mode in result.modes:
        latencies = [
            row["latency_ms"] for row in result.rows_by_mode.get(mode, [])
            if "latency_ms" in row
        ]
        if not latencies:
            continue
        fig.add_trace(go.Histogram(
            x=latencies, name=_label(mode),
            marker_color=_colour(mode), opacity=0.65,
            nbinsx=12,
        ))
        # P50 marker as a vertical line annotation
        p50 = float(np.median(latencies))
        fig.add_vline(
            x=p50, line_dash="dash", line_color=_colour(mode),
            annotation_text=f"{_label(mode)} p50: {p50:.0f} ms",
            annotation_position="top",
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        height=560,
        title_text="Per-question latency distribution",
    )
    fig.update_xaxes(_AXIS_DEFAULTS, title_text="Latency (ms)")
    fig.update_yaxes(_AXIS_DEFAULTS, title_text="Number of questions")
    return fig


# ---------------------------------------------------------------------------
# 6. Citation health - stacked bars showing valid vs invalid per question
# ---------------------------------------------------------------------------

def fig_citation_health(result: EvalRunResult) -> go.Figure:
    """
    Horizontal stacked bars per question. For each question and each mode,
    valid citations are stacked in green and invalid in red. Empty bars
    mean that question produced no citations.
    """
    questions = result.questions
    n_q = len(questions)
    if n_q == 0:
        return _empty_fig("No data")

    fig = go.Figure()

    # We render two traces per mode (valid + invalid) and offset the
    # bars per mode using a fake category axis: "<qid> :: <mode_label>".
    y_categories: list[str] = []
    valid_vals: list[int] = []
    invalid_vals: list[int] = []
    bar_colours: list[str] = []
    mode_labels_per_bar: list[str] = []

    for q_idx, q in enumerate(questions):
        for mode in result.modes:
            rows = result.rows_by_mode.get(mode, [])
            row = rows[q_idx] if q_idx < len(rows) else {}
            n_cit = int(row.get("n_citations", 0) or 0)
            validity = float(row.get("citation_validity", 1.0) or 1.0)
            v = round(n_cit * validity)
            iv = n_cit - v
            label = f"{q.question_id} ({_label(mode)})"
            y_categories.append(label)
            valid_vals.append(v)
            invalid_vals.append(iv)
            bar_colours.append(_colour(mode))
            mode_labels_per_bar.append(_label(mode))

    fig.add_trace(go.Bar(
        y=y_categories, x=valid_vals,
        orientation="h", name="Valid citations",
        marker_color="#2ca02c",
        hovertemplate="%{y}<br>Valid: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=y_categories, x=invalid_vals,
        orientation="h", name="Invalid citations",
        marker_color="#d62728",
        hovertemplate="%{y}<br>Invalid: %{x}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="stack",
        height=max(520, 28 * len(y_categories) + 160),
        title_text="Citation health per question",
    )
    fig.update_xaxes(_AXIS_DEFAULTS, title_text="Citation count")
    fig.update_yaxes(_AXIS_DEFAULTS, autorange="reversed",
                     showgrid=False)
    return fig
