"""ProTeGi Experiment Results — Table + Seaborn Visualizations.

Reads data/protegi_experiments.jsonl (one JSON record per line, never modified)
and produces:
    1. Console table  — ranked by improvement %
    2. Figure 1       — Baseline vs Final composite  (grouped bar chart)
    3. Figure 2       — Improvement % per run        (horizontal bar, colored by mode)
    4. Figure 3       — Heatmap all metrics           (baseline AND final side by side)
    5. Figure 4       — Baseline vs Improvement       (scatter — key ProTeGi insight)
    6. Figure 5       — Faithfulness & Coverage before/after (top-6 runs)

All figures saved to data/plots/ and shown interactively.

Usage:
    python visualize_protegi.py
    python visualize_protegi.py --no-show      # save only, no interactive window
    python visualize_protegi.py --file path/to/other.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JSONL = PROJECT_ROOT / "data" / "protegi_experiments.jsonl"
PLOTS_DIR     = PROJECT_ROOT / "data" / "plots"

# ── Lazy imports (fail early with clear message) ──────────────────────────────
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import pandas as pd
    import numpy as np
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Install with:  pip install matplotlib seaborn pandas numpy")
    sys.exit(1)


# ============================================================
# Palette & helpers
# ============================================================

# Mode colors
_MODE_COLORS = {
    "structured": "#4A90D9",   # blue
    "narrative":  "#E8834A",   # orange
}

# Provider colors (for scatter)
_PROVIDER_COLORS = {
    "anthropic": "#1a3a5c",
    "openai":    "#10a37f",
    "groq":      "#f55036",
    "nebius":    "#7c3aed",
    "mixed":     "#888888",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def _short_name(run_name: str, max_len: int = 26) -> str:
    """Truncate long run names for axis labels."""
    return run_name if len(run_name) <= max_len else run_name[:max_len - 1] + "…"


def _detect_mode(rec: dict) -> str:
    """Determine if the run used narrative or structured mode."""
    # Narrative runs: use_rouge=True in params OR "narrative" in run_name
    if rec["params"].get("use_rouge", False):
        return "narrative+rouge"
    if "narrative" in rec["run_name"].lower():
        return "narrative"
    return "structured"


def _detect_provider(rec: dict) -> str:
    """Detect dominant provider from model names."""
    models = rec.get("models", {})
    providers = set()
    for model_str in models.values():
        if model_str and "/" in model_str:
            providers.add(model_str.split("/")[0])
    if len(providers) == 1:
        return providers.pop()
    return "mixed"


def _improvement_arrow(pct: float) -> str:
    if pct > 1.0:
        return f"(+) +{pct:.1f}%"
    if pct < -0.5:
        return f"(-) {pct:.1f}%"
    return f"(=) {pct:+.1f}%"


# ============================================================
# Load & build DataFrame
# ============================================================


def load_records(path: Path) -> list[dict]:
    """Load all records from the JSONL file."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skipping malformed line: {exc}")
    if not records:
        print("[ERROR] No valid records found in the file.")
        sys.exit(1)
    print(f"Loaded {len(records)} experiment(s) from {path}\n")
    return records


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten records into a tidy DataFrame."""
    rows = []
    for rec in records:
        mode     = _detect_mode(rec)
        provider = _detect_provider(rec)
        rows.append({
            "run_name":         rec["run_name"],
            "short_name":       _short_name(rec["run_name"]),
            "timestamp":        rec.get("timestamp", ""),
            "mode":             mode,
            "provider":         provider,
            "summarize_model":  rec["models"].get("summarize", "?"),
            "critic_model":     rec["models"].get("critic", "?"),
            "judge_model":      rec["models"].get("judge", "?"),
            "n_train":          rec["params"].get("n_train", 0),
            "n_val":            rec["params"].get("n_val", 0),
            "n_iter":           rec["params"].get("n_iter", 0),
            "use_rouge":        rec["params"].get("use_rouge", False),
            # Baseline
            "base_composite":   rec["baseline"].get("composite", 0.0),
            "base_faithful":    rec["baseline"].get("faithfulness", 0.0),
            "base_coverage":    rec["baseline"].get("coverage", 0.0),
            "base_rougeL":      rec["baseline"].get("rougeL", 0.0),
            # Final
            "final_composite":  rec["final"].get("composite", 0.0),
            "final_faithful":   rec["final"].get("faithfulness", 0.0),
            "final_coverage":   rec["final"].get("coverage", 0.0),
            "final_rougeL":     rec["final"].get("rougeL", 0.0),
            # Improvement
            "improvement_pct":  rec.get("improvement_pct", 0.0),
            "duration_min":     round(rec.get("duration_seconds", 0) / 60, 1),
        })
    df = pd.DataFrame(rows)
    df["delta_composite"]  = df["final_composite"]  - df["base_composite"]
    df["delta_faithful"]   = df["final_faithful"]   - df["base_faithful"]
    df["delta_coverage"]   = df["final_coverage"]   - df["base_coverage"]
    return df


# ============================================================
# Console table
# ============================================================


def print_table(df: pd.DataFrame) -> None:
    """Print a ranked comparison table to the console."""
    sep  = "=" * 120
    sep2 = "-" * 120

    print(sep)
    print("  ProTeGi Experiment Results -- Ranked by Improvement %")
    print(sep)
    print(
        f"  {'#':<3} {'Run Name':<32} {'Mode':<16} {'Summarizer':<30} "
        f"{'Baseline':>9} {'Final':>7} {'Impr':>8} {'Duration':>9}"
    )
    print(sep2)

    sorted_df = df.sort_values("improvement_pct", ascending=False)
    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        arrow = _improvement_arrow(row["improvement_pct"])
        print(
            f"  {rank:<3} {row['run_name']:<32} {row['mode']:<16} "
            f"{row['summarize_model']:<30} "
            f"{row['base_composite']*100:>8.1f}% "
            f"{row['final_composite']*100:>6.1f}% "
            f"{arrow:>9} "
            f"{row['duration_min']:>7.1f}m"
        )

    print(sep)
    best = sorted_df.iloc[0]
    print(
        f"  Best run  : {best['run_name']}  "
        f"({best['base_composite']*100:.1f}% -> {best['final_composite']*100:.1f}%, "
        f"+{best['improvement_pct']:.1f}%)"
    )
    print(
        f"  Total runs: {len(df)}  |  "
        f"Avg improvement: {df['improvement_pct'].mean():.1f}%  |  "
        f"Runs improved: {(df['improvement_pct'] > 0.5).sum()}/{len(df)}"
    )
    print(sep)
    print()


# ============================================================
# Figure 1 — Baseline vs Final composite (grouped bar)
# ============================================================


def fig_baseline_vs_final(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Grouped bar chart: baseline vs final composite per run, sorted by improvement."""
    df_s = df.sort_values("improvement_pct", ascending=True).copy()
    labels = df_s["short_name"].tolist()
    x      = np.arange(len(labels))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_b = ax.barh(x - width / 2, df_s["base_composite"]  * 100,
                     width, label="Baseline", color="#8ab4d8", edgecolor="white")
    bars_f = ax.barh(x + width / 2, df_s["final_composite"] * 100,
                     width, label="Optimized", color="#2563eb", edgecolor="white")

    # Improvement labels on the right
    for i, (_, row) in enumerate(df_s.iterrows()):
        pct = row["improvement_pct"]
        color = "#16a34a" if pct > 1 else ("#dc2626" if pct < -0.5 else "#6b7280")
        ax.text(
            max(row["final_composite"] * 100, row["base_composite"] * 100) + 0.5,
            i,
            f"{pct:+.1f}%",
            va="center", ha="left", fontsize=9, color=color, fontweight="bold",
        )

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Composite Score (%)", fontsize=11)
    ax.set_title(
        "ProTeGi — Baseline vs Optimized Composite Score\n(sorted by improvement %)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 115)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = save_dir / "fig1_baseline_vs_final.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Figure 2 — Improvement % colored by mode
# ============================================================


def fig_improvement_by_mode(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Horizontal bar chart: improvement % per run, colored by mode."""
    df_s = df.sort_values("improvement_pct", ascending=True).copy()

    mode_palette = {
        "structured":    "#4A90D9",
        "narrative":     "#E8834A",
        "narrative+rouge": "#E84A8A",
    }
    colors = [mode_palette.get(m, "#888") for m in df_s["mode"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_s["short_name"], df_s["improvement_pct"],
                   color=colors, edgecolor="white", height=0.65)

    # Zero line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Value labels
    for bar, val in zip(bars, df_s["improvement_pct"]):
        offset = 0.3 if val >= 0 else -0.3
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, fontweight="bold",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4A90D9", label="Structured (6 fields)"),
        mpatches.Patch(color="#E8834A", label="Narrative (Judge only)"),
        mpatches.Patch(color="#E84A8A", label="Narrative (ROUGE + Judge)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    ax.set_xlabel("Relative Improvement (%)", fontsize=11)
    ax.set_title(
        "ProTeGi — Improvement % per Experiment\n(colored by summary mode)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.set_tick_params(labelsize=9)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = save_dir / "fig2_improvement_by_mode.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Figure 3 — Heatmap: all metrics baseline + final
# ============================================================


def fig_metrics_heatmap(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Heatmap of all metrics — baseline (top) and final (bottom)."""
    metrics     = ["composite", "faithful", "coverage", "rougeL"]
    metric_labels = ["Composite", "Faithfulness", "Coverage", "ROUGE-L"]
    df_s        = df.sort_values("improvement_pct", ascending=False).copy()
    run_labels  = df_s["short_name"].tolist()

    base_data  = df_s[["base_composite", "base_faithful",  "base_coverage",  "base_rougeL"]].values
    final_data = df_s[["final_composite", "final_faithful", "final_coverage", "final_rougeL"]].values

    # Stack: [baseline rows, final rows] with NaN separator row
    sep        = np.full((1, 4), np.nan)
    combined   = np.vstack([base_data, sep, final_data])
    row_labels = [f"BASE  {n}" for n in run_labels] + [""] + [f"FINAL {n}" for n in run_labels]

    fig, ax = plt.subplots(figsize=(10, max(8, len(run_labels) * 0.8 + 2)))
    sns.heatmap(
        combined,
        ax=ax,
        xticklabels=metric_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Score [0–1]", "shrink": 0.6},
        mask=np.isnan(combined),
    )

    # Separator line between baseline and final blocks
    ax.axhline(len(run_labels) + 1, color="black", linewidth=2)

    ax.set_title(
        "ProTeGi — All Metrics Heatmap\n(top = Baseline  |  bottom = Optimized)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xticklabels(metric_labels, rotation=0, fontsize=10)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
    fig.tight_layout()

    path = save_dir / "fig3_metrics_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Figure 4 — Scatter: Baseline vs Improvement (key insight)
# ============================================================


def fig_baseline_vs_improvement(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Scatter plot: baseline composite vs improvement %.

    This is the KEY scientific insight:
        Higher baseline → less room → less improvement.
    """
    mode_palette = {
        "structured":     "#4A90D9",
        "narrative":      "#E8834A",
        "narrative+rouge": "#E84A8A",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, grp in df.groupby("mode"):
        color = mode_palette.get(mode, "#888")
        ax.scatter(
            grp["base_composite"] * 100,
            grp["improvement_pct"],
            c=color, s=120, edgecolors="white", linewidths=1.5,
            label=mode, zorder=3,
        )
        # Annotate each point with run name
        for _, row in grp.iterrows():
            ax.annotate(
                _short_name(row["run_name"], 18),
                (row["base_composite"] * 100, row["improvement_pct"]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7.5, color="#333",
            )

    # Trend line (all points)
    x_all = df["base_composite"].values * 100
    y_all = df["improvement_pct"].values
    if len(x_all) > 2:
        z     = np.polyfit(x_all, y_all, 1)
        p     = np.poly1d(z)
        x_line = np.linspace(x_all.min() - 2, x_all.max() + 2, 100)
        ax.plot(x_line, p(x_line), "--", color="#94a3b8", linewidth=1.5,
                label=f"Trend (slope={z[0]:.1f})")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Baseline Composite Score (%)", fontsize=11)
    ax.set_ylabel("Improvement % after ProTeGi", fontsize=11)
    ax.set_title(
        "ProTeGi — Baseline Capability vs Optimization Gain\n"
        "Key insight: stronger baselines leave less room for improvement",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()

    path = save_dir / "fig4_baseline_vs_improvement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Figure 5 — Faithfulness & Coverage: before/after (top 6)
# ============================================================


def fig_faithfulness_coverage(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Side-by-side bars: faithfulness & coverage before/after, top-6 by improvement."""
    top6 = df.nlargest(6, "improvement_pct").sort_values("improvement_pct")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    width = 0.38

    for ax, metric, title, base_col, final_col in [
        (axes[0], "Faithfulness", "Faithfulness — Before vs After",
         "base_faithful",  "final_faithful"),
        (axes[1], "Coverage",     "Coverage — Before vs After",
         "base_coverage",  "final_coverage"),
    ]:
        x = np.arange(len(top6))
        ax.barh(x - width / 2, top6[base_col]  * 100, width,
                label="Baseline",   color="#93c5fd", edgecolor="white")
        ax.barh(x + width / 2, top6[final_col] * 100, width,
                label="Optimized",  color="#1d4ed8", edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(top6["short_name"], fontsize=9)
        ax.set_xlabel(f"{metric} (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 115)
        ax.legend(fontsize=9)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Top-6 Runs — Faithfulness & Coverage Before/After ProTeGi",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = save_dir / "fig5_faithfulness_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Figure 6 — Summary table (matplotlib)
# ============================================================


def fig_summary_table(df: pd.DataFrame, save_dir: Path, show: bool) -> None:
    """Render a professional summary table as a PNG figure.

    Columns: rank, run name, mode, summarizer, baseline %, final %, improvement %.
    Rows sorted by improvement % descending.
    Color coding:
        - Improvement > 20%  : green row
        - Improvement 1-20%  : light green
        - Improvement ~0%    : white
        - Improvement < 0%   : light red
    """
    df_s = df.sort_values("improvement_pct", ascending=False).copy()
    df_s.insert(0, "rank", range(1, len(df_s) + 1))

    # Build display values
    col_labels = [
        "#", "Run Name", "Mode",
        "Summarizer", "Baseline", "Final", "Impr %", "Duration"
    ]

    def _short_model(s: str, n: int = 28) -> str:
        parts = s.split("/")
        model = parts[-1] if parts else s
        return model if len(model) <= n else model[:n - 1] + "…"

    table_data = []
    for _, row in df_s.iterrows():
        pct = row["improvement_pct"]
        arrow = f"+{pct:.1f}%" if pct > 0.5 else (f"{pct:.1f}%" if pct < -0.5 else f"~{pct:.1f}%")
        table_data.append([
            str(int(row["rank"])),
            _short_name(row["run_name"], 30),
            row["mode"],
            _short_model(row["summarize_model"]),
            f"{row['base_composite']*100:.1f}%",
            f"{row['final_composite']*100:.1f}%",
            arrow,
            f"{row['duration_min']:.0f} min",
        ])

    n_rows = len(table_data)
    n_cols = len(col_labels)

    # Row colors
    def _row_color(pct: float) -> str:
        if pct >= 20:
            return "#bbf7d0"   # strong green
        if pct >= 5:
            return "#d1fae5"   # light green
        if pct >= 1:
            return "#ecfdf5"   # very light green
        if pct < -0.5:
            return "#fee2e2"   # light red
        return "#f9fafb"       # near white

    row_colors = [
        [_row_color(df_s.iloc[i]["improvement_pct"])] * n_cols
        for i in range(n_rows)
    ]

    # Column widths (relative)
    col_widths = [0.04, 0.18, 0.11, 0.20, 0.08, 0.08, 0.08, 0.08]

    fig_h = max(3.5, n_rows * 0.42 + 1.2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=row_colors,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Header style
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")

    # Rank column: center-align
    for row_idx in range(1, n_rows + 1):
        tbl[row_idx, 0].set_text_props(ha="center")

    # Numeric columns: right-align
    for row_idx in range(1, n_rows + 1):
        for col_idx in [4, 5, 6, 7]:
            tbl[row_idx, col_idx].set_text_props(ha="right")

    ax.set_title(
        "ProTeGi Experiments — Summary Table  (sorted by Improvement %)",
        fontsize=13, fontweight="bold", pad=14, y=1.01,
    )

    # Color legend
    legend_patches = [
        mpatches.Patch(color="#bbf7d0", label=">= 20% improvement"),
        mpatches.Patch(color="#d1fae5", label="5–20% improvement"),
        mpatches.Patch(color="#ecfdf5", label="1–5% improvement"),
        mpatches.Patch(color="#f9fafb", label="~0% improvement"),
        mpatches.Patch(color="#fee2e2", label="regression"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()

    path = save_dir / "fig6_summary_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ProTeGi experiment results from a JSONL file."
    )
    parser.add_argument(
        "--file", type=Path, default=DEFAULT_JSONL,
        help=f"Path to the JSONL experiments file (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save figures to disk only — do not open interactive windows.",
    )
    args = parser.parse_args()

    show = not args.no_show

    # ── Load ──────────────────────────────────────────────────────────────────
    records = load_records(args.file)
    df      = build_dataframe(records)

    # ── Console table ─────────────────────────────────────────────────────────
    print_table(df)

    # ── Output directory ──────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {PLOTS_DIR}\n")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    fig_summary_table(df,             PLOTS_DIR, show)
    fig_baseline_vs_final(df,         PLOTS_DIR, show)
    fig_improvement_by_mode(df,       PLOTS_DIR, show)
    fig_metrics_heatmap(df,           PLOTS_DIR, show)
    fig_baseline_vs_improvement(df,   PLOTS_DIR, show)
    fig_faithfulness_coverage(df,     PLOTS_DIR, show)

    print(f"\nDone. All figures saved in: {PLOTS_DIR}")
    print("Files generated:")
    for f in sorted(PLOTS_DIR.glob("fig*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
