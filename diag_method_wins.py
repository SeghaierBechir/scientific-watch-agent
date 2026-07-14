"""Per-domain win/loss diagnostic table for meta-learning methods.

Reads data/metalearning/phase8c_results.json and produces a colour-coded table:
  - Rows    : 19 domains, sorted by Borda delta (best to worst)
  - Columns : Transfer (ref), kNN(1-3), Ridge, BayesRidge, GPR,
              SoftVote, HardVote, Borda, RRF, En+R, En+B, Optuna* (ref)
  - Cells   : delta NDCG@15 vs Transfer Direct
  - Green   : beats Transfer Direct (win)
  - Red     : below Transfer Direct (fail)

Usage
-----
    python diag_method_wins.py
    python diag_method_wins.py --no-show     # save only, no popup
    python diag_method_wins.py --absolute    # show absolute NDCG instead of delta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_PATH = PROJECT_ROOT / "data" / "metalearning" / "phase8c_results.json"
PLOTS_DIR    = PROJECT_ROOT / "data" / "plots"

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Install with:  pip install matplotlib numpy")
    sys.exit(1)

matplotlib.rcParams.update({"font.family": "sans-serif"})

SHORT_LABELS: dict[str, str] = {
    "fake_news_detection":              "Fake news",
    "graph_neural_networks":            "Graph NN",
    "federated_learning":               "Federated",
    "medical_image_segmentation":       "Medical img",
    "neural_architecture_search":       "NAS",
    "explainable_ai":                   "XAI",
    "knowledge_graph_embedding":        "KG embed",
    "graph_attention_networks":         "GAT",
    "object_detection_deep_learning":   "Object det.",
    "sentiment_analysis":               "Sentiment",
    "transfer_learning":                "Transfer L.",
    "generative_adversarial_networks":  "GAN",
    "anomaly_detection":                "Anomaly",
    "deep_reinforcement_learning":      "Deep RL",
    "text_summarization":               "Text sum.",
    "speech_recognition_asr":           "Speech ASR",
    "computational_biology":            "Comp. Bio.",
    "medical_nlp":                      "Medical NLP",
    "quantitative_finance_ml":          "Quant. Fin.",
}

# Methods to evaluate (columns of coloured cells)
EVAL_METHODS = [
    "knn1", "knn2", "knn3",
    "ridge", "bayesian_ridge", "gpr",
    "soft_vote", "hard_vote", "borda", "rrf",
    "ensemble", "ensemble_bayes",
]
EVAL_LABELS = [
    "kNN(1)", "kNN(2)", "kNN(3)",
    "Ridge", "BayesRidge", "GPR",
    "SoftVote", "HardVote", "Borda", "RRF",
    "En+R", "En+B",
]

# Sort rows by Borda delta (the best confirmed meta-learner — 15/19 wins)
_SORT_METHOD = "borda"

# Reference columns (not coloured, shown for context)
REF_METHODS = ["transfer_direct", "optuna"]
REF_LABELS  = ["Transfer*", "Optuna**"]

WIN_COLOR    = "#b7f0b1"   # green
FAIL_COLOR   = "#f5b3b3"   # red
TIE_COLOR    = "#e8e8e8"   # grey
REF_COLOR    = "#dce8f5"   # blue-grey for reference columns
HEADER_COLOR = "#2c3e50"   # dark navy
HEADER_TEXT  = "white"
ROW_EVEN     = "#fafaf8"
ROW_ODD      = "#f0f0ed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-domain win/loss diagnostic table")
    parser.add_argument("--no-show",  action="store_true", help="Save PNG only, no popup")
    parser.add_argument("--absolute", action="store_true", help="Show absolute NDCG instead of delta")
    parser.add_argument("--file", default=str(RESULTS_PATH), help="Path to phase8c_results.json")
    args = parser.parse_args()

    results_path = Path(args.file)
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        print("Run phase8c.py first to generate results.")
        sys.exit(1)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    loo     = data["loo_ndcg15"]
    domains = data["domains"]

    # Sort rows: Borda delta descending (best wins at top, biggest losses at bottom)
    domains = sorted(
        domains,
        key=lambda d: loo[d].get(_SORT_METHOD, 0.0) - loo[d].get("transfer_direct", 0.0),
        reverse=True,
    )

    n = len(domains)
    n_eval = len(EVAL_METHODS)
    n_ref  = len(REF_METHODS)
    n_cols  = n_eval + n_ref  # total columns (ref Transfer + evals + ref Optuna)

    # Build data arrays
    td_vals   = np.array([loo[d].get("transfer_direct", 0.0) for d in domains])
    eval_vals = np.array([[loo[d].get(m, 0.0) for m in EVAL_METHODS] for d in domains])
    ref_vals  = np.array([[loo[d].get(m, 0.0) for m in REF_METHODS]  for d in domains])
    deltas    = eval_vals - td_vals[:, None]  # shape (n, n_eval)

    wins = (deltas > 1e-6).sum(axis=0)  # win count per method

    # ── Build cell text and colours ───────────────────────────────────────────

    all_cell_text   = []
    all_cell_colors = []

    for i, d in enumerate(domains):
        row_text   = []
        row_colors = []

        for j in range(n_eval):
            if args.absolute:
                val = eval_vals[i, j]
                row_text.append(f"{val:.3f}")
            else:
                delta = deltas[i, j]
                sign  = "+" if delta > 1e-6 else ""
                row_text.append(f"{sign}{delta:.3f}")

            delta = deltas[i, j]
            if delta > 1e-6:
                row_colors.append(WIN_COLOR)
            elif delta < -1e-6:
                row_colors.append(FAIL_COLOR)
            else:
                row_colors.append(TIE_COLOR)

        # Reference columns (Transfer and Optuna) — always shown absolute, no colour coding
        for k in range(n_ref):
            row_text.append(f"{ref_vals[i, k]:.3f}")
            row_colors.append(REF_COLOR)

        all_cell_text.append(row_text)
        all_cell_colors.append(row_colors)

    # ── Column headers ────────────────────────────────────────────────────────

    col_headers = []
    for j, label in enumerate(EVAL_LABELS):
        col_headers.append(f"{label}\n{wins[j]}/{n} wins")
    for label in REF_LABELS:
        col_headers.append(label)

    # Reorder: Transfer first, then eval methods, then Optuna
    # Currently: eval[0..4], ref[transfer, optuna]
    # Reorder to: ref[transfer], eval[0..4], ref[optuna]
    def _reorder(lst: list) -> list:
        return [lst[n_eval]] + lst[:n_eval] + [lst[n_eval + 1]]

    col_headers_ordered = _reorder(col_headers)
    all_cell_text_ordered   = [_reorder(row) for row in all_cell_text]
    all_cell_colors_ordered = [_reorder(row) for row in all_cell_colors]

    # ── Figure ────────────────────────────────────────────────────────────────

    row_labels = [SHORT_LABELS.get(d, d) for d in domains]
    # Compute height so table rows fill most of the canvas
    # Each row ~0.30 in at scale_y=1.55; add 1.2 in for title + footer
    fig_h = max(5.0, (n + 1) * 0.30 + 1.2)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")

    # Axes fills figure between footer (5%) and title (6% from top)
    ax.set_position([0.06, 0.055, 0.93, 0.885])

    table = ax.table(
        cellText=all_cell_text_ordered,
        cellColours=all_cell_colors_ordered,
        rowLabels=row_labels,
        colLabels=col_headers_ordered,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.55)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color=HEADER_TEXT, fontweight="bold", fontsize=7.5)

    # Alternating row shading for row labels
    for i in range(n):
        label_cell = table[i + 1, -1]
        label_cell.set_facecolor(ROW_EVEN if i % 2 == 0 else ROW_ODD)
        label_cell.set_text_props(fontweight="bold")

    # Win separator line (visual split between winners and losers based on Borda delta)
    borda_idx = EVAL_METHODS.index(_SORT_METHOD)
    n_borda_wins = int(wins[borda_idx])
    if 0 < n_borda_wins < n:
        for j in range(n_cols):
            table[n_borda_wins + 1, j].set_edgecolor("#555")
            table[n_borda_wins + 1, j].set_linewidth(1.8)

    mode_label = "absolute NDCG@15" if args.absolute else "delta NDCG@15 vs Transfer Direct"
    fig.text(
        0.5, 0.975,
        f"Meta-learner per-domain wins/losses  ({mode_label})  —  {n} domains, LOO-CV",
        ha="center", va="top", fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, 0.955,
        "Green = beats Transfer Direct   |   Red = below Transfer Direct   |   "
        "Blue-grey = reference column   |   Rows sorted by Borda delta",
        ha="center", va="top", fontsize=8.5, color="#444",
    )

    # Footer: win counts
    summary = "   |   ".join(
        f"{EVAL_LABELS[j]}: {wins[j]}/{n}"
        for j in range(n_eval)
    )
    fig.text(0.5, 0.018, f"Win counts (vs Transfer Direct):  {summary}",
             ha="center", va="bottom", fontsize=8.5, color="#444",
             bbox=dict(facecolor="#f5f5f0", edgecolor="#ccc", boxstyle="round,pad=0.3"))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "diag_method_wins.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {save_path}")

    if not args.no_show:
        plt.show()
    plt.close()

    # ── Console summary ───────────────────────────────────────────────────────

    header_cols = "  ".join(f"{lbl[:6]:>8}" for lbl in EVAL_LABELS)
    print(f"\n  Win/Loss per domain (delta vs Transfer Direct):")
    print(f"  {'Domain':<22}  {header_cols}  Transfer  Optuna*")
    sep_w = 24 + len(EVAL_LABELS) * 10 + 22
    print(f"  {'-'*sep_w}")
    for i, d in enumerate(domains):
        td  = loo[d].get("transfer_direct", 0.0)
        opt = loo[d].get("optuna", 0.0)
        row = f"  {SHORT_LABELS.get(d, d):<22}"
        for j in range(n_eval):
            delta = deltas[i, j]
            sign  = "+" if delta > 1e-6 else ""
            mark  = "W" if delta > 1e-6 else ("L" if delta < -1e-6 else "=")
            row += f"  {sign}{delta:+.3f}{mark}"
        row += f"  {td:.3f}    {opt:.3f}"
        print(row)

    print(f"  {'-'*sep_w}")
    print(f"  {'Wins':<22}  ", end="")
    for j in range(n_eval):
        print(f"  {wins[j]:>5}/{n}  ", end="")
    print()


if __name__ == "__main__":
    main()
