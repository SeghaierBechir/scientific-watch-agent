"""Phase 8c — Meta-learner results visualizer.

Reads data/metalearning/phase8c_results.json and produces three figures:

    Figure 1  kNN(1) delta bar — Δ NDCG@15 vs Transfer Direct, sorted,
              annotated with the LOO nearest neighbor for each domain.
    Figure 2  Scatter kNN(1) vs Transfer Direct — parity diagonal,
              points coloured by win / tie / loss.
    Figure 3  All methods heatmap — NDCG@15 per domain × method.

All figures saved to data/plots/ and optionally shown interactively.

Usage
-----
    python visualize_phase8c.py
    python visualize_phase8c.py --no-show     # save only, no popup window
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "data" / "metalearning" / "phase8c_results.json"
PLOTS_DIR    = PROJECT_ROOT / "data" / "plots"

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Install with:  pip install matplotlib numpy")
    sys.exit(1)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.7,
})

WIN_COLOR  = "#1baf7a"
LOSS_COLOR = "#e34948"
TIE_COLOR  = "#888780"
GRAY       = "#c3c2b7"

SHORT_LABELS: dict[str, str] = {
    "fake_news_detection":          "Fake news",
    "graph_neural_networks":        "Graph NN",
    "federated_learning":           "Federated",
    "medical_image_segmentation":   "Medical img",
    "neural_architecture_search":   "NAS",
    "explainable_ai":               "XAI",
    "knowledge_graph_embedding":    "KG embed",
    "graph_attention_networks":     "GAT",
    "object_detection_deep_learning": "Object det.",
    "sentiment_analysis":           "Sentiment",
    "transfer_learning":            "Transfer L.",
    "generative_adversarial_networks": "GAN",
    "anomaly_detection":            "Anomaly det.",
    "deep_reinforcement_learning":  "Deep RL",
    "text_summarization":           "Text sum.",
    "speech_recognition_asr":       "Speech ASR",
}

METHOD_LABELS: dict[str, str] = {
    "default":         "Default",
    "transfer_direct": "Transfer",
    "knn1":            "kNN(1)",
    "knn2":            "kNN(2)",
    "knn3":            "kNN(3)",
    "ridge":           "Ridge",
    "bayesian_ridge":  "BayesRidge",
    "optuna":          "Optuna*",
}


# ── LOO nearest-neighbour computation ────────────────────────────────────────

def _compute_loo_neighbors(
    meta_features: dict[str, dict],
    feature_names: list[str],
) -> dict[str, dict]:
    """Return {domain_id: {'neighbor': domain_id, 'dist': float}}."""
    domains = list(meta_features.keys())
    raw: dict[str, list[float]] = {
        d: [meta_features[d][f] for f in feature_names]
        for d in domains
    }
    nf = len(feature_names)
    result: dict[str, dict] = {}

    for i, test_d in enumerate(domains):
        train_d = [d for j, d in enumerate(domains) if j != i]
        train_vecs = [raw[d] for d in train_d]

        mins = [min(v[j] for v in train_vecs) for j in range(nf)]
        maxs = [max(v[j] for v in train_vecs) for j in range(nf)]

        def _scale(vec: list[float]) -> list[float]:
            return [
                (vec[j] - mins[j]) / (maxs[j] - mins[j] + 1e-10)
                for j in range(nf)
            ]

        ts = _scale(raw[test_d])
        best_d, best_dist = "", float("inf")
        for td in train_d:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ts, _scale(raw[td]))))
            if dist < best_dist:
                best_dist = dist
                best_d = td
        result[test_d] = {"neighbor": best_d, "dist": best_dist}

    return result


# ── Figure 1 — Delta bar chart with neighbour annotations ────────────────────

def fig_delta_bars(
    loo: dict[str, dict[str, float]],
    neighbors: dict[str, dict],
    save_dir: Path,
    show: bool,
) -> None:
    domains = list(loo.keys())
    deltas = {
        d: loo[d].get("knn1", 0) - loo[d].get("transfer_direct", 0)
        for d in domains
    }
    sorted_d = sorted(domains, key=lambda d: deltas[d], reverse=True)

    labels, values, colors = [], [], []
    for d in sorted_d:
        delta = deltas[d]
        nb = SHORT_LABELS.get(neighbors[d]["neighbor"], neighbors[d]["neighbor"])
        dist = neighbors[d]["dist"]
        labels.append(f"{SHORT_LABELS.get(d, d)}\n→ {nb}  d={dist:.2f}")
        values.append(delta)
        colors.append(WIN_COLOR if delta > 1e-5 else (LOSS_COLOR if delta < -1e-5 else TIE_COLOR))

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.65, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=1.2, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ NDCG@15 (kNN(1) − Transfer Direct)", fontsize=10)
    ax.set_title(
        "kNN(1) meta-learner — LOO performance vs Transfer Direct\n"
        "each label shows the nearest neighbour used for weight prediction",
        fontsize=11,
    )

    n_wins = sum(1 for v in values if v > 1e-5)
    n_total = len(values)
    threshold = math.ceil(0.75 * n_total)
    h3_ok = n_wins >= threshold
    verdict = f"H3: {'SUPPORTED' if h3_ok else 'REJECTED'} — {n_wins}/{n_total} wins ({100*n_wins/n_total:.0f}%)"
    ax.text(
        0.98, 0.02, verdict,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10, fontweight="bold",
        color=WIN_COLOR if h3_ok else LOSS_COLOR,
    )

    legend_patches = [
        mpatches.Patch(color=WIN_COLOR, label=f"kNN(1) wins ({n_wins})"),
        mpatches.Patch(color=TIE_COLOR, label=f"Tie ({sum(1 for v in values if abs(v) <= 1e-5)})"),
        mpatches.Patch(color=LOSS_COLOR, label=f"kNN(1) loses ({sum(1 for v in values if v < -1e-5)})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = save_dir / "phase8c_fig1_delta_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close()


# ── Figure 2 — Scatter kNN(1) vs Transfer Direct ─────────────────────────────

def fig_scatter(
    loo: dict[str, dict[str, float]],
    save_dir: Path,
    show: bool,
) -> None:
    domains = list(loo.keys())
    xs = [loo[d].get("transfer_direct", 0) for d in domains]
    ys = [loo[d].get("knn1", 0) for d in domains]
    cs = [
        WIN_COLOR if y > x + 1e-5 else (LOSS_COLOR if y < x - 1e-5 else TIE_COLOR)
        for x, y in zip(xs, ys)
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, c=cs, s=80, zorder=4, edgecolors="white", linewidths=0.5)

    top = max(max(xs), max(ys)) * 1.05
    ax.plot([0, top], [0, top], color=GRAY, linewidth=1.2, linestyle="--", zorder=3, label="y = x")

    for d, x, y in zip(domains, xs, ys):
        lbl = SHORT_LABELS.get(d, d)
        offset_x = 0.004
        offset_y = -0.006 if y > x else 0.006
        ax.text(x + offset_x, y + offset_y, lbl, fontsize=7.5, color="#52514e", ha="left")

    ax.set_xlabel("Transfer Direct (NDCG@15)", fontsize=10)
    ax.set_ylabel("kNN(1) (NDCG@15)", fontsize=10)
    ax.set_title("kNN(1) vs Transfer Direct — points above diagonal = kNN wins", fontsize=11)

    legend_patches = [
        mpatches.Patch(color=WIN_COLOR, label="kNN(1) wins"),
        mpatches.Patch(color=TIE_COLOR, label="Tie"),
        mpatches.Patch(color=LOSS_COLOR, label="kNN(1) loses"),
        mpatches.Patch(color=GRAY, label="Parity y = x"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = save_dir / "phase8c_fig2_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close()


# ── Figure 3 — Heatmap all methods ───────────────────────────────────────────

def fig_heatmap(
    loo: dict[str, dict[str, float]],
    save_dir: Path,
    show: bool,
) -> None:
    methods = ["default", "transfer_direct", "knn1", "knn2", "knn3", "ridge", "bayesian_ridge", "optuna"]
    domains = sorted(loo.keys(), key=lambda d: loo[d].get("knn1", 0) - loo[d].get("transfer_direct", 0), reverse=True)

    data = np.array([[loo[d].get(m, 0) for m in methods] for d in domains])
    y_labels = [SHORT_LABELS.get(d, d) for d in domains]
    x_labels = [METHOD_LABELS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)

    for i in range(len(domains)):
        for j in range(len(methods)):
            v = data[i, j]
            text_color = "white" if v > 0.35 else "#333"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_title("NDCG@15 per domain × method  (sorted by kNN(1) delta)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6, label="NDCG@15")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.tight_layout()
    path = save_dir / "phase8c_fig3_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase 8c meta-learner results")
    parser.add_argument("--no-show", action="store_true", help="Save figures only, no interactive window")
    parser.add_argument("--file", default=str(RESULTS_PATH), help="Path to phase8c_results.json")
    args = parser.parse_args()

    results_path = Path(args.file)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("Run phase8c.py first to generate the results.")
        sys.exit(1)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    loo = data["loo_ndcg15"]
    meta_features = data["meta_features"]
    feature_names = data.get("effective_features", [
        "median_year", "citation_median", "grade2_ratio",
        "year_std", "pct_high_cited", "pct_recent", "pct_q1",
    ])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    print(f"\n  Phase 8c visualizer — {len(loo)} domains, {len(feature_names)} features")
    print(f"  Results : {results_path}")
    print(f"  Output  : {PLOTS_DIR}\n")

    neighbors = _compute_loo_neighbors(meta_features, feature_names)

    print("  [1/3] Delta bar chart + nearest neighbours ...")
    fig_delta_bars(loo, neighbors, PLOTS_DIR, show)

    print("  [2/3] Scatter kNN(1) vs Transfer Direct ...")
    fig_scatter(loo, PLOTS_DIR, show)

    print("  [3/3] Heatmap all methods ...")
    fig_heatmap(loo, PLOTS_DIR, show)

    print(f"\n  Done.  3 figures saved in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
