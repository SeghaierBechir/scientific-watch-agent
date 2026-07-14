"""Visualize the V2 multi-agent architecture.

Generates 3 diagrams in one figure:
    1. Pipeline flow  — agents as nodes, edges as data flow
    2. State ownership — what each agent reads and writes
    3. LLM cost map   — which model, how many calls, estimated cost

Run:
    python visualize_architecture.py
    python visualize_architecture.py --save architecture.png
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec


# ============================================================
# Color palette
# ============================================================

C = {
    "bg":         "#0f1117",
    "panel":      "#1a1d27",
    "border":     "#2d3148",
    "blue":       "#4c9be8",
    "green":      "#4caf7d",
    "orange":     "#e8934c",
    "purple":     "#9b59b6",
    "red":        "#e74c3c",
    "yellow":     "#f1c40f",
    "text":       "#e8eaf6",
    "subtext":    "#8892b0",
    "arrow":      "#4c9be8",
    "haiku":      "#4caf7d",
    "sonnet":     "#4c9be8",
    "opus":       "#9b59b6",
    "openai":     "#e8934c",
    "no_llm":     "#636e72",
}

# ============================================================
# Data definitions
# ============================================================

AGENTS = [
    {
        "name": "Searcher",
        "number": "①",
        "color": C["no_llm"],
        "llm": "No LLM",
        "llm_color": C["no_llm"],
        "calls": "0 calls",
        "reads": ["topic", "config.n_raw", "config.from_year"],
        "writes": ["raw_articles"],
        "output": "20-30 Articles",
        "source": "OpenAlex API",
    },
    {
        "name": "QualityCritic",
        "number": "②",
        "color": C["no_llm"],
        "llm": "No LLM",
        "llm_color": C["no_llm"],
        "calls": "0 calls",
        "reads": ["topic", "raw_articles", "config.weights", "config.top_n"],
        "writes": ["quality_scores", "top_articles", "top_scores"],
        "output": "Top-5 Articles",
        "source": "Scoring algo",
    },
    {
        "name": "Summarizer",
        "number": "③",
        "color": C["haiku"],
        "llm": "claude-haiku-4-5",
        "llm_color": C["haiku"],
        "calls": "5 calls",
        "reads": ["top_articles"],
        "writes": ["summaries"],
        "output": "5 Summaries",
        "source": "Claude Haiku",
    },
    {
        "name": "Synthesizer",
        "number": "④",
        "color": C["sonnet"],
        "llm": "claude-sonnet-4-6",
        "llm_color": C["sonnet"],
        "calls": "1 call",
        "reads": ["topic", "summaries"],
        "writes": ["synthesis"],
        "output": "Global Synthesis",
        "source": "Claude Sonnet",
    },
    {
        "name": "TrendAnalyst",
        "number": "⑤",
        "color": C["sonnet"],
        "llm": "claude-sonnet-4-6",
        "llm_color": C["sonnet"],
        "calls": "1 call",
        "reads": ["summaries", "synthesis"],
        "writes": ["trend_analysis"],
        "output": "Trends + Gaps",
        "source": "Claude Sonnet",
    },
]

STATE_SECTIONS = [
    ("topic / config",    C["subtext"],  "INPUT",        "──────────"),
    ("raw_articles",      C["no_llm"],   "← Searcher",   "list[Article]"),
    ("quality_scores",    C["no_llm"],   "← QualityCritic", "list[QualityScore]"),
    ("top_articles",      C["no_llm"],   "← QualityCritic", "list[Article]"),
    ("summaries",         C["haiku"],    "← Summarizer", "list[ArticleSummary]"),
    ("synthesis",         C["sonnet"],   "← Synthesizer","Synthesis"),
    ("trend_analysis",    C["sonnet"],   "← TrendAnalyst","TrendAnalysis"),
    ("logs / errors",     C["purple"],   "← all agents", "list[AgentLog]"),
]

LLM_DATA = [
    ("claude-haiku-4-5",  C["haiku"],  "Summarizer",  "5 calls",  "~$0.05",  "Fast, cheap"),
    ("claude-sonnet-4-6", C["sonnet"], "Synthesizer\n+ TrendAnalyst", "2 calls", "~$0.08", "Large context"),
    ("claude-opus-4-7",   C["opus"],   "Judge (V3)",  "—",        "~$0.00",  "Not used yet"),
    ("gpt-4o-mini",       C["openai"], "query_expansion", "—",    "~$0.00",  "Phase 3+"),
]


# ============================================================
# Drawing helpers
# ============================================================

def rounded_box(ax, x, y, w, h, color, alpha=0.15, lw=2, text=None,
                fontsize=9, text_color=None, radius=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize,
            color=text_color or C["text"],
            fontweight="bold",
            zorder=3,
        )


def arrow(ax, x1, y1, x2, y2, color=None, label=None):
    color = color or C["arrow"]
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=4,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.02, label, ha="center", va="bottom",
                fontsize=7, color=C["subtext"], zorder=5)


# ============================================================
# Panel 1 — Pipeline flow
# ============================================================

def draw_pipeline(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("① Pipeline Flow", color=C["text"], fontsize=12,
                 fontweight="bold", pad=10, loc="left")

    # User input
    rounded_box(ax, 0.1, 2.9, 1.5, 0.45, C["yellow"], alpha=0.25,
                text="USER\ntopic", fontsize=8, text_color=C["yellow"])

    # LangGraph container
    rounded_box(ax, 0.1, 0.15, 9.8, 2.65, C["border"], alpha=0.5, lw=1)
    ax.text(0.25, 2.72, "LangGraph — StateGraph(WatchState)", color=C["subtext"],
            fontsize=7.5, zorder=5)

    # Agent boxes
    positions = [0.25, 2.25, 4.25, 6.25, 8.25]
    box_w = 1.65
    box_h = 2.0
    box_y = 0.35

    for i, (agent, x) in enumerate(zip(AGENTS, positions)):
        col = agent["color"]

        # Main box
        rounded_box(ax, x, box_y, box_w, box_h, col, alpha=0.18, lw=2)

        # Number badge
        rounded_box(ax, x + 0.05, box_y + box_h - 0.35, 0.32, 0.28,
                    col, alpha=0.6, lw=1)
        ax.text(x + 0.21, box_y + box_h - 0.21, agent["number"],
                ha="center", va="center", fontsize=9, color=C["text"],
                fontweight="bold", zorder=5)

        # Agent name
        ax.text(x + box_w / 2, box_y + box_h - 0.2, agent["name"],
                ha="center", va="center", fontsize=8.5, color=C["text"],
                fontweight="bold", zorder=5)

        # Output
        ax.text(x + box_w / 2, box_y + 1.35, agent["output"],
                ha="center", va="center", fontsize=7.5,
                color=col, fontweight="bold", zorder=5)

        # LLM badge
        badge_col = agent["llm_color"]
        rounded_box(ax, x + 0.15, box_y + 0.62, box_w - 0.3, 0.52,
                    badge_col, alpha=0.3, lw=1)
        ax.text(x + box_w / 2, box_y + 0.92, agent["llm"],
                ha="center", va="center", fontsize=6.8,
                color=badge_col, fontweight="bold", zorder=5)
        ax.text(x + box_w / 2, box_y + 0.72, agent["calls"],
                ha="center", va="center", fontsize=6.5,
                color=C["subtext"], zorder=5)

        # Source
        ax.text(x + box_w / 2, box_y + 0.28, agent["source"],
                ha="center", va="center", fontsize=6.5,
                color=C["subtext"], zorder=5)

        # Arrow to next agent
        if i < len(AGENTS) - 1:
            ax.annotate(
                "",
                xy=(positions[i + 1] - 0.05, box_y + box_h / 2),
                xytext=(x + box_w + 0.05, box_y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.8, mutation_scale=14),
                zorder=4,
            )

    # Arrow from user to Searcher
    ax.annotate(
        "",
        xy=(positions[0] + box_w / 2, box_y + box_h),
        xytext=(positions[0] + box_w / 2, 2.9),
        arrowprops=dict(arrowstyle="-|>", color=C["yellow"],
                        lw=1.5, mutation_scale=12),
        zorder=4,
    )

    # WatchState bar at bottom
    rounded_box(ax, 0.15, 0.18, 9.7, 0.14, C["purple"], alpha=0.25, lw=1,
                text="WatchState (Shared State — LangGraph)",
                fontsize=7, text_color=C["purple"])


# ============================================================
# Panel 2 — State ownership
# ============================================================

def draw_state(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STATE_SECTIONS) + 1.2)
    ax.axis("off")
    ax.set_title("② WatchState — ownership (Single Writer rule)",
                 color=C["text"], fontsize=12, fontweight="bold", pad=10, loc="left")

    n = len(STATE_SECTIONS)
    ax.text(0.2, n + 0.8, "Field", color=C["subtext"], fontsize=8, fontweight="bold")
    ax.text(3.5, n + 0.8, "Owner", color=C["subtext"], fontsize=8, fontweight="bold")
    ax.text(6.0, n + 0.8, "Type", color=C["subtext"], fontsize=8, fontweight="bold")

    ax.axhline(n + 0.6, color=C["border"], lw=1, xmin=0.02, xmax=0.98)

    for i, (field, color, owner, dtype) in enumerate(reversed(STATE_SECTIONS)):
        y = i + 0.4
        # Row background
        rounded_box(ax, 0.1, y - 0.02, 9.8, 0.75, color, alpha=0.08, lw=1)
        # Color dot
        ax.plot(0.35, y + 0.35, "o", color=color, markersize=7, zorder=5)
        # Field name
        ax.text(0.6, y + 0.35, field, color=C["text"], fontsize=9,
                va="center", fontweight="bold", zorder=5)
        # Owner
        ax.text(3.5, y + 0.35, owner, color=color, fontsize=8.5,
                va="center", fontweight="bold", zorder=5)
        # Type
        ax.text(6.0, y + 0.35, dtype, color=C["subtext"], fontsize=8,
                va="center", zorder=5)


# ============================================================
# Panel 3 — LLM cost map
# ============================================================

def draw_llm_map(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("③ LLM Layer — model mapping & cost per run",
                 color=C["text"], fontsize=12, fontweight="bold", pad=10, loc="left")

    # Total cost box
    rounded_box(ax, 0.1, 4.8, 9.8, 0.9, C["yellow"], alpha=0.15, lw=2)
    ax.text(5.0, 5.25, "Estimated cost per run (top_n=5) :  ~$0.10 – $0.20",
            ha="center", va="center", fontsize=10.5, color=C["yellow"],
            fontweight="bold", zorder=5)

    # Header
    headers = ["Model", "Used by", "Calls/run", "Cost", "Note"]
    xs = [0.4, 2.8, 5.6, 7.0, 8.2]
    ax.axhline(4.55, color=C["border"], lw=1, xmin=0.02, xmax=0.98)
    for h, x in zip(headers, xs):
        ax.text(x, 4.65, h, color=C["subtext"], fontsize=8, fontweight="bold")

    for i, (model, color, used_by, calls, cost, note) in enumerate(LLM_DATA):
        y = 3.4 - i * 0.82
        rounded_box(ax, 0.1, y - 0.05, 9.8, 0.7, color, alpha=0.12, lw=1)
        ax.plot(0.3, y + 0.3, "o", color=color, markersize=8, zorder=5)
        vals = [model, used_by, calls, cost, note]
        for v, x in zip(vals, xs):
            ax.text(x, y + 0.3, v, color=C["text"] if v != model else color,
                    fontsize=8.5, va="center",
                    fontweight="bold" if v == model else "normal", zorder=5)

    # Prompt caching note
    rounded_box(ax, 0.1, 0.1, 9.8, 0.65, C["green"], alpha=0.12, lw=1)
    ax.text(0.4, 0.43,
            "Prompt caching (Anthropic) enabled on all system prompts  →  "
            "90% off on cache reads after the 1st call",
            color=C["green"], fontsize=8.5, va="center", zorder=5)


# ============================================================
# Main figure assembly
# ============================================================

def build_figure():
    fig = plt.figure(figsize=(18, 20), facecolor=C["bg"])
    fig.suptitle(
        "Scientific Watch Agent  —  V2 Multi-Agent Architecture",
        color=C["text"], fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        hspace=0.12,
        top=0.95, bottom=0.02,
        left=0.03, right=0.97,
        height_ratios=[1.0, 1.1, 0.9],
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(C["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
            spine.set_linewidth(1.5)

    draw_pipeline(ax1)
    draw_state(ax2)
    draw_llm_map(ax3)

    return fig


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize V2 agent architecture")
    parser.add_argument("--save", metavar="FILE", help="Save to PNG instead of showing")
    args = parser.parse_args()

    fig = build_figure()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight",
                    facecolor=C["bg"], edgecolor="none")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
