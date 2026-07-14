"""Detailed internal architecture of each agent.

Shows for each agent:
    - What it reads from WatchState
    - Internal steps: tool call / LLM call / parse
    - Prompt caching markers
    - Structured output mechanism (Tool Use vs JSON Schema)
    - Planned Reflexion feedback loop (Phase 6)

Run:
    python visualize_agent_internals.py
    python visualize_agent_internals.py --save agent_internals.png
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ============================================================
# Palette
# ============================================================

C = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "blue":     "#58a6ff",
    "green":    "#3fb950",
    "orange":   "#d29922",
    "purple":   "#bc8cff",
    "red":      "#f85149",
    "yellow":   "#e3b341",
    "cyan":     "#39c5cf",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
    "haiku":    "#3fb950",
    "sonnet":   "#58a6ff",
    "no_llm":   "#6e7681",
    "tool":     "#d29922",
    "state":    "#bc8cff",
    "cache":    "#39c5cf",
    "parse":    "#f0883e",
}


# ============================================================
# Low-level drawing helpers
# ============================================================

def box(ax, x, y, w, h, fc, ec, alpha=1.0, lw=1.5, radius=0.3, zorder=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)


def label(ax, x, y, text, color=None, size=8, weight="normal",
          ha="center", va="center", zorder=5):
    ax.text(x, y, text, color=color or C["text"], fontsize=size,
            fontweight=weight, ha=ha, va=va, zorder=zorder,
            fontfamily="monospace")


def arr(ax, x1, y1, x2, y2, color, lw=1.5, style="-|>", rad=0.0, zorder=4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=12,
        ),
        zorder=zorder,
    )


def dashed_arr(ax, x1, y1, x2, y2, color, lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0",
            mutation_scale=12,
        ),
        zorder=4,
    )


def badge(ax, x, y, w, h, text, fc, ec, tcolor=None, size=7.5, zorder=5):
    box(ax, x, y, w, h, fc=fc, ec=ec, alpha=0.9, lw=1.2, radius=0.15, zorder=zorder)
    label(ax, x + w / 2, y + h / 2, text, color=tcolor or C["text"],
          size=size, weight="bold", zorder=zorder + 1)


# ============================================================
# Panel 1 — Full agent flow (internal steps)
# ============================================================

def draw_full_flow(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 17)
    ax.axis("off")
    ax.set_title(
        "Agent Internals — Tool Calls, LLM Calls, Structured Output",
        color=C["text"], fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    # ---- column X positions ----
    col = {
        "state_in":  0.3,
        "agent":     3.2,
        "tool":      6.8,
        "llm_call":  9.8,
        "llm_api":   13.5,
        "parse":     17.0,
        "state_out": 19.8,
    }
    ROW_H = 2.9
    top = 15.8

    # ---- column headers ----
    headers = [
        (col["state_in"] + 1.1,  "WatchState IN",   C["state"]),
        (col["agent"] + 1.1,     "Agent",            C["text"]),
        (col["tool"] + 1.1,      "Tool / API",       C["tool"]),
        (col["llm_call"] + 1.6,  "LLM Call",         C["haiku"]),
        (col["llm_api"] + 1.6,   "Provider API",     C["blue"]),
        (col["parse"] + 1.0,     "Parse + Validate", C["parse"]),
        (col["state_out"] + 1.0, "WatchState OUT",   C["state"]),
    ]
    for hx, ht, hc in headers:
        label(ax, hx, top + 0.5, ht, color=hc, size=7.5, weight="bold")
    ax.axhline(top + 0.15, color=C["border"], lw=1, xmin=0.01, xmax=0.99)

    # ---- agent rows ----
    agents = [
        {
            "name": "Searcher",
            "color": C["no_llm"],
            "state_in": ["topic", "config.n_raw"],
            "tool": "OpenAlex API\nGET /works?search=...",
            "tool_color": C["tool"],
            "llm": None,
            "parse": "list[dict]\n-> list[Article]",
            "state_out": "raw_articles\n(20-30 items)",
        },
        {
            "name": "QualityCritic",
            "color": C["no_llm"],
            "state_in": ["raw_articles", "config.weights"],
            "tool": "score_article()\nfilter_top_n()\n[local functions]",
            "tool_color": C["no_llm"],
            "llm": None,
            "parse": "QualityScore[]\n-> top-N sort",
            "state_out": "top_articles\ntop_scores",
        },
        {
            "name": "Summarizer",
            "color": C["haiku"],
            "state_in": ["top_articles\n(5 items)"],
            "tool": None,
            "tool_color": None,
            "llm": {
                "model": "claude-haiku-4-5",
                "calls": "x 5 calls",
                "system": "SYSTEM_PROMPT\n[CACHED]",
                "user": "title + abstract",
                "schema": "ArticleSummary",
                "mechanism": "Tool Use\n(forced)",
                "color": C["haiku"],
            },
            "parse": "Pydantic\nmodel_validate()\n-> ArticleSummary",
            "state_out": "summaries\n(5 items)",
        },
        {
            "name": "Synthesizer",
            "color": C["sonnet"],
            "state_in": ["topic", "summaries\n(5 items)"],
            "tool": None,
            "tool_color": None,
            "llm": {
                "model": "claude-sonnet-4-6",
                "calls": "x 1 call",
                "system": "SYSTEM_PROMPT\n[CACHED]",
                "user": "all summaries\nas JSON",
                "schema": "Synthesis",
                "mechanism": "Tool Use\n(forced)",
                "color": C["sonnet"],
            },
            "parse": "Pydantic\nmodel_validate()\n-> Synthesis",
            "state_out": "synthesis",
        },
        {
            "name": "TrendAnalyst",
            "color": C["sonnet"],
            "state_in": ["summaries", "synthesis"],
            "tool": None,
            "tool_color": None,
            "llm": {
                "model": "claude-sonnet-4-6",
                "calls": "x 1 call",
                "system": "SYSTEM_PROMPT\n[CACHED]",
                "user": "synthesis +\nsummaries JSON",
                "schema": "TrendAnalysis",
                "mechanism": "Tool Use\n(forced)",
                "color": C["sonnet"],
            },
            "parse": "Pydantic\nmodel_validate()\n-> TrendAnalysis",
            "state_out": "trend_analysis",
        },
    ]

    for i, ag in enumerate(agents):
        y_base = top - 0.6 - i * ROW_H
        row_cy = y_base - ROW_H / 2 + 0.2
        col_c = ag["color"]

        # ---- state in ----
        box(ax, col["state_in"], y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
            fc=C["state"], ec=C["state"], alpha=0.08, lw=1, radius=0.2)
        for j, s in enumerate(ag["state_in"]):
            label(ax, col["state_in"] + 1.1,
                  y_base - 0.5 - j * 0.55, s,
                  color=C["state"], size=7.5)

        # ---- agent box ----
        box(ax, col["agent"], y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
            fc=col_c, ec=col_c, alpha=0.15, lw=2, radius=0.2)
        label(ax, col["agent"] + 1.1, row_cy + 0.55, ag["name"],
              color=col_c, size=9, weight="bold")
        badge(ax, col["agent"] + 0.3, row_cy - 0.4, 1.6, 0.4,
              "AgentLog" , fc=C["purple"], ec=C["purple"],
              tcolor=C["text"], size=6.5)

        # arrow state_in -> agent
        arr(ax, col["agent"] - 0.1, row_cy, col["agent"],
            row_cy, C["state"])
        arr(ax, col["state_in"] + 2.2, row_cy,
            col["agent"], row_cy, C["state"], lw=1.2)

        if ag["tool"]:
            # ---- tool call ----
            box(ax, col["tool"], y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
                fc=ag["tool_color"], ec=ag["tool_color"], alpha=0.12, lw=1.5,
                radius=0.2)
            label(ax, col["tool"] + 1.1, row_cy, ag["tool"],
                  color=ag["tool_color"], size=7.5)
            arr(ax, col["agent"] + 2.2, row_cy,
                col["tool"], row_cy, ag["tool_color"], lw=1.5)

            # ---- parse (no LLM row) ----
            box(ax, col["parse"] - 3.5, y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
                fc=C["parse"], ec=C["parse"], alpha=0.12, lw=1.5, radius=0.2)
            label(ax, col["parse"] - 3.5 + 1.1, row_cy, ag["parse"],
                  color=C["parse"], size=7.5)
            arr(ax, col["tool"] + 2.2, row_cy,
                col["parse"] - 3.5, row_cy, C["parse"], lw=1.2)

            # ---- state out ----
            box(ax, col["state_out"] - 3.5, y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
                fc=C["state"], ec=C["state"], alpha=0.08, lw=1, radius=0.2)
            label(ax, col["state_out"] - 3.5 + 1.1, row_cy, ag["state_out"],
                  color=C["state"], size=7.5)
            arr(ax, col["parse"] - 3.5 + 2.2, row_cy,
                col["state_out"] - 3.5, row_cy, C["state"], lw=1.2)

        else:
            llm = ag["llm"]
            # ---- LLM call box ----
            box(ax, col["llm_call"], y_base - ROW_H + 0.35, 3.2, ROW_H - 0.55,
                fc=llm["color"], ec=llm["color"], alpha=0.12, lw=2, radius=0.2)

            # system prompt badge (cached)
            badge(ax, col["llm_call"] + 0.15, row_cy + 0.35, 2.0, 0.42,
                  llm["system"], fc=C["cache"], ec=C["cache"],
                  tcolor=C["bg"], size=6.5)
            # user message badge
            badge(ax, col["llm_call"] + 0.15, row_cy - 0.22, 2.0, 0.42,
                  "user: " + llm["user"], fc=llm["color"], ec=llm["color"],
                  tcolor=C["bg"], size=6.5)
            # schema / calls
            label(ax, col["llm_call"] + 1.6, row_cy - 0.75,
                  f"schema: {llm['schema']}  |  {llm['calls']}",
                  color=llm["color"], size=6.5)

            arr(ax, col["agent"] + 2.2, row_cy,
                col["llm_call"], row_cy, llm["color"], lw=1.5)

            # ---- provider API box ----
            box(ax, col["llm_api"], y_base - ROW_H + 0.35, 3.2, ROW_H - 0.55,
                fc=C["blue"], ec=C["blue"], alpha=0.12, lw=2, radius=0.2)
            label(ax, col["llm_api"] + 1.6, row_cy + 0.55,
                  "Anthropic API", color=C["blue"], size=8, weight="bold")
            badge(ax, col["llm_api"] + 0.3, row_cy + 0.02, 2.6, 0.42,
                  llm["mechanism"], fc=C["orange"], ec=C["orange"],
                  tcolor=C["bg"], size=7)
            label(ax, col["llm_api"] + 1.6, row_cy - 0.55,
                  llm["model"], color=C["subtext"], size=7)

            arr(ax, col["llm_call"] + 3.2, row_cy,
                col["llm_api"], row_cy, C["blue"], lw=1.5)

            # ---- parse box ----
            box(ax, col["parse"], y_base - ROW_H + 0.35, 2.2, ROW_H - 0.55,
                fc=C["parse"], ec=C["parse"], alpha=0.12, lw=1.5, radius=0.2)
            label(ax, col["parse"] + 1.1, row_cy, ag["parse"],
                  color=C["parse"], size=7.5)
            arr(ax, col["llm_api"] + 3.2, row_cy,
                col["parse"], row_cy, C["parse"], lw=1.2)

            # ---- state out ----
            box(ax, col["state_out"], y_base - ROW_H + 0.35, 2.0, ROW_H - 0.55,
                fc=C["state"], ec=C["state"], alpha=0.08, lw=1, radius=0.2)
            label(ax, col["state_out"] + 1.0, row_cy, ag["state_out"],
                  color=C["state"], size=7.5)
            arr(ax, col["parse"] + 2.2, row_cy,
                col["state_out"], row_cy, C["state"], lw=1.2)

        # row separator
        ax.axhline(y_base - ROW_H + 0.25, color=C["border"], lw=0.5,
                   xmin=0.01, xmax=0.99, linestyle="--", alpha=0.4)


# ============================================================
# Panel 2 — LLM structured output detail
# ============================================================

def draw_llm_internals(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "How Structured Output Works  (Claude Tool Use vs OpenAI JSON Schema)",
        color=C["text"], fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    # ---- Claude side ----
    box(ax, 0.3, 0.4, 9.8, 6.0, fc=C["haiku"], ec=C["haiku"],
        alpha=0.07, lw=2, radius=0.4)
    label(ax, 5.2, 6.05, "Claude  —  Forced Tool Use", color=C["haiku"],
          size=10, weight="bold")

    steps_claude = [
        (1.0, 4.6, 7.5, 0.8, C["cache"],
         "system: [SYSTEM_PROMPT, cache_control: ephemeral]  <- prompt cached (90% off)"),
        (1.0, 3.6, 7.5, 0.8, C["haiku"],
         "user:   'Title: ...  Abstract: ...'"),
        (1.0, 2.6, 7.5, 0.8, C["orange"],
         "tools:  [{name: emit_articlesummary, input_schema: ArticleSummary.schema()}]"),
        (1.0, 1.6, 7.5, 0.8, C["orange"],
         "tool_choice: {type: tool, name: emit_articlesummary}   <- forced"),
    ]
    for bx, by, bw, bh, bc, bt in steps_claude:
        box(ax, bx, by, bw, bh, fc=bc, ec=bc, alpha=0.18, lw=1, radius=0.2)
        label(ax, bx + bw / 2, by + bh / 2, bt, color=C["text"], size=7.5)

    # response
    box(ax, 1.0, 0.55, 7.5, 0.82, fc=C["green"], ec=C["green"], alpha=0.18,
        lw=1.5, radius=0.2)
    label(ax, 5.2, 0.96,
          "Response: content[0].type = tool_use  ->  content[0].input = {problem: ..., method: ...}",
          color=C["green"], size=7.5)

    arr(ax, 5.2, 1.55, 5.2, 1.37, C["green"], lw=2)
    label(ax, 5.2, 1.45, "Pydantic model_validate()", color=C["parse"], size=7)

    # ---- OpenAI side ----
    box(ax, 11.5, 0.4, 9.8, 6.0, fc=C["orange"], ec=C["orange"],
        alpha=0.07, lw=2, radius=0.4)
    label(ax, 16.4, 6.05, "OpenAI  —  JSON Schema strict", color=C["orange"],
          size=10, weight="bold")

    steps_oai = [
        (12.0, 4.6, 7.5, 0.8, C["subtext"],
         "system: 'You are a ...'  (no caching)"),
        (12.0, 3.6, 7.5, 0.8, C["orange"],
         "user:   'Title: ...  Abstract: ...'"),
        (12.0, 2.6, 7.5, 0.8, C["orange"],
         "response_format: {type: json_schema, name: ArticleSummary}"),
        (12.0, 1.6, 7.5, 0.8, C["orange"],
         "strict: True  ->  additionalProperties:false on all objects"),
    ]
    for bx, by, bw, bh, bc, bt in steps_oai:
        box(ax, bx, by, bw, bh, fc=bc, ec=bc, alpha=0.18, lw=1, radius=0.2)
        label(ax, bx + bw / 2, by + bh / 2, bt, color=C["text"], size=7.5)

    box(ax, 12.0, 0.55, 7.5, 0.82, fc=C["green"], ec=C["green"],
        alpha=0.18, lw=1.5, radius=0.2)
    label(ax, 16.4, 0.96,
          "Response: choices[0].message.content = '{\"problem\": ..., \"method\": ...}'",
          color=C["green"], size=7.5)

    arr(ax, 16.4, 1.55, 16.4, 1.37, C["green"], lw=2)
    label(ax, 16.4, 1.45, "json.loads() + Pydantic model_validate()", color=C["parse"], size=7)

    # divider
    ax.axvline(11.0, color=C["border"], lw=1, ymin=0.05, ymax=0.95,
               linestyle="--", alpha=0.5)


# ============================================================
# Panel 3 — Reflexion feedback loop (Phase 6 planned)
# ============================================================

def draw_reflexion(ax):
    ax.set_facecolor(C["panel"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title(
        "Reflexion Pattern — Phase 6 (planned)  |  Critic feedback on Synthesizer",
        color=C["text"], fontsize=11, fontweight="bold", pad=8, loc="left",
    )

    # planned badge
    badge(ax, 18.0, 4.7, 3.5, 0.55, "PHASE 6 — not yet implemented",
          fc=C["orange"], ec=C["orange"], tcolor=C["bg"], size=8)

    # nodes
    nodes = [
        (1.0, 1.8, 2.4, 1.4, C["haiku"],  "Summarizer",   "summaries"),
        (4.8, 1.8, 2.4, 1.4, C["sonnet"],  "Synthesizer",  "synthesis\nv1, v2..."),
        (8.6, 1.8, 2.4, 1.4, C["red"],     "Critic",       "CriticFeedback\n{needs_revision}"),
        (12.4, 1.8, 2.4, 1.4, C["sonnet"], "Synthesizer\n(revised)", "synthesis\nfinal"),
        (16.2, 1.8, 2.4, 1.4, C["sonnet"], "TrendAnalyst", "trend_analysis"),
    ]

    for nx, ny, nw, nh, nc, name, out in nodes:
        box(ax, nx, ny, nw, nh, fc=nc, ec=nc, alpha=0.18, lw=2, radius=0.3)
        label(ax, nx + nw / 2, ny + nh - 0.32, name, color=nc, size=9,
              weight="bold")
        label(ax, nx + nw / 2, ny + 0.35, out, color=C["subtext"], size=7)

    # arrows forward
    for i in range(len(nodes) - 1):
        nx, ny, nw, nh = nodes[i][:4]
        nx2 = nodes[i + 1][0]
        cy = ny + nh / 2
        arr(ax, nx + nw, cy, nx2, cy, C["blue"], lw=2)

    # feedback arrow: Critic -> Synthesizer (loop back)
    # Critic box
    cx, cy_b, cw, ch = nodes[2][:4]
    sx, sy_b, sw, sh = nodes[1][:4]

    ax.annotate(
        "",
        xy=(sx + sw / 2, sy_b + sh),
        xytext=(cx + cw / 2, cy_b + ch),
        arrowprops=dict(
            arrowstyle="-|>", color=C["red"], lw=2,
            connectionstyle="arc3,rad=-0.5",
            linestyle="dashed",
            mutation_scale=14,
        ),
        zorder=4,
    )
    label(ax, 7.0, 4.0, "needs_revision = True\n(max 3 iterations)",
          color=C["red"], size=8, weight="bold")

    # condition diamond
    box(ax, 11.0, 1.9, 1.2, 1.2, fc=C["red"], ec=C["red"],
        alpha=0.2, lw=1.5, radius=0.15)
    label(ax, 11.6, 2.5, "needs\nrevision?", color=C["red"], size=7, weight="bold")
    label(ax, 11.6, 1.7, "False -> continue", color=C["green"], size=7)

    # iteration counter
    box(ax, 0.5, 0.2, 20.8, 1.2, fc=C["orange"], ec=C["orange"],
        alpha=0.06, lw=1, radius=0.3)

    items = [
        ("iteration 0", C["subtext"], 2.5),
        ("Synthesizer produces synthesis v1", C["sonnet"], 6.0),
        ("Critic evaluates -> needs_revision=True", C["red"], 10.5),
        ("Synthesizer revises with feedback", C["sonnet"], 15.0),
        ("Critic evaluates -> needs_revision=False", C["green"], 19.5),
    ]
    for txt, col, x in items:
        label(ax, x, 0.82, txt, color=col, size=7.5)


# ============================================================
# Legend
# ============================================================

def draw_legend(fig):
    legend_items = [
        (C["state"],   "WatchState read/write"),
        (C["tool"],    "External tool call (API / function)"),
        (C["cache"],   "Prompt cached (Anthropic, 90% off)"),
        (C["haiku"],   "LLM: Claude Haiku (cheap, fast)"),
        (C["sonnet"],  "LLM: Claude Sonnet (large context)"),
        (C["orange"],  "Structured output mechanism"),
        (C["parse"],   "Pydantic validation"),
        (C["red"],     "Reflexion / feedback (Phase 6)"),
        (C["no_llm"],  "No LLM (algorithm only)"),
    ]
    handles = [
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.8, label=l)
        for c, l in legend_items
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=5, frameon=True,
        facecolor=C["panel"], edgecolor=C["border"],
        labelcolor=C["text"], fontsize=7.5,
        bbox_to_anchor=(0.5, 0.0),
    )


# ============================================================
# Main
# ============================================================

def build_figure():
    fig = plt.figure(figsize=(22, 30), facecolor=C["bg"])
    fig.suptitle(
        "Scientific Watch Agent V2  —  Agent Internals: Tool Calls, LLM, Feedback",
        color=C["text"], fontsize=15, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.09,
        top=0.985, bottom=0.04,
        left=0.01, right=0.99,
        height_ratios=[2.6, 1.0, 0.8],
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(C["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
            spine.set_linewidth(1.5)

    draw_full_flow(ax1)
    draw_llm_internals(ax2)
    draw_reflexion(ax3)
    draw_legend(fig)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="FILE", default="agent_internals.png")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.save, dpi=150, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    print(f"Saved: {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
