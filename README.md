# Scientific Watch Agent

> **Automated scientific literature monitoring with cross-domain meta-learning**
> Powered by **Nebius Serverless AI Endpoints** (DeepSeek-V4, Qwen3-30B)

A 7-agent LangGraph pipeline that turns any research topic into a complete literature review in under 2 minutes — with a meta-learner that adapts scoring to new domains without retraining.

---

## The Problem

A researcher starting a PhD faces 2+ million new papers per year in CS/AI alone.
A proper literature review takes **3–6 weeks of manual work** — before writing a single thesis line.

This project automates the entire process end-to-end.

---

## Quick Demo

```bash
python run_pipeline.py "graph neural networks"
```

```
[QueryExpander]  4 iterations  →  6 diversified queries (ReAct loop)
[Searcher]                     →  213 papers from OpenAlex
[QualityCritic]                →  top 15 selected (6-feature scoring)
[Summarizer]     15 papers     →  structured summaries via Nebius Qwen3
[Synthesizer]    Reflexion ×2  →  full domain synthesis via Nebius DeepSeek-V4
[TrendAnalyst]                 →  3 trends · 4 gaps · 5 future directions

Runtime: 91s   Cost: $0.048
```

Output: a structured Markdown report with summaries, synthesis, and trend analysis.

---

## Nebius Integration

All LLM inference runs exclusively on **Nebius Serverless AI Endpoints**.

| Task | Model | Cost |
|---|---|---|
| Summarization (×15 papers) | `Qwen/Qwen3-30B-A3B-Instruct` | $0.14/M tokens |
| Synthesis + Reflexion critic | `deepseek-ai/DeepSeek-V4` | $0.27/M tokens |
| Trend analysis | `deepseek-ai/DeepSeek-V4` | $0.27/M tokens |
| Query expansion | `Qwen/Qwen3-30B-A3B-Instruct` | $0.14/M tokens |

**Total per pipeline run: ~$0.05** (vs ~$0.42 with Claude/GPT-4 equivalent)

```python
# src/llm/nebius_client.py
client = NebiusClient(
    model="Qwen/Qwen3-30B-A3B-Instruct",
    base_url="https://api.studio.nebius.com/v1",
    api_key=os.environ["NEBIUS_API_KEY"],
)
result, log = client.chat_structured(
    system=SUMMARIZER_PROMPT,
    messages=[{"role": "user", "content": article_text}],
    schema=ArticleSummary,
)
# log.cost_usd, log.tokens_used tracked per call
```

`NebiusClient` extends `OpenAIClient` with a custom `base_url` — zero migration friction from the OpenAI SDK.

---

## Architecture

```
Topic (string)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ QueryExpander — ReAct loop (max 4 iterations)        │
│ Thought → Search OpenAlex → Observe → Adapt          │
└────────────────────┬────────────────────────────────┘
                     │ 4–8 search queries
                     ▼
┌─────────────────────────────────────────────────────┐
│ Searcher — parallel fetch + deduplication            │
│ 80–300 papers from OpenAlex (free, 250M+ works)     │
└────────────────────┬────────────────────────────────┘
                     │ corpus
                     ▼
┌─────────────────────────────────────────────────────┐
│ QualityCritic — 6-feature weighted scorer            │
│ venue · authors · impact · velocity · recency ·      │
│ relevance (semantic embeddings)                      │
│ Weights loaded from Optuna-learned JSON per domain   │
└────────────────────┬────────────────────────────────┘
                     │ top 15 papers
                     ▼
┌─────────────────────────────────────────────────────┐
│ Summarizer — Nebius Qwen3 (×15)                      │
│ Structured: problem / method / dataset / results     │
│ Or narrative prose (150–250 words)                   │
└────────────────────┬────────────────────────────────┘
                     │ summaries
                     ▼
┌─────────────────────────────────────────────────────┐
│ Synthesizer  ◄──────────────── Critic (Reflexion)    │
│ Nebius DeepSeek-V4               4 evaluation axes   │
│ Revises if quality < "good"      max 3 iterations    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ TrendAnalyst — Nebius DeepSeek-V4                    │
│ Emerging trends · Research gaps · Future directions  │
└─────────────────────────────────────────────────────┘
```

Orchestrated with **LangGraph** (StateGraph with conditional Reflexion edge).
All I/O typed with **Pydantic v2** — the inter-agent communication contract.

---

## Meta-Learning: Cross-Domain Generalization

**The core scientific contribution.**

Optimal scoring weights differ significantly across domains:

| Domain | venue | authors | impact | velocity | recency | relevance |
|---|---|---|---|---|---|---|
| Fake news detection | 0.03 | 0.06 | 0.04 | **0.70** | 0.10 | 0.02 |
| Federated learning | 0.20 | **0.30** | 0.18 | 0.14 | 0.15 | 0.04 |
| Graph neural networks | 0.20 | 0.13 | 0.08 | 0.22 | **0.33** | 0.03 |
| Quantitative finance | **0.28** | 0.15 | 0.07 | 0.11 | 0.03 | **0.35** |

Instead of re-running optimization for every new domain, we extract 7 **meta-features** from the corpus (citation distribution, author expertise, recency ratio, etc.) and train a **Borda Count ensemble** (kNN + BayesianRidge) to predict the right weights — with no oracle needed at inference time.

### Results — Leave-One-Out CV on 19 domains

| Method | Win rate | Mean NDCG@15 | vs Default |
|---|---|---|---|
| **Borda Ensemble ✅** | **15/19 = 79%** | **0.282** | **+23.1%** |
| RRF | 14/19 = 74% | 0.254 | +10.9% |
| Transfer Direct (baseline) | — | 0.247 | +7.7% |
| Default weights (baseline) | — | 0.229 | — |
| *Optuna oracle (upper bound)* | — | *0.387* | *+68.7%* |

**Wilcoxon signed-rank test: W⁺ = 149, p = 0.014, r_rb = +0.57 (large effect)**

The system learns to configure itself. A practitioner on a new domain gets near-optimal rankings without running any optimization.

---

## Evaluation

### H1 — AutoML beats expert weights ✅ Confirmed

Optuna TPE (150 trials, 6 features, simplex constraint) beats uniform weights on all 19 domains.
Mean improvement: **+68.7% NDCG@15**. Minimum improvement: +20%. Maximum: +538%.

### H3 — Meta-learning beats transfer ✅ Confirmed

Borda ensemble wins on 15/19 domains (79% ≥ 75% threshold), statistically significant
at α = 0.05 (p = 0.014, large effect size r_rb = +0.57).

---

## Setup

### Requirements

- Python 3.10+
- A Nebius API key ([studio.nebius.com](https://studio.nebius.com))
- OpenAlex email (free, for polite pool access)

### Install

```bash
git clone https://github.com/SeghaierBechir/scientific-watch-agent
cd scientific-watch-agent

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:
```
NEBIUS_API_KEY=your_key_here
OPENALEX_EMAIL=your_email@example.com
```

### Generate a Report

The main output is a **professional PDF report** saved to `output/` with 7 sections:

```
Cover page        topic · date · stats (papers, tokens, cost, reflexion iterations)
1. Trends         emerging trends table · supporting articles · research gaps · future directions
2. Synthesis      global overview · main approaches · reference datasets · key findings
3. Top Articles   ranked table + score breakdown (venue / authors / impact / relevance)
4. Summaries      one card per paper — structured fields OR narrative prose (see modes below)
5. Queries        expanded search queries from the ReAct loop
6. Reflexion      per-iteration critic feedback (quality, issues, suggestions)
7. Logs           duration · tokens · API calls per agent
```

#### Structured mode (default)

Each paper summary has 6 labeled fields: **Problem · Method · Dataset · Results · Limits · Contributions**.

```bash
python run_pipeline.py "graph neural networks"
```

```
Topic       : graph neural networks
Papers      : fetch 30 → keep top 10
Mode        : structured summaries
Running pipeline...

[QueryExpander]  4 iterations  →  6 diversified queries
[Searcher]                     →  187 papers from OpenAlex
[QualityCritic]                →  top 10 selected
[Summarizer]     ×10 papers    →  structured summaries  (Nebius Qwen3)
[Synthesizer]    Reflexion ×2  →  global synthesis      (Nebius DeepSeek-V4)
[TrendAnalyst]                 →  3 trends · 4 gaps · 5 directions

Runtime: 87s   Cost: $0.042
Generating PDF report...
Report saved → output/graph_neural_networks_20260714_1423.pdf
```

#### Narrative mode

Each paper summary is a **prose paragraph** (150–250 words) — better for a reading-flow report.

```bash
python run_pipeline.py "federated learning" --narrative
```

#### More papers, earlier date range

```bash
python run_pipeline.py "fake news detection" --n-raw 50 --top-n 15 --from-year 2018
```

#### Send report by email

```bash
python run_pipeline.py "graph neural networks" --email you@example.com
```

Requires `EMAIL_SENDER` and `EMAIL_PASSWORD` (Gmail App Password) in `.env`.

#### All options

```
usage: run_pipeline.py [-h] [--n-raw N] [--top-n N] [--from-year YEAR]
                       [--narrative] [--output FILE] [--email ADDRESS] [--no-save]
                       topic

positional arguments:
  topic              Research topic (e.g. "graph neural networks")

options:
  --n-raw   N        Papers to fetch from OpenAlex (default: 30)
  --top-n   N        Papers to keep after quality scoring (default: 10)
  --from-year YEAR   Earliest publication year (default: 2023)
  --narrative        Prose paragraph summaries instead of structured 6-field cards
  --output FILE      Custom PDF path (default: output/<slug>_<timestamp>.pdf)
  --email ADDRESS    Send the PDF to this address after generation
  --no-save          Print terminal summary only, do not generate PDF
```

#### Via REST API (optional)

```bash
# Start the API server
uvicorn api:app --port 8080

# Call it
curl -X POST http://localhost:8080/watch \
  -H "Content-Type: application/json" \
  -d '{"topic": "graph neural networks", "top_n": 5, "narrative_mode": false}'
```

Interactive docs at `http://localhost:8080/docs`.

---

## Reproducibility

All experiments run without LLM calls (pre-computed oracles and sub-scores).

### Reproduce H1 (AutoML, one domain, ~30s)

```bash
python demo_phase3.py --domain fake_news_detection
# Expected: NDCG@15 improvement ~+155% over default weights
```

### Reproduce H3 (meta-learning, all 19 domains, ~5 min)

```bash
python phase8a_8b.py      # extract meta-features + Pearson correlations
python phase8c.py         # LOO-CV: 15 methods × 19 domains
python phase8d_wilcoxon.py # Wilcoxon test → data/plots/phase8d_wilcoxon.png
```

### Run the test suite (no API keys needed)

```bash
pytest tests/ -v
# 266 tests, all mocked
```

---

## Project Structure

```
scientific-watch-agent/
├── src/
│   ├── agents/
│   │   ├── query_expander.py   # ReAct loop
│   │   ├── summarizer.py       # structured + narrative modes
│   │   ├── synthesizer.py      # Reflexion
│   │   ├── critic.py           # 4-axis quality evaluation
│   │   └── trend_analyst.py
│   ├── llm/
│   │   ├── nebius_client.py    # ← Nebius Serverless AI Endpoints
│   │   ├── base.py             # LLMClient Protocol
│   │   └── factory.py          # task → model routing
│   ├── scoring/
│   │   ├── quality_scorer.py   # 6-feature weighted scorer
│   │   └── automl_scorer.py    # Optuna TPE
│   └── metalearning/
│       ├── meta_features.py    # corpus characterization (7 features)
│       └── meta_learner.py     # Borda + 11 other methods
├── data/
│   ├── oracle/                 # 19 annotated domains
│   ├── weights/                # Optuna weights per domain (JSON)
│   └── metalearning/           # meta-features + LOO-CV results
├── tests/                      # 266 tests (all mocked)
├── run_pipeline.py
├── requirements.txt
└── .env.example
```

---

## Why Nebius

| Factor | Detail |
|---|---|
| **Cost** | 10–20× cheaper than Claude/GPT for identical tasks |
| **Compatibility** | OpenAI-compatible API — one-line swap in factory.py |
| **Scale** | 285 LOO-CV evaluation runs feasible within budget |
| **Models** | DeepSeek-V4 + Qwen3 match GPT-4-class quality on structured output |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | **Nebius Serverless AI Endpoints** |
| Agent orchestration | LangGraph |
| Data contract | Pydantic v2 |
| Scientific data | OpenAlex API (free, 250M+ papers) |
| Semantic embeddings | all-MiniLM-L6-v2 (local, no API cost) |
| AutoML | Optuna TPE |
| Evaluation | NDCG@15 · Wilcoxon signed-rank |
| Tests | pytest (266 tests, fully mocked) |

---

## License

MIT
