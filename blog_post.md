# I Built a 7-Agent AI System That Reads 200 Papers and Writes a Research Report in 90 Seconds — Powered by Nebius

*Tags: #NebiusServerlessChallenge #AI #LLM #MultiAgent #Research*

---

## The Problem No One Talks About

Before writing a single line of a thesis, a PhD student faces an invisible wall: **literature review**.

There are over 2 million new CS/AI papers published every year. Searching, filtering, reading, and synthesizing them into a coherent picture takes 3 to 6 weeks of manual work — before the actual research even begins. I've seen it happen. It's exhausting, error-prone, and frankly, a waste of human intelligence.

So I built **Scientific Watch Agent**: a multi-agent AI system that takes a research topic as input and produces a full, structured research report as output — in under 2 minutes, for about $0.05.

The entire LLM inference runs on **Nebius Serverless AI Endpoints**, using DeepSeek-V4 and Qwen3-30B. Here's how it works, and what I learned building it.

---

## Architecture: 7 Agents, One Pipeline

The system is orchestrated with **LangGraph** — a framework that lets you define agents as nodes in a directed graph, with a shared state flowing between them.

```
Topic (string)
    ↓
QueryExpander   [ReAct loop, 4 iterations]
    ↓
Searcher        [OpenAlex API, 80–300 papers]
    ↓
QualityCritic   [6-feature scorer, Optuna weights]
    ↓
Summarizer      [×15 papers, Nebius Qwen3-30B]
    ↓
Synthesizer ←──── Critic  [Reflexion loop, max 3 iterations]
    ↓
TrendAnalyst    [Nebius DeepSeek-V4]
    ↓
PDF Report
```

Each agent has a single responsibility and communicates exclusively through a typed shared state — a `TypedDict` validated with **Pydantic v2**. No agent talks to another directly. This single-writer / multiple-readers contract made the system much easier to test and debug.

### Agent 1 — QueryExpander (ReAct)

The user types "graph neural networks". That's too vague for a good literature search. The QueryExpander uses the **ReAct pattern** (Reason → Act → Observe) to iteratively refine search queries:

1. It generates a query candidate
2. Probes OpenAlex with a small 5-paper sample
3. Observes what concepts appear
4. Adapts the next query accordingly

After 4 iterations, it produces 4–8 diversified queries covering different angles of the topic.

### Agent 2 — QualityCritic (AutoML Scoring)

Not all papers are equal. The QualityCritic scores each paper on 6 dimensions:
- **Venue**: journal quartile (Q1–Q4) from Scimago data
- **Authors**: maximum h-index in the author list
- **Impact**: citations per year (log-normalized)
- **Velocity**: citation momentum — is this paper gaining traction?
- **Recency**: exponential decay with a 3-year half-life
- **Relevance**: cosine similarity with the topic query (all-MiniLM-L6-v2 embeddings)

The weights for these 6 dimensions are **learned per domain** using **Optuna TPE** (Tree-structured Parzen Estimator). A domain like "fake news detection" needs high velocity weight (the field moves fast). "Quantitative finance ML" needs high venue + relevance weights (established journals, precise topic). These learned weights are stored as JSON files and loaded at inference time — no retraining needed.

### Agents 3–5 — Summarizer + Synthesizer + Critic (Reflexion)

The Summarizer calls **Nebius Qwen3-30B** for each of the top 15 papers. Two modes are supported:
- **Structured**: 6-field cards (Problem / Method / Dataset / Results / Limits / Contributions)
- **Narrative**: prose paragraphs (150–250 words), better for reading flow

The Synthesizer then reads all summaries and produces a global synthesis — overview, main approaches, reference datasets, key findings — using **Nebius DeepSeek-V4**.

Here's where **Reflexion** (Shinn et al., 2023) comes in. After each synthesis, a Critic agent evaluates the output on 4 axes: *fidelity, completeness, specificity, consistency*. If quality is below "good", it sends detailed feedback to the Synthesizer, which revises. This loop runs up to 3 times. In practice, it improves synthesis quality significantly on specialized or niche domains.

### Agent 6 — TrendAnalyst

The final agent identifies emerging trends (with maturity labels), research gaps (with importance badges), and future research directions. It reads both the summaries and the synthesis — giving it the broadest possible context before drawing conclusions.

---

## Why Nebius?

I chose **Nebius Serverless AI Endpoints** for three reasons:

**Cost.** A full pipeline run costs ~$0.05 with Nebius. The equivalent with Claude Sonnet + GPT-4o would cost ~$0.42. That's an 8× difference — critical when you're running 285 evaluation experiments across 19 domains.

**Compatibility.** Nebius uses an OpenAI-compatible API. My `NebiusClient` is literally a 5-line subclass of `OpenAIClient` with a different `base_url`. Zero migration friction.

**Model quality.** DeepSeek-V4 produces synthesis quality comparable to Claude Sonnet on structured scientific tasks. Qwen3-30B handles repetitive summarization tasks at $0.14/M tokens.

```python
class NebiusClient(OpenAIClient):
    def __init__(self, model: str = "Qwen/Qwen3-30B-A3B-Instruct"):
        super().__init__(
            model=model,
            base_url="https://api.studio.nebius.com/v1",
            api_key=os.environ["NEBIUS_API_KEY"],
        )
```

The factory routes each task to the right model automatically — agents never know which provider they're talking to.

---

## The Scientific Contribution: Meta-Learning

The most original part of this project isn't the pipeline. It's the **cross-domain generalization**.

The insight: optimal scoring weights differ dramatically across research domains. "Fake news detection" needs high velocity weight (0.70). "Federated learning" needs high authors weight (0.30). If you use the same weights everywhere, you get mediocre results everywhere.

I built a **meta-learner** that predicts optimal weights for a new domain from 7 corpus statistics (citation distribution, author expertise ratio, recency ratio, etc.) — without running Optuna optimization on that domain.

The meta-learner is a **Borda Count ensemble** (kNN(1) + kNN(2) + kNN(3) + BayesianRidge rank fusion), evaluated on 19 domains by Leave-One-Out Cross-Validation.

**Results:**
- Borda ensemble beats Transfer Direct (using another domain's weights) on **15/19 = 79%** of domains
- Mean NDCG@15: **0.282** vs 0.229 (default weights) — **+23.1%**
- Wilcoxon signed-rank test: **W⁺ = 149, p = 0.014, r_rb = +0.57** (large effect, statistically significant at α = 0.05)

The system learns to configure itself for new domains. A researcher on a new topic gets near-optimal paper rankings with no manual tuning.

---

## What the Output Looks Like

Running this command:

```bash
python run_pipeline.py "fake news detection" --n-raw 50 --top-n 15 --from-year 2018
```

Produces a **7-section PDF report** in ~90 seconds:

1. **Trends** — emerging trends table with supporting article digests, research gaps with importance badges
2. **Global Synthesis** — field overview, main approaches, reference datasets, key findings
3. **Top Articles** — ranked table with score breakdown per dimension
4. **Summaries** — structured cards or narrative prose per paper
5. **Query Expansion** — the ReAct search queries that were used
6. **Reflexion Log** — how many revision cycles the synthesizer ran
7. **Agent Logs** — duration, tokens, API calls per agent

The report a PhD student would spend 3 weeks writing manually: generated in 90 seconds, for $0.05.

---

## Try It Yourself

The full code is open source. You only need a **Nebius API key** (free tier available) and an email for OpenAlex.

🔗 **Repository**: (https://github.com/SeghaierBechir/scientific-watch-agent)

```bash
git clone https://github.com/SeghaierBechir/scientific-watch-agent
cd scientific-watch-agent
pip install -r requirements.txt
cp .env.example .env
# Add your NEBIUS_API_KEY to .env

python run_pipeline.py "graph neural networks" --top-n 10
```

No GPU required. No fine-tuning. No infrastructure to manage. Just a Nebius API key and a research topic.

---

*Built for the Nebius Serverless AI Challenge. All experiments reproducible — see the repository for the full evaluation campaign scripts.*

*#NebiusServerlessChallenge #MultiAgentAI #LLM #LangGraph #ResearchAutomation #DeepSeek #Qwen3*
