# Scientific Watch Agent - Project Specification

> **Document type**: Formal project specification
> **Audience**: Memoire reviewers, supervisors, future contributors
> **Status**: Living document - updated at each phase milestone

---

## 1. Executive summary

This project develops a **multi-agent system for automated scientific
literature monitoring**. Given a research topic provided by a user (e.g. "fake
news detection"), the system retrieves relevant scientific articles, filters
them by quality, summarizes each one, produces a global synthesis, and
identifies research trends and perspectives.

The system targets **early-stage PhD researchers** who need to rapidly
understand a new field. It evolves through four versions:
- **V1**: Mono-agent pipeline with AutoML-tuned scoring
- **V2**: Multi-agent collaborative architecture
- **V3**: Hybrid collaborative + concurrent (Best-of-N)
- **V4**: Prompt optimization and meta-learning for cross-domain adaptation

**Academic context**: Master's thesis in Computer Science / AI / Data Science,
3-4 month timeline.

---

## 2. Motivation

### Problem statement

PhD researchers in their early stages face a recurring challenge: rapidly
mapping the state of the art in a new research area. The traditional process is
manual and time-consuming:

1. Search Google Scholar / databases with rough queries
2. Skim dozens of abstracts to assess relevance
3. Read the most promising papers in depth
4. Manually identify recurring themes, dominant approaches, gaps
5. Form a mental synthesis of the field

This process can take **weeks**, with the risk of missing important works or
forming a biased view of the field.

### Why now?

Three convergent factors make this project tractable today:

1. **LLM capabilities**: modern LLMs (Claude, GPT) can summarize and synthesize
   scientific text at near-expert quality
2. **Open scholarly data**: OpenAlex provides ~250M articles with rich
   bibliometric metadata, free and via API
3. **Multi-agent frameworks**: LangGraph and similar tools make orchestration
   of complex agent pipelines accessible

### Why a multi-agent approach?

A monolithic LLM-based solution has limitations:
- Difficult to evaluate each step independently
- Hard to debug when output quality drops
- Single point of failure for prompt engineering
- No mechanism for cross-validation between perspectives

A multi-agent architecture addresses these by **separating concerns** (search,
filter, summarize, synthesize, analyze trends) and enabling **patterns like
Reflexion and Best-of-N** that improve robustness.

---

## 3. Objectives

### Primary objective : cross-domain generalization via meta-learning

The central scientific objective is to demonstrate that a scientific watch
agent can **generalize across research domains** without per-domain
re-engineering, thanks to meta-learning techniques applied to the system's
configurations (scoring weights, prompts, model choices).

The core idea: rather than learning optimal parameters **for one domain**
(e.g. fake news detection), we learn a **function predicting good parameters
from meta-features of the domain** (technical vocabulary, field maturity,
citation distribution, etc.).

Three levels of generalization are studied:
1. **Direct transfer** - apply parameters learned on Domain A as-is on Domain B
2. **Few-shot fine-tuning** - adjust parameters with very few samples on B
3. **Meta-learning** - learn `meta_features(domain) → optimal_params` from
   multiple domains

### Secondary objectives (in service of the primary)

1. **Multi-agent architecture** that allows isolating *where* generalization
   breaks (which agent degrades the most cross-domain?)
2. **Application of AutoML** (Optuna) to learn per-domain optimal weights
   (foundation for meta-learning)
3. **Reflexion pattern** to detect and correct cross-domain hallucinations
4. **Cross-LLM benchmark** comparing Claude's vs GPT's generalization profiles
5. **Production-quality system** that researchers can actually use

### Non-objectives

- Not a search engine (we use OpenAlex as backend)
- Not a peer-review tool (we don't judge the validity of claims)
- Not a chatbot (we produce structured reports, not conversation)
- Not a Google Scholar replacement (we focus on synthesis, not retrieval volume)
- Not a single-domain expert (the whole point is generalization)

---

## 4. Research hypotheses

These hypotheses will be tested empirically and reported in the thesis. They
are organized around the central meta-learning objective.

### H1 - Per-domain AutoML beats expert weights

**Statement**: Quality-scoring weights learned by AutoML (Optuna) on a
labeled dataset for a given domain achieve higher NDCG@15 than fixed
expert-defined weights.

**Test method**: Compare baseline (expert weights from scientometrics
literature) vs AutoML-learned weights on a held-out test set.

**Success criterion**: AutoML weights show ≥5% relative improvement on NDCG@15.

### H2 - Direct transfer degrades predictably

**Statement**: Using parameters learned on Domain A directly on Domain B
yields performance between 60% and 85% of B's optimal performance, with
degradation correlated to a measurable distance metric between domains.

**Test method**: Train on Domain A, evaluate on Domains B, C, D, E without
retraining. Compute pairwise domain distance via meta-features.

**Success criterion**: Statistically significant correlation (Pearson or
Spearman, p < 0.05) between domain distance and performance gap.

### H3 - Meta-learning beats direct transfer (CENTRAL HYPOTHESIS)

**Statement**: Predicting parameters for a new domain via a learned function
`meta_features(D) → params` outperforms direct transfer on at least 75% of
test domains.

**Test method**: Train meta-learner on 3-4 source domains, evaluate on
held-out target domains. Compare meta-learning predictions vs direct transfer
from the most similar source.

**Success criterion**: Meta-learning ≥ direct transfer on ≥75% of test cases,
with average improvement > 5%.

### H4 - Multi-agent architecture localizes generalization bottlenecks

**Statement**: The agent-by-agent analysis (V2 architecture + Reflexion
feedback) reveals that some pipeline components generalize well (e.g. quality
scoring) while others degrade more (e.g. trend identification).

**Test method**: Per-agent quality metrics across domains. Identify which
agents have the highest variance across domains.

**Success criterion**: Identify at least 2 agents with significantly
different cross-domain robustness (variance ratio > 2).

### H5 - Claude vs GPT have different generalization profiles

**Statement**: Claude and GPT exhibit complementary cross-domain robustness:
one model generalizes better on tasks requiring deep technical understanding,
the other on tasks requiring breadth of coverage.

**Test method**: Systematic A/B testing across all 5 evaluation domains using
`BENCHMARK_PAIRS` configuration.

**Success criterion**: Find at least one task where Claude > GPT
significantly, AND one where GPT > Claude significantly (each p < 0.05).

### H6 - Concurrent V3 helps most where direct transfer fails

**Statement**: The V3 concurrent multi-agent mode (Best-of-N + Judge) provides
the largest quality improvement on the cross-domain settings where direct
transfer underperforms most.

**Test method**: Compute V3-vs-V2 improvement on each domain, correlate with
direct transfer performance gap.

**Success criterion**: Negative correlation (the worse direct transfer, the
more V3 helps) with p < 0.05.

---

## 5. Meta-learning approach (CENTRAL CONTRIBUTION)

### Conceptual framework

The system learns to **predict optimal configurations** for new domains
rather than relying on a single fixed configuration. This is operationalized
via three mechanisms:

#### Mechanism 1 - Domain meta-features

For each domain, we compute a vector of **meta-features** describing its
characteristics:
- Vocabulary richness (TF-IDF entropy on top retrieved articles)
- Field maturity (median publication year, citation distribution)
- Citation density (average cites per paper)
- Author concentration (Gini coefficient on author productivity)
- Concept diversity (number of distinct OpenAlex concepts)
- Preprint ratio
- Open access ratio

These meta-features serve as input to the meta-learner.

#### Mechanism 2 - Per-domain AutoML

For each training domain, Optuna learns optimal weights `(w_v, w_a, w_i, w_r)`
maximizing NDCG@15. This produces a labeled dataset:

```
{(meta_features(D_1), optimal_weights(D_1)),
 (meta_features(D_2), optimal_weights(D_2)),
 ...}
```

#### Mechanism 3 - Meta-learner

A simple regression model (multi-output Random Forest or small MLP) is
trained to predict optimal weights from meta-features:

```
optimal_weights = f(meta_features(domain))
```

For a new domain, we compute its meta-features and predict the weights -
no per-domain Optuna run needed.

### Validation protocol

- **Leave-one-domain-out cross-validation**: train meta-learner on N-1
  domains, evaluate on the held-out domain
- **Compare against baselines**:
  - Fixed expert weights (no learning)
  - Per-domain AutoML upper bound (impractical at runtime, gives the ceiling)
  - Direct transfer from most similar source domain
  - Average of source domain weights

### Expected outcomes

If H3 holds, the meta-learner closes most of the gap between direct transfer
and per-domain AutoML, at runtime cost of a single inference call.

---

## 6. System architecture

### High-level pipeline

```
┌──────────┐    ┌────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│  Topic   │ -> │ Search │ -> │ Quality │ -> │Summary  │ -> │Synthesis │ -> │  Trends  │
│  input   │    │ Agent  │    │ Filter  │    │  Agent  │    │  Agent   │    │  Agent   │
└──────────┘    └────────┘    └─────────┘    └─────────┘    └──────────┘    └──────────┘
                    |              |              |              |              |
                    v              v              v              v              v
                                       SHARED STATE (Pydantic)
```

### Component responsibilities

| Component | Input | Output | Tech |
|---|---|---|---|
| Search Agent | topic | raw articles | OpenAlex API |
| Quality Filter | raw articles | top-N scored articles | AutoML scorer |
| Summary Agent | top-N articles | structured summaries | Claude Haiku |
| Synthesis Agent | summaries | global synthesis | Claude Sonnet |
| Trend Agent | summaries + synthesis | trends + gaps | Claude Sonnet |
| Critic Agent (V2+) | synthesis | feedback (Reflexion) | Claude Sonnet |
| Judge Agent (V3) | multiple outputs | best/fused output | Claude Opus |

### Communication contract

All inter-agent communication is mediated by a **typed shared state**
(LangGraph TypedDict) with **Pydantic v2 schemas** for validation.

Key schemas (defined in `src/schemas.py`):
- `Article` - central data entity
- `QualityScore` - 4 sub-scores + final score
- `ArticleSummary` - structured (problem, method, dataset, results, limitations)
- `Synthesis` - overview, approaches, findings
- `TrendAnalysis` - trends, gaps, perspectives
- `CriticFeedback` - for Reflexion loop
- `AgentLog` - traces (tokens, cost, duration)

**Ownership rule**: Single Writer / Multiple Readers per state section.

---

## 7. Quality scoring methodology

### Four dimensions

The quality of a scientific article is scored on four dimensions, each in
[0, 1]:

| Dimension | What it captures | Data source |
|---|---|---|
| **Venue** | Journal/conference prestige | OpenAlex quartile + SJR (Phase 3+) |
| **Authors** | Researcher credibility | OpenAlex h-index (max across authors) |
| **Impact** | Citation impact normalized by age | OpenAlex citation count / age |
| **Relevance** | Topic match | Keyword matching (V1) → embeddings (later) |

### Combination formula

```
final_score = w_venue · venue_score
            + w_authors · authors_score
            + w_impact · impact_score
            + w_relevance · relevance_score
```

Where weights `w_*` are:
- **V1**: fixed expert-defined defaults (`{0.25, 0.20, 0.25, 0.30}`)
- **Phase 3+**: learned by AutoML (Optuna with NDCG@15 objective)

### AutoML setup (Phase 3)

- **Optimizer**: Optuna with TPE sampler
- **Search space**: 4 weights in [0, 1], constrained to sum = 1
- **Metric**: NDCG@15 (Normalized Discounted Cumulative Gain at K=15)
- **Labels**: combination of expert oracle + citation forward signal

---

## 8. Multi-LLM strategy

### Why two LLMs?

1. **Diversity**: different training data and biases reduce overconfidence
2. **Empirical comparison**: generates research data for the thesis
3. **Natural multi-agent pattern**: Best-of-N with truly diverse outputs

### Three operating modes

1. **Production (specialization)**: each task uses its best-fit model
2. **Benchmark**: A/B testing on `BENCHMARK_PAIRS`
3. **Ensemble V3**: both models run, Judge agent fuses or selects

### Cost optimizations

- **Prompt caching**: 90% off on cached system prompts
- **Batch API**: 50% off for asynchronous workloads
- **Model tiering**: Haiku for repetitive tasks, Sonnet for complex ones

### Estimated budget for full thesis experimentation

- 100 production runs: ~$21
- 50 benchmark comparisons: ~$21
- 200 evaluation runs across domains: ~$42
- **Total estimated**: $80-100

---

## 9. Evaluation protocol

### Quantitative metrics

| Metric | What it measures | When |
|---|---|---|
| NDCG@15 | Quality of article ranking | After Phase 3 (AutoML) |
| Tokens used | Cost efficiency | Every run |
| Latency | User experience | Every run |
| Coverage rate | % of relevant papers found | Per topic |

### Qualitative evaluation

For synthesis quality, use a **structured human evaluation grid**:

1. Completeness (is the synthesis missing important aspects?)
2. Coherence (does the structure make sense?)
3. Depth (is the analysis surface-level or substantive?)
4. Accuracy (any factual errors or misrepresentations?)
5. Novelty (does it reveal non-obvious patterns?)

Each criterion scored on a 1-5 Likert scale.

### Validation domains

To rigorously test cross-domain generalization (H2, H3), the system will be
evaluated on **at least 5 domains** spanning different fields, maturities,
and characteristics:

| # | Domain (suggested) | Purpose | Field characteristics |
|---|---|---|---|
| 1 | Fake news detection | Source domain (training) | Mature, NLP-heavy, high citation density |
| 2 | Recommender systems | In-distribution test | Mature, ML-heavy, similar to D1 |
| 3 | Medical imaging | Out-of-distribution test | Mature, vision/medical, different vocab |
| 4 | Quantum machine learning | Emerging field | Recent, sparse citations, fast-evolving |
| 5 | AI safety / alignment | Conceptual field | Recent, less empirical, debate-heavy |

The diversity is intentional: D1-D2 test in-distribution behavior, D3 tests
domain shift in vocabulary, D4 tests behavior under sparse data, D5 tests
behavior on conceptual rather than empirical content.

Each domain is evaluated on:
- Quality of top-15 articles selected (NDCG@15)
- Quality of generated synthesis (LLM-as-judge protocol)
- Identified trends (validated against curated lists from review papers)
- Cross-domain weight transfer behavior

---

## 10. Risks and mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| OpenAlex API changes/downtime | Low | High | Add arXiv as backup; cache results |
| LLM hallucinations in synthesis | High | Medium | Reflexion pattern (V2); cite sources |
| AutoML overfitting on training domain | Medium | High | Cross-domain validation; regularization |
| Budget exhaustion | Medium | Medium | Prompt caching; batch API; budget monitoring |
| Multi-agent coordination bugs | High | Medium | Strong typing (Pydantic); extensive tests |
| Memoir scope creep | Medium | High | Strict phase roadmap; weekly reviews |

---

## 11. Project timeline

### Months 1: Foundations (DONE / IN PROGRESS)

- ✅ Phase 0: Project setup
- ✅ Phase 1: OpenAlex source wrapper
- ✅ Phase 2: Quality scoring features
- 🔜 Phase 3: AutoML scorer with Optuna

### Month 2: Multi-agent V2

- Phase 4: LLM layer (Claude + GPT abstraction)
- Phase 4: Summarizer + Synthesizer
- Phase 5: LangGraph orchestration (V2)
- Phase 6: Reflexion pattern

### Month 3: Concurrent V3 + evaluation

- Phase 7: V3 concurrent agents (Best-of-N + Judge)
- Phase 8: Multi-domain evaluation campaign
- Statistical analysis of results

### Month 4: Memoire writing + finalization

- Final experiments
- Memoire writing (chapters: intro, related work, methodology,
  experiments, results, discussion, conclusion)
- Demo preparation
- Defense rehearsal

---

## 12. Deliverables

### Code

- Open-source repository on GitHub (suggested license: MIT)
- 56+ unit tests passing
- Demo scripts for each phase
- README with setup instructions

### Documentation

- This specification document (`PROJECT_SPEC.md`)
- `CLAUDE.md` for future maintainers/AI assistants
- Architectural diagrams
- API documentation

### Memoire

- ~80-100 pages
- Standard PFE structure (introduction, related work, methodology,
  experiments, results, discussion, conclusion, references)
- Figures, tables, evaluation results

### Demo

- Live demo with at least 3 different topics
- Performance dashboard (cost, latency, quality metrics)
- Comparison across V1/V2/V3 architectures

---

## 13. Open questions and decisions pending

(Updated as the project progresses)

- [ ] Final selection of 2 additional validation domains
- [ ] Strategy for h-index enrichment (cost vs benefit)
- [ ] Decision on Scimago CSV integration timing
- [ ] Embedding provider choice for relevance V2 (Voyage AI vs local)
- [ ] Final report format (Markdown / HTML / PDF)
- [ ] User interface choice (CLI / Streamlit / web)
- [ ] Hosting strategy for demo (HuggingFace Spaces? local?)

---

**Document version**: 1.0
**Last updated**: April 2026 (after Phase 2 completion)
