"""Centralized configuration for the Scientific Watch Agent.

Loads environment variables from .env file and exposes them as typed constants.
Centralizing config here makes it easy to mock in tests and to change defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# API Keys & credentials
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
NEBIUS_API_KEY    = os.getenv("NEBIUS_API_KEY", "")   # Nebius AI Studio
OPENALEX_EMAIL    = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_API_KEY  = os.getenv("OPENALEX_API_KEY", "")
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")

# ============================================================
# Multi-LLM strategy: which model handles which task
# Strategy: specialization (production) + ensemble V3 + benchmark mode
# ============================================================

# Format: (provider, model_name)
TASK_MODELS = {
    "query_expansion":  ("openai",    "gpt-4o-mini"),       # cheap, fast
    "summarize":        ("nebius", "Qwen/Qwen3-30B-A3B-Instruct-2507"),  # repetitive, cheap
    "synthesize":       ("nebius", "deepseek-ai/DeepSeek-V4-Pro"),          # large context
    "trend_analysis":   ("nebius", "deepseek-ai/DeepSeek-V4-Pro"),          # complex reasoning
    "critic":           ("nebius", "deepseek-ai/DeepSeek-V4-Pro"),  # for Reflexion
    "judge":            ("nebius", "deepseek-ai/DeepSeek-V4-Pro"),  # same model = cheap + coherent
}

# Benchmark pairs: which (taskA, taskB) to compare in benchmark mode
BENCHMARK_PAIRS = {
    "summarize":      ("anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"),
    "synthesize":     ("anthropic/claude-sonnet-4-6", "openai/gpt-4o"),
    "trend_analysis": ("anthropic/claude-sonnet-4-6", "openai/gpt-4o"),
}

# Approximate pricing per 1M tokens (USD), for cost tracking in AgentLog
# Update if pricing changes; cf. /pricing pages of each provider
MODEL_PRICING = {
    "claude-haiku-4-5":   {"input": 1.00,  "output":  5.00},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":    {"input": 5.00,  "output": 25.00},
    "gpt-4o-mini":              {"input": 0.15,  "output":  0.60},
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    # Groq models (ultra-fast, cheap)
    "llama-3.1-8b-instant":     {"input": 0.05,  "output":  0.08},
    "llama-3.3-70b-versatile":  {"input": 0.59,  "output":  0.79},
    "llama3-8b-8192":           {"input": 0.05,  "output":  0.08},
    "mixtral-8x7b-32768":       {"input": 0.24,  "output":  0.24},
    # Nebius AI Studio — available models (May 2026)
    "deepseek-ai/DeepSeek-V3.2":                       {"input": 0.14,  "output":  0.28},
    "deepseek-ai/DeepSeek-V3.2-fast":                  {"input": 0.10,  "output":  0.20},
    "deepseek-ai/DeepSeek-V4-Pro":                     {"input": 0.27,  "output":  1.10},
    "meta-llama/Llama-3.3-70B-Instruct":               {"input": 0.12,  "output":  0.30},
    "Qwen/Qwen3-235B-A22B-Thinking-2507-fast":         {"input": 0.20,  "output":  0.60},
    "Qwen/Qwen3-32B":                                  {"input": 0.10,  "output":  0.30},
}

# ============================================================
# OpenAlex defaults
# ============================================================

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_DEFAULT_PER_PAGE = 25  # Max 100; 25 is a good default
OPENALEX_MAX_RETRIES = 5
OPENALEX_TIMEOUT = 30  # seconds

# ============================================================
# Scoring defaults (will be overridden by AutoML in Phase 3)
# ============================================================

# Level 3: 6-feature default weights (venue, authors, impact, velocity, recency, relevance).
# Velocity and recency share the weight previously held by impact alone.
# Optuna will learn the optimal split per domain.
DEFAULT_WEIGHTS = {
    "venue":     0.15,
    "authors":   0.15,
    "impact":    0.20,
    "velocity":  0.15,   # Level 3 — citation momentum (linear, saturation 20 cpy)
    "recency":   0.15,   # Level 3 — publication freshness (exp decay, half-life 3y)
    "relevance": 0.20,   # V2 semantic (sentence-transformers cosine similarity)
}

DEFAULT_TOP_N = 30  # Top-N articles after filtering

# Minimum relevance score required to keep an article after scoring.
# Articles below this threshold are EXCLUDED from top_articles regardless
# of their venue / author / impact scores, preventing off-topic papers
# (e.g. pure RNN/LSTM papers when the topic is about attention mechanisms)
# from slipping through on the strength of citation counts alone.
# Range: [0, 1].  0.20 is a good default; lower it for broad exploratory
# searches, raise it for very specific queries.
MIN_RELEVANCE_SCORE = 0.20

# Earliest publication year accepted in searches.
# Articles older than this are excluded at the OpenAlex query level — simpler
# and cheaper than a recency score: old papers never enter the pipeline.
# Adjust per domain: fast-moving fields (LLMs) → 2023, mature fields → 2018.
DEFAULT_FROM_YEAR = 2023  # covers 2022-2026 by default

# ============================================================
# ReAct pattern (QueryExpander — hybrid with Reflexion)
# ============================================================

# Maximum number of search probes in the ReAct QueryExpander loop.
# Each probe = 1 OpenAlex request + 1 LLM call.
# 4 is a good balance: covers 3-4 distinct angles before stopping.
MAX_REACT_ITERATIONS = 4

# Number of articles fetched per probe (small — just enough to observe
# concepts and titles; the Searcher does the full fetch afterwards).
REACT_PROBE_N = 5

# ============================================================
# Reflexion pattern (Phase 6)
# ============================================================

# Maximum number of Synthesizer revision cycles.
# 0 = Reflexion disabled (single-shot, same as before Phase 6).
# 3 = recommended: good quality/cost tradeoff.
MAX_REFLEXION_ITERATIONS = 3

# Minimum quality level accepted without revision.
# "good" means the Critic must rate the synthesis at least "good" to stop.
# "acceptable" is more lenient (fewer API calls on average).
REFLEXION_MIN_QUALITY = "good"   # "poor" | "acceptable" | "good" | "excellent"

# ============================================================
# Email delivery (optional — used by export_to_pdf.py --email)
# Gmail requires an App Password (NOT your regular password):
#   Google Account → Security → 2-Step Verification → App passwords
#   Select "Mail" → generate a 16-char password → put it here
# ============================================================

EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")    # your Gmail address
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")  # Gmail App Password (16 chars)
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")  # default recipient (optional)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587   # STARTTLS

# ============================================================
# ============================================================
# Level 3 feature calibration
# ============================================================

# velocity_score: citations/year at which score saturates to 1.0.
# Lower than impact_score's 50 → more sensitive to fast-rising papers.
VELOCITY_SATURATION = 20.0

# recency_score: publication age (years) at which score = 0.5.
# 3 years is appropriate for fast-moving CS/AI fields.
RECENCY_HALF_LIFE = 3.0

# ============================================================
# Relevance scoring V2 — semantic embeddings (sentence-transformers)
# ============================================================

# Model used for dense-vector relevance scoring.
# all-MiniLM-L6-v2 : fast (~10 ms/article CPU), 80 MB, 384-dim, good quality.
# allenai-specter  : trained on scientific citations, 430 MB, stronger for
#                    citation-prediction tasks but overkill for relevance.
# all-mpnet-base-v2: slightly better quality, 420 MB.
RELEVANCE_V2_MODEL = "all-MiniLM-L6-v2"

# Set True to use semantic (V2) relevance. Set False to force V1 keywords.
# If sentence-transformers is not installed, V1 is used automatically regardless.
USE_SEMANTIC_RELEVANCE = True

# Max embeddings cached in memory per process (LRU eviction).
# Each vector: 384 dims × 4 bytes = 1.5 KB → 512 entries ≈ 0.75 MB total.
RELEVANCE_CACHE_SIZE = 512

# ============================================================
# Phase 3 — AutoML (Optuna)
# ============================================================

# Number of Optuna trials per domain optimisation run.
# With only 4 weight parameters, 150 trials is typically enough for the TPE
# sampler to converge; each trial takes <1 ms (pure in-memory scoring).
OPTUNA_N_TRIALS = 150

# Wall-clock timeout per study (seconds). Prevents runaway optimisations.
# Set to 0 to disable the timeout and rely solely on n_trials.
OPTUNA_TIMEOUT = 120   # 2 minutes max per domain

# Minimum relative NDCG@15 improvement over DEFAULT_WEIGHTS to consider
# the optimised weights useful. Below this threshold, DEFAULT_WEIGHTS are kept.
OPTUNA_MIN_IMPROVEMENT = 0.01   # 1 % relative gain

# Earliest year accepted in the oracle gold-standard corpus.
# Older papers are excluded because citation counts are not comparable.
ORACLE_FROM_YEAR = 2018

# ============================================================
# Paths
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ORACLE_DIR  = DATA_DIR / "oracle"    # domain configs + gold articles
WEIGHTS_DIR = DATA_DIR / "weights"   # learned weights per domain (JSON)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
ORACLE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
