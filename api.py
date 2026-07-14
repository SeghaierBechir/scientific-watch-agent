"""Scientific Watch Agent — FastAPI endpoint.

Wraps the LangGraph pipeline as an HTTP API compatible with Nebius Serverless
Containers. The pipeline is synchronous (LangGraph invoke) and runs in a thread
pool so the event loop stays free.

Endpoints:
    GET  /              — API metadata
    GET  /health        — liveness probe
    POST /watch         — run the full pipeline (60-120s)

Usage (local):
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload

Usage (Docker):
    docker build -t scientific-watch-agent .
    docker run -p 8080:8080 \
        -e NEBIUS_API_KEY=... \
        -e OPENAI_API_KEY=... \
        -e OPENALEX_EMAIL=... \
        scientific-watch-agent
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Scientific Watch Agent",
    description="Automated scientific literature monitoring with cross-domain meta-learning.",
    version="1.0.0",
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class WatchRequest(BaseModel):
    topic: str = Field(..., description="Research topic, e.g. 'graph neural networks'")
    n_raw: int = Field(30, ge=5, le=200, description="Papers to fetch from OpenAlex")
    top_n: int = Field(10, ge=1, le=30,  description="Papers to keep after quality scoring")
    from_year: int | None = Field(None,  description="Earliest publication year (default from config)")
    narrative_mode: bool = Field(False,  description="Prose summaries instead of structured fields")


class AgentStats(BaseModel):
    papers_fetched: int
    papers_selected: int
    reflexion_iterations: int
    total_tokens: int
    cost_usd: float
    errors: list[str]
    elapsed_seconds: float


class WatchResponse(BaseModel):
    topic: str
    synthesis: dict[str, Any] | None
    trend_analysis: dict[str, Any] | None
    summaries: list[dict[str, Any]]
    top_articles: list[dict[str, Any]]
    stats: AgentStats


# ── Helpers ────────────────────────────────────────────────────────────────────

def _serialize(obj: Any) -> Any:
    """Recursively serialize Pydantic v2 objects to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _build_response(state: dict, topic: str, elapsed: float) -> WatchResponse:
    logs = state.get("logs", [])
    total_tokens = sum(getattr(l, "tokens_used", 0) or 0 for l in logs)
    cost_usd = sum(getattr(l, "cost_usd", 0) or 0 for l in logs)

    summaries = state.get("summaries") or state.get("narrative_summaries") or []

    return WatchResponse(
        topic=topic,
        synthesis=_serialize(state.get("synthesis")),
        trend_analysis=_serialize(state.get("trend_analysis")),
        summaries=[_serialize(s) for s in summaries],
        top_articles=[_serialize(a) for a in state.get("top_articles", [])],
        stats=AgentStats(
            papers_fetched=len(state.get("raw_articles", [])),
            papers_selected=len(state.get("top_articles", [])),
            reflexion_iterations=state.get("synthesis_iteration", 0),
            total_tokens=total_tokens,
            cost_usd=round(cost_usd, 6),
            errors=state.get("errors", []),
            elapsed_seconds=round(elapsed, 2),
        ),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {
        "name": "Scientific Watch Agent",
        "version": "1.0.0",
        "description": "Automated literature monitoring powered by Nebius AI Endpoints.",
        "endpoints": {
            "POST /watch": "Run the full pipeline on a research topic",
            "GET /health": "Liveness probe",
        },
        "powered_by": ["Nebius DeepSeek-V3", "Nebius Qwen3-30B", "LangGraph", "OpenAlex"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/watch", response_model=WatchResponse)
async def watch(req: WatchRequest) -> WatchResponse:
    """Run the full 7-agent pipeline on a research topic.

    The pipeline typically takes 60–120 seconds:
    - QueryExpander (ReAct, 4 iterations)
    - Searcher (multi-query OpenAlex fetch)
    - QualityCritic (6-feature scoring)
    - Summarizer ×top_n (Nebius Qwen3)
    - Synthesizer + Critic (Reflexion loop, max 3 iterations, Nebius DeepSeek-V3)
    - TrendAnalyst (Nebius DeepSeek-V3)
    """
    try:
        from src.agents.graph import run_pipeline
        from src.config import DEFAULT_FROM_YEAR
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline import error: {exc}")

    from_year = req.from_year or DEFAULT_FROM_YEAR

    logger.info(
        "POST /watch topic=%r n_raw=%d top_n=%d from_year=%d",
        req.topic, req.n_raw, req.top_n, from_year,
    )

    t0 = time.time()
    try:
        # run_pipeline is synchronous; offload to thread pool so the event loop
        # stays free for health checks and concurrent requests.
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                req.topic,
                n_raw=req.n_raw,
                top_n=req.top_n,
                from_year=from_year,
                narrative_mode=req.narrative_mode,
            ),
        )
    except Exception as exc:
        logger.exception("Pipeline failed for topic=%r", req.topic)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - t0
    logger.info("Pipeline done in %.1fs for topic=%r", elapsed, req.topic)

    return _build_response(state, req.topic, elapsed)
