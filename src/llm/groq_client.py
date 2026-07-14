"""Groq LLM client — wraps OpenAI-compatible Groq API.

Groq provides ultra-fast inference for open-source models (Llama, Mixtral…).
Its API is OpenAI-compatible, so we reuse OpenAI's SDK with a custom base_url.

Key difference vs OpenAIClient:
    Llama models on Groq do NOT support OpenAI's strict JSON-schema mode
    (`response_format={"type":"json_schema","strict":true}`).
    We fall back to json_object mode + schema-in-prompt + Pydantic validation.

Available models (May 2026):
    llama-3.1-8b-instant    — fastest, cheapest  (~$0.05/$0.08 per 1M tokens)
    llama-3.3-70b-versatile — best quality        (~$0.59/$0.79 per 1M tokens)
    mixtral-8x7b-32768      — 32K context window
    gemma2-9b-it            — Google Gemma 2

Usage:
    from src.llm.groq_client import GroqClient
    llm = GroqClient(model="llama-3.1-8b-instant")
    resp = llm.chat("You are helpful.", [Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import GROQ_API_KEY
from src.llm.base import (
    LLMResponse,
    LLMStructuredOutputError,
    Message,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq free-tier limits (May 2026):
#   llama-3.1-8b-instant    : 30 RPM, 6 000 TPM
#   llama-3.3-70b-versatile : 30 RPM, 6 000 TPM
#
# TPM is the binding constraint for ProTeGi, not RPM.
# Each summarization call uses ~1 000–1 500 tokens (system prompt + article + output).
# At 6 000 TPM cap:
#   5 RPM (12s/req) → 5 × 1 500 = 7 500 tokens/min  ← EXCEEDS the 6 000 TPM cap
#   3 RPM (20s/req) → 3 × 1 500 = 4 500 tokens/min  ← safe margin under 6 000 TPM
#
# 3 RPM is the right default: it keeps us ~25% under the TPM cap even with
# long articles. ProTeGi runs take a few minutes longer but never hit 429.
GROQ_DEFAULT_RPM = 3

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class _RateLimiter:
    """Global rate limiter for Groq — enforces minimum interval + 429 backoff.

    Groq enforces TWO limits simultaneously:
        - RPM  (requests per minute)   — e.g. 30 RPM free tier
        - TPM  (tokens per minute)     — e.g. 6 000 TPM free tier ← often binding

    With a schema-injected system prompt, each ProTeGi call uses ~1 000–1 500
    tokens, so the TPM cap kicks in at only 4–6 requests/minute.

    Strategy:
        1. Proactive: enforce a minimum interval between requests (RPM floor).
        2. Reactive : when a 429 is received, read the `retry-after` value from
           the exception and force ALL pending calls to wait that long before
           retrying.  This is the critical fix — without it, other requests in
           the loop keep firing while the failed request is waiting to retry.
        3. Adaptive: when the backoff is long (> 60s), Groq is enforcing a
           longer-window limit (5-10 min window or daily token cap).  We
           automatically slow down to 1 RPM (60s/req) after recovery so that
           the very first request after the wait does not immediately trigger
           another long-window 429.
    """

    # Backoff threshold above which we consider Groq is enforcing a longer-
    # window (multi-minute) limit rather than the per-minute TPM cap.
    _LONG_BACKOFF_THRESHOLD = 60.0   # seconds
    _SLOW_INTERVAL           = 60.0  # seconds/req = 1 RPM

    def __init__(self, requests_per_minute: int) -> None:
        self._base_interval = 60.0 / max(requests_per_minute, 1)
        self._min_interval  = self._base_interval
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def reconfigure(self, requests_per_minute: int) -> None:
        """Update the RPM cap (takes effect on the next call)."""
        with self._lock:
            self._base_interval = 60.0 / max(requests_per_minute, 1)
            self._min_interval  = self._base_interval

    def wait(self) -> None:
        """Block until it is safe to fire the next request."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_for = self._min_interval - elapsed
                logger.debug("Groq rate-limiter: sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
            self._last_call = time.monotonic()

    def backoff(self, seconds: float) -> None:
        """Force all subsequent calls to wait at least `seconds` from now.

        Called immediately when a 429 is received so that every other pending
        request in the pipeline also pauses — not just the failed one.

        Adaptive behaviour: when `seconds > _LONG_BACKOFF_THRESHOLD` (default
        60s) the limiter switches to 1 RPM (60s/req) for subsequent calls.
        This prevents the pattern:
            backoff 256s → one success → 429 again 20s later
        because Groq is still within a longer-window (5-min or daily) limit.
        """
        with self._lock:
            # Push _last_call into the future so the next wait() sleeps >= seconds
            self._last_call = time.monotonic() + seconds

            if seconds > self._LONG_BACKOFF_THRESHOLD:
                # Long backoff = Groq long-window limit hit.
                # Slow down to 1 RPM so the first request after recovery doesn't
                # immediately trigger another long-window 429.
                new_interval = max(self._min_interval, self._SLOW_INTERVAL)
                if new_interval > self._min_interval:
                    self._min_interval = new_interval
                    logger.warning(
                        "Groq long backoff (%.0fs > %.0fs) → adaptive slowdown "
                        "to 1 RPM (%.0fs/req) after recovery",
                        seconds, self._LONG_BACKOFF_THRESHOLD, self._min_interval,
                    )
                else:
                    logger.warning(
                        "Groq 429 → global backoff %.1fs (all requests paused)", seconds
                    )
            else:
                logger.warning(
                    "Groq 429 → global backoff %.1fs (all requests paused)", seconds
                )


# ── Global shared rate limiter ────────────────────────────────────────────────
# Groq enforces limits at the API-key level, not per SDK instance.
# All GroqClient instances MUST share ONE limiter so they coordinate.
# Default: 3 RPM (20s/req) — safe under the 6 000 TPM free-tier cap.
# After a long backoff (>60s) the limiter self-adapts to 1 RPM (60s/req).
_GLOBAL_GROQ_LIMITER = _RateLimiter(GROQ_DEFAULT_RPM)


class GroqClient:
    """LLMClient backed by Groq's inference API (OpenAI-compatible).

    Structured outputs use JSON-mode + schema-in-prompt because Llama models
    do not support OpenAI's strict JSON schema format.

    Built-in rate limiter prevents 429 errors on the free tier (default: 25 RPM).
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        requests_per_minute: int = GROQ_DEFAULT_RPM,
    ):
        """Init.

        Args:
            model               : Groq model ID (default: llama-3.1-8b-instant).
            api_key             : optional override; defaults to env GROQ_API_KEY.
            requests_per_minute : rate limit cap (default: 25, safe for free tier).
                                  Set higher if you have a paid Groq plan.
        """
        key = api_key or GROQ_API_KEY
        if not key:
            raise ValueError("GROQ_API_KEY is missing — set it in .env")
        # max_retries=0 : disable the OpenAI SDK's built-in retry so that
        # every 429 surfaces as RateLimitError to our _call() method.
        # Without this, the SDK silently retries after its own delay and our
        # _GLOBAL_GROQ_LIMITER.backoff() is never called — other GroqClient
        # instances keep firing and hitting 429 in a loop.
        self._client = openai.OpenAI(
            api_key=key,
            base_url=GROQ_BASE_URL,
            max_retries=0,
        )
        self._model = model
        # All instances share the same global limiter (Groq limits per API key).
        # Reconfigure it if the caller wants a custom RPM.
        if requests_per_minute != GROQ_DEFAULT_RPM:
            _GLOBAL_GROQ_LIMITER.reconfigure(requests_per_minute)
        logger.debug(
            "GroqClient ready: model=%s rpm=%d (%.2fs/req)",
            model, requests_per_minute, 60.0 / requests_per_minute,
        )

    @property
    def provider(self) -> Literal["groq"]:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Plain text completion."""
        raw = self._call(
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )
        text = raw.choices[0].message.content or ""
        return self._build_response(raw, content=text)

    def chat_structured(
        self,
        system: str,
        messages: list[Message],
        schema: type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_attempts: int = 3,
    ) -> tuple[T, LLMResponse]:
        """Structured output via JSON-mode + schema injected in the system prompt.

        Groq/Llama does not support strict JSON schema (OpenAI's
        `response_format.type = "json_schema"`), so we:
          1. Append the JSON schema to the system prompt as instructions
          2. Use `response_format={"type": "json_object"}` (supported)
          3. Parse + validate the response with Pydantic

        Retried up to `max_attempts` times on LLMStructuredOutputError (null
        fields, empty JSON, invalid JSON).  Transient network/rate errors are
        handled separately by the tenacity retry in _call().
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system}\n\n"
            f"CRITICAL INSTRUCTION: Your response MUST be a valid JSON object "
            f"that exactly matches this schema. Output ONLY the JSON, no markdown, "
            f"no explanation, no code fences.\n\n"
            f"Required JSON schema:\n{schema_json}"
        )

        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, max_attempts + 1):
            try:
                raw = self._attempt_call(
                    enhanced_system=enhanced_system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = (raw.choices[0].message.content or "").strip()
                text = _extract_json(text)

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise LLMStructuredOutputError(
                        f"Groq/{self._model} returned invalid JSON: {text[:300]!r}"
                    ) from exc

                try:
                    parsed = schema.model_validate(payload)
                except ValidationError as exc:
                    raise LLMStructuredOutputError(
                        f"Groq/{self._model} output failed Pydantic validation: {exc}\n"
                        f"Raw JSON: {text[:300]}"
                    ) from exc

                return parsed, self._build_response(raw, content=text)

            except LLMStructuredOutputError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    logger.warning(
                        "chat_structured attempt %d/%d failed for %s: %s — retrying",
                        attempt, max_attempts, self._model, exc,
                    )
        raise last_exc

    def _attempt_call(self, enhanced_system, messages, temperature, max_tokens):
        """Single structured-output API call, with json_object → plain fallback."""
        try:
            return self._call(
                system=enhanced_system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.BadRequestError as exc:
            logger.debug("json_object mode unsupported (%s), falling back to plain", exc)
            return self._call(
                system=enhanced_system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=None,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    def _call(
        self,
        system: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ):
        # Proactive: wait if we're sending requests too fast
        _GLOBAL_GROQ_LIMITER.wait()

        api_messages = [{"role": "system", "content": system}] if system else []
        api_messages.extend(m.model_dump() for m in messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            return self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            # Reactive: parse retry-after and force a global pause so ALL
            # other requests also stop — not just this one.
            # Default = 65s (> 60s TPM window) so the token budget fully resets
            # before any retry, preventing cascading 429s during beam search.
            wait_sec = _parse_retry_after(exc) or 65.0
            _GLOBAL_GROQ_LIMITER.backoff(wait_sec)
            raise  # let tenacity retry after the backoff

    def _build_response(self, raw, content: str) -> LLMResponse:
        usage = raw.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost = estimate_cost_usd(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider="openai",   # use "openai" to satisfy LLMResponse Literal
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


# ============================================================
# Helper: parse retry-after from a RateLimitError
# ============================================================


def _parse_retry_after(exc: openai.RateLimitError) -> float | None:
    """Extract the retry-after wait time (seconds) from a Groq 429 response.

    Groq sends one of:
        - `Retry-After: 30`              (integer seconds)
        - message containing "try again in 25.3s"
    Returns None if we can't parse it (caller should use a safe default).
    """
    import re

    # Try the response headers first
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            return float(ra)
    except Exception:
        pass

    # Fall back to parsing the error message
    msg = str(exc)
    match = re.search(r"(\d+(?:\.\d+)?)\s*s", msg)
    if match:
        return float(match.group(1)) + 2.0   # add 2s safety margin

    return None


# ============================================================
# Helper: extract JSON from model output
# ============================================================


def _extract_json(text: str) -> str:
    """Strip markdown code fences and isolate the JSON object.

    Some models wrap their JSON in ```json ... ``` even when instructed not to.
    This function handles:
        - ```json { ... } ```
        - ``` { ... } ```
        - Plain { ... }
    """
    # Remove fenced code block wrappers
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
                break

    # Find the first { and last } to isolate the JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx: end_idx + 1]

    return text.strip()
