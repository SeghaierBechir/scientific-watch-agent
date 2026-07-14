"""Nebius AI Studio LLM client — wraps OpenAI-compatible Nebius API.

Nebius AI Studio hosts popular open-source models (DeepSeek, Llama, Qwen…)
via an OpenAI-compatible REST API.  We reuse OpenAI's SDK with a custom
base_url, exactly like GroqClient.

Key differences vs GroqClient:
    - No aggressive rate limiting (Nebius does not enforce the same tight TPM
      cap as Groq's free tier; a light exponential retry is sufficient).
    - DeepSeek-R1 wraps its reasoning in <think>...</think> blocks before
      the final answer.  We strip those automatically so callers always get
      clean text / valid JSON.

Available models on Nebius AI Studio (May 2026):
    deepseek-ai/DeepSeek-V3           ~$0.14/$0.28 per 1M tokens — strong general model
    deepseek-ai/DeepSeek-R1           ~$0.55/$2.19 per 1M tokens — reasoning model
    meta-llama/Llama-3.3-70B-Instruct ~$0.12/$0.30 per 1M tokens — strong open-source
    Qwen/Qwen2.5-72B-Instruct         ~$0.12/$0.30 per 1M tokens — strong on structured tasks

Structured output strategy (same as GroqClient):
    1. Append JSON schema to system prompt.
    2. Call with response_format={"type": "json_object"} (supported by most models).
    3. Fallback to plain text if json_object mode returns BadRequestError.
    4. Parse + validate with Pydantic; retry up to max_attempts on failures.

Usage:
    from src.llm.nebius_client import NebiusClient
    llm = NebiusClient(model="deepseek-ai/DeepSeek-V3")
    resp = llm.chat("You are helpful.", [Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import NEBIUS_API_KEY
from src.llm.base import (
    LLMResponse,
    LLMStructuredOutputError,
    Message,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"

# DeepSeek-R1 wraps its chain-of-thought in <think>...</think> blocks placed
# BEFORE the final answer.  We remove them so the caller receives only the
# clean response (or valid JSON for structured calls).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class NebiusClient:
    """LLMClient backed by Nebius AI Studio (OpenAI-compatible).

    Structured outputs use JSON-mode + schema-in-system-prompt because
    open-source models do not support OpenAI's strict JSON-schema format.
    DeepSeek-R1 <think> reasoning blocks are stripped automatically.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        api_key: str | None = None,
    ):
        """Init.

        Args:
            model   : Nebius model ID (default: deepseek-ai/DeepSeek-V3).
            api_key : optional override; defaults to env NEBIUS_API_KEY.
        """
        key = api_key or NEBIUS_API_KEY
        if not key:
            raise ValueError("NEBIUS_API_KEY is missing — set it in .env")
        # max_retries=0: surface every 429 to our tenacity retry so the caller
        # can handle rate-limit errors explicitly if needed.
        self._client = openai.OpenAI(
            api_key=key,
            base_url=NEBIUS_BASE_URL,
            max_retries=0,
        )
        self._model = model
        logger.debug("NebiusClient ready: model=%s", model)

    @property
    def provider(self) -> Literal["openai"]:
        # Nebius is OpenAI-compatible — returning "openai" keeps LLMResponse
        # provider valid without a schema change.
        return "openai"

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
        text = _strip_think(raw.choices[0].message.content or "")
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

        Same strategy as GroqClient:
          1. Append the JSON schema to the system prompt.
          2. Call with response_format={"type": "json_object"}.
          3. Strip DeepSeek-R1 <think> blocks.
          4. Parse + validate with Pydantic.
          5. Retry up to max_attempts on LLMStructuredOutputError.
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
                # Strip R1 think blocks, then isolate the JSON object
                text = _strip_think((raw.choices[0].message.content or "").strip())
                text = _extract_json(text)

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise LLMStructuredOutputError(
                        f"Nebius/{self._model} returned invalid JSON: {text[:300]!r}"
                    ) from exc

                try:
                    parsed = schema.model_validate(payload)
                except ValidationError as exc:
                    raise LLMStructuredOutputError(
                        f"Nebius/{self._model} output failed Pydantic validation: {exc}\n"
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
            logger.debug(
                "json_object mode unsupported for %s (%s), falling back to plain",
                self._model, exc,
            )
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
        stop=stop_after_attempt(5),
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

        return self._client.chat.completions.create(**kwargs)

    def _build_response(self, raw, content: str) -> LLMResponse:
        usage = raw.usage
        input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost = estimate_cost_usd(
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


# ============================================================
# Helpers
# ============================================================


def _strip_think(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> reasoning blocks.

    R1 places its chain-of-thought between <think> tags before the final
    answer.  Stripping them ensures callers receive clean text or valid JSON.
    """
    return _THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> str:
    """Strip markdown code fences and isolate the JSON object.

    Handles:
        - ```json { ... } ```
        - ``` { ... } ```
        - Plain { ... }
    """
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
                break

    start_idx = text.find("{")
    end_idx   = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx: end_idx + 1]

    return text.strip()
