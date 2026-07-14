"""Tests for the LLM layer.

We never call the real APIs in tests; ClaudeClient and OpenAIClient are
exercised against mocked SDK responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.llm.base import (
    LLMResponse,
    LLMStructuredOutputError,
    Message,
    estimate_cost_usd,
)
from src.llm.claude_client import ClaudeClient
from src.llm.openai_client import OpenAIClient, _to_strict_json_schema


# ============================================================
# Fixture schema
# ============================================================


class DummyOut(BaseModel):
    answer: str
    confidence: float


# ============================================================
# Cost estimator
# ============================================================


class TestEstimateCost:
    def test_known_model_basic(self):
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        # 1000 input * 0.15/1M + 500 output * 0.60/1M = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045, rel=1e-3)

    def test_unknown_model_returns_zero(self):
        assert estimate_cost_usd("not-a-real-model", 100, 100) == 0.0

    def test_cache_read_is_cheaper(self):
        no_cache = estimate_cost_usd(
            "claude-haiku-4-5", input_tokens=1000, output_tokens=0
        )
        with_cache = estimate_cost_usd(
            "claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=0,
            cache_read_tokens=900,
        )
        assert with_cache < no_cache

    def test_cache_creation_more_expensive_than_fresh(self):
        fresh = estimate_cost_usd("claude-sonnet-4-6", 1000, 0)
        all_creation = estimate_cost_usd(
            "claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=0,
            cache_creation_tokens=1000,
        )
        assert all_creation > fresh


# ============================================================
# JSON Schema tightening for OpenAI strict mode
# ============================================================


class TestStrictJsonSchema:
    def test_adds_additional_properties_false(self):
        schema = _to_strict_json_schema(DummyOut)
        assert schema["additionalProperties"] is False

    def test_all_props_required(self):
        schema = _to_strict_json_schema(DummyOut)
        assert set(schema["required"]) == {"answer", "confidence"}


# ============================================================
# ClaudeClient
# ============================================================


def _fake_anthropic_text(text: str = "hello"):
    """Build a fake Anthropic Message-like response with a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    raw = MagicMock()
    raw.content = [block]
    raw.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return raw


def _fake_anthropic_tool_use(tool_name: str, payload: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = payload
    raw = MagicMock()
    raw.content = [block]
    raw.usage = MagicMock(
        input_tokens=20,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return raw


class TestClaudeClient:
    def test_chat_returns_response_with_usage(self, monkeypatch):
        client = ClaudeClient(model="claude-haiku-4-5", api_key="sk-ant-test")
        client._client.messages.create = MagicMock(  # type: ignore
            return_value=_fake_anthropic_text("hi there")
        )

        resp = client.chat(system="sys", messages=[Message(role="user", content="hi")])
        assert resp.content == "hi there"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.provider == "anthropic"
        assert resp.cost_usd > 0

    def test_chat_structured_extracts_tool_input(self):
        client = ClaudeClient(model="claude-haiku-4-5", api_key="sk-ant-test")
        client._client.messages.create = MagicMock(  # type: ignore
            return_value=_fake_anthropic_tool_use(
                "emit_dummyout", {"answer": "yes", "confidence": 0.9}
            )
        )

        parsed, resp = client.chat_structured(
            system="sys",
            messages=[Message(role="user", content="q")],
            schema=DummyOut,
        )
        assert isinstance(parsed, DummyOut)
        assert parsed.answer == "yes"
        assert parsed.confidence == 0.9
        assert resp.provider == "anthropic"

    def test_chat_structured_raises_on_missing_tool(self):
        client = ClaudeClient(model="claude-haiku-4-5", api_key="sk-ant-test")
        # Returns text only, no tool_use block.
        client._client.messages.create = MagicMock(  # type: ignore
            return_value=_fake_anthropic_text("oops")
        )

        with pytest.raises(LLMStructuredOutputError):
            client.chat_structured(
                system="sys",
                messages=[Message(role="user", content="q")],
                schema=DummyOut,
            )

    def test_cache_control_added_when_enabled(self):
        client = ClaudeClient(
            model="claude-haiku-4-5", api_key="sk-ant-test", cache_system=True
        )
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _fake_anthropic_text("ok")

        client._client.messages.create = MagicMock(side_effect=fake_create)  # type: ignore
        client.chat(system="my system", messages=[Message(role="user", content="x")])

        assert isinstance(captured["system"], list)
        assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_disabled_uses_plain_string(self):
        client = ClaudeClient(
            model="claude-haiku-4-5", api_key="sk-ant-test", cache_system=False
        )
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _fake_anthropic_text("ok")

        client._client.messages.create = MagicMock(side_effect=fake_create)  # type: ignore
        client.chat(system="my system", messages=[Message(role="user", content="x")])

        assert captured["system"] == "my system"


# ============================================================
# OpenAIClient
# ============================================================


def _fake_openai_response(content: str):
    raw = MagicMock()
    raw.choices = [MagicMock(message=MagicMock(content=content))]
    raw.usage = MagicMock(prompt_tokens=12, completion_tokens=6)
    return raw


class TestOpenAIClient:
    def test_chat_returns_text(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
        client._client.chat.completions.create = MagicMock(  # type: ignore
            return_value=_fake_openai_response("hello world")
        )
        resp = client.chat(system="s", messages=[Message(role="user", content="hi")])
        assert resp.content == "hello world"
        assert resp.input_tokens == 12
        assert resp.provider == "openai"

    def test_chat_structured_parses_json(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
        client._client.chat.completions.create = MagicMock(  # type: ignore
            return_value=_fake_openai_response('{"answer":"yes","confidence":0.7}')
        )
        parsed, _ = client.chat_structured(
            system="s",
            messages=[Message(role="user", content="q")],
            schema=DummyOut,
        )
        assert parsed.answer == "yes"
        assert parsed.confidence == 0.7

    def test_chat_structured_raises_on_invalid_json(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
        client._client.chat.completions.create = MagicMock(  # type: ignore
            return_value=_fake_openai_response("not json")
        )
        with pytest.raises(LLMStructuredOutputError):
            client.chat_structured(
                system="s",
                messages=[Message(role="user", content="q")],
                schema=DummyOut,
            )
