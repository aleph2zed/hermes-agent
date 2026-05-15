"""Tests for agent/task_router.py"""

import pytest
from agent.task_router import (
    Tier,
    RouteDecision,
    classify_turn_complexity,
    find_model_for_tier,
    route_turn,
)


class TestClassifyTurnComplexity:
    """Test turn complexity classification."""

    def test_short_prompt_is_fast(self):
        result = classify_turn_complexity("hi", context_pct=0.0)
        assert result.tier == Tier.FAST

    def test_slash_command_is_fast(self):
        result = classify_turn_complexity("/help", context_pct=0.0)
        assert result.tier == Tier.FAST

    def test_standard_prompt(self):
        result = classify_turn_complexity(
            "Can you help me fix this bug in the authentication code?",
            context_pct=0.1,
        )
        assert result.tier == Tier.STANDARD

    def test_deep_signal_think_carefully(self):
        result = classify_turn_complexity(
            "Please think carefully about the architecture here"
        )
        assert result.tier == Tier.DEEP
        assert "deep reasoning signal" in result.reason

    def test_deep_signal_step_by_step(self):
        result = classify_turn_complexity(
            "Walk me through this step by step"
        )
        assert result.tier == Tier.DEEP

    def test_high_context_triggers_deep(self):
        result = classify_turn_complexity(
            "continue",
            context_pct=0.7,
        )
        assert result.tier == Tier.DEEP
        assert "context usage" in result.reason

    def test_many_tool_calls_triggers_deep(self):
        result = classify_turn_complexity(
            "keep going",
            recent_tool_calls=12,
        )
        assert result.tier == Tier.DEEP
        assert "tool activity" in result.reason

    def test_long_prompt_with_code_is_deep(self):
        long_prompt = "x" * 1200
        result = classify_turn_complexity(
            long_prompt,
            has_code_context=True,
        )
        assert result.tier == Tier.DEEP


class TestFindModelForTier:
    """Test model selection from available pool."""

    def test_finds_haiku_for_fast(self):
        available = [
            {"provider": "bedrock", "model": "global.anthropic.claude-sonnet-4-6"},
            {"provider": "bedrock", "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0"},
        ]
        result = find_model_for_tier(Tier.FAST, available)
        assert result is not None
        assert "haiku" in result[1].lower()

    def test_finds_nova_lite_for_fast(self):
        available = [
            {"provider": "bedrock", "model": "global.anthropic.claude-sonnet-4-6"},
            {"provider": "bedrock", "model": "us.amazon.nova-lite-v1:0"},
        ]
        result = find_model_for_tier(Tier.FAST, available)
        assert result is not None
        assert "nova-lite" in result[1].lower()

    def test_finds_opus_for_deep(self):
        available = [
            {"provider": "bedrock", "model": "global.anthropic.claude-sonnet-4-6"},
            {"provider": "bedrock", "model": "global.anthropic.claude-opus-4-7"},
        ]
        result = find_model_for_tier(Tier.DEEP, available)
        assert result is not None
        assert "opus" in result[1].lower()

    def test_finds_nova_premier_for_deep(self):
        available = [
            {"provider": "bedrock", "model": "global.anthropic.claude-sonnet-4-6"},
            {"provider": "bedrock", "model": "us.amazon.nova-premier-v1:0"},
        ]
        result = find_model_for_tier(Tier.DEEP, available)
        assert result is not None
        assert "nova-premier" in result[1].lower()

    def test_returns_none_when_no_match(self):
        available = [
            {"provider": "bedrock", "model": "global.anthropic.claude-sonnet-4-6"},
        ]
        result = find_model_for_tier(Tier.FAST, available)
        assert result is None


class TestRouteTurn:
    """Test full routing decision."""

    def test_disabled_routing_returns_none(self):
        result = route_turn(
            prompt="think carefully about this",
            primary_provider="bedrock",
            primary_model="global.anthropic.claude-sonnet-4-6",
            fallback_chain=[],
            enabled=False,
        )
        assert result is None

    def test_switches_to_fast_for_trivial(self):
        result = route_turn(
            prompt="hi",
            primary_provider="bedrock",
            primary_model="global.anthropic.claude-sonnet-4-6",
            fallback_chain=[
                {"provider": "bedrock", "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0"},
            ],
            enabled=True,
        )
        assert result is not None
        provider, model, reason = result
        assert "haiku" in model.lower()

    def test_switches_to_deep_for_complex(self):
        result = route_turn(
            prompt="please think carefully about the architecture",
            primary_provider="bedrock",
            primary_model="global.anthropic.claude-sonnet-4-6",
            fallback_chain=[
                {"provider": "bedrock", "model": "global.anthropic.claude-opus-4-7"},
            ],
            enabled=True,
        )
        assert result is not None
        provider, model, reason = result
        assert "opus" in model.lower()

    def test_no_switch_when_already_on_correct_tier(self):
        # Primary is Opus, task needs deep → no switch
        result = route_turn(
            prompt="think carefully",
            primary_provider="bedrock",
            primary_model="global.anthropic.claude-opus-4-7",
            fallback_chain=[],
            enabled=True,
        )
        assert result is None

    def test_no_switch_when_no_alternative(self):
        # Needs fast but no Haiku in chain
        result = route_turn(
            prompt="hi",
            primary_provider="bedrock",
            primary_model="global.anthropic.claude-sonnet-4-6",
            fallback_chain=[
                {"provider": "bedrock", "model": "global.anthropic.claude-opus-4-7"},
            ],
            enabled=True,
        )
        assert result is None
