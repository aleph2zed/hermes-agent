"""Task-based model routing for intelligent model selection.

Classifies each turn's complexity and selects the most cost-effective model
capable of handling it. Three tiers:

  FAST (Haiku / Nova-Lite / Nova-Micro)
    - Simple lookups, title generation, yes/no questions
    - Short prompts with no tool context
    - Signals: prompt < 200 chars, no code blocks, no tool output

  STANDARD (Sonnet / Nova-Pro) — default
    - Multi-step agent work, coding, analysis
    - Most turns fall here

  DEEP (Opus / Nova-Premier)
    - Complex reasoning, architecture, debugging multi-file issues
    - Extended context analysis
    - Signals: explicit depth request, high context %, "think carefully"

The router is opt-in via config.yaml `model_routing.enabled: true` and respects
the available fallback chain — it won't pick a model that isn't provisioned.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Tier(Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


# Model mappings by tier and provider family
# Each entry: (provider, model_pattern) — pattern is case-insensitive partial match
TIER_MODELS: Dict[Tier, List[Tuple[str, str]]] = {
    Tier.FAST: [
        ("bedrock", "haiku"),
        ("bedrock", "nova-lite"),
        ("bedrock", "nova-micro"),
        ("anthropic", "haiku"),
        ("openrouter", "haiku"),
    ],
    Tier.STANDARD: [
        ("bedrock", "sonnet"),
        ("bedrock", "nova-pro"),
        ("anthropic", "sonnet"),
        ("openrouter", "sonnet"),
    ],
    Tier.DEEP: [
        ("bedrock", "opus"),
        ("bedrock", "nova-premier"),
        ("bedrock", "deepseek"),
        ("bedrock", "kimi"),
        ("anthropic", "opus"),
        ("openrouter", "opus"),
    ],
}

# Patterns that signal deep reasoning is needed
DEEP_SIGNALS = [
    r"\bthink\s+(carefully|deeply|through)\b",
    r"\banalyze\s+(carefully|in\s+depth)\b",
    r"\barchitect(ure)?\b",
    r"\bdesign\s+(decision|review|pattern)\b",
    r"\bdebug(ging)?\s+.*(complex|hairy|tricky)\b",
    r"\brefactor(ing)?\s+.*(major|large|significant)\b",
    r"\bexplain\s+(why|how|in\s+detail)\b",
    r"\bstep\s+by\s+step\b",
    r"\bcomprehensive\b",
    r"\bthorough(ly)?\b",
]

# Patterns that signal a trivial/fast query
FAST_SIGNALS = [
    r"^(yes|no|true|false)\??$",
    r"^what\s+is\s+\w+\??$",
    r"^(hi|hello|hey|thanks|thank\s+you)\b",
    r"^(ok|okay|got\s+it|understood)\b",
    r"^/\w+",  # slash commands
]


@dataclass
class RouteDecision:
    """Result of task classification."""
    tier: Tier
    reason: str
    suggested_model: Optional[str] = None
    suggested_provider: Optional[str] = None


def classify_turn_complexity(
    prompt: str,
    context_pct: float = 0.0,
    recent_tool_calls: int = 0,
    has_code_context: bool = False,
) -> RouteDecision:
    """Classify a user turn to determine optimal model tier.
    
    Args:
        prompt: The user's message text
        context_pct: Current context window usage (0.0-1.0)
        recent_tool_calls: Number of tool calls in last N turns
        has_code_context: Whether recent context contains code blocks
        
    Returns:
        RouteDecision with tier and reasoning
    """
    prompt_lower = prompt.lower().strip()
    prompt_len = len(prompt)
    
    # Check for explicit deep reasoning signals
    for pattern in DEEP_SIGNALS:
        if re.search(pattern, prompt_lower):
            return RouteDecision(
                tier=Tier.DEEP,
                reason=f"prompt contains deep reasoning signal: {pattern[:30]}",
            )
    
    # High context usage suggests complex, long-running task
    if context_pct > 0.6:
        return RouteDecision(
            tier=Tier.DEEP,
            reason=f"high context usage ({context_pct:.0%})",
        )
    
    # Many recent tool calls suggests iterative complex work
    if recent_tool_calls > 8:
        return RouteDecision(
            tier=Tier.DEEP,
            reason=f"high tool activity ({recent_tool_calls} recent calls)",
        )
    
    # Check for trivial/fast signals
    for pattern in FAST_SIGNALS:
        if re.match(pattern, prompt_lower):
            return RouteDecision(
                tier=Tier.FAST,
                reason=f"trivial prompt pattern: {pattern[:30]}",
            )
    
    # Very short prompts without code context → fast
    # But only really trivial ones (< 50 chars) to avoid misclassifying
    # reasonable questions as trivial
    if prompt_len < 50 and not has_code_context:
        return RouteDecision(
            tier=Tier.FAST,
            reason=f"short prompt ({prompt_len} chars), no code context",
        )
    
    # Short-medium prompts → standard
    if prompt_len < 500:
        return RouteDecision(
            tier=Tier.STANDARD,
            reason="standard complexity prompt",
        )
    
    # Long prompts (> 500 chars) might need more capability
    # but default to standard unless other signals present
    if has_code_context and prompt_len > 1000:
        return RouteDecision(
            tier=Tier.DEEP,
            reason=f"long prompt ({prompt_len} chars) with code context",
        )
    
    return RouteDecision(
        tier=Tier.STANDARD,
        reason="default tier",
    )


def find_model_for_tier(
    tier: Tier,
    available_models: List[Dict[str, str]],
    current_provider: str = "",
    current_model: str = "",
) -> Optional[Tuple[str, str]]:
    """Find a model matching the requested tier from available options.
    
    Args:
        tier: Desired complexity tier
        available_models: List of {"provider": ..., "model": ...} dicts
        current_provider: Current provider (prefer same if possible)
        current_model: Current model (avoid if switching tiers)
        
    Returns:
        (provider, model) tuple or None if no match found
    """
    tier_patterns = TIER_MODELS.get(tier, [])
    
    # First pass: find matches in the available list
    matches = []
    for avail in available_models:
        avail_provider = (avail.get("provider") or "").lower()
        avail_model = (avail.get("model") or "").lower()
        
        for prov_pattern, model_pattern in tier_patterns:
            if prov_pattern in avail_provider and model_pattern in avail_model:
                matches.append((avail.get("provider"), avail.get("model")))
                break
    
    if not matches:
        return None
    
    # Prefer same provider family if possible
    current_provider_lower = current_provider.lower()
    for prov, model in matches:
        if current_provider_lower and current_provider_lower in prov.lower():
            return (prov, model)
    
    # Return first match
    return matches[0]


def route_turn(
    prompt: str,
    primary_provider: str,
    primary_model: str,
    fallback_chain: List[Dict[str, str]],
    context_pct: float = 0.0,
    recent_tool_calls: int = 0,
    has_code_context: bool = False,
    enabled: bool = True,
) -> Optional[Tuple[str, str, str]]:
    """Main entry point: decide if we should switch models for this turn.
    
    Args:
        prompt: User's message
        primary_provider: Session's primary provider
        primary_model: Session's primary model
        fallback_chain: Available fallback models from config
        context_pct: Context window usage (0.0-1.0)
        recent_tool_calls: Tool calls in recent turns
        has_code_context: Whether context has code
        enabled: Whether routing is enabled (False = always return None)
        
    Returns:
        (provider, model, reason) if a switch is recommended, else None
    """
    if not enabled:
        return None
    
    # Classify the turn
    decision = classify_turn_complexity(
        prompt=prompt,
        context_pct=context_pct,
        recent_tool_calls=recent_tool_calls,
        has_code_context=has_code_context,
    )
    
    # Build available pool: primary + fallbacks
    available = [{"provider": primary_provider, "model": primary_model}]
    available.extend(fallback_chain)
    
    # Determine what tier the current model is
    current_tier = None
    for tier, patterns in TIER_MODELS.items():
        for prov_pattern, model_pattern in patterns:
            if (prov_pattern in primary_provider.lower() and 
                model_pattern in primary_model.lower()):
                current_tier = tier
                break
        if current_tier:
            break
    
    # If current model already matches the decision tier, no switch needed
    if current_tier == decision.tier:
        logger.debug(
            "Task router: tier=%s matches current model, no switch",
            decision.tier.value,
        )
        return None
    
    # Find a model for the desired tier
    match = find_model_for_tier(
        tier=decision.tier,
        available_models=available,
        current_provider=primary_provider,
        current_model=primary_model,
    )
    
    if not match:
        logger.debug(
            "Task router: no %s-tier model available in pool",
            decision.tier.value,
        )
        return None
    
    new_provider, new_model = match
    
    # Don't switch if we'd end up on the same model
    if new_provider.lower() == primary_provider.lower() and new_model == primary_model:
        return None
    
    logger.info(
        "Task router: %s → %s/%s (%s)",
        decision.tier.value,
        new_provider,
        new_model,
        decision.reason,
    )
    
    return (new_provider, new_model, decision.reason)
