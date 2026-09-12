# ruff: noqa: I001 -- sys.path bootstrap below keeps imports unsorted by design
"""
Tests for agents/swarm.py's run_aegis_pipeline — the 7-agent orchestration
is the core business logic and must complete even when LLM outputs are
malformed, logging schema-validation warnings rather than raising.
"""
import logging
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.swarm import run_aegis_pipeline  # noqa: E402


SAMPLE_EVENT = {
    "oil_price_change_pct": 18.0,
    "shipping_disruption": "Strait of Hormuz disruption",
    "news_headline": "Regional conflict near Persian Gulf",
    "severity": "HIGH",
    "disruption_factor": 0.7,
}

SAMPLE_FORECAST = {
    "summary": {"final_price": 95.0, "pct_change": 5.0, "risk_score": 50.0},
    "forecast": [],
}

EXPECTED_AGENTS = {
    "signal", "intelligence", "forecast",
    "simulation", "decision", "alert", "execution",
}

# One deliberately type-wrong payload per agent call (each agent calls _llm
# exactly once, in order). signal/intelligence/decision validate against
# Pydantic schemas -> must log "failed schema validation"; simulation's
# non-list scenarios trigger its defaulting warning; forecast/alert/execution
# have no schema validation (documented behavior, no warning expected).
BAD_LLM_OUTPUTS = [
    '{"confidence": "high"}',                # signal: confidence must be int
    '{"escalation_probability": "likely"}',  # intelligence: int expected
    '{"supply_risk_score": "high"}',         # forecast: no validation step
    '{"scenarios": "none"}',                 # simulation: non-list scenarios
    '{"threat_level": ["CRITICAL"]}',        # decision: str expected
    '{"slack_message": "ok"}',               # alert: no validation step
    '{"autonomous_actions_count": 1}',       # execution: no validation step
]


@pytest.mark.asyncio
async def test_pipeline_returns_all_seven_agents(caplog):
    """run_aegis_pipeline returns {"event": ..., "agents": {...}} — every
    one of the 7 agents must appear under result["agents"], tagged with
    its own name, even on fully bad LLM output."""
    with (
        patch("agents.swarm._llm", new=AsyncMock(side_effect=BAD_LLM_OUTPUTS)),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
        caplog.at_level(logging.WARNING),
    ):
        result = await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)

    assert "event" in result and "agents" in result
    assert set(result["agents"].keys()) == EXPECTED_AGENTS
    for name, agent_result in result["agents"].items():
        assert agent_result["agent"] == name


@pytest.mark.asyncio
async def test_pipeline_logs_schema_warnings_on_bad_llm_types(caplog):
    """Type-mismatched LLM output must produce logged schema-validation
    warnings (signal, intelligence, decision), not exceptions that kill
    the pipeline."""
    with (
        patch("agents.swarm._llm", new=AsyncMock(side_effect=BAD_LLM_OUTPUTS)),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
        caplog.at_level(logging.WARNING),
    ):
        await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)

    schema_warnings = [r for r in caplog.records if "schema validation" in r.message]
    assert len(schema_warnings) >= 3


@pytest.mark.asyncio
async def test_pipeline_never_calls_real_groq_client():
    """Regression guard: if any code path reaches the real Groq client
    constructor, this test must blow up — pipeline tests must stay
    offline and credential-free."""
    mock_llm = AsyncMock(side_effect=BAD_LLM_OUTPUTS)
    with (
        patch("agents.swarm._llm", new=mock_llm),
        patch("agents.swarm._client", side_effect=AssertionError("real Groq client must not be instantiated")),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
    ):
        await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)
    assert mock_llm.called
