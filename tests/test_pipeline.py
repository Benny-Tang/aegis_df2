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


@pytest.mark.asyncio
async def test_pipeline_returns_all_seven_agent_keys(caplog):
    """With LLM outputs stubbed to empty JSON, _validated() falls back to
    defaults per agent — the pipeline must still return every agent's key."""
    with (
        patch("agents.swarm._llm", new=AsyncMock(return_value="{}")),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
        caplog.at_level(logging.WARNING),
    ):
        result = await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)

    assert set(result.keys()) == {
        "signal", "intelligence", "forecast",
        "simulation", "decision", "alert", "execution",
    }


@pytest.mark.asyncio
async def test_pipeline_logs_schema_warnings_on_malformed_llm_output(caplog):
    """A malformed agent output must produce a logged validation warning,
    not an exception that kills the pipeline."""
    with (
        patch("agents.swarm._llm", new=AsyncMock(return_value="{}")),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
        caplog.at_level(logging.WARNING),
    ):
        await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)

    assert any("schema" in r.message.lower() or "validation" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_pipeline_does_not_call_real_groq(caplog):
    """Guard against regressions where a test accidentally hits the real
    Groq API: _llm must be intercepted in every pipeline test."""
    mock_llm = AsyncMock(return_value="{}")
    with (
        patch("agents.swarm._llm", new=mock_llm),
        patch("agents.swarm.scrape_marine_traffic", return_value={"items": []}),
    ):
        await run_aegis_pipeline(SAMPLE_EVENT, SAMPLE_FORECAST)
    assert mock_llm.called
