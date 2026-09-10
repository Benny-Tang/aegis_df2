"""
Tests for agents.swarm — the scraper consolidation and schema validation
added when closing out the DataFactor Score architecture/robustness gap.
All network calls are mocked; these should run identically offline or in CI.
"""
from unittest.mock import MagicMock, patch

import requests

from agents.swarm import (
    MARITIME_SOURCES,
    SIMULATED_ITEMS,
    DecisionOutput,
    _json,
    _scrape_source,
    _validated,
    scrape_marine_traffic,
)


def _mock_response(status_code=200, html="<h2>A sufficiently long test headline for scraping</h2>"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    return resp


def test_scrape_source_returns_items_on_success():
    source = {
        "name": "gCaptain", "url": "https://gcaptain.com/", "tags": ["h2", "h3"],
        "type": "maritime_news", "min_len": 5, "max_len": 300, "limit": 8,
    }
    with patch("agents.swarm.requests.get", return_value=_mock_response()):
        items = _scrape_source(source)
    assert len(items) == 1
    assert items[0]["source"] == "gCaptain"
    assert items[0]["type"] == "maritime_news"


def test_scrape_source_returns_empty_list_on_network_error():
    source = MARITIME_SOURCES[0]
    with patch("agents.swarm.requests.get", side_effect=requests.RequestException("boom")):
        items = _scrape_source(source)
    assert items == []


def test_scrape_source_returns_empty_list_on_non_200():
    source = MARITIME_SOURCES[1]
    with patch("agents.swarm.requests.get", return_value=_mock_response(status_code=503)):
        items = _scrape_source(source)
    assert items == []


def test_scrape_marine_traffic_falls_back_to_simulated_when_all_sources_fail():
    with patch("agents.swarm.requests.get", side_effect=requests.RequestException("offline")):
        result = scrape_marine_traffic()
    assert result["status"] == "simulated"
    assert result["items"] == SIMULATED_ITEMS
    assert result["sources_ok"] == []


def test_scrape_marine_traffic_reports_live_status_when_a_source_succeeds():
    def fake_get(url, headers=None, timeout=None):
        if "gcaptain" in url:
            return _mock_response()
        raise requests.RequestException("offline")

    with patch("agents.swarm.requests.get", side_effect=fake_get):
        result = scrape_marine_traffic()
    assert result["status"] == "live"
    assert "gCaptain" in result["sources_ok"]


def test_validated_returns_data_unchanged_on_schema_mismatch():
    """Schema validation failures are logged, not raised — the pipeline
    should keep working off the same dict shape it always did."""
    bad_data = {"threat_level": 123, "unexpected_field": "x"}  # wrong type for threat_level
    result = _validated(DecisionOutput, bad_data, "decision_agent")
    assert result == bad_data


def test_validated_passes_through_on_valid_schema():
    good_data = {"threat_level": "HIGH", "executive_summary": "Test summary"}
    result = _validated(DecisionOutput, good_data, "decision_agent")
    assert result == good_data


def test_json_parses_clean_json():
    import asyncio

    async def run():
        with patch("agents.swarm._llm", return_value='{"key": "value"}'):
            return await _json("system", "user")

    result = asyncio.run(run())
    assert result == {"key": "value"}


def test_json_recovers_from_markdown_fenced_json():
    import asyncio

    async def run():
        with patch("agents.swarm._llm", return_value='```json\n{"key": "value"}\n```'):
            return await _json("system", "user")

    result = asyncio.run(run())
    assert result == {"key": "value"}


def test_json_falls_back_to_raw_on_unparseable_response():
    import asyncio

    async def run():
        with patch("agents.swarm._llm", return_value="not json at all"):
            return await _json("system", "user")

    result = asyncio.run(run())
    assert "raw" in result
