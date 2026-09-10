"""
Tests for api.server — the FastAPI app exposing Aegis's endpoints.
Uses FastAPI's TestClient so these run without a live server or network
access (forecast/health endpoints don't touch the network; marine
scraping is exercised separately and falls back to simulated data when
offline, so it's safe to hit in CI too).
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint_returns_200_and_expected_keys():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "system", "version", "agents", "platform", "model", "forecast"):
        assert key in body
    assert body["status"] == "online"
    assert body["agents"] == 7


def test_health_works_without_groq_api_key(monkeypatch):
    """Dependency-free smoke test: the service must boot and answer
    /health with no external credentials configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_ready_endpoint_reports_forecaster_state():
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["forecaster_ready"] is True
    assert body["status"] == "ready"
    assert "gpu" in body


def test_forecast_endpoint_returns_200_and_expected_shape():
    response = client.post("/api/forecast", json={"oil_shock_pct": 0.0, "disruption_factor": 0.0, "horizon_days": 7})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["data"]["forecast"]) == 7


def test_forecast_endpoint_validates_horizon_days():
    """horizon_days is constrained to 1-90 by the Pydantic model."""
    response = client.post("/api/forecast", json={"horizon_days": 999})
    assert response.status_code == 422


def test_status_endpoint_returns_200_and_expected_keys():
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    for key in ("agents_online", "platform", "oil_price", "risk_level", "groq_model", "forecast_model"):
        assert key in body


def test_marine_endpoint_returns_200_even_when_offline():
    """scrape_marine_traffic() falls back to simulated data on any
    network failure, so this endpoint should never itself 500 due to
    connectivity issues in a CI environment."""
    response = client.get("/api/marine")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "items" in body["data"]
