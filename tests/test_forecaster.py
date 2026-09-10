"""
Tests for models.forecaster.AegisForecaster — the ARIMA+XGBoost hybrid
forecasting model at the core of Aegis's crisis simulation.
"""
import pytest

from models.forecaster import AegisForecaster, get_forecaster


@pytest.fixture(scope="module")
def forecaster():
    return AegisForecaster().fit()


def test_forecast_returns_correct_horizon_length(forecaster):
    result = forecaster.forecast(horizon_days=14, crisis_shock=0.0, disruption_factor=0.0)
    assert len(result["forecast"]) == 14


def test_forecast_bounds_are_consistent(forecaster):
    """For every forecasted day, lower <= price <= upper should hold."""
    result = forecaster.forecast(horizon_days=14, crisis_shock=0.0, disruption_factor=0.0)
    for day in result["forecast"]:
        assert day["lower"] <= day["price"] <= day["upper"]


def test_forecast_with_crisis_shock_raises_shocked_price(forecaster):
    baseline = forecaster.forecast(horizon_days=7, crisis_shock=0.0, disruption_factor=0.0)
    shocked = forecaster.forecast(horizon_days=7, crisis_shock=18.0, disruption_factor=0.7)
    assert shocked["shocked_price"] > baseline["shocked_price"]
    assert shocked["summary"]["risk_score"] >= baseline["summary"]["risk_score"]


def test_forecast_summary_has_expected_keys(forecaster):
    result = forecaster.forecast(horizon_days=14, crisis_shock=18.0, disruption_factor=0.7)
    summary = result["summary"]
    for key in ("final_price", "pct_change", "peak_price", "peak_day", "risk_score", "delay_prob", "cost_impact"):
        assert key in summary


def test_get_forecaster_returns_singleton():
    fc1 = get_forecaster()
    fc2 = get_forecaster()
    assert fc1 is fc2
    assert fc1.fitted is True


def test_forecast_horizon_bounds_respected(forecaster):
    """horizon_days is validated at the API layer (1-90) but the model
    itself should also behave sanely at the edges."""
    result = forecaster.forecast(horizon_days=1, crisis_shock=0.0, disruption_factor=0.0)
    assert len(result["forecast"]) == 1
