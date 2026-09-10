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


def test_forecast_output_matches_validated_schema(forecaster):
    """forecast() returns the Pydantic-validated ForecastResult shape."""
    result = forecaster.forecast(horizon_days=7, crisis_shock=0.0, disruption_factor=0.0)
    for key in ("base_price", "shocked_price", "horizon_days", "gpu_accelerated", "device", "forecast", "summary", "model"):
        assert key in result
    for day in result["forecast"]:
        assert set(day) == {"date", "price", "lower", "upper"}


def test_blend_weight_overrides_isolate_model_contributions(forecaster):
    """blend_weight=0/1 must reproduce pure ARIMA / pure XGBoost, and the
    default crisis blend must equal the explicit weighted combination —
    this pins the ablation logic the experiment relies on."""
    w0 = forecaster.forecast(horizon_days=7, crisis_shock=18.0, disruption_factor=0.7, blend_weight=0.0)
    w1 = forecaster.forecast(horizon_days=7, crisis_shock=18.0, disruption_factor=0.7, blend_weight=1.0)
    default = forecaster.forecast(horizon_days=7, crisis_shock=18.0, disruption_factor=0.7)
    p0 = [d["price"] for d in w0["forecast"]]
    p1 = [d["price"] for d in w1["forecast"]]
    pdef = [d["price"] for d in default["forecast"]]
    assert p0 != p1  # the override actually changes the forecast
    for i in range(7):
        # crisis default weight is 0.7: 0.7*xgb + 0.3*arima (2-dp rounding tol)
        assert abs(0.7 * p1[i] + 0.3 * p0[i] - pdef[i]) < 0.02


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

