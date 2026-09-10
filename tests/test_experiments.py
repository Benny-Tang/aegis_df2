"""
Tests for experiments/run_baseline_comparison.py — the config-driven
ablation must stay reproducible and must emit the per-day error and
crisis-scenario breakdowns.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.run_baseline_comparison as exp  # noqa: E402


def test_run_comparison_emits_per_day_error():
    config = {"seed": 42, "history_days": 150, "holdout_days": 14, "base_price": 82.0}
    results = exp.run_comparison(config)
    for name, variant in results["variants"].items():
        assert len(variant["per_day_error"]) == config["holdout_days"]
        assert variant["mae"] > 0
        assert variant["rmse"] >= variant["mae"]  # RMSE >= MAE always
    assert results["best_variant_by_mae"] in results["variants"]


def test_run_comparison_is_reproducible_for_same_seed():
    config = {"seed": 42, "history_days": 150, "holdout_days": 14, "base_price": 82.0}
    r1 = exp.run_comparison(config)
    r2 = exp.run_comparison(config)
    assert r1["holdout_actual"] == r2["holdout_actual"]
    assert r1["variants"]["arima_only"]["mae"] == r2["variants"]["arima_only"]["mae"]


def test_run_comparison_includes_crisis_scenario():
    config = {"seed": 42, "history_days": 150, "holdout_days": 14, "base_price": 82.0}
    results = exp.run_comparison(config)
    crisis = results["crisis_scenario"]
    assert set(crisis["results"]) == {"arima_only", "hybrid_crisis_blend"}
    assert crisis["shock_pct"] > 0
    for variant in crisis["results"].values():
        assert len(variant["per_day_error"]) == config["holdout_days"]
