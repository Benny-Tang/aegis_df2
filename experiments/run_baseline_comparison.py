"""
Baseline / ablation comparison for AegisForecaster.

Compares three variants on a held-out window of the synthetic price
series:
  - ARIMA-only   (blend_weight=0.0)
  - XGBoost-only (blend_weight=1.0)
  - Hybrid       (blend_weight=None -> the model's normal default blend)

Beyond MAE/RMSE, each variant reports per-day absolute error (error by
horizon day) and a crisis-scenario stress test: ARIMA-only vs the
hybrid's crisis blend under an 18% shock / 0.7 disruption, to check the
blend actually responds to shock inputs the way it was designed to.

Config lives in experiments/config.yaml; results are written to
experiments/results.json with a timestamp and the exact config used, so
a run is reproducible and comparable against prior runs.

Usage:
    python experiments/run_baseline_comparison.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.forecaster import AegisForecaster, _gen_history  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"

VARIANTS = {
    "arima_only": 0.0,
    "xgboost_only": 1.0,
    "hybrid_default": None,
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _eval_variant(forecaster: AegisForecaster, holdout_actual: list, shock: float, disruption: float, weight):
    """Forecast with the given shock/disruption/blend and score against holdout."""
    forecast = forecaster.forecast(
        horizon_days=len(holdout_actual), crisis_shock=shock, disruption_factor=disruption, blend_weight=weight
    )
    predicted = [day["price"] for day in forecast["forecast"]]
    errors = np.abs(np.array(predicted) - np.array(holdout_actual))
    return {
        "mae": round(float(np.mean(errors)), 4),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
        "per_day_error": [round(float(e), 4) for e in errors],
        "mean_late_horizon_error": round(float(np.mean(errors[-3:])), 4),  # final 3 days
        "mean_forecast_price": round(float(np.mean(predicted)), 2),
        "predicted": predicted,
    }


def run_comparison(config: dict) -> dict:
    full_history = _gen_history(n=config["history_days"], base=config["base_price"], seed=config["seed"])

    holdout_days = config["holdout_days"]
    train = full_history.iloc[:-holdout_days]
    holdout_actual = full_history["price"].iloc[-holdout_days:].tolist()

    variant_results = {}
    for variant_name, weight in VARIANTS.items():
        forecaster = AegisForecaster()
        forecaster.fit(df=train)
        r = _eval_variant(forecaster, holdout_actual, shock=0.0, disruption=0.0, weight=weight)
        r.pop("mean_forecast_price")  # only meaningful in the crisis block
        variant_results[variant_name] = r

    # Crisis-scenario stress test: does the hybrid's crisis blend (w=0.7)
    # behave differently from ARIMA-only when the shock the blend was
    # designed for is present? Scored against the calm holdout as a
    # shock-response reference, not a matched forecast target.
    shock = config.get("crisis_shock_pct", 18.0)
    disruption = config.get("crisis_disruption_factor", 0.7)
    crisis_scenario = {"shock_pct": shock, "disruption_factor": disruption, "results": {}}
    for label, weight in [("arima_only", 0.0), ("hybrid_crisis_blend", None)]:
        forecaster = AegisForecaster()
        forecaster.fit(df=train)
        crisis_scenario["results"][label] = _eval_variant(forecaster, holdout_actual, shock=shock, disruption=disruption, weight=weight)

    best_variant = min(variant_results, key=lambda v: variant_results[v]["mae"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "holdout_actual": holdout_actual,
        "variants": variant_results,
        "best_variant_by_mae": best_variant,
        "crisis_scenario": crisis_scenario,
    }


def main():
    config = load_config()
    results = run_comparison(config)

    print(f"Baseline comparison ({config['holdout_days']}-day holdout, seed={config['seed']})")
    print("-" * 60)
    for name, r in results["variants"].items():
        print(f"  {name:20s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  late-horizon MAE={r['mean_late_horizon_error']:.4f}")
    print("-" * 60)
    print(f"Best variant by MAE: {results['best_variant_by_mae']}")
    cs = results["crisis_scenario"]
    print(f"\nCrisis scenario (shock={cs['shock_pct']}%, disruption={cs['disruption_factor']}):")
    for name, r in cs["results"].items():
        print(f"  {name:20s}  MAE={r['mae']:.4f}  mean price={r['mean_forecast_price']:.2f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
