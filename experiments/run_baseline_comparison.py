"""
Baseline / ablation comparison for AegisForecaster.

Compares three variants on a held-out window of the synthetic price
series:
  - ARIMA-only   (blend_weight=0.0)
  - XGBoost-only (blend_weight=1.0)
  - Hybrid       (blend_weight=None -> the model's normal default blend)

This addresses the gap DataFactor's report flagged: "no experiment
tracking, no config-driven runs, no baselines/ablations." Config lives
in experiments/config.yaml; results are written to
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


def run_comparison(config: dict) -> dict:
    full_history = _gen_history(n=config["history_days"], base=config["base_price"], seed=config["seed"])

    holdout_days = config["holdout_days"]
    train = full_history.iloc[:-holdout_days]
    holdout_actual = full_history["price"].iloc[-holdout_days:].tolist()

    variant_results = {}
    for variant_name, weight in VARIANTS.items():
        forecaster = AegisForecaster()
        forecaster.fit(df=train)
        forecast = forecaster.forecast(horizon_days=holdout_days, crisis_shock=0.0, disruption_factor=0.0, blend_weight=weight)
        predicted = [day["price"] for day in forecast["forecast"]]

        mae = float(np.mean(np.abs(np.array(predicted) - np.array(holdout_actual))))
        rmse = float(np.sqrt(np.mean((np.array(predicted) - np.array(holdout_actual)) ** 2)))

        variant_results[variant_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "predicted": predicted,
        }

    best_variant = min(variant_results, key=lambda v: variant_results[v]["mae"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "holdout_actual": holdout_actual,
        "variants": variant_results,
        "best_variant_by_mae": best_variant,
    }


def main():
    config = load_config()
    results = run_comparison(config)

    print(f"Baseline comparison ({config['holdout_days']}-day holdout, seed={config['seed']})")
    print("-" * 60)
    for name, r in results["variants"].items():
        print(f"  {name:20s}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}")
    print("-" * 60)
    print(f"Best variant by MAE: {results['best_variant_by_mae']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
