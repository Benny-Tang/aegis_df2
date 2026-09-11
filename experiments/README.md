# Experiments

## Baseline / ablation comparison

`run_baseline_comparison.py` compares three forecasting variants on a
held-out window of the synthetic price series:

- **arima_only** (`blend_weight=0.0`)
- **xgboost_only** (`blend_weight=1.0`)
- **hybrid_default** (the model's normal blend — 0.7 during a crisis shock, 0.4 otherwise)

Config lives in `config.yaml` — change `seed`, `history_days`,
`holdout_days`, or `base_price` and re-run; no code changes needed.
Results are written to `results.json` with the timestamp and exact
config used, so a run is fully reproducible.

```bash
python experiments/run_baseline_comparison.py
```

### Finding (seed=42, 14-day holdout, no crisis shock)

| Variant | MAE | RMSE |
|---|---|---|
| arima_only | 1.00 | 1.21 |
| hybrid_default | 1.45 | 1.64 |
| xgboost_only | 2.31 | 2.46 |

ARIMA-only outperforms both XGBoost-only and the hybrid blend on
this no-shock holdout window. This makes some sense: the synthetic
history is a mean-reverting random walk with no real exogenous
features for XGBoost to learn from beyond lagged price — ARIMA is
well-suited to exactly that kind of series, while XGBoost has no
signal advantage here and mostly adds noise to the blend.

This is a genuine, if inconvenient, result: it suggests the
hardcoded 0.7/0.4 blend weights in `forecaster.py` aren't necessarily
optimal, at least not for calm (non-crisis) periods. It doesn't
necessarily hold once real crisis-shock data or real exogenous
features (actual shipping/news signals, not just lagged price) are
in play, which is the scenario the hybrid was originally designed for
— but it's worth re-running this comparison periodically, especially
once a live price feed replaces the synthetic history, rather than
assuming the original blend weights are correct indefinitely.

### Per-day error and crisis-scenario analysis (round 3)

`results.json` now stores, per variant:
- `per_day_error` — absolute error for each horizon day (day 1..14)
- `mean_late_horizon_error` — mean error over the final 3 days, a
  rough proxy for how error compounds with horizon length

It also stores a `crisis_scenario` block: ARIMA-only vs the hybrid's
crisis blend (`blend_weight=None`, `crisis_shock=18%`,
`disruption=0.7` — both configurable in `config.yaml`) run against
the same calm holdout, with `mean_forecast_price` capturing the
shock response.

**How to read it:** the round-2 finding (ARIMA wins on a calm holdout)
holds by construction — the synthetic series is a mean-reverting walk
with no exogenous features, and MAE against a calm holdout cannot
reward shock responsiveness. The crisis block exists to verify the
hybrid does what it was built for: respond to shock/disruption inputs
ARIMA never sees. A working hybrid shows a materially higher
`mean_forecast_price` under shock than ARIMA-only; similar prices
would mean the blend is effectively ignoring its crisis inputs.

The comparison that actually justifies the blend weights needs a real
price feed and real crisis windows — re-run this script once the
synthetic `_gen_history()` is replaced.
