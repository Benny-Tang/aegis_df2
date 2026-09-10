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
