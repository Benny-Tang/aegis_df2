Markdown
Code
Preview
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
Finding (seed=42, 14-day holdout, no crisis shock)
Table
Variant	MAE	RMSE
arima_only	1.00	1.21
hybrid_default	1.45	1.64
xgboost_only	2.31	2.46
ARIMA-only outperforms both XGBoost-only and the hybrid blend on
this no-shock holdout window. This makes some sense: the synthetic
history is a mean-reverting random walk with no real exogenous
features for XGBoost to learn from beyond lagged price — ARIMA is
well-suited to exactly that kind of series, while XGBoost has no
signal advantage here and mostly adds noise to the blend.
This is a genuine, if inconvenient, result: it suggests the
hardcoded 0.7/0.4 blend weights in forecaster.py aren't necessarily
optimal, at least not for calm (non-crisis) periods. It doesn't
necessarily hold once real crisis-shock data or real exogenous
features (actual shipping/news signals, not just lagged price) are
in play, which is the scenario the hybrid was originally designed for
— but it's worth re-running this comparison periodically, especially
once a live price feed replaces the synthetic history, rather than
assuming the original blend weights are correct indefinitely.
Per-day error and crisis-scenario analysis (round 3)
results.json now stores, per variant:
per_day_error — absolute error for each horizon day (day 1..14)
mean_late_horizon_error — mean error over the final 3 days, a
rough proxy for how error compounds with horizon length
It also stores a crisis_scenario block: ARIMA-only vs the hybrid's
crisis blend (blend_weight=None, crisis_shock=18%,
disruption=0.7 — both configurable in config.yaml) run against
the same calm holdout, with mean_forecast_price capturing the
shock response.
How to read it: the round-2 finding (ARIMA wins on a calm holdout)
holds by construction — the synthetic series is a mean-reverting walk
with no exogenous features, and MAE against a calm holdout cannot
reward shock responsiveness. The crisis block exists to verify the
hybrid does what it was built for: respond to shock/disruption inputs
ARIMA never sees. A working hybrid shows a materially higher
mean_forecast_price under shock than ARIMA-only; similar prices
would mean the blend is effectively ignoring its crisis inputs.
The comparison that actually justifies the blend weights needs a real
price feed and real crisis windows — re-run this script once the
synthetic _gen_history() is replaced.
plain

## 15. `README.md` — **FULL REPLACEMENT**

````markdown
# Aegis — Autonomous Enterprise Crisis Management
**Shield Against Chaos**

Aegis is a 7-agent autonomous pipeline that monitors global shipping
intelligence and commodity markets in real time, forecasts supply-chain
disruption (e.g. a Strait of Hormuz closure), simulates response scenarios,
and recommends ranked, cost-quantified actions — without a human analyst
in the loop.

Built for the AMD Developer Cloud hackathon track (2026), running on
AMD Instinct MI300X GPUs.

## Architecture
[MarineTraffic + Lloyd's List]        [Oil price / market data]
|                                |
+----------------+---------------+
|
[Signal Agent]        <- Agent 1: Watcher
|
[Intelligence Agent]      <- Agent 2: Interpreter
|
[Forecast Agent]        <- Agent 3: ARIMA + XGBoost
|
[Simulation Agent]       <- Agent 4: Strategist (3 scenarios)
|
[Decision Agent]        <- Agent 5: Brain (ranked actions)
|
[Alert Agent]          <- Agent 6: Communicator
|
[Execution Agent]        <- Agent 7: Operator (ERP workflows)
|
[FastAPI + SSE]
|
[Dashboard UI]
plain

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) | Reasoning for all 7 agents |
| Forecasting | ARIMA(2,1,2) + XGBoost | 14-day oil price + delay prediction |
| GPU | AMD Instinct MI300X | Model inference / forecasting acceleration |
| Backend | FastAPI + Server-Sent Events | Real-time agent streaming |
| Marine intel | MarineTraffic + Lloyd's List scrape | Live shipping disruption signals |
| Frontend | Single-page dashboard (vanilla JS) | Live agent monitor, crisis injection |

## Project structure
aegis_df2/
├── agents/
│   └── swarm.py           # 7-agent pipeline logic + parameterized 4-source
│                           #   marine scraper (MarineTraffic, gCaptain,
│                           #   TradeWinds, Reuters) with Pydantic-validated
│                           #   agent output schemas
├── api/
│   └── server.py          # FastAPI app: /health, /ready, /api/marine,
│                           #   /api/crisis, /api/stream (SSE), /api/status —
│                           #   typed Pydantic responses, structured JSON
│                           #   logging, optional Sentry error tracking
├── models/
│   └── forecaster.py      # ARIMA(2,1,2) + XGBoost hybrid forecaster with
│                           #   Pydantic-validated output schema
├── experiments/
│   ├── config.yaml         # Config-driven experiment parameters
│   ├── run_baseline_comparison.py  # ARIMA vs XGBoost vs hybrid ablation with
│   │                               #   per-day error + crisis-scenario analysis
│   ├── results.json         # Latest run's output (reproducible via config.yaml)
│   └── README.md            # Findings write-up
├── tests/
│   ├── test_forecaster.py # Forecast horizon, bounds, crisis-shock, schema,
│   │                      #   blend_weight isolation
│   ├── test_server.py     # Health (incl. credential-free smoke test), ready,
│   │                      #   forecast, status, marine endpoint tests
│   ├── test_swarm.py      # Scraper consolidation + schema validation tests
│   └── test_experiments.py# Experiment reproducibility + per-day error tests
├── docker/
│   ├── Dockerfile          # All-in-one image: locked deps, HEALTHCHECK
│   ├── Dockerfile.dockerignore
│   └── docker-compose.yml  # One-command local stack
├── .github/
│   ├── workflows/ci.yml    # Lint (ruff) + test (pytest-cov, 60% gate) +
│   │                       #   dependency audit (pip-audit) — manual trigger
│   └── dependabot.yml      # Weekly pip + GitHub Actions update PRs
├── frontend.html           # Dashboard UI: live agents, marine feed,
│                           #   business value panel, GTM strategy panel
├── requirements.txt        # Loose runtime pins (source for the lockfile)
├── requirements-dev.txt    # pytest, pytest-cov, ruff, pip-audit, PyYAML
├── requirements-lock.txt   # pip-compile generated, hashed, reproducible
├── pyproject.toml          # pytest + coverage configuration
├── ruff.toml
├── .env.example            # GROQ_API_KEY, LOG_LEVEL, SENTRY_DSN
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
plain

## Testing & CI

```bash
pip install -r requirements-lock.txt -r requirements-dev.txt
pytest -q --cov --cov-report=term-missing
ruff check .
pip-audit -r requirements-lock.txt
The CI workflow (.github/workflows/ci.yml) runs all three as separate
jobs (lint / test / audit) and is triggered manually from the
Actions tab → CI → "Run workflow" — nothing runs automatically on push.
The test step fails the build if coverage drops below 60%.
See CONTRIBUTING.md for setup details.
Business case (from the original hackathon pitch)
Table
Metric	Value
Total addressable market	$1.5T
Supply chain market (2028)	$19.3T
Estimated ROI, year 1	70x
Savings per crisis prevented	$1.7M
Annual subscription (proposed)	$24K/yr
Response time vs. human analysts	2 sec vs. 4–8 hours
Target segments: maritime insurance underwriters, commodity traders,
manufacturers with $50M+ procurement exposure, and sovereign wealth funds
managing oil-revenue exposure.
Running locally
bash
cp .env.example .env   # then fill in GROQ_API_KEY
pip install -r requirements-lock.txt   # reproducible, hashed pins
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
Or with Docker (no local Python needed):
bash
docker compose -f docker/docker-compose.yml up --build
# or standalone:
docker build -f docker/Dockerfile -t aegis .
docker run -p 8000:8000 --env-file .env aegis
Open http://localhost:8000 for the live dashboard. Click Inject Crisis
Scenario to trigger the full 7-agent pipeline against a simulated Strait
of Hormuz disruption event.
API endpoints
Table
Endpoint	Method	Description
/health	GET	Liveness: system status, model, platform info
/ready	GET	Readiness: confirms the forecaster is fitted and usable
/api/marine	GET	Live MarineTraffic + Lloyd's List shipping feed
/api/forecast	POST	Run the forecasting model standalone
/api/crisis	POST	Run the full 7-agent pipeline synchronously
/api/stream	GET	Run the pipeline via Server-Sent Events (used by the dashboard)
/api/status	GET	Live oil price + system status snapshot
License
MIT
plain

## 16. `CONTRIBUTING.md` — **FULL REPLACEMENT**

```markdown
# Contributing

## Setup

```bash
git clone <this-repo>
cd aegis_df2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # reproducible, hashed pins
pip install -r requirements-dev.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Running the app

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker: `docker compose -f docker/docker-compose.yml up --build`

## Running tests, lint, and audit

```bash
pytest -q --cov --cov-report=term-missing   # fails under 60% coverage
ruff check .
pip-audit -r requirements-lock.txt
```

All three also run in CI (`.github/workflows/ci.yml`) as separate jobs —
triggered manually from the Actions tab, since this repo keeps all
workflow runs manual. Run the workflow on a branch before merging if you
want CI verification for a change.

## Commit conventions

- Keep each feature or fix in its own small commit (or PR), including the
  tests that verify it — avoid bulk commits that mix formatting,
  refactors, and features together.
- If you're pairing or a teammate contributes, please commit under your
  own identity/email so the history reflects actual authorship.

## Project structure

See `README.md` for the architecture diagram and directory layout.
```

## 17. `CHANGELOG.md` — **FULL REPLACEMENT**

```markdown
# Changelog

## [Unreleased] — round 3

### Added
- `.github/workflows/ci.yml` — real CI pipeline: ruff lint, pytest-cov with a 60% coverage gate, and pip-audit against the hashed lockfile. Triggered manually via `workflow_dispatch` (no automatic push/PR runs, by repo policy)
- `.github/dependabot.yml` — weekly pip and GitHub Actions dependency-update PRs
- `.env.example` — self-documenting environment variables (`GROQ_API_KEY`, `LOG_LEVEL`, `SENTRY_DSN`); fixes the fresh-clone self-configuration gap
- `docker/` — all-in-one `Dockerfile` (locked deps, `HEALTHCHECK`), `docker-compose.yml`, and a Dockerfile-specific ignore file; `docker compose -f docker/docker-compose.yml up --build` runs the whole stack
- `/ready` readiness endpoint complementing `/health` liveness — confirms the fitted forecaster is actually usable; startup switched from deprecated `@app.on_event` to a `lifespan` context manager
- Optional Sentry error tracking via `SENTRY_DSN` — the service boots fully without it (guarded optional import)
- Pydantic schema validation on forecaster output (`ForecastResult` / `ForecastPoint` / `ForecastSummary` in `models/forecaster.py`) so every consumer sees a type-checked structure
- Experiment analysis beyond MAE/RMSE: per-day absolute error, `mean_late_horizon_error`, and a crisis-scenario stress test (ARIMA-only vs the hybrid's crisis blend under an 18% shock / 0.7 disruption) in `experiments/run_baseline_comparison.py`
- New tests: credential-free `/health` smoke test, `/ready` probe, blend_weight isolation, forecaster output schema, and experiment reproducibility/per-day-error tests (`tests/test_experiments.py`)

### Changed
- ARIMA fit failures in `models/forecaster.py` are now logged as warnings instead of failing silently
- README/CONTRIBUTING synced with the actual repo layout (test_swarm.py, docker/, .github/, .env.example) and now reference installing from `requirements-lock.txt`

## [Unreleased] — round 2

### Added
- `experiments/` — config-driven baseline/ablation comparison (ARIMA-only vs XGBoost-only vs hybrid) with reproducible results (`config.yaml` + `results.json`); see `experiments/README.md` for the finding
- Structured JSON logging (replacing plain-text log formatting) across the API layer
- Pydantic schema validation on agent LLM outputs (`SignalOutput`, `IntelligenceOutput`, `DecisionOutput` in `agents/swarm.py`) — validation failures are logged as warnings rather than silently accepted
- `pytest-cov` with a 60% coverage threshold (`pyproject.toml`)
- Real hashed lockfile via `pip-compile` (`requirements-lock.txt`) — the previous `pip freeze` dump wasn't being recognized as a lockfile

### Changed
- Consolidated `agents/swarm.py`'s 4 near-identical scrape try/except blocks (MarineTraffic, gCaptain, TradeWinds, Reuters) into one parameterized `_scrape_source()` helper driven by a `MARITIME_SOURCES` config list
- Scrape failures are now logged (`logger.info`) instead of silently swallowed by bare `except: pass`
- `_gen_history()`'s random seed is now a parameter instead of hardcoded, so experiment runs are genuinely reproducible end-to-end
- `AegisForecaster.forecast()` gained an optional `blend_weight` override, used by the ablation experiment to isolate each model's contribution without duplicating forecasting logic

## [Unreleased] — round 1

### Added
- Test suite: `tests/test_forecaster.py` (forecast horizon, bounds, crisis-shock behavior) and `tests/test_server.py` (health, forecast, status, marine endpoints)
- `ruff.toml` lint configuration
- Typed Pydantic response models for all API endpoints (`HealthResponse`, `MarineResponse`, `ForecastResponse`, `CrisisResponse`, `StatusResponse`)
- Structured logging (`logging` module) replacing `print()` statements throughout the API layer

### Changed
- Error responses no longer leak internal tracebacks to clients (`/api/crisis` previously returned `traceback.format_exc()` in the HTTP response body) — full details are now logged server-side only, clients get a generic safe message
- Swapped the decommissioned Groq model `llama-3.3-70b-versatile` for `openai/gpt-oss-120b`
- Converted from hardcoded absolute paths (`/opt/aegis`, `/content/aegis`) to a relative, portable repo structure

### Fixed
- Groq client was previously instantiated fresh on every LLM call; now a single client instance is reused
- MarineTraffic scraper: each source (MarineTraffic, gCaptain, TradeWinds, Reuters) now fails independently instead of one broad try/except covering all sources
- Frontend JS: fixed a malformed string-concatenation bug in the `log()` function that broke the opening `<div>` tag for log entries
- `models/forecaster.py`'s synthetic date index had no frequency set, causing `ARIMA.forecast()` to throw "No supported index is available" — `/api/forecast` was silently broken; fixed via `pd.date_range(freq="D")`
- `datetime.utcnow()` deprecation warnings, fixed to timezone-aware `datetime.now(timezone.utc)`

## [2.0.0] — initial hackathon submission (AMD Developer Cloud track)

- 7-agent crisis-simulation pipeline (Signal → Intelligence → Forecast → Simulation → Decision → Alert → Execution)
- ARIMA(2,1,2) + XGBoost hybrid forecaster
- FastAPI backend with Server-Sent Events streaming
- Single-page dashboard with live agent monitor, crisis injection, business-value and go-to-market panels
```

---

## New full repo structure

```
aegis_df2/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                 # NEW — manual-trigger CI (lint/test/audit)
│   └── dependabot.yml             # NEW — weekly update PRs (optional, see note)
├── agents/
│   └── swarm.py                   # unchanged
├── api/
│   └── server.py                  # REPLACED — /ready, lifespan, optional Sentry
├── docker/                        # NEW folder
│   ├── Dockerfile
│   ├── Dockerfile.dockerignore
│   └── docker-compose.yml
├── experiments/
│   ├── README.md                  # REPLACED
│   ├── config.yaml                # REPLACED
│   ├── results.json               # regenerate: python experiments/run_baseline_comparison.py
│   └── run_baseline_comparison.py # REPLACED
├── models/
│   └── forecaster.py              # REPLACED — validated output schema
├── tests/
│   ├── __init__.py                # unchanged
│   ├── test_experiments.py        # NEW
│   ├── test_forecaster.py         # REPLACED
│   ├── test_server.py             # REPLACED
│   └── test_swarm.py              # unchanged
├── .env.example                   # NEW
├── CHANGELOG.md                   # REPLACED
├── CONTRIBUTING.md                # REPLACED
├── README.md                      # REPLACED
├── frontend.html                  # unchanged
├── pyproject.toml                 # unchanged
├── requirements.txt               # unchanged
├── requirements-dev.txt           # unchanged
├── requirements-lock.txt          # unchanged (already committed)
└── ruff.toml                      # unchanged
```

**Before pushing, in order:**
1. Apply all files locally, then run: `pip install -r requirements-lock.txt -r requirements-dev.txt && pytest -q --cov --cov-report=term-missing && ruff check .` — everything must be green.
2. Regenerate `experiments/results.json` once (`python experiments/run_baseline_comparison.py`) so the new per-day/crisis keys are in the committed output.
3. Commit in the 7-step sequence from earlier (CI config → .env.example → server → forecaster → docker → experiments → docs) — keep using real `git commit`, never web upload, so the history starts mining properly.
