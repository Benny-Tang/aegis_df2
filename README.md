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

[MarineTraffic + Lloyd's List] [Oil price / market data]
| |
+----------------+---------------+
|
[Signal Agent] <- Agent 1: Watcher
|
[Intelligence Agent] <- Agent 2: Interpreter
|
[Forecast Agent] <- Agent 3: ARIMA + XGBoost
|
[Simulation Agent] <- Agent 4: Strategist (3 scenarios)
|
[Decision Agent] <- Agent 5: Brain (ranked actions)
|
[Alert Agent] <- Agent 6: Communicator
|
[Execution Agent] <- Agent 7: Operator (ERP workflows)
|
[FastAPI + SSE]
|
[Dashboard UI]

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

aegis_df4/
├── agents/
│ └── swarm.py # 7-agent pipeline logic + 4-source marine scraper
│ # (MarineTraffic, gCaptain, TradeWinds, Reuters),
│ # deduplicated into one parameterized scraper
│ # with Pydantic-validated agent output schemas
├── api/
│ └── server.py # FastAPI app: /health, /ready, /api/marine,
│ # /api/forecast, /api/crisis, /api/stream (SSE),
│ # /api/status — typed Pydantic responses,
│ # structured JSON logging
├── models/
│ └── forecaster.py # ARIMA(2,1,2) + XGBoost hybrid forecasting model
├── experiments/
│ ├── config.yaml # Config-driven experiment parameters
│ ├── run_baseline_comparison.py # ARIMA-only vs XGBoost-only vs hybrid ablation
│ ├── results.json # Latest run's output (reproducible via config.yaml)
│ └── README.md # Findings write-up
├── tests/
│ ├── test_forecaster.py # Forecast horizon, bounds, crisis-shock, schema,
│ │ # blend_weight isolation, HAS_MODELS fallback
│ ├── test_server.py # Health, ready, forecast, status, marine endpoints
│ ├── test_swarm.py # Scraper consolidation + schema validation tests
│ └── test_experiments.py # Experiment reproducibility + per-day error tests
├── .github/
│ └── workflows/ci.yml # Lint (ruff) + test+coverage (pytest-cov, fails
│ # under 65%) + dependency audit (pip-audit) —
│ # runs on every push and pull request
├── .devcontainer/
│ └── devcontainer.json # VS Code / Codespaces dev container config
├── frontend.html # Dashboard UI: live agents, marine feed,
│ # business value panel, GTM strategy panel
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt # pytest, pytest-cov, ruff, pip-audit, PyYAML
├── requirements-lock.txt # pip-compile generated, hashed, reproducible
├── pyproject.toml # pytest + coverage configuration
├── ruff.toml
├── .env.example
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md


## Testing & CI

```bash
pip install -r requirements-lock.txt -r requirements-dev.txt
pytest -q --cov --cov-report=term-missing
ruff check .
pip-audit -r requirements-lock.txt
```

All three run automatically via GitHub Actions (`.github/workflows/ci.yml`)
on every push and pull request — the test step fails the build if
coverage drops below 65%. See `CONTRIBUTING.md` for setup details.

## Business case (from the original hackathon pitch)

| Metric | Value |
|---|---|
| Total addressable market | $1.5T |
| Supply chain market (2028) | $19.3T |
| Estimated ROI, year 1 | 70x |
| Savings per crisis prevented | $1.7M |
| Annual subscription (proposed) | $24K/yr |
| Response time vs. human analysts | 2 sec vs. 4–8 hours |

Target segments: maritime insurance underwriters, commodity traders,
manufacturers with $50M+ procurement exposure, and sovereign wealth funds
managing oil-revenue exposure.

## Running locally

```bash
git clone <this-repo>
cd aegis_df4
pip install -r requirements-lock.txt
cp .env.example .env   # then fill in GROQ_API_KEY
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker (no local Python needed):

```bash
docker compose up --build
```

Open `http://localhost:8000` for the live dashboard. Click **Inject Crisis
Scenario** to trigger the full 7-agent pipeline against a simulated Strait
of Hormuz disruption event.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness: system status, model, platform info |
| `/ready` | GET | Readiness: confirms the forecaster is fitted and usable |
| `/api/marine` | GET | Live MarineTraffic + Lloyd's List shipping feed |
| `/api/forecast` | POST | Run the forecasting model standalone |
| `/api/crisis` | POST | Run the full 7-agent pipeline synchronously |
| `/api/stream` | GET | Run the pipeline via Server-Sent Events (used by the dashboard) |
| `/api/status` | GET | Live oil price + system status snapshot |

## License

MIT
