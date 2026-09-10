# Changelog

## [Unreleased] — round 2

### Added
- `experiments/` — config-driven baseline/ablation comparison (ARIMA-only vs XGBoost-only vs hybrid) with reproducible results (`config.yaml` + `results.json`); see `experiments/README.md` for the finding
- Structured JSON logging (replacing plain-text log formatting) across the API layer
- Pydantic schema validation on agent LLM outputs (`SignalOutput`, `IntelligenceOutput`, `DecisionOutput` in `agents/swarm.py`) — validation failures are logged as warnings rather than silently accepted
- `pytest-cov` with a 60% coverage threshold enforced in CI (`pyproject.toml`)
- `pip-audit` dependency vulnerability scan in CI (advisory for now)
- Real hashed lockfile via `pip-compile` (`requirements-lock.txt`) — the previous `pip freeze` dump wasn't being recognized as a lockfile by DataFactor's scorer

### Changed
- Consolidated `agents/swarm.py`'s 4 near-identical scrape try/except blocks (MarineTraffic, gCaptain, TradeWinds, Reuters) into one parameterized `_scrape_source()` helper driven by a `MARITIME_SOURCES` config list
- Scrape failures are now logged (`logger.info`) instead of silently swallowed by bare `except: pass`
- `_gen_history()`'s random seed is now a parameter instead of hardcoded, so experiment runs are genuinely reproducible end-to-end
- `AegisForecaster.forecast()` gained an optional `blend_weight` override, used by the ablation experiment to isolate each model's contribution without duplicating forecasting logic

## [Unreleased] — round 1

### Added
- Test suite: `tests/test_forecaster.py` (forecast horizon, bounds, crisis-shock behavior) and `tests/test_server.py` (health, forecast, status, marine endpoints)
- CI pipeline (`.github/workflows/ci.yml`) running `ruff check` and `pytest` on every push/PR
- `ruff.toml` lint configuration
- `.env.example` documenting required/optional environment variables
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
