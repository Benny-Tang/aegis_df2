Here are all files complete and final — copy-paste each into the exact path shown. New files marked **NEW**, the rest are **full replacements** of existing ones.

---

## 1. `.github/workflows/ci.yml` — **NEW** (manual trigger only, no automatic runs)

```yaml
name: CI

# Manually triggered only. Go to: Actions tab -> CI -> "Run workflow" -> pick branch.
# Nothing runs automatically on push or pull_request.
on:
  workflow_dispatch:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dev tooling
        run: pip install -r requirements-dev.txt
      - name: Ruff lint
        run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install locked dependencies
        run: pip install -r requirements-lock.txt -r requirements-dev.txt
      - name: Run tests with 60% coverage gate
        run: pytest -q --cov --cov-report=term-missing
        env:
          LOG_LEVEL: WARNING

  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install pip-audit
        run: pip install pip-audit
      - name: Dependency vulnerability scan
        run: pip-audit -r requirements-lock.txt
```

## 2. `.github/dependabot.yml` — **NEW** (⚠️ read note below)

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

> **Note:** This is *not* a workflow — it never executes anything. It only opens update PRs weekly (that is precisely what the scorer's `dep_update_tooling` check looks for). Since your CI is manual-only, those PRs won't auto-test; just run the CI workflow manually on the dependabot branch before merging. If you'd rather not have any automation at all, delete this file — you'll just leave the Dependency Health points on the table.

## 3. `.env.example` — **NEW**

```bash
# Required for the 7-agent pipeline (agents/swarm.py -> Groq API).
# Get a key at https://console.groq.com/keys
GROQ_API_KEY=

# Optional: log verbosity (DEBUG | INFO | WARNING | ERROR)
LOG_LEVEL=INFO

# Optional: Sentry DSN for error tracking. Leave blank to disable.
SENTRY_DSN=
```

## 4. `docker/Dockerfile` — **NEW**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5. `docker/Dockerfile.dockerignore` — **NEW**

```
.venv
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.env
.git
.github
tests
experiments
```

## 6. `docker/docker-compose.yml` — **NEW**

```yaml
services:
  aegis:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - ../.env
    environment:
      LOG_LEVEL: INFO
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
```

## 7. `api/server.py` — **FULL REPLACEMENT**

```python
"""
Aegis - FastAPI service layer: REST endpoints + Server-Sent Events streaming
for the 7-agent crisis-management pipeline.
"""
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agents.swarm import (
    MODEL,
    alert_agent,
    decision_agent,
    execution_agent,
    forecast_agent,
    intelligence_agent,
    run_aegis_pipeline,
    scrape_marine_traffic,
    signal_agent,
    simulation_agent,
)
from models.forecaster import get_forecaster


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter. Plain-text logs are hard to query/alert
    on in any real log aggregation system (CloudWatch, Datadog, ELK, etc)
    — JSON lines are the standard for anything beyond local dev.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), handlers=[_handler])
logger = logging.getLogger("aegis")

# --- Optional error tracking -------------------------------------------------
# Enabled only when SENTRY_DSN is set. The service boots fully (and
# credential-free) without it — sentry_sdk is an optional import.
try:
    import sentry_sdk

    if os.environ.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=0.2)
        logger.info("Sentry error tracking enabled")
except ImportError:
    pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Aegis starting on AMD Developer Cloud...")
    try:
        get_forecaster()
        logger.info("Forecaster ready")
    except Exception as e:
        logger.warning("Forecaster initialization warning: %s", e)
    yield


app = FastAPI(title="Aegis", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000, 1)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    response.headers["X-Powered-By"] = "AMD Instinct MI300X"
    logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


class CrisisEvent(BaseModel):
    oil_price_change_pct: float = Field(default=18.0)
    shipping_disruption: str = Field(default="Strait of Hormuz disruption")
    news_headline: str = Field(default="Regional conflict near Persian Gulf")
    severity: str = Field(default="HIGH")
    disruption_factor: float = Field(default=0.7, ge=0, le=1)
    horizon_days: int = Field(default=14, ge=1, le=90)


class ForecastRequest(BaseModel):
    oil_shock_pct: float = Field(default=0.0)
    disruption_factor: float = Field(default=0.0, ge=0, le=1)
    horizon_days: int = Field(default=14, ge=1, le=90)


# --- Typed response models ---

class HealthResponse(BaseModel):
    status: str
    system: str
    version: str
    timestamp: str
    agents: int
    platform: str
    model: str
    forecast: str


class ReadyResponse(BaseModel):
    status: str
    forecaster_ready: bool
    gpu: bool
    model: str


class MarineResponse(BaseModel):
    success: bool
    data: dict[str, Any]


class ForecastResponse(BaseModel):
    success: bool
    data: dict[str, Any]


class CrisisResponse(BaseModel):
    success: bool
    timestamp: str
    data: dict[str, Any]


class StatusResponse(BaseModel):
    agents_online: int
    platform: str
    oil_price: float
    risk_level: str
    groq_model: str
    forecast_model: str
    tam: str
    roi: str


def _safe_error(exc: Exception, context: str) -> HTTPException:
    """
    Logs the full exception server-side (with traceback) but returns only a
    generic, non-leaking message to the client.
    """
    logger.exception("Error in %s: %s", context, exc)
    return HTTPException(status_code=500, detail=f"Internal error in {context}. See server logs for details.")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe: process is up."""
    return HealthResponse(
        status="online",
        system="Aegis",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        agents=7,
        platform="AMD Developer Cloud",
        model=MODEL,
        forecast="ARIMA(2,1,2) + XGBoost hybrid",
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness probe: /health says the process is up; /ready confirms the
    heavy dependency (the fitted forecaster) is actually usable."""
    fc = get_forecaster()
    return ReadyResponse(
        status="ready" if fc.fitted else "not_ready",
        forecaster_ready=fc.fitted,
        gpu=fc.gpu,
        model=MODEL,
    )


@app.get("/api/marine", response_model=MarineResponse)
async def marine_feed():
    """Live MarineTraffic shipping news feed."""
    try:
        data = scrape_marine_traffic()
        return MarineResponse(success=True, data=data)
    except Exception as e:
        raise _safe_error(e, "marine_feed")


@app.post("/api/forecast", response_model=ForecastResponse)
async def forecast_only(req: ForecastRequest):
    try:
        fc = get_forecaster()
        result = fc.forecast(req.horizon_days, req.oil_shock_pct, req.disruption_factor)
        return ForecastResponse(success=True, data=result)
    except Exception as e:
        raise _safe_error(e, "forecast_only")


@app.post("/api/crisis", response_model=CrisisResponse)
async def run_crisis(event: CrisisEvent):
    try:
        fc = get_forecaster()
        forecast_data = fc.forecast(event.horizon_days, event.oil_price_change_pct, event.disruption_factor)
        results = await run_aegis_pipeline(event.model_dump(), forecast_data)
        return CrisisResponse(success=True, timestamp=datetime.now(timezone.utc).isoformat(), data=results)
    except Exception as e:
        raise _safe_error(e, "run_crisis")


@app.get("/api/stream")
async def stream_crisis(oil_change: float = 18.0, disruption: float = 0.7, severity: str = "HIGH"):
    async def gen() -> AsyncGenerator[str, None]:
        def sse(d):
            return f"data: {json.dumps(d)}\n\n"

        yield sse({"type": "start", "message": "Aegis pipeline initiated"})
        try:
            fc = get_forecaster()
            fd = fc.forecast(14, oil_change, disruption)
            yield sse({"type": "forecast_ready", "data": fd["summary"]})

            ev = {
                "oil_price_change_pct": oil_change,
                "shipping_disruption": "Strait of Hormuz",
                "news_headline": "Geopolitical escalation",
                "severity": severity,
                "disruption_factor": disruption,
            }

            yield sse({"type": "agent_start", "agent": "signal", "index": 1})
            sig = await signal_agent(ev)
            yield sse({"type": "agent_done", "agent": "signal", "data": sig})

            yield sse({"type": "agent_start", "agent": "intelligence", "index": 2})
            intel = await intelligence_agent(sig, ev)
            yield sse({"type": "agent_done", "agent": "intelligence", "data": intel})

            yield sse({"type": "agent_start", "agent": "forecast", "index": 3})
            fore = await forecast_agent(intel, fd)
            yield sse({"type": "agent_done", "agent": "forecast", "data": fore})

            yield sse({"type": "agent_start", "agent": "simulation", "index": 4})
            sim = await simulation_agent(fore, ev)
            yield sse({"type": "agent_done", "agent": "simulation", "data": sim})

            yield sse({"type": "agent_start", "agent": "decision", "index": 5})
            dec = await decision_agent(sim, fore, intel)
            yield sse({"type": "agent_done", "agent": "decision", "data": dec})

            yield sse({"type": "agent_start", "agent": "alert", "index": 6})
            alert = await alert_agent(dec, fore)
            yield sse({"type": "agent_done", "agent": "alert", "data": alert})

            yield sse({"type": "agent_start", "agent": "execution", "index": 7})
            exe = await execution_agent(dec)
            yield sse({"type": "agent_done", "agent": "execution", "data": exe})

            yield sse({
                "type": "complete",
                "message": "All 7 agents finished",
                "summary": dec.get("executive_summary", ""),
                "threat": dec.get("threat_level", ""),
            })
        except Exception as e:
            logger.exception("Error in stream_crisis pipeline: %s", e)
            yield sse({"type": "error", "message": "Pipeline error — see server logs for details."})

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        agents_online=7,
        platform="AMD Developer Cloud",
        oil_price=round(82 + random.uniform(-2, 2), 2),
        risk_level="LOW",
        groq_model=MODEL,
        forecast_model="ARIMA(2,1,2)+XGBoost",
        tam="$1.5T",
        roi="70x",
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE_DIR, "frontend.html")) as f:
        return f.read()
```

## 8. `models/forecaster.py` — **FULL REPLACEMENT**

```python
"""
Aegis - ARIMA + XGBoost hybrid forecaster for oil price / supply-chain
disruption scenarios. Runs on GPU (CUDA) if available, falls back to CPU.
"""
import logging
import subprocess
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel

warnings.filterwarnings("ignore")

logger = logging.getLogger("aegis.forecaster")

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.arima.model import ARIMA
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def _detect_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True, "cuda"
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return False, "cpu"


GPU_AVAILABLE, XGB_DEVICE = _detect_gpu()


def _gen_history(n=120, base=82.0, seed=42):
    """Synthetic oil-price history used to seed the model when no real
    price feed is wired in. Replace with a live price feed for production use."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    prices = [base]
    for _ in range(n - 1):
        shock = np.random.normal(0, 1.2)
        drift = 0.05 * (base - prices[-1])
        prices.append(max(prices[-1] + drift + shock, 40))
    return pd.DataFrame({"price": prices}, index=dates)


def _features(df):
    df = df.copy()
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    df["rm7"] = df["price"].rolling(7).mean()
    df["rs7"] = df["price"].rolling(7).std()
    df["rm14"] = df["price"].rolling(14).mean()
    df["pct3"] = df["price"].pct_change(3)
    df["dow"] = df.index.dayofweek
    return df.dropna()


FEAT = ["lag_1", "lag_3", "lag_7", "lag_14", "rm7", "rs7", "rm14", "pct3", "dow"]


# --- Validated output schema -------------------------------------------------
# Guarantees every consumer (API layer, experiments, dashboard) sees a
# consistent, type-checked structure from forecast().

class ForecastPoint(BaseModel):
    date: str
    price: float
    lower: float
    upper: float


class ForecastSummary(BaseModel):
    final_price: float
    pct_change: float
    peak_price: float
    peak_day: int
    risk_score: float
    delay_prob: float
    cost_impact: float


class ForecastResult(BaseModel):
    base_price: float
    shocked_price: float
    horizon_days: int
    gpu_accelerated: bool
    device: str
    forecast: list[ForecastPoint]
    summary: ForecastSummary
    model: str


class AegisForecaster:
    def __init__(self):
        self.arima = self.xgb = self.scaler = None
        self.hist = None
        self.fitted = False
        self.gpu = GPU_AVAILABLE
        self.device = XGB_DEVICE

    def fit(self, df=None):
        self.hist = df if df is not None else _gen_history()
        if not HAS_MODELS:
            self.fitted = True
            return self
        try:
            self.arima = ARIMA(self.hist["price"], order=(2, 1, 2)).fit()
        except Exception as e:
            logger.warning("ARIMA fit failed, using drift fallback in forecast(): %s", e)
            self.arima = None

        fd = _features(self.hist)
        X = fd[FEAT].values
        y = fd["price"].values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        if self.gpu:
            params["device"] = "cuda"
        params["tree_method"] = "hist"

        self.xgb = xgb.XGBRegressor(**params).fit(Xs, y)
        self.fitted = True
        return self

    def forecast(self, horizon_days=14, crisis_shock=0.0, disruption_factor=0.0, blend_weight=None):
        """
        blend_weight overrides the ARIMA/XGBoost blend ratio (0.0 = pure
        ARIMA, 1.0 = pure XGBoost). Left as None in normal use, where the
        weight is chosen automatically based on crisis_shock (see below).
        This override exists so experiments/run_baseline_comparison.py can
        isolate each model's contribution for an ablation comparison
        without duplicating the forecasting logic.
        """
        if not self.fitted:
            self.fit()
        base = float(self.hist["price"].iloc[-1])
        shocked = base * (1 + crisis_shock / 100)
        dates = [datetime.today() + timedelta(days=i + 1) for i in range(horizon_days)]

        if self.arima:
            arima_p = list(self.arima.forecast(steps=horizon_days))
        else:
            arima_p = [base + np.random.normal(0, 1) * (i + 1) ** 0.5 for i in range(horizon_days)]

        rw = list(self.hist["price"].tail(14).values)
        if crisis_shock > 0:
            rw[-1] = shocked

        xgb_p = []
        for step in range(horizon_days):
            pw = rw[-14:]
            f = np.array([[
                pw[-1], pw[-3], pw[-7], pw[0],
                np.mean(pw[-7:]), np.std(pw[-7:]), np.mean(pw),
                (pw[-1] - pw[-4]) / pw[-4] if pw[-4] != 0 else 0,
                (datetime.today().weekday() + step + 1) % 7,
            ]])
            pred = float(self.xgb.predict(self.scaler.transform(f))[0])
            pred *= (1 + disruption_factor * 0.8 * (1 - step / horizon_days))
            xgb_p.append(pred)
            rw.append(pred)

        w = blend_weight if blend_weight is not None else (0.7 if crisis_shock > 0 else 0.4)
        prices = [round(w * xgb_p[i] + (1 - w) * arima_p[i], 2) for i in range(horizon_days)]
        std = float(self.hist["price"].pct_change().std()) * base
        lower = [round(p - 1.96 * std * ((i + 1) ** 0.4), 2) for i, p in enumerate(prices)]
        upper = [round(p + 1.96 * std * ((i + 1) ** 0.4), 2) for i, p in enumerate(prices)]

        fp = prices[-1]
        pct = round((fp - base) / base * 100, 1)

        result = {
            "base_price": round(base, 2),
            "shocked_price": round(shocked, 2),
            "horizon_days": horizon_days,
            "gpu_accelerated": self.gpu,
            "device": "GPU (CUDA)" if self.gpu else "CPU",
            "forecast": [
                {"date": d.strftime("%Y-%m-%d"), "price": p, "lower": lo, "upper": u}
                for d, p, lo, u in zip(dates, prices, lower, upper)
            ],
            "summary": {
                "final_price": fp,
                "pct_change": pct,
                "peak_price": round(max(prices), 2),
                "peak_day": prices.index(max(prices)) + 1,
                "risk_score": min(100, round(abs(pct) * 1.5 + disruption_factor * 40 + (crisis_shock / 100) * 30, 1)),
                "delay_prob": min(99, round(disruption_factor * 65 + (pct / 100) * 20, 1)),
                "cost_impact": round(pct * 0.35 + disruption_factor * 18, 1),
            },
            "model": "ARIMA+XGBoost hybrid (GPU)" if self.gpu else "ARIMA+XGBoost hybrid (CPU)",
        }
        return ForecastResult(**result).model_dump()


_fc = None


def get_forecaster():
    global _fc
    if _fc is None:
        _fc = AegisForecaster().fit()
    return _fc
```

## 9. `tests/test_server.py` — **FULL REPLACEMENT**

```python
"""
Tests for api.server — the FastAPI app exposing Aegis's endpoints.
Uses FastAPI's TestClient so these run without a live server or network
access (forecast/health endpoints don't touch the network; marine
scraping is exercised separately and falls back to simulated data when
offline, so it's safe to hit in CI too).
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint_returns_200_and_expected_keys():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "system", "version", "agents", "platform", "model", "forecast"):
        assert key in body
    assert body["status"] == "online"
    assert body["agents"] == 7


def test_health_works_without_groq_api_key(monkeypatch):
    """Dependency-free smoke test: the service must boot and answer
    /health with no external credentials configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_ready_endpoint_reports_forecaster_state():
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["forecaster_ready"] is True
    assert body["status"] == "ready"
    assert "gpu" in body


def test_forecast_endpoint_returns_200_and_expected_shape():
    response = client.post("/api/forecast", json={"oil_shock_pct": 0.0, "disruption_factor": 0.0, "horizon_days": 7})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["data"]["forecast"]) == 7


def test_forecast_endpoint_validates_horizon_days():
    """horizon_days is constrained to 1-90 by the Pydantic model."""
    response = client.post("/api/forecast", json={"horizon_days": 999})
    assert response.status_code == 422


def test_status_endpoint_returns_200_and_expected_keys():
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    for key in ("agents_online", "platform", "oil_price", "risk_level", "groq_model", "forecast_model"):
        assert key in body


def test_marine_endpoint_returns_200_even_when_offline():
    """scrape_marine_traffic() falls back to simulated data on any
    network failure, so this endpoint should never itself 500 due to
    connectivity issues in a CI environment."""
    response = client.get("/api/marine")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "items" in body["data"]
```

## 10. `tests/test_forecaster.py` — **FULL REPLACEMENT**

```python
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
```

## 11. `tests/test_experiments.py` — **NEW**

```python
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
```

## 12. `experiments/run_baseline_comparison.py` — **FULL REPLACEMENT**

```python
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
```

## 13. `experiments/config.yaml` — **FULL REPLACEMENT**

```yaml
# Config for experiments/run_baseline_comparison.py
# Change these and re-run to reproduce or extend the comparison —
# nothing in the script itself needs editing for a different run.

seed: 42
history_days: 150      # total synthetic days generated
holdout_days: 14        # last N days held out as ground truth to forecast against
base_price: 82.0        # starting synthetic oil price ($/bbl)

# Crisis-scenario stress test parameters (crisis_scenario block in results)
crisis_shock_pct: 18.0          # oil price shock applied during the stress test
crisis_disruption_factor: 0.7   # disruption factor applied during the stress test
```

## 14. `experiments/README.md` — **FULL REPLACEMENT**

```markdown
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

## Finding (seed=42, 14-day holdout, no crisis shock)

| Variant | MAE | RMSE |
|---|---|---|
| **arima_only** | **1.00** | **1.21** |
| hybrid_default | 1.45 | 1.64 |
| xgboost_only | 2.31 | 2.46 |

**ARIMA-only outperforms both XGBoost-only and the hybrid blend** on
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

## Per-day error and crisis-scenario analysis (round 3)

`results.json` now stores, per variant:

- `per_day_error` — absolute error for each horizon day (day 1..14)
- `mean_late_horizon_error` — mean error over the final 3 days, a
  rough proxy for how error compounds with horizon length

It also stores a `crisis_scenario` block: ARIMA-only vs the hybrid's
crisis blend (`blend_weight=None`, `crisis_shock=18%`,
`disruption=0.7` — both configurable in `config.yaml`) run against
the same calm holdout, with `mean_forecast_price` capturing the
shock response.

How to read it: the round-2 finding (ARIMA wins on a calm holdout)
holds by construction — the synthetic series is a mean-reverting walk
with no exogenous features, and MAE against a *calm* holdout cannot
reward shock responsiveness. The crisis block exists to verify the
hybrid does what it was built for: respond to shock/disruption inputs
ARIMA never sees. A working hybrid shows a materially higher
`mean_forecast_price` under shock than ARIMA-only; similar prices
would mean the blend is effectively ignoring its crisis inputs.

The comparison that actually justifies the blend weights needs a real
price feed and real crisis windows — re-run this script once the
synthetic `_gen_history()` is replaced.
```

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

```
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
```

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

```
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
```

## Testing & CI

```bash
pip install -r requirements-lock.txt -r requirements-dev.txt
pytest -q --cov --cov-report=term-missing
ruff check .
pip-audit -r requirements-lock.txt
```

The CI workflow (`.github/workflows/ci.yml`) runs all three as separate
jobs (lint / test / audit) and is triggered **manually** from the
Actions tab → CI → "Run workflow" — nothing runs automatically on push.
The test step fails the build if coverage drops below 60%.
See `CONTRIBUTING.md` for setup details.

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
cp .env.example .env   # then fill in GROQ_API_KEY
pip install -r requirements-lock.txt   # reproducible, hashed pins
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker (no local Python needed):

```bash
docker compose -f docker/docker-compose.yml up --build
# or standalone:
docker build -f docker/Dockerfile -t aegis .
docker run -p 8000:8000 --env-file .env aegis
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
````

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
