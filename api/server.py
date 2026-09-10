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

