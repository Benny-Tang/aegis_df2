"""
Aegis - Autonomous Enterprise Crisis Management
7-agent swarm pipeline (Signal -> Intelligence -> Forecast -> Simulation
-> Decision -> Alert -> Execution), backed by Groq inference.
"""
import datetime
import json
import logging
import os
import re

import requests
from bs4 import BeautifulSoup
from groq import AsyncGroq
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("aegis.swarm")

# openai/gpt-oss-120b — llama-3.3-70b-versatile was decommissioned by Groq
# on 2026-08-16; this is one of Groq's recommended replacements.
MODEL = "openai/gpt-oss-120b"

# Single shared client instead of one per call.
_CLIENT = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# One entry per maritime-news source. Consolidating what used to be 4
# near-identical try/except blocks into data + one shared scrape function
# below — same behavior, far less duplication, and failures are now
# logged instead of silently swallowed.
MARITIME_SOURCES = [
    {
        "name": "MarineTraffic", "url": "https://www.marinetraffic.com/blog/",
        "tags": ["article"], "type": "shipping_news", "min_len": 15, "max_len": 10_000, "limit": 5,
    },
    {
        "name": "gCaptain", "url": "https://gcaptain.com/",
        "tags": ["h2", "h3"], "type": "maritime_news", "min_len": 20, "max_len": 300, "limit": 8,
    },
    {
        "name": "TradeWinds", "url": "https://www.tradewindsnews.com/",
        "tags": ["h2", "h3", "h4"], "type": "shipping_intelligence", "min_len": 20, "max_len": 300, "limit": 6,
    },
    {
        "name": "Reuters", "url": "https://www.reuters.com/business/aerospace-defense/",
        "tags": ["h2", "h3"], "type": "geopolitical_news", "min_len": 20, "max_len": 300, "limit": 6,
        "keywords": ["ship", "tanker", "oil", "hormuz", "gulf", "port", "cargo", "iran", "strait", "supply"],
    },
]

SIMULATED_ITEMS = [
    {"source": "MarineTraffic", "headline": "3 tankers rerouted away from Strait of Hormuz amid escalating tensions", "type": "shipping_disruption", "url": "https://www.marinetraffic.com"},
    {"source": "MarineTraffic", "headline": "Iranian Revolutionary Guard patrol boats spotted near major shipping lanes", "type": "security_alert", "url": "https://www.marinetraffic.com"},
    {"source": "gCaptain", "headline": "War risk insurance premiums surge 40% for Gulf vessels", "type": "financial_impact", "url": "https://gcaptain.com"},
    {"source": "TradeWinds", "headline": "Major shipping lines suspend bookings through Hormuz indefinitely", "type": "operational_disruption", "url": "https://www.tradewindsnews.com"},
    {"source": "Reuters", "headline": "Oil tanker traffic through Strait of Hormuz drops 60% as conflict escalates", "type": "market_impact", "url": "https://reuters.com"},
]


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


async def _llm(system, user, temperature=0.3):
    r = await _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=600,
    )
    return r.choices[0].message.content.strip()


async def _json(system, user):
    raw = await _llm(system + "\n\nRespond ONLY with valid JSON. No markdown.", user)
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning("LLM response was not valid JSON, returning raw text. Raw: %s", raw[:200])
        return {"raw": raw}


def _validated(model_cls, data, agent_name):
    """
    Validates an agent's LLM-produced dict against its expected schema.
    Logs a warning (rather than raising) on mismatch, since a schema
    miss shouldn't crash the pipeline — but it should be visible in
    logs/metrics rather than silently accepted. Returns the original
    dict either way so downstream agents keep working off the same
    shape they always have.
    """
    try:
        model_cls(**data)
    except ValidationError as e:
        logger.warning("%s output failed schema validation: %s", agent_name, e)
    return data


def _scrape_source(source: dict) -> list[dict]:
    """Scrapes a single maritime-news source per its config. Returns a
    list of item dicts (possibly empty) — never raises; network/parse
    failures are logged and treated as "this source found nothing"."""
    items = []
    try:
        r = requests.get(source["url"], headers=HEADERS, timeout=10)
        if r.status_code != 200:
            logger.info("%s returned HTTP %s", source["name"], r.status_code)
            return items

        soup = BeautifulSoup(r.text, "lxml")
        if source["name"] == "MarineTraffic":
            elements = (
                soup.find_all("article", limit=source["limit"])
                or soup.find_all("div", class_=lambda x: x and "post" in str(x).lower(), limit=source["limit"])
            )
        else:
            elements = soup.find_all(source["tags"], limit=source["limit"])

        keywords = source.get("keywords")
        for el in elements:
            if source["name"] == "MarineTraffic":
                title_el = (
                    el.find("h2") or el.find("h3") or el.find("h1")
                    or el.find(class_=lambda x: x and "title" in str(x).lower())
                )
                text = title_el.get_text(strip=True) if title_el else ""
            else:
                text = el.get_text(strip=True)

            if not (source["min_len"] < len(text) < source["max_len"]):
                continue
            if keywords and not any(k in text.lower() for k in keywords):
                continue

            items.append({
                "source": source["name"],
                "headline": text[:250],
                "type": source["type"],
                "url": source["url"],
            })
    except requests.RequestException as e:
        logger.info("Scrape failed for %s: %s", source["name"], e)
    return items


def scrape_marine_traffic():
    """
    Scrapes multiple maritime news sources for real-time shipping
    intelligence: MarineTraffic blog, gCaptain, TradeWinds, and Reuters
    (filtered for shipping/oil-relevant keywords). Falls back to
    representative simulated Hormuz-crisis data if every source fails,
    so the pipeline never breaks on a missing external dependency.
    """
    news_items = []
    sources_tried = []
    sources_ok = []

    for source in MARITIME_SOURCES:
        sources_tried.append(source["name"])
        found = _scrape_source(source)
        if found:
            sources_ok.append(source["name"])
            news_items.extend(found)

    if news_items:
        return {
            "status": "live",
            "sources_tried": sources_tried,
            "sources_ok": sources_ok,
            "items": news_items[:8],
            "total_found": len(news_items),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    logger.info("All maritime sources returned nothing (tried: %s) — using simulated data.", sources_tried)
    return {
        "status": "simulated",
        "source": "Aegis Crisis Simulation",
        "note": "Live scraping unavailable — showing Hormuz crisis simulation",
        "sources_tried": sources_tried,
        "sources_ok": [],
        "items": SIMULATED_ITEMS,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# --- Schemas for validating each agent's LLM output shape ---

class SignalOutput(BaseModel):
    severity: str = Field(default="LOW")
    signal_type: str = Field(default="")
    confidence: int = Field(default=0, ge=0, le=100)
    summary: str = Field(default="")


class IntelligenceOutput(BaseModel):
    supply_chain_impact: str = Field(default="LOW")
    escalation_probability: int = Field(default=0, ge=0, le=100)
    time_to_impact_days: int = Field(default=0)


class DecisionOutput(BaseModel):
    threat_level: str = Field(default="LOW")
    executive_summary: str = Field(default="")


async def signal_agent(event):
    marine_data = scrape_marine_traffic()
    event = {**event, "marine_traffic": marine_data}
    r = await _json(
        "You are the Signal Agent for Aegis. Classify incoming signals.\n"
        "You have access to live MarineTraffic shipping news.\n"
        "Return JSON: severity(LOW|MEDIUM|HIGH|CRITICAL), signal_type, anomalies(list), "
        "confidence(0-100), summary(1 sentence), shipping_alerts(list of strings from marine data).",
        f"Event: {json.dumps(event)}",
    )
    r = _validated(SignalOutput, r, "signal_agent")
    r["agent"] = "signal"
    r["marine_data"] = marine_data
    return r


async def intelligence_agent(signal, event):
    r = await _json(
        "You are the Intelligence Agent for Aegis. Interpret signals for business risk.\n"
        "Return JSON: root_cause, affected_regions(list), supply_chain_impact(LOW|MEDIUM|HIGH|SEVERE), "
        "escalation_probability(0-100), geopolitical_context(1-2 sentences), time_to_impact_days(int).",
        f"Signal: {json.dumps(signal)}\nEvent: {json.dumps(event)}",
    )
    r = _validated(IntelligenceOutput, r, "intelligence_agent")
    r["agent"] = "intelligence"
    return r


async def forecast_agent(intel, forecast_data):
    r = await _json(
        "You are the Forecast Agent for Aegis. Interpret ML forecasts in business terms.\n"
        "Return JSON: oil_outlook(string), price_trajectory(RISING|STABLE|FALLING|VOLATILE), "
        "supply_risk_score(0-100), delay_probability(0-100), cost_impact_pct(float), "
        "confidence_level(LOW|MEDIUM|HIGH), key_assumptions(list).",
        f"Intel: {json.dumps(intel)}\nML summary: {json.dumps(forecast_data.get('summary', {}))}",
    )
    r["agent"] = "forecast"
    r["ml_summary"] = forecast_data.get("summary", {})
    r["ml_forecast"] = forecast_data.get("forecast", [])[:14]
    return r


async def simulation_agent(forecast, event):
    r = await _json(
        "You are the Simulation Agent for Aegis. Run 3 supply chain scenarios.\n"
        "Return JSON: scenarios(list of: name,probability,cost_impact_pct,lead_time_increase_days,"
        "description,mitigation_available).",
        f"Forecast: {json.dumps(forecast)}\nEvent: {json.dumps(event)}",
    )
    r["agent"] = "simulation"
    if not isinstance(r.get("scenarios"), list):
        logger.warning("simulation_agent returned non-list scenarios, defaulting to empty list")
        r["scenarios"] = []
    return r


async def decision_agent(simulation, forecast, intel):
    r = await _json(
        "You are the Decision Agent for Aegis — the final brain.\n"
        "Return JSON: threat_level(LOW|MEDIUM|HIGH|CRITICAL), "
        "recommended_actions(list of: priority,action,rationale,estimated_savings_usd,time_to_implement,risk), "
        "executive_summary(2-3 sentences), do_nothing_cost_usd(int).",
        f"Scenarios: {json.dumps(simulation.get('scenarios', []))}\n"
        f"Forecast: {json.dumps(forecast.get('ml_summary', {}))}\nIntel: {json.dumps(intel)}",
    )
    r = _validated(DecisionOutput, r, "decision_agent")
    r["agent"] = "decision"
    return r


async def alert_agent(decision, forecast):
    r = await _json(
        "You are the Alert Agent for Aegis. Generate stakeholder alerts.\n"
        "Return JSON: slack_message(string,max 200 chars), email_subject, email_body(3-4 sentences), "
        "severity_emoji, notification_channels(list).",
        f"Decision: {json.dumps(decision)}\nForecast: {json.dumps(forecast.get('ml_summary', {}))}",
    )
    r["agent"] = "alert"
    return r


async def execution_agent(decision):
    r = await _json(
        "You are the Execution Agent for Aegis. Determine workflows to trigger.\n"
        "Return JSON: triggered_workflows(list of: system,action,api_endpoint,payload_summary,status,"
        "requires_human_approval), autonomous_actions_count(int), pending_approvals_count(int), "
        "execution_summary(1 sentence).",
        f"Actions: {json.dumps(decision.get('recommended_actions', [])[:3])}",
    )
    r["agent"] = "execution"
    return r


async def run_aegis_pipeline(event, forecast_data):
    results = {"event": event, "agents": {}}
    sig = await signal_agent(event)
    results["agents"]["signal"] = sig
    intel = await intelligence_agent(sig, event)
    results["agents"]["intelligence"] = intel
    fore = await forecast_agent(intel, forecast_data)
    results["agents"]["forecast"] = fore
    sim = await simulation_agent(fore, event)
    results["agents"]["simulation"] = sim
    dec = await decision_agent(sim, fore, intel)
    results["agents"]["decision"] = dec
    alert = await alert_agent(dec, fore)
    results["agents"]["alert"] = alert
    exe = await execution_agent(dec)
    results["agents"]["execution"] = exe
    return results
