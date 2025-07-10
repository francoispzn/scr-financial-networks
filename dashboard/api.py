"""
FastAPI backend for the SCR Financial Networks dashboard.

Endpoints
---------
GET  /health                   — liveness probe
GET  /simulation/state         — current bank + system state
POST /simulation/run           — run N steps
POST /simulation/shock         — apply a named or custom shock
POST /simulation/reset         — reset to initial state
GET  /spectral                 — full spectral analysis
POST /analysis/llm             — LLM narrative analysis

Run with::

    uvicorn dashboard.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import simulation_state as state
from .demo_data import SHOCK_SCENARIOS
from .data_loader import ALL_BANKS
from .llm import analyze_system_state, build_snapshot

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SCR Financial Networks API",
    description="REST API for the Spectral Coarse-Graining financial networks dashboard.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    steps: int = Field(10, ge=1, le=500, description="Number of simulation steps")
    shock_scenario: Optional[str] = Field(
        None, description="Named shock scenario to apply at step 1"
    )


class ShockRequest(BaseModel):
    scenario: Optional[str] = Field(None, description="Named shock scenario key")
    custom_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom shock parameters (bank_id → {field: delta})"
    )


class ReloadRequest(BaseModel):
    start_date: str = Field("2020-01-01", description="Start date YYYY-MM-DD")
    end_date: str = Field("2024-12-31", description="End date YYYY-MM-DD")
    bank_list: Optional[List[str]] = Field(None, description="Bank IDs to include")
    snapshot_date: Optional[str] = Field(None, description="Snapshot date YYYY-MM-DD")


class LLMRequest(BaseModel):
    model: Optional[str] = Field(None, description="Cerebras model ID override")
    api_key: Optional[str] = Field(None, description="Cerebras API key override")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/simulation/state")
def get_state() -> Dict[str, Any]:
    """Return current bank states, system metrics, and network graph data."""
    sim = state.get_simulation()
    network_data = state.get_network_graph_data()
    system_metrics = sim.get_system_metrics()
    system_metrics["time"] = sim.time
    return {
        "time": sim.time,
        "nodes": network_data["nodes"],
        "edges": network_data["edges"],
        "system_metrics": system_metrics,
    }


@app.post("/simulation/run")
def run_simulation(req: RunRequest) -> Dict[str, Any]:
    """Run the simulation for *steps* steps."""
    shocks = None
    if req.shock_scenario:
        if req.shock_scenario not in SHOCK_SCENARIOS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown shock scenario '{req.shock_scenario}'. "
                       f"Valid options: {list(SHOCK_SCENARIOS)}",
            )
        shocks = {1: SHOCK_SCENARIOS[req.shock_scenario]["params"]}

    history = state.run_steps(req.steps, shocks=shocks)
    return {"steps_run": req.steps, "current_time": state.get_simulation().time,
            "history_length": len(history)}


@app.post("/simulation/shock")
def apply_shock(req: ShockRequest) -> Dict[str, str]:
    """Apply a named or custom shock to the simulation."""
    if req.scenario:
        if req.scenario not in SHOCK_SCENARIOS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown shock scenario '{req.scenario}'.",
            )
        params = SHOCK_SCENARIOS[req.scenario]["params"]
    elif req.custom_params:
        params = req.custom_params
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'scenario' or 'custom_params'.",
        )
    state.apply_shock(params)
    return {"status": "shock applied"}


@app.post("/simulation/reset")
def reset() -> Dict[str, str]:
    state.reset_simulation()
    return {"status": "reset"}


@app.post("/simulation/reload")
def reload_data(req: ReloadRequest) -> Dict[str, Any]:
    """Re-fetch data from the pipeline with updated parameters."""
    banks = req.bank_list or ALL_BANKS
    state.reload_data(
        start_date=req.start_date,
        end_date=req.end_date,
        bank_list=banks,
        snapshot_date=req.snapshot_date,
    )
    sim = state.get_simulation()
    return {
        "status": "reloaded",
        "banks": list(sim.banks.keys()),
        "config": state.get_config(),
    }


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return state.get_config()


@app.get("/spectral")
def get_spectral() -> Dict[str, Any]:
    """Return full spectral analysis of the current network."""
    return state.get_spectral_data()


@app.get("/scenarios")
def list_scenarios() -> List[Dict[str, str]]:
    """List available shock scenarios."""
    return [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in SHOCK_SCENARIOS.items()
    ]


@app.post("/analysis/llm")
def llm_analysis(req: LLMRequest) -> Dict[str, str]:
    """Generate a narrative analysis of the current state using Cerebras LLM."""
    sim = state.get_simulation()
    network_data = state.get_network_graph_data()
    system_metrics = sim.get_system_metrics()
    system_metrics["time"] = sim.time
    spectral_data = state.get_spectral_data()

    snapshot = build_snapshot(network_data, system_metrics, spectral_data)
    narrative = analyze_system_state(snapshot, model=req.model, api_key=req.api_key)
    return {"narrative": narrative}
