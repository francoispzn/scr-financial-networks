"""
Dash dashboard for SCR Financial Networks.

Tabs
----
1. Overview      — system health cards + interactive network graph
2. ABM Simulation — controls, run/shock/reset, time-series charts
3. Spectral Analysis — eigenvalue spectrum + diffusion distance heatmap
4. AI Analysis   — Cerebras LLM narrative

Run with::

    python -m dashboard.app
    # or
    python dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, ctx

# Ensure the repo root is on the path when run directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dashboard import simulation_state as sim_state
from dashboard.demo_data import SHOCK_SCENARIOS
from dashboard.llm import analyze_system_state, build_snapshot

# ── App initialisation ───────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="SCR Financial Networks",
    suppress_callback_exceptions=True,
)
server = app.server  # expose WSGI app for gunicorn / uvicorn


# ── Colour palette ───────────────────────────────────────────────────────────

COLORS = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "green": "#00b894",
    "amber": "#fdcb6e",
    "text": "#e0e0e0",
}

CET1_COLORSCALE = [
    [0.0, "#e94560"],   # red  — distressed
    [0.4, "#fdcb6e"],   # amber
    [1.0, "#00b894"],   # green — healthy
]

# ── Helper: network figure ───────────────────────────────────────────────────

def _layout_circle(n: int) -> list[tuple[float, float]]:
    """Evenly spaced positions on a unit circle."""
    angles = [2 * np.pi * i / n for i in range(n)]
    return [(np.cos(a), np.sin(a)) for a in angles]


def build_network_figure(graph_data: dict) -> go.Figure:
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    n = len(nodes)
    if n == 0:
        return go.Figure()

    positions = {nd["id"]: pos for nd, pos in zip(nodes, _layout_circle(n))}

    # ── Edge traces ──
    edge_traces = []
    max_weight = max((e["weight"] for e in edges), default=1.0)
    for edge in edges:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        w = edge["weight"] / max_weight
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(0.5, w * 4), color=f"rgba(200,200,200,{0.2 + 0.5 * w})"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # ── Node trace ──
    max_assets = max(nd["total_assets"] for nd in nodes)
    node_x = [positions[nd["id"]][0] for nd in nodes]
    node_y = [positions[nd["id"]][1] for nd in nodes]
    node_colors = [nd["CET1_ratio"] for nd in nodes]
    node_sizes = [20 + 60 * nd["total_assets"] / max_assets for nd in nodes]
    node_text = [
        f"<b>{nd['label']}</b><br>"
        f"CET1: {nd['CET1_ratio']:.1f}%<br>"
        f"LCR: {nd['LCR']:.0f}%<br>"
        f"Assets: €{nd['total_assets']/1e12:.2f}tn<br>"
        f"{'⚠ Insolvent' if not nd['solvent'] else ''}"
        f"{'⚠ Illiquid' if not nd['liquid'] else ''}"
        for nd in nodes
    ]
    node_labels = [nd["id"] for nd in nodes]

    cet1_min = min(node_colors) - 0.1
    cet1_max = max(node_colors) + 0.1

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale=CET1_COLORSCALE,
            cmin=cet1_min, cmax=cet1_max,
            colorbar=dict(title="CET1 (%)", tickfont=dict(color=COLORS["text"])),
            line=dict(width=2, color="white"),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(color="white", size=11),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font_color=COLORS["text"],
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=480,
    )
    return fig


# ── Helper: eigenvalue spectrum figure ──────────────────────────────────────

def build_eigenvalue_figure(spectral: dict) -> go.Figure:
    evals = spectral.get("eigenvalues", [])
    gap_idx = spectral.get("gap_index", 0)
    colors = [
        COLORS["highlight"] if i == gap_idx else COLORS["green"]
        for i in range(len(evals))
    ]
    fig = go.Figure(
        go.Bar(
            x=list(range(len(evals))),
            y=evals,
            marker_color=colors,
            hovertemplate="λ_%{x} = %{y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=gap_idx + 0.5,
        line_dash="dash",
        line_color=COLORS["amber"],
        annotation_text=f"Spectral gap (k={gap_idx})",
        annotation_font_color=COLORS["amber"],
    )
    fig.update_layout(
        title="Eigenvalue Spectrum",
        xaxis_title="Index", yaxis_title="λ",
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"], height=350,
    )
    return fig


# ── Helper: diffusion distance heatmap ──────────────────────────────────────

def build_diffusion_heatmap(spectral: dict) -> go.Figure:
    dd = spectral.get("diffusion_distance", [])
    bank_ids = spectral.get("bank_ids", [])
    if not dd:
        return go.Figure()
    fig = go.Figure(
        go.Heatmap(
            z=dd, x=bank_ids, y=bank_ids,
            colorscale="Viridis",
            hovertemplate="%{y} ↔ %{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Diffusion Distance Matrix",
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"], height=350,
    )
    return fig


# ── Helper: time-series figure ───────────────────────────────────────────────

def build_timeseries_figure(history: list, metric: str) -> go.Figure:
    fig = go.Figure()
    if not history:
        return fig
    bank_ids = list(history[0]["bank_states"].keys())
    palette = px.colors.qualitative.Plotly
    for i, bid in enumerate(bank_ids):
        times = [h["time"] for h in history]
        values = [h["bank_states"][bid].get(metric, None) for h in history]
        fig.add_trace(go.Scatter(
            x=times, y=values, name=bid,
            line=dict(color=palette[i % len(palette)], width=2),
            mode="lines+markers",
        ))
    fig.update_layout(
        title=f"{metric} over time",
        xaxis_title="Step", yaxis_title=metric,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"], height=320, legend_font_color=COLORS["text"],
    )
    return fig


# ── Metric card helper ───────────────────────────────────────────────────────

def metric_card(title: str, value: str, color: str = COLORS["green"]) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, style={"color": color, "fontWeight": "bold"}),
        ]),
        style={"background": COLORS["card"], "border": f"1px solid {color}"},
        className="mb-2",
    )


# ── Layout ───────────────────────────────────────────────────────────────────

def _overview_tab() -> dbc.Tab:
    return dbc.Tab(label="Overview", tab_id="tab-overview", children=[
        dbc.Row(id="overview-cards", className="mt-3 g-2"),
        dbc.Row([
            dbc.Col([
                html.H5("Interbank Network", style={"color": COLORS["text"]}),
                dcc.Graph(id="network-graph"),
            ], width=12),
        ], className="mt-2"),
    ])


def _simulation_tab() -> dbc.Tab:
    scenario_opts = [{"label": "None", "value": ""}] + [
        {"label": v["label"], "value": k} for k, v in SHOCK_SCENARIOS.items()
    ]
    return dbc.Tab(label="ABM Simulation", tab_id="tab-sim", children=[
        dbc.Row([
            # ── Controls ──
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Simulation Controls",
                                   style={"background": COLORS["accent"],
                                          "color": "white"}),
                    dbc.CardBody([
                        dbc.Label("Steps to run", style={"color": COLORS["text"]}),
                        dcc.Slider(1, 50, 1, value=10, id="steps-slider",
                                   marks={1: "1", 10: "10", 25: "25", 50: "50"},
                                   tooltip={"always_visible": True}),
                        html.Hr(style={"borderColor": COLORS["accent"]}),
                        dbc.Label("Shock scenario", style={"color": COLORS["text"]}),
                        dcc.Dropdown(
                            id="shock-dropdown",
                            options=scenario_opts,
                            value="",
                            clearable=False,
                            style={"background": COLORS["card"], "color": "black"},
                        ),
                        html.Div(id="shock-description",
                                 style={"color": COLORS["amber"], "fontSize": "0.8rem",
                                        "marginTop": "0.5rem"}),
                        html.Hr(style={"borderColor": COLORS["accent"]}),
                        dbc.Button("▶ Run", id="btn-run", color="success",
                                   className="me-2 w-100 mb-2"),
                        dbc.Button("⚡ Apply Shock", id="btn-shock", color="warning",
                                   className="me-2 w-100 mb-2",
                                   disabled=True),
                        dbc.Button("↺ Reset", id="btn-reset", color="secondary",
                                   className="w-100"),
                        html.Div(id="sim-status", className="mt-2",
                                 style={"color": COLORS["green"], "fontSize": "0.85rem"}),
                    ]),
                ], style={"background": COLORS["card"]}),
            ], width=3),

            # ── Charts ──
            dbc.Col([
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[
                        {"label": "CET1 Ratio (%)", "value": "CET1_ratio"},
                        {"label": "LCR (%)", "value": "LCR"},
                        {"label": "NSFR (%)", "value": "NSFR"},
                        {"label": "Cash (€)", "value": "cash"},
                        {"label": "Interbank Assets (€)", "value": "interbank_assets"},
                        {"label": "Interbank Liabilities (€)", "value": "interbank_liabilities"},
                    ],
                    value="CET1_ratio",
                    clearable=False,
                    style={"background": COLORS["card"], "color": "black"},
                    className="mb-2",
                ),
                dcc.Graph(id="timeseries-graph"),
                dcc.Graph(id="system-metrics-graph"),
            ], width=9),
        ], className="mt-3"),
    ])


def _spectral_tab() -> dbc.Tab:
    return dbc.Tab(label="Spectral Analysis", tab_id="tab-spectral", children=[
        dbc.Row(id="spectral-cards", className="mt-3 g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="eigenvalue-graph"), width=6),
            dbc.Col(dcc.Graph(id="diffusion-heatmap"), width=6),
        ], className="mt-2"),
    ])


def _llm_tab() -> dbc.Tab:
    return dbc.Tab(label="AI Analysis", tab_id="tab-llm", children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cerebras LLM Financial Analyst",
                                   style={"background": COLORS["accent"], "color": "white"}),
                    dbc.CardBody([
                        dbc.Label("Model", style={"color": COLORS["text"]}),
                        dcc.Dropdown(
                            id="llm-model-dropdown",
                            options=[
                                {"label": "Llama 4 Scout 17B", "value": "llama-4-scout-17b-16e-instruct"},
                                {"label": "Llama 3.1 70B", "value": "llama3.1-70b"},
                                {"label": "Llama 3.1 8B", "value": "llama3.1-8b"},
                            ],
                            value="llama-4-scout-17b-16e-instruct",
                            clearable=False,
                            style={"color": "black"},
                        ),
                        html.Br(),
                        dbc.Button("Analyse Current State", id="btn-llm",
                                   color="primary", className="w-100"),
                        html.Div(id="llm-spinner", className="mt-2"),
                    ]),
                ], style={"background": COLORS["card"]}),
            ], width=3),
            dbc.Col([
                dcc.Loading(
                    id="llm-loading",
                    type="circle",
                    color=COLORS["highlight"],
                    children=dcc.Markdown(
                        id="llm-output",
                        style={
                            "background": COLORS["card"],
                            "color": COLORS["text"],
                            "padding": "1.5rem",
                            "borderRadius": "8px",
                            "minHeight": "400px",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                ),
            ], width=9),
        ], className="mt-3"),
    ])


app.layout = dbc.Container(
    fluid=True,
    style={"background": COLORS["bg"], "minHeight": "100vh", "padding": "1rem"},
    children=[
        # ── Header ──
        dbc.Row([
            dbc.Col([
                html.H2("SCR Financial Networks",
                        style={"color": "white", "marginBottom": "0"}),
                html.Small("Spectral Coarse-Graining · ABM · AI Analysis",
                           style={"color": COLORS["text"]}),
            ], width=9),
            dbc.Col([
                html.Div(id="header-time",
                         style={"color": COLORS["text"], "textAlign": "right",
                                "paddingTop": "0.5rem"}),
            ], width=3),
        ], className="mb-3"),

        # ── Tabs ──
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-overview",
            children=[
                _overview_tab(),
                _simulation_tab(),
                _spectral_tab(),
                _llm_tab(),
            ],
            style={"borderBottom": f"2px solid {COLORS['accent']}"},
        ),

        # ── Hidden stores ──
        dcc.Store(id="sim-history-store", data=[]),
        dcc.Interval(id="auto-refresh", interval=5_000, disabled=True),
    ],
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("shock-description", "children"),
    Output("btn-shock", "disabled"),
    Input("shock-dropdown", "value"),
)
def update_shock_description(scenario: str):
    if scenario and scenario in SHOCK_SCENARIOS:
        desc = SHOCK_SCENARIOS[scenario]["description"]
        return desc, False
    return "", True


@callback(
    Output("sim-status", "children"),
    Output("sim-history-store", "data"),
    Input("btn-run", "n_clicks"),
    Input("btn-shock", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("steps-slider", "value"),
    State("shock-dropdown", "value"),
    State("sim-history-store", "data"),
    prevent_initial_call=True,
)
def handle_simulation_actions(
    run_clicks, shock_clicks, reset_clicks,
    steps, scenario, current_history
):
    triggered = ctx.triggered_id
    if triggered == "btn-reset":
        sim_state.reset_simulation()
        return "Reset to initial state.", []

    if triggered == "btn-shock" and scenario:
        params = SHOCK_SCENARIOS[scenario]["params"]
        sim_state.apply_shock(params)
        return f"Shock applied: {SHOCK_SCENARIOS[scenario]['label']}", current_history

    if triggered == "btn-run":
        history = sim_state.run_steps(steps)
        # Serialise history for dcc.Store (keep last 100 steps to stay lean)
        serialised = [
            {
                "time": h["time"],
                "bank_states": h["bank_states"],
                "system_indicators": h["system_indicators"],
            }
            for h in history[-100:]
        ]
        sim = sim_state.get_simulation()
        return f"Ran {steps} steps. Current time: {sim.time}", serialised

    return dash.no_update, dash.no_update


@callback(
    Output("overview-cards", "children"),
    Output("network-graph", "figure"),
    Output("header-time", "children"),
    Input("main-tabs", "active_tab"),
    Input("sim-history-store", "data"),
)
def update_overview(active_tab, _history):
    sim = sim_state.get_simulation()
    metrics = sim.get_system_metrics()
    graph_data = sim_state.get_network_graph_data()

    cet1 = metrics.get("avg_CET1_ratio", 0)
    lcr = metrics.get("avg_LCR", 0)
    ciss = metrics.get("CISS", metrics.get("CISS", 0))
    density = metrics.get("network_density", 0)
    sentiment = sim.calculate_market_sentiment()

    cet1_color = COLORS["green"] if cet1 >= 10.5 else (COLORS["amber"] if cet1 >= 8.0 else COLORS["highlight"])
    lcr_color = COLORS["green"] if lcr >= 100 else COLORS["highlight"]
    ciss_color = COLORS["highlight"] if ciss > 0.4 else (COLORS["amber"] if ciss > 0.2 else COLORS["green"])

    cards = [
        dbc.Col(metric_card("Simulation Step", str(sim.time)), width=2),
        dbc.Col(metric_card("Avg CET1 Ratio", f"{cet1:.2f}%", cet1_color), width=2),
        dbc.Col(metric_card("Avg LCR", f"{lcr:.1f}%", lcr_color), width=2),
        dbc.Col(metric_card("CISS", f"{ciss:.3f}", ciss_color), width=2),
        dbc.Col(metric_card("Network Density", f"{density:.3f}"), width=2),
        dbc.Col(metric_card("Market Sentiment", f"{sentiment:.2f}"), width=2),
    ]

    fig = build_network_figure(graph_data)
    header = f"Step {sim.time} | {len(sim.banks)} banks"
    return cards, fig, header


@callback(
    Output("timeseries-graph", "figure"),
    Output("system-metrics-graph", "figure"),
    Input("metric-dropdown", "value"),
    Input("sim-history-store", "data"),
)
def update_timeseries(metric, history):
    ts_fig = build_timeseries_figure(history, metric)

    # System metrics (CISS + avg CET1)
    sys_fig = go.Figure()
    if history:
        times = [h["time"] for h in history]
        ciss_vals = [h["system_indicators"].get("CISS", 0) for h in history]
        sys_fig.add_trace(go.Scatter(
            x=times, y=ciss_vals, name="CISS",
            line=dict(color=COLORS["highlight"], width=2), mode="lines",
        ))
    sys_fig.update_layout(
        title="System Stress (CISS)",
        xaxis_title="Step", yaxis_title="CISS",
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        font_color=COLORS["text"], height=200,
    )

    return ts_fig, sys_fig


@callback(
    Output("spectral-cards", "children"),
    Output("eigenvalue-graph", "figure"),
    Output("diffusion-heatmap", "figure"),
    Input("main-tabs", "active_tab"),
    Input("sim-history-store", "data"),
)
def update_spectral(active_tab, _history):
    if active_tab != "tab-spectral":
        return dash.no_update, dash.no_update, dash.no_update

    spectral = sim_state.get_spectral_data()

    cards = [
        dbc.Col(metric_card(
            "Algebraic Connectivity",
            f"{spectral['algebraic_connectivity']:.4f}",
        ), width=3),
        dbc.Col(metric_card(
            "Spectral Gap",
            f"{spectral['gap_size']:.4f} (k={spectral['gap_index']})",
            COLORS["amber"],
        ), width=3),
        dbc.Col(metric_card(
            "Spectral Radius",
            f"{spectral['spectral_radius']:.4f}",
        ), width=3),
        dbc.Col(metric_card(
            "Suggested Clusters",
            str(spectral["gap_index"] + 1),
            COLORS["highlight"],
        ), width=3),
    ]

    return cards, build_eigenvalue_figure(spectral), build_diffusion_heatmap(spectral)


@callback(
    Output("llm-output", "children"),
    Input("btn-llm", "n_clicks"),
    State("llm-model-dropdown", "value"),
    prevent_initial_call=True,
)
def run_llm_analysis(n_clicks, model):
    sim = sim_state.get_simulation()
    graph_data = sim_state.get_network_graph_data()
    system_metrics = sim.get_system_metrics()
    system_metrics["time"] = sim.time
    spectral_data = sim_state.get_spectral_data()

    snapshot = build_snapshot(graph_data, system_metrics, spectral_data)
    return analyze_system_state(snapshot, model=model)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8050))
    debug = os.environ.get("DASHBOARD_DEBUG", "true").lower() == "true"
    print(f"Starting SCR Financial Networks dashboard on http://localhost:{port}")
    app.run(debug=debug, port=port)
