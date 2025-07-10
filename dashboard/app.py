"""
SCR Financial Networks — Production Dashboard
"""
from __future__ import annotations

import os
import sys
import threading
import time as _time
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx, no_update, clientside_callback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_KEY = "csk-525d56feckfxwwc4cd5n4kd63mrfd334n2cdh5jjnypfkj4x"
if not os.environ.get("CEREBRAS_API_KEY"):
    os.environ["CEREBRAS_API_KEY"] = _KEY

from dashboard import simulation_state as sim_state
from dashboard.demo_data import SHOCK_SCENARIOS
from dashboard.data_loader import ALL_BANKS, BANK_LABELS
from dashboard.llm import (
    analyze_system_state, build_snapshot,
    fetch_bank_data_via_llm, fetch_bank_features_for_gnn,
)
from dashboard.gnn_export import build_and_export
from dashboard.data_api import build_simulation_inputs_from_api
from dashboard.prediction import (
    generate_evolution_data, train_predictor, build_scg_vs_basel_comparison,
    compute_scg_reconstruction_accuracy,
)
from scr_financial.ml.gnn_predictor import TARGET_NAMES

# ─── Training progress (shared between background thread and polling callback)
_training_state = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "test_loss": 0.0,
    "pct": 0.0,
    "done": False,
    "error": None,
    "start_time": 0.0,
    "ram_mb": 0.0,
}
_training_lock = threading.Lock()

# ─── Tokens ───────────────────────────────────────────────────────────────────
BG     = "#060a12"
SURF   = "#0b1220"
SURF2  = "#0f1a2e"
SURF3  = "#131f36"
BORDER = "#182435"
BORD2  = "#1e2f47"

BLUE   = "#3b82f6"
BLUE2  = "#60a5fa"
GREEN  = "#10b981"
GREEN2 = "#34d399"
RED    = "#ef4444"
RED2   = "#f87171"
AMBER  = "#f59e0b"
AMBER2 = "#fbbf24"
PURP   = "#8b5cf6"
PURP2  = "#a78bfa"
TEAL   = "#14b8a6"
TEAL2  = "#2dd4bf"

TEXT   = "#e2e8f0"
TEXT2  = "#94a3b8"
TEXT3  = "#475569"
TEXT4  = "#2a3a52"

MONO = "'JetBrains Mono','Fira Code','Courier New',monospace"
SANS = "'Inter','Segoe UI',system-ui,sans-serif"

NAV_W = "68px"

# ─── Chart base ───────────────────────────────────────────────────────────────
_CL = dict(
    paper_bgcolor=SURF, plot_bgcolor=BG,
    font=dict(family=SANS, color=TEXT2, size=11),
    margin=dict(l=50, r=16, t=36, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=TEXT3, family=MONO, size=10), linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=TEXT3, family=MONO, size=10), linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT2, size=10),
                bordercolor=BORDER, borderwidth=1),
    hoverlabel=dict(bgcolor=SURF2, font_color=TEXT, bordercolor=BORD2, font_family=SANS),
)
LP = [BLUE2, GREEN2, RED2, AMBER2, PURP2, TEAL2, "#fb923c", "#34d399", "#f472b6"]
CET1_CS = [[0.0, RED], [0.35, AMBER], [0.65, GREEN], [1.0, BLUE2]]


def _fig(**kw) -> go.Figure:
    fig = go.Figure()
    layout = {**_CL, **kw}
    for k in ("xaxis", "yaxis", "margin"):
        if k in kw and k in _CL:
            layout[k] = {**_CL[k], **kw[k]}
    fig.update_layout(**layout)
    return fig


def _blank_fig(msg: str = "") -> go.Figure:
    fig = _fig()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=TEXT3, size=11),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


# ─── Spring layout ────────────────────────────────────────────────────────────
def _spring(bank_ids, edges, seed=7):
    n = len(bank_ids)
    if n == 0:
        return {}
    idx = {b: i for i, b in enumerate(bank_ids)}
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2)) * 0.25

    if not edges:
        ang = [2 * np.pi * i / n for i in range(n)]
        return {b: (np.cos(a) * 0.8, np.sin(a) * 0.8) for b, a in zip(bank_ids, ang)}

    max_w = max(e["weight"] for e in edges) or 1.0
    adj = np.zeros((n, n))
    for e in edges:
        if e["source"] in idx and e["target"] in idx:
            w = e["weight"] / max_w
            adj[idx[e["source"]], idx[e["target"]]] = w
            adj[idx[e["target"]], idx[e["source"]]] = w

    k, t = np.sqrt(1.0 / n), 0.2
    for _ in range(100):
        diff = pos[:, None] - pos[None]           # [n,n,2]
        dist = np.sqrt((diff**2).sum(-1))          # [n,n]
        np.fill_diagonal(dist, 1e-9)
        rep  = k * k / dist
        attr = dist * dist / k * adj
        disp = ((rep - attr)[:, :, None] * diff / dist[:, :, None]).sum(1)
        d = np.sqrt((disp**2).sum(-1, keepdims=True)); d[d == 0] = 1e-9
        pos += disp / d * np.minimum(d, t); t *= 0.93

    pos -= pos.mean(0)
    sc = np.abs(pos).max()
    if sc > 0:
        pos /= sc / 0.85
    return {b: (float(pos[i, 0]), float(pos[i, 1])) for b, i in idx.items()}


# ─── Figures ─────────────────────────────────────────────────────────────────
def network_figure(gd: dict) -> go.Figure:
    nodes, edges = gd["nodes"], gd["edges"]
    n = len(nodes)
    if not n:
        return _fig(height=480)
    bids = [nd["id"] for nd in nodes]
    pos  = _spring(bids, edges)
    fig  = _fig(height=480, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG)

    if edges:
        max_w = max(e["weight"] for e in edges) or 1
        thresh = max_w * 0.12
        for e in edges:
            if e["weight"] < thresh or e["source"] not in pos or e["target"] not in pos:
                continue
            x0, y0 = pos[e["source"]]; x1, y1 = pos[e["target"]]
            w = e["weight"] / max_w
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(width=0.4 + 2.6*w, color=f"rgba(59,130,246,{0.05+0.35*w})"),
                hoverinfo="none", showlegend=False,
            ))

    max_a = max(nd["total_assets"] for nd in nodes) or 1
    cet1  = [nd["CET1_ratio"] for nd in nodes]
    sizes = [20 + 38*(nd["total_assets"]/max_a)**0.38 for nd in nodes]
    bcols = [RED if not nd["solvent"] or not nd["liquid"] else BORD2 for nd in nodes]
    htxt  = [
        f"<b>{nd['label']}</b><br>"
        f"<span style='color:{TEXT3}'>CET1   </span><b>{nd['CET1_ratio']:.2f}%</b><br>"
        f"<span style='color:{TEXT3}'>LCR    </span><b>{nd['LCR']:.1f}%</b><br>"
        f"<span style='color:{TEXT3}'>NSFR   </span><b>{nd['NSFR']:.1f}%</b><br>"
        f"<span style='color:{TEXT3}'>Assets </span><b>€{nd['total_assets']/1e12:.2f}tn</b><br>"
        f"{'⚠ Insolvent' if not nd['solvent'] else '✓ Solvent'}  "
        f"{'⚠ Illiquid' if not nd['liquid'] else '✓ Liquid'}"
        for nd in nodes if nd["id"] in pos
    ]
    fig.add_trace(go.Scatter(
        x=[pos[nd["id"]][0] for nd in nodes if nd["id"] in pos],
        y=[pos[nd["id"]][1] for nd in nodes if nd["id"] in pos],
        mode="markers+text",
        marker=dict(
            size=sizes, color=cet1, colorscale=CET1_CS,
            cmin=min(cet1)*0.9, cmax=max(cet1)*1.05,
            colorbar=dict(title=dict(text="CET1%", font=dict(color=TEXT2, size=10)),
                          tickfont=dict(color=TEXT3, family=MONO, size=9),
                          thickness=5, len=0.5, x=1.01),
            line=dict(width=1.8, color=bcols), opacity=0.93,
        ),
        text=[nd["id"] for nd in nodes if nd["id"] in pos],
        textfont=dict(color=TEXT, size=8, family=MONO),
        textposition="top center",
        hovertext=htxt, hoverinfo="text", showlegend=False,
    ))
    return fig


def eigenvalue_figure(s: dict) -> go.Figure:
    ev = s.get("eigenvalues", [])
    k  = s.get("gap_index", 0)
    if not ev:
        return _fig(height=300)
    fig = _fig(xaxis_title="k", yaxis_title="λₖ", height=300,
               title_text="Laplacian Eigenvalue Spectrum")
    fig.add_trace(go.Bar(
        x=list(range(len(ev))), y=ev,
        marker=dict(color=[AMBER2 if i <= k else BLUE2 for i in range(len(ev))],
                    opacity=0.88, line=dict(width=0)),
        hovertemplate="k=%{x}  λ=%{y:.5f}<extra></extra>",
    ))
    fig.add_vline(x=k+0.5, line_dash="dot", line_color=AMBER2, line_width=1.2,
                  annotation_text=f"gap  k={k}",
                  annotation_font=dict(color=AMBER2, size=10),
                  annotation_position="top right")
    return fig


def diffusion_heatmap(s: dict) -> go.Figure:
    dd = s.get("diffusion_distance", []); ids = s.get("bank_ids", [])
    if not dd:
        return _fig(height=300)
    fig = _fig(title_text="Diffusion Distance Matrix", height=300,
               margin=dict(l=70, r=16, t=36, b=60))
    fig.add_trace(go.Heatmap(
        z=dd, x=ids, y=ids,
        colorscale=[[0, BG], [0.4, "#1e3a6e"], [1, BLUE2]],
        hovertemplate="%{y} ↔ %{x}: %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=5, tickfont=dict(color=TEXT3, family=MONO, size=9)),
    ))
    return fig


def timeseries_figure(history: list, metric: str) -> go.Figure:
    fig = _fig(title_text=metric.replace("_", " ").title(),
               xaxis_title="Step", yaxis_title=metric, height=260)
    if not history:
        return fig
    for i, bid in enumerate(history[0]["bank_states"]):
        fig.add_trace(go.Scatter(
            x=[h["time"] for h in history],
            y=[h["bank_states"][bid].get(metric) for h in history],
            name=BANK_LABELS.get(bid, bid), mode="lines",
            line=dict(color=LP[i % len(LP)], width=1.7),
        ))
    return fig


def sys_metrics_figure(history: list) -> go.Figure:
    fig = _fig(title_text="System Metrics", xaxis_title="Step", height=220,
               margin=dict(l=50, r=80, t=36, b=40))
    if not history:
        return fig
    times = [h["time"] for h in history]
    for key, label, color in [
        ("CISS", "CISS", RED2),
        ("avg_CET1_ratio", "CET1 %", BLUE2),
        ("network_density", "Density", TEAL2),
    ]:
        vals = [h["system_indicators"].get(key, 0) for h in history]
        fig.add_trace(go.Scatter(x=times, y=vals, name=label, mode="lines",
                                 line=dict(color=color, width=1.6)))
    fig.add_hline(y=0.4, line_dash="dot", line_color=AMBER2, line_width=0.7)
    return fig


def ciss_figure(history: list) -> go.Figure:
    fig = _fig(xaxis_title="Step", yaxis_title="CISS", height=130,
               margin=dict(l=50, r=16, t=14, b=36))
    if not history:
        return fig
    times = [h["time"] for h in history]
    ciss  = [h["system_indicators"].get("CISS", 0) for h in history]
    fig.add_trace(go.Scatter(x=times, y=ciss, mode="lines", fill="tozeroy",
                             line=dict(color=RED2, width=1.5),
                             fillcolor="rgba(239,68,68,.09)",
                             hovertemplate="step %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0.4, line_dash="dot", line_color=AMBER2, line_width=0.8)
    return fig


# ─── Primitives ───────────────────────────────────────────────────────────────
def _col(v, lo, hi):
    if v is None:
        return TEXT3
    return GREEN2 if v >= hi else (AMBER2 if v >= lo else RED2)


def kpi_strip(label, value, color=TEXT2) -> html.Div:
    return html.Div([
        html.Div(label, style={"color": TEXT3, "fontSize": "8px", "fontWeight": "700",
                               "textTransform": "uppercase", "letterSpacing": "0.09em",
                               "marginBottom": "1px"}),
        html.Div(value, style={"color": color, "fontSize": "13px", "fontWeight": "700",
                               "fontFamily": MONO, "lineHeight": "1"}),
    ], style={"padding": "0 12px", "borderRight": f"1px solid {BORDER}",
              "flexShrink": "0"})


def metric_card(label, value, color, sub="") -> html.Div:
    return html.Div([
        html.Div(label, style={"color": TEXT3, "fontSize": "8px", "fontWeight": "700",
                               "textTransform": "uppercase", "letterSpacing": "0.1em",
                               "marginBottom": "5px"}),
        html.Div(value, style={"color": color, "fontSize": "19px", "fontWeight": "700",
                               "fontFamily": MONO, "lineHeight": "1",
                               "letterSpacing": "-0.02em"}),
        html.Div(sub, style={"color": TEXT4, "fontSize": "9px",
                             "marginTop": "4px", "fontFamily": MONO}),
    ], style={"background": SURF, "border": f"1px solid {BORDER}",
              "borderTop": f"2px solid {color}50",
              "padding": "10px 14px", "flex": "1", "minWidth": "100px"})


def sec_hdr(title) -> html.Div:
    return html.Div(title, style={
        "color": TEXT3, "fontSize": "8px", "fontWeight": "700",
        "textTransform": "uppercase", "letterSpacing": "0.12em",
        "padding": "10px 0 5px", "borderBottom": f"1px solid {BORDER}",
        "marginBottom": "8px",
    })


def btn(label, id_, color=BLUE2) -> html.Button:
    return html.Button(label, id=id_, n_clicks=0, style={
        "width": "100%", "padding": "6px 10px", "marginBottom": "5px",
        "background": "transparent", "border": f"1px solid {color}",
        "color": color, "borderRadius": "3px", "cursor": "pointer",
        "fontSize": "11px", "fontWeight": "600", "fontFamily": SANS,
        "textAlign": "left",
    })


def status(id_) -> html.Div:
    return html.Div(id=id_, style={
        "color": GREEN2, "fontSize": "10px", "marginTop": "7px",
        "lineHeight": "1.6", "fontFamily": MONO, "whiteSpace": "pre-wrap",
        "wordBreak": "break-word",
    })


def sidebar(*s) -> html.Div:
    return html.Div(s, style={
        "width": "210px", "flexShrink": "0",
        "borderRight": f"1px solid {BORDER}",
        "padding": "12px 12px 12px 0",
        "overflowY": "auto", "background": BG,
    })


def panel(title, *children) -> html.Div:
    return html.Div([
        html.Div(title, style={
            "color": TEXT3, "fontSize": "8px", "fontWeight": "700",
            "textTransform": "uppercase", "letterSpacing": "0.1em",
            "marginBottom": "9px", "borderBottom": f"1px solid {BORDER}",
            "paddingBottom": "7px",
        }),
        *children,
    ], style={"background": SURF, "border": f"1px solid {BORDER}",
              "borderRadius": "4px", "padding": "11px 13px",
              "marginBottom": "10px"})


def inrow(label, id_, value="") -> html.Div:
    return html.Div([
        html.Label(label, htmlFor=id_,
                   style={"color": TEXT3, "fontSize": "8px", "display": "block",
                          "marginBottom": "3px", "fontWeight": "700",
                          "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        dbc.Input(id=id_, value=value, size="sm",
                  style={"background": SURF2, "borderColor": BORD2, "color": TEXT,
                         "fontFamily": MONO, "fontSize": "11px",
                         "borderRadius": "2px", "padding": "3px 7px"}),
    ], style={"marginBottom": "7px"})


def badge(text, color) -> html.Span:
    return html.Span(text, style={
        "background": f"{color}18", "color": color,
        "border": f"1px solid {color}40",
        "borderRadius": "2px", "fontSize": "8px", "fontWeight": "700",
        "padding": "1px 5px", "fontFamily": MONO,
    })


# ─── Bank health table ────────────────────────────────────────────────────────
def _bank_table():
    cols = [
        {"name": "Bank",       "id": "label",  "type": "text"},
        {"name": "CET1 %",     "id": "cet1",   "type": "numeric"},
        {"name": "LCR %",      "id": "lcr",    "type": "numeric"},
        {"name": "NSFR %",     "id": "nsfr",   "type": "numeric"},
        {"name": "Assets €bn", "id": "assets", "type": "numeric"},
        {"name": "Status",     "id": "status", "type": "text"},
    ]
    cell = {"backgroundColor": SURF, "color": TEXT, "fontSize": "11px",
            "fontFamily": MONO, "border": f"1px solid {BORDER}",
            "padding": "5px 9px", "textAlign": "right"}
    return dash_table.DataTable(
        id="bank-health-table",
        columns=cols, data=[],
        sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "320px"},
        style_header={
            "backgroundColor": SURF2, "color": TEXT3, "fontSize": "8px",
            "fontWeight": "700", "textTransform": "uppercase",
            "letterSpacing": "0.08em", "border": f"1px solid {BORDER}",
            "fontFamily": SANS, "padding": "6px 9px",
        },
        style_cell=cell,
        style_cell_conditional=[
            {"if": {"column_id": "label"}, "textAlign": "left",
             "color": TEXT2, "fontFamily": SANS, "fontWeight": "500"},
            {"if": {"column_id": "status"}, "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{cet1} < 8"},    "color": RED2},
            {"if": {"filter_query": "{cet1} >= 12"},  "color": GREEN2},
            {"if": {"filter_query": "{lcr} < 100"},   "color": RED2},
            {"if": {"filter_query": "{nsfr} < 100"},  "color": RED2},
            {"if": {"filter_query": '{status} = "STRESS"'},
             "color": RED2, "fontWeight": "700"},
            {"if": {"filter_query": '{status} = "OK"'}, "color": GREEN2},
            {"if": {"row_index": "odd"}, "backgroundColor": SURF2},
        ],
    )


# ─── Page content ────────────────────────────────────────────────────────────
def _page_overview():
    bank_opts = [{"label": f"{BANK_LABELS[b]} ({b})", "value": b} for b in ALL_BANKS]
    cfg = sim_state.get_config()

    kpi_row = html.Div(id="kpi-cards-row",
                       style={"display": "flex", "gap": "1px", "flexShrink": "0",
                              "borderBottom": f"1px solid {BORDER}"})

    ctrl = sidebar(
        sec_hdr("Data Source"),
        inrow("Start date", "cfg-start-date", cfg["start_date"]),
        inrow("End / snapshot", "cfg-end-date", cfg["end_date"]),
        html.Div([
            html.Label("Banks", style={"color": TEXT3, "fontSize": "8px", "display": "block",
                                       "marginBottom": "3px", "fontWeight": "700",
                                       "textTransform": "uppercase"}),
            dcc.Dropdown(id="cfg-bank-list", options=bank_opts,
                         value=cfg["bank_list"], multi=True,
                         style={"fontSize": "11px"}),
        ], style={"marginBottom": "10px"}),
        sec_hdr("Actions"),
        btn("↻  Pipeline Reload",        "btn-reload-data"),
        btn("📡  Market APIs",            "btn-reload-api",  TEAL2),
        btn("🔍  AI Data Fetch",          "btn-fetch-llm",   AMBER2),
        btn("🧠  Export GNN Dataset",     "btn-gnn-export",  PURP2),
        status("reload-status"),
    )

    body = html.Div([
        # Network graph
        html.Div([
            html.Div("INTERBANK EXPOSURE NETWORK",
                     style={"color": TEXT3, "fontSize": "8px", "fontWeight": "700",
                            "letterSpacing": "0.12em", "marginBottom": "8px"}),
            dcc.Graph(id="network-graph", config={"displayModeBar": False}),
        ], style={"flex": "1", "minWidth": "0", "padding": "12px 0 12px 16px"}),

        # Right column: bank table + spectral mini
        html.Div([
            panel("Bank Health Monitor", _bank_table()),
            panel("SCG Metrics", html.Div(id="spectral-mini")),
        ], style={"width": "310px", "flexShrink": "0",
                  "padding": "12px 14px 12px 0", "overflowY": "auto"}),
    ], style={"display": "flex", "flex": "1", "minHeight": "0", "overflow": "hidden"})

    return html.Div([kpi_row,
                     html.Div([ctrl, body],
                              style={"display": "flex", "flex": "1",
                                     "minHeight": "0", "overflow": "hidden"})],
                    style={"display": "flex", "flexDirection": "column", "height": "100%"})


def _page_simulation():
    sc_opts = [{"label": "— none —", "value": ""}] + [
        {"label": v["label"], "value": k} for k, v in SHOCK_SCENARIOS.items()
    ]
    ctrl = sidebar(
        sec_hdr("Run"),
        html.Div([
            html.Label("Steps", style={"color": TEXT3, "fontSize": "8px",
                                       "fontWeight": "700", "textTransform": "uppercase"}),
            dcc.Slider(1, 50, 1, value=10, id="steps-slider",
                       marks={1: "1", 10: "10", 25: "25", 50: "50"},
                       tooltip={"always_visible": False},
                       className="mt-2 mb-3"),
        ]),
        btn("▶  Run Simulation",    "btn-run",   GREEN2),
        btn("↺  Reset to Snapshot", "btn-reset", TEXT3),
        html.Div(style={"marginTop": "6px"}),
        sec_hdr("Shock Scenario"),
        dcc.Dropdown(id="shock-dropdown", options=sc_opts, value="",
                     clearable=False, style={"fontSize": "11px", "marginBottom": "6px"}),
        html.Div(id="shock-description",
                 style={"color": AMBER2, "fontSize": "10px", "lineHeight": "1.6",
                        "marginBottom": "8px", "minHeight": "26px"}),
        btn("⚡  Apply Shock", "btn-shock", AMBER2),
        status("sim-status"),
    )
    charts = html.Div([
        html.Div([
            dcc.Dropdown(id="metric-dropdown",
                         options=[
                             {"label": "CET1 Ratio (%)",              "value": "CET1_ratio"},
                             {"label": "LCR (%)",                     "value": "LCR"},
                             {"label": "NSFR (%)",                    "value": "NSFR"},
                             {"label": "Cash (€)",                    "value": "cash"},
                             {"label": "Interbank Assets (€)",        "value": "interbank_assets"},
                             {"label": "Interbank Liabilities (€)",   "value": "interbank_liabilities"},
                         ],
                         value="CET1_ratio", clearable=False,
                         style={"fontSize": "11px", "width": "220px"}),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Div([
                panel("Bank Metrics",
                      dcc.Graph(id="timeseries-graph",
                                config={"displayModeBar": False})),
            ], style={"flex": "3", "minWidth": "0"}),
            html.Div([
                panel("System Overview",
                      dcc.Graph(id="sys-metrics-graph",
                                config={"displayModeBar": False})),
                panel("CISS Stress",
                      dcc.Graph(id="ciss-graph",
                                config={"displayModeBar": False})),
            ], style={"flex": "2", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px"}),
    ], style={"flex": "1", "minWidth": "0", "padding": "12px 14px 12px 16px",
              "overflowY": "auto"})

    return html.Div([ctrl, charts],
                    style={"display": "flex", "height": "100%", "overflow": "hidden"})


def _page_spectral():
    note = html.Div([
        badge("SCG", BLUE2),
        html.Span(
            "  Schmidt, Caccioli & Aste (2024) — coarse-grained Laplacian "
            "L̃ = Σ_{λ≤λₖ} λu uᵀ.  "
            "Error: ε(t) ≤ e^{−tλₖ₊₁}·ε₀.",
            style={"color": TEXT3, "fontSize": "9px", "fontFamily": MONO}),
    ], style={"padding": "8px 16px", "borderTop": f"1px solid {BORDER}",
              "flexShrink": "0"})

    return html.Div([
        html.Div(id="spectral-kpi-row",
                 style={"display": "flex", "gap": "1px", "flexShrink": "0",
                        "borderBottom": f"1px solid {BORDER}"}),
        html.Div([
            html.Div([
                panel("Laplacian Eigenvalue Spectrum",
                      html.Div("Highlighted (amber) bars span the coarse-grained subspace.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="eigenvalue-graph", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div([
                panel("Diffusion Distance Matrix",
                      html.Div("Low distance = banks in same SCG cluster, "
                               "strongly coupled under stress.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="diffusion-heatmap", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px", "flex": "1",
                  "padding": "12px 14px 6px", "overflow": "hidden"}),
        # ── SCG comparison row ──
        html.Div([
            html.Div([
                panel("SCG Eigenvalue Comparison",
                      html.Div("Original (blue) vs coarse-grained (amber) eigenvalues.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="scg-eigenvalue-compare", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div([
                panel("Diffusion Error Decay",
                      html.Div("ε(t) ≤ e^{−tλₖ₊₁} — CG diffusion error should "
                               "decay exponentially.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="scg-diffusion-error", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px", "flex": "1",
                  "padding": "0 14px 12px", "overflow": "hidden"}),
        note,
    ], style={"display": "flex", "flexDirection": "column",
              "height": "100%", "overflow": "hidden"})


def _page_data():
    model_opts = [
        {"label": "Qwen-3 235B — flagship",  "value": "qwen-3-235b-a22b-instruct-2507"},
        {"label": "GPT-OSS 120B",            "value": "gpt-oss-120b"},
        {"label": "ZAI GLM-4.7",             "value": "zai-glm-4.7"},
        {"label": "Llama 3.1 8B — fast",     "value": "llama3.1-8b"},
    ]
    ctrl = sidebar(
        sec_hdr("Model"),
        dcc.Dropdown(id="llm-model-dropdown", options=model_opts,
                     value="qwen-3-235b-a22b-instruct-2507", clearable=False,
                     style={"fontSize": "11px", "marginBottom": "10px"}),
        sec_hdr("AI Analysis"),
        btn("🧠  Analyse System State", "btn-llm", BLUE2),
        html.Div("Cerebras LLM + DuckDuckGo web search to contextualise the "
                 "current state with live market data.",
                 style={"color": TEXT3, "fontSize": "9px", "lineHeight": "1.7",
                        "marginTop": "6px"}),
        html.Div(style={"marginTop": "12px"}),
        sec_hdr("Data Sources"),
        *[html.Div([badge(*b),
                    html.Span(t, style={"color": TEXT3, "fontSize": "9px",
                                        "marginLeft": "5px"})],
                   style={"marginBottom": "5px"})
          for b, t in [
              (("API", TEAL2),  "yfinance  (prices, balance sheet)"),
              (("API", TEAL2),  "ECB SDW  (sovereign yields)"),
              (("LLM", PURP2),  "Cerebras  (15-field GNN features)"),
              (("ABM", AMBER2), "EBA pipeline  (CET1, LCR, NSFR)"),
          ]],
        html.Div(style={"marginTop": "12px"}),
        sec_hdr("GNN Export"),
        btn("📤  Export GNN Dataset", "btn-gnn-export-ai", PURP2),
        status("ai-status"),
    )
    output = html.Div([
        dcc.Loading(type="circle", color=BLUE2, children=[
            dcc.Markdown(id="llm-output",
                         style={"background": SURF, "border": f"1px solid {BORDER}",
                                "borderRadius": "4px", "padding": "20px 22px",
                                "minHeight": "480px", "color": TEXT,
                                "fontSize": "13px", "lineHeight": "1.85",
                                "whiteSpace": "pre-wrap", "fontFamily": SANS}),
        ]),
    ], style={"flex": "1", "minWidth": "0", "padding": "12px 14px 12px 16px",
              "overflowY": "auto"})

    return html.Div([ctrl, output],
                    style={"display": "flex", "height": "100%", "overflow": "hidden"})


def _page_evolve():
    _lbl = {"color": TEXT3, "fontSize": "8px", "fontFamily": MONO, "marginBottom": "2px"}
    _inp = {"background": SURF2, "border": f"1px solid {BORDER}", "color": TEXT,
            "fontSize": "10px", "fontFamily": MONO, "width": "100%", "padding": "3px 6px",
            "borderRadius": "3px"}
    ctrl = sidebar(
        sec_hdr("Data Source"),
        dcc.Dropdown(
            id="evolve-source", value="market",
            options=[{"label": "Real Market (yfinance)", "value": "market"},
                     {"label": "ABM Simulation", "value": "abm"}],
            style={"fontSize": "10px", "marginBottom": "8px"},
            className="dash-dropdown",
        ),
        html.Div([
            html.Label("Corr Window (days)", style=_lbl),
            dcc.Input(id="evolve-corr-window", type="number", value=60, min=20, max=252, step=10, style=_inp),
        ], style={"marginBottom": "6px"}),
        html.Div([
            html.Label("ABM Steps (if ABM)", style=_lbl),
            dcc.Slider(id="evolve-steps", min=100, max=2000, step=100, value=500,
                       marks={100: "100", 500: "500", 1000: "1k", 2000: "2k"},
                       tooltip={"placement": "bottom"}),
        ], style={"marginBottom": "8px"}),
        btn("Load Data", "btn-run-evolve", BLUE2),
        html.Div(style={"marginTop": "10px"}),
        sec_hdr("GNN Hyperparameters"),
        html.Div([
            html.Label("Hidden Dim", style=_lbl),
            dcc.Input(id="gnn-hidden-dim", type="number", value=64, min=16, max=256, step=16, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("GCN Layers", style=_lbl),
            dcc.Input(id="gnn-gcn-layers", type="number", value=3, min=1, max=6, step=1, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("LSTM Layers", style=_lbl),
            dcc.Input(id="gnn-lstm-layers", type="number", value=2, min=1, max=4, step=1, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("Sequence Length", style=_lbl),
            dcc.Input(id="gnn-seq-len", type="number", value=10, min=5, max=30, step=1, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("Epochs", style=_lbl),
            dcc.Input(id="gnn-epochs", type="number", value=200, min=50, max=1000, step=50, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("Learning Rate", style=_lbl),
            dcc.Input(id="gnn-lr", type="number", value=0.003, min=0.0001, max=0.01, step=0.0001, style=_inp),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Label("Dropout", style=_lbl),
            dcc.Input(id="gnn-dropout", type="number", value=0.1, min=0.0, max=0.5, step=0.05, style=_inp),
        ], style={"marginBottom": "8px"}),
        btn("Train GNN", "btn-train-pred", PURP2),
        html.Div(style={"marginTop": "6px"}),
        btn("Compare SCG vs Basel", "btn-scg-vs-basel", AMBER2),
        status("evolve-status"),
    )
    main = html.Div([
        # Polling interval for training progress (disabled by default)
        dcc.Interval(id="evolve-train-interval", interval=500, disabled=True),
        # Training progress banner with progress bar
        html.Div(id="evolve-train-banner",
                 style={"display": "none", "padding": "8px 14px",
                        "background": f"{PURP2}22", "borderBottom": f"1px solid {PURP2}",
                        "fontSize": "11px", "fontFamily": MONO, "color": PURP2}),
        html.Div([
            html.Div(id="evolve-progress-bar",
                     style={"height": "3px", "background": PURP2,
                            "width": "0%", "transition": "width 0.3s ease",
                            "borderRadius": "2px"}),
        ], id="evolve-progress-track",
           style={"display": "none", "height": "3px",
                  "background": f"{PURP2}15", "width": "100%"}),
        # KPI row
        html.Div(id="evolve-kpi-row",
                 style={"display": "flex", "gap": "1px", "flexShrink": "0",
                        "borderBottom": f"1px solid {BORDER}"}),
        # Row 1: spectral metrics + test scatter
        html.Div([
            html.Div([
                panel("Spectral Metrics Evolution",
                      html.Div("Actual (solid) + GNN predicted (dashed). Vertical line = train/test boundary.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-spectral-ts", config={"displayModeBar": False})),
            ], style={"flex": "2", "minWidth": "0"}),
            html.Div([
                panel("Test Set: Predicted vs Actual",
                      html.Div("Scatter of GNN predictions on held-out 20% test data.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-test-scatter", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px", "flexShrink": "0"}),
        # Row 2: training loss curve + risk comparison
        html.Div([
            html.Div([
                panel("GNN Training Loss",
                      html.Div("Train vs test loss per epoch (log scale).",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-loss-curve", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div([
                panel("SCG Risk Score vs Basel Stress",
                      html.Div("SCG risk = 1 − λ₂/ρ. Basel stress = count(CET1 < 8%).",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-scg-vs-basel", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px", "flexShrink": "0"}),
        # Row 3: SCG reconstruction + CET1
        html.Div([
            html.Div([
                panel("SCG Reconstruction Accuracy",
                      html.Div("R² and RMSE of CG diffusion vs original signal over time.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-recon-accuracy", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
            html.Div([
                panel("Spectral Feature Distribution",
                      html.Div("λ₂, spectral gap, and spectral radius over training data.",
                               style={"color": TEXT3, "fontSize": "9px", "marginBottom": "7px"}),
                      dcc.Graph(id="evolve-cet1-ts", config={"displayModeBar": False})),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "10px", "flexShrink": "0"}),
    ], style={"display": "flex", "flexDirection": "column", "gap": "10px",
              "flex": "1", "minWidth": "0", "padding": "12px 14px",
              "overflowY": "auto"})

    return html.Div([ctrl, main],
                    style={"display": "flex", "height": "100%", "overflow": "hidden"})


# ─── Nav rail ─────────────────────────────────────────────────────────────────
_NAV = [
    ("overview",   "⬡",  "NETWORK"),
    ("simulation", "▶",  "SIMULATE"),
    ("spectral",   "λ",  "SPECTRAL"),
    ("evolve",     "⟳",  "EVOLVE"),
    ("data",       "◈",  "AI/DATA"),
]


def _nav_btn(page_id, icon, label, active=False) -> html.Button:
    return html.Button(
        [html.Div(icon, style={"fontSize": "17px", "fontFamily": MONO, "lineHeight": "1",
                               "marginBottom": "3px"}),
         html.Div(label, style={"fontSize": "7px", "fontWeight": "700",
                                "letterSpacing": "0.07em"})],
        id=f"nav-{page_id}", n_clicks=0,
        style={
            "width": "100%", "padding": "13px 4px",
            "background": f"{BLUE}18" if active else "transparent",
            "border": "none",
            "borderLeft": f"2px solid {BLUE2}" if active else "2px solid transparent",
            "color": BLUE2 if active else TEXT3,
            "cursor": "pointer", "textAlign": "center",
            "fontFamily": SANS, "transition": "all 0.15s",
        },
    )


def _nav_rail() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("SCR", style={"color": BLUE2, "fontSize": "12px",
                                    "fontWeight": "800", "fontFamily": MONO}),
            html.Div("FN",  style={"color": TEXT4, "fontSize": "8px",
                                    "fontWeight": "700", "letterSpacing": "0.15em"}),
        ], style={"padding": "12px 4px", "textAlign": "center",
                  "borderBottom": f"1px solid {BORDER}"}),
        html.Div([_nav_btn(p, i, l, k == 0) for k, (p, i, l) in enumerate(_NAV)],
                 id="nav-items"),
        html.Div("v0.1", style={"position": "absolute", "bottom": "10px",
                                  "width": "100%", "textAlign": "center",
                                  "color": TEXT4, "fontSize": "8px", "fontFamily": MONO}),
    ], style={
        "width": NAV_W, "flexShrink": "0", "background": SURF,
        "borderRight": f"1px solid {BORDER}",
        "height": "100vh", "overflowY": "hidden",
        "display": "flex", "flexDirection": "column", "position": "relative",
    })


# ─── Topbar + statusbar ───────────────────────────────────────────────────────
def _topbar() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.Span("SCR ", style={"color": BLUE2, "fontWeight": "700",
                                          "fontSize": "12px", "letterSpacing": "0.06em"}),
                html.Span("Financial Networks",
                           style={"color": TEXT3, "fontSize": "12px"}),
                html.Span("  BETA", style={"color": TEXT4, "fontSize": "7px",
                                            "fontFamily": MONO, "letterSpacing": "0.1em"}),
            ], style={"flexShrink": "0", "paddingRight": "14px",
                      "borderRight": f"1px solid {BORDER}"}),
            html.Div(id="header-kpis",
                     style={"display": "flex", "alignItems": "center",
                            "flex": "1", "overflow": "hidden"}),
            html.Div([
                html.Div(id="header-step",
                         style={"color": TEXT3, "fontSize": "9px", "fontFamily": MONO}),
                html.Div(id="header-ts",
                         style={"color": TEXT4, "fontSize": "8px", "fontFamily": MONO}),
            ], style={"flexShrink": "0", "paddingLeft": "14px",
                      "borderLeft": f"1px solid {BORDER}",
                      "display": "flex", "flexDirection": "column",
                      "justifyContent": "center", "gap": "2px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "height": "46px", "padding": "0 18px"}),
    ], style={"background": SURF, "borderBottom": f"1px solid {BORDER}",
              "position": "sticky", "top": "0", "zIndex": "100"})


def _statusbar() -> html.Div:
    return html.Div([
        html.Div([
            badge("SPECTRAL", BLUE2),
            html.Span(" Coarse-Graining Active",
                      style={"color": TEXT4, "fontSize": "8px",
                             "fontFamily": MONO, "marginLeft": "5px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span(s, style={"color": TEXT4, "fontSize": "8px", "fontFamily": MONO})
            for s in ["Cerebras LLM", "  ·  ", "yfinance", "  ·  ",
                      "ECB SDW", "  ·  ",
                      datetime.now(timezone.utc).strftime("%Y-%m-%d")]
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "height": "24px", "background": BG, "borderTop": f"1px solid {BORDER}",
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between", "padding": "0 18px", "flexShrink": "0",
    })


# ─── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
    ],
    title="SCR Financial Networks",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
    <style>
        *, *::before, *::after {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; height: 100%;
                      background: {BG}; color: {TEXT};
                      font-family: {SANS}; font-size: 13px; overflow: hidden; }}
        ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
        ::-webkit-scrollbar-track {{ background: {SURF}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORD2}; border-radius: 2px; }}
        .Select-control {{ background: {SURF2} !important; border-color: {BORD2} !important;
                           border-radius: 2px !important; }}
        .Select-menu-outer {{ background: {SURF2} !important; border-color: {BORD2} !important; }}
        .Select-option {{ color: {TEXT} !important; font-size: 11px !important; }}
        .Select-option.is-focused {{ background: {SURF3} !important; }}
        .Select-value-label {{ color: {TEXT} !important; font-size: 11px !important; }}
        .Select-placeholder {{ color: {TEXT3} !important; font-size: 11px !important; }}
        .rc-slider-track {{ background: {BLUE} !important; height: 2px !important; }}
        .rc-slider-handle {{ border-color: {BLUE2} !important; background: {BLUE2} !important;
                             width: 10px !important; height: 10px !important; margin-top: -4px !important; }}
        .rc-slider-rail {{ background: {BORD2} !important; height: 2px !important; }}
        .form-control {{ background: {SURF2} !important; border-color: {BORD2} !important;
                         color: {TEXT} !important; font-size: 11px !important; }}
        .form-control:focus {{ border-color: {BLUE2} !important;
                               box-shadow: 0 0 0 2px rgba(96,165,250,.12) !important; }}
        #nav-items button:hover {{ color: {TEXT2} !important; }}
    </style>
</head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>"""


# ─── Layout — all pages pre-rendered, visibility toggled ──────────────────────
def _page_wrap(page_id, content) -> html.Div:
    return html.Div(content,
                    id=f"page-{page_id}",
                    style={"display": "flex", "flexDirection": "column",
                           "height": "100%", "overflow": "hidden"})


app.layout = html.Div([
    dcc.Store(id="active-page", data="overview"),
    dcc.Store(id="sim-history-store", data=[]),

    _nav_rail(),

    html.Div([
        _topbar(),
        html.Div([
            _page_wrap("overview",   _page_overview()),
            _page_wrap("simulation", _page_simulation()),
            _page_wrap("spectral",   _page_spectral()),
            _page_wrap("evolve",     _page_evolve()),
            _page_wrap("data",       _page_data()),
        ], style={"flex": "1", "overflow": "hidden", "position": "relative"}),
        _statusbar(),
    ], style={"flex": "1", "display": "flex", "flexDirection": "column",
              "overflow": "hidden", "minWidth": "0"}),
], style={"display": "flex", "height": "100vh", "overflow": "hidden", "background": BG})


# ─── Navigation ───────────────────────────────────────────────────────────────
_PAGES = ["overview", "simulation", "spectral", "evolve", "data"]

_NAV_ACTIVE = {
    "width": "100%", "padding": "13px 4px",
    "background": f"{BLUE}18",
    "border": "none", "borderLeft": f"2px solid {BLUE2}",
    "color": BLUE2, "cursor": "pointer", "textAlign": "center",
    "fontFamily": SANS,
}
_NAV_IDLE = {
    "width": "100%", "padding": "13px 4px",
    "background": "transparent",
    "border": "none", "borderLeft": "2px solid transparent",
    "color": TEXT3, "cursor": "pointer", "textAlign": "center",
    "fontFamily": SANS,
}


@callback(
    Output("active-page", "data"),
    [Input(f"nav-{p}", "n_clicks") for p in _PAGES],
    prevent_initial_call=True,
)
def nav_route(*_):
    return ctx.triggered_id.replace("nav-", "")


@callback(
    [Output(f"page-{p}", "style") for p in _PAGES] +
    [Output(f"nav-{p}",  "style") for p in _PAGES],
    Input("active-page", "data"),
)
def toggle_pages(page):
    page_styles = []
    for p in _PAGES:
        if p == page:
            page_styles.append({"display": "flex", "flexDirection": "column",
                                 "height": "100%", "overflow": "hidden"})
        else:
            page_styles.append({"display": "none"})
    nav_styles = [_NAV_ACTIVE if p == page else _NAV_IDLE for p in _PAGES]
    return page_styles + nav_styles


# ─── Header KPIs ──────────────────────────────────────────────────────────────
@callback(
    Output("header-kpis", "children"),
    Output("header-step", "children"),
    Output("header-ts",   "children"),
    Input("active-page",       "data"),
    Input("sim-history-store", "data"),
)
def header_kpis(_p, _h):
    sim  = sim_state.get_simulation()
    m    = sim.get_system_metrics()
    cet1 = m.get("avg_CET1_ratio", 0)
    lcr  = m.get("avg_LCR", 0)
    ciss = m.get("CISS", 0)
    sent = sim.calculate_market_sentiment()
    dens = m.get("network_density", 0)
    try:
        s    = sim_state.get_spectral_data()
        lam2 = s.get("algebraic_connectivity", 0)
        gap  = s.get("gap_size", 0)
    except Exception:
        lam2 = gap = 0

    cards = [
        kpi_strip("CET1",    f"{cet1:.2f}%",  _col(cet1, 8, 12)),
        kpi_strip("LCR",     f"{lcr:.1f}%",   _col(lcr, 100, 130)),
        kpi_strip("CISS",    f"{ciss:.3f}",   RED2 if ciss > 0.4 else (AMBER2 if ciss > 0.2 else GREEN2)),
        kpi_strip("λ₂",      f"{lam2:.4f}",   TEAL2),
        kpi_strip("Gap δ",   f"{gap:.4f}",    AMBER2),
        kpi_strip("Sent.",   f"{sent:.2f}",   _col(sent, 0.4, 0.7)),
        kpi_strip("Density", f"{dens:.3f}",   TEXT2),
    ]
    return (cards,
            f"Step {sim.time}  ·  {len(sim.banks)} banks",
            datetime.now(timezone.utc).strftime("%H:%M UTC"))


# ─── Overview ────────────────────────────────────────────────────────────────
@callback(
    Output("network-graph",    "figure"),
    Output("bank-health-table","data"),
    Output("kpi-cards-row",    "children"),
    Output("spectral-mini",    "children"),
    Input("active-page",       "data"),
    Input("sim-history-store", "data"),
)
def update_overview(page, _h):
    if page != "overview":
        return no_update, no_update, no_update, no_update

    sim = sim_state.get_simulation()
    m   = sim.get_system_metrics()
    gd  = sim_state.get_network_graph_data()

    # bank table
    rows = []
    stressed = 0
    for nd in gd["nodes"]:
        ok = nd["solvent"] and nd["liquid"]
        if not ok:
            stressed += 1
        rows.append({
            "label":  BANK_LABELS.get(nd["id"], nd["id"]),
            "cet1":   round(nd["CET1_ratio"], 2),
            "lcr":    round(nd["LCR"], 1),
            "nsfr":   round(nd["NSFR"], 1),
            "assets": round(nd["total_assets"] / 1e9, 0),
            "status": "OK" if ok else "STRESS",
        })

    cet1 = m.get("avg_CET1_ratio", 0)
    lcr  = m.get("avg_LCR", 0)
    ciss = m.get("CISS", 0)
    try:
        s   = sim_state.get_spectral_data()
        lam2, gap, gap_k, srad = (
            s.get("algebraic_connectivity", 0),
            s.get("gap_size", 0),
            s.get("gap_index", 0),
            s.get("spectral_radius", 0),
        )
    except Exception:
        lam2 = gap = srad = 0; gap_k = 0

    n_total = len(gd["nodes"])
    cards = html.Div([
        metric_card("AVG CET1",        f"{cet1:.2f}%",  _col(cet1, 8, 12), "min 8%"),
        metric_card("AVG LCR",         f"{lcr:.1f}%",   _col(lcr, 100, 130), "min 100%"),
        metric_card("CISS",            f"{ciss:.3f}",
                    RED2 if ciss > 0.4 else (AMBER2 if ciss > 0.2 else GREEN2),
                    "IT-DE spread proxy"),
        metric_card("λ₂  Alg. Conn.", f"{lam2:.4f}",   GREEN2, "higher = more robust"),
        metric_card("Spectral Gap δ",  f"{gap:.4f}",    AMBER2, f"at k = {gap_k}"),
        metric_card("STRESSED BANKS",  str(stressed),
                    RED2 if stressed else GREEN2, f"of {n_total} total"),
    ], style={"display": "flex", "gap": "1px"})

    mini = html.Div([
        _kv("Spectral radius",   f"{srad:.4f}",     PURP2),
        _kv("SCG clusters",      str(gap_k + 1),    GREEN2),
        _kv("Algebraic conn.",   f"{lam2:.4f}",     TEAL2),
        _kv("Error ε(t=1)",      f"{np.exp(-gap):.4f}", BLUE2),
        html.Div(
            "Spectral gap separates the strongly-coupled cluster "
            "from the weakly-coupled periphery.",
            style={"color": TEXT3, "fontSize": "9px", "lineHeight": "1.6",
                   "borderTop": f"1px solid {BORDER}", "paddingTop": "8px",
                   "marginTop": "6px"}),
    ])

    return network_figure(gd), rows, cards, mini


def _kv(label, value, color) -> html.Div:
    return html.Div([
        html.Span(label + "  ", style={"color": TEXT3, "fontSize": "9px"}),
        html.Span(value, style={"color": color, "fontFamily": MONO,
                                "fontSize": "12px", "fontWeight": "600"}),
    ], style={"marginBottom": "4px"})


@callback(
    Output("reload-status",     "children"),
    Output("sim-history-store", "data", allow_duplicate=True),
    Input("btn-reload-data",    "n_clicks"),
    Input("btn-reload-api",     "n_clicks"),
    Input("btn-fetch-llm",      "n_clicks"),
    Input("btn-gnn-export",     "n_clicks"),
    State("cfg-start-date",     "value"),
    State("cfg-end-date",       "value"),
    State("cfg-bank-list",      "value"),
    State("llm-model-dropdown", "value"),
    prevent_initial_call=True,
)
def data_actions(_r, _ra, _f, _g, start, end, banks, model):
    trig  = ctx.triggered_id
    banks = banks or ALL_BANKS
    model = model or "qwen-3-235b-a22b-instruct-2507"

    if trig == "btn-reload-api":
        try:
            bd, nd, si = build_simulation_inputs_from_api(bank_ids=banks)
            sim_state.load_from_data(bd, nd, si)
            n = len(sim_state.get_simulation().banks)
            return f"✓ {n} banks — market APIs (yfinance + ECB)", []
        except Exception as e:
            return f"✗ {e}", no_update

    if trig == "btn-reload-data":
        try:
            sim_state.reload_data(start_date=start, end_date=end, bank_list=banks)
            n = len(sim_state.get_simulation().banks)
            return f"✓ {n} banks loaded  (snapshot {end})", []
        except Exception as e:
            return f"✗ {e}", no_update

    if trig == "btn-fetch-llm":
        try:
            data = fetch_bank_data_via_llm(banks, model=model)
            if "error" in data:
                return f"✗ {data['error']}", no_update
            sim_state.apply_llm_bank_data(data)
            sim_state.get_simulation().record_state()
            return f"✓ AI updated {len(data)} banks", _ser(sim_state.get_simulation().history)
        except Exception as e:
            return f"✗ {e}", no_update

    if trig == "btn-gnn-export":
        try:
            gnn_feats = fetch_bank_features_for_gnn(banks, model=model)
            if "error" in gnn_feats:
                return f"✗ {gnn_feats['error']}", no_update
            sd = {b: {k: v for k, v in f.items()
                      if v is not None and k in ("CET1_ratio", "LCR", "NSFR", "total_assets")}
                  for b, f in gnn_feats.items()}
            sim_state.apply_llm_bank_data(sd)
            sim_state.get_simulation().record_state()
            graph = sim_state.get_network_graph_data()
            out   = os.path.join(_REPO_ROOT, "data", "gnn_datasets")
            info  = build_and_export(gnn_feats, graph, output_dir=out)
            return (f"✓ GNN saved — {info['n_nodes']}n · {info['n_edges']}e · "
                    f"{info['n_features']}f\n→ {info['output_dir']}",
                    _ser(sim_state.get_simulation().history))
        except Exception as e:
            return f"✗ {e}", no_update

    return no_update, no_update


# ─── Simulation ───────────────────────────────────────────────────────────────
@callback(
    Output("shock-description", "children"),
    Input("shock-dropdown", "value"),
)
def shock_desc(sc):
    return SHOCK_SCENARIOS[sc]["description"] if sc and sc in SHOCK_SCENARIOS else ""


def _ser(history):
    return [{"time": h["time"], "bank_states": h["bank_states"],
             "system_indicators": h["system_indicators"]}
            for h in history[-120:]]


@callback(
    Output("sim-status",        "children"),
    Output("sim-history-store", "data"),
    Input("btn-run",   "n_clicks"),
    Input("btn-shock", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("steps-slider",   "value"),
    State("shock-dropdown", "value"),
    prevent_initial_call=True,
)
def sim_actions(_, __, ___, steps, scenario):
    trig = ctx.triggered_id
    if trig == "btn-reset":
        sim_state.reset_simulation()
        return "↺  Reset.", []
    if trig == "btn-shock":
        if not scenario:
            return "⚠  Select a scenario first.", no_update
        history = sim_state.apply_shock_and_record(SHOCK_SCENARIOS[scenario]["params"])
        return (f"⚡  {SHOCK_SCENARIOS[scenario]['label']}  →  step "
                f"{sim_state.get_simulation().time}", _ser(history))
    if trig == "btn-run":
        history = sim_state.run_steps(steps)
        return f"▶  +{steps} steps  →  step {sim_state.get_simulation().time}", _ser(history)
    return no_update, no_update


@callback(
    Output("timeseries-graph", "figure"),
    Output("sys-metrics-graph","figure"),
    Output("ciss-graph",       "figure"),
    Input("metric-dropdown",   "value"),
    Input("sim-history-store", "data"),
)
def update_sim_charts(metric, history):
    return timeseries_figure(history, metric), sys_metrics_figure(history), ciss_figure(history)


# ─── Spectral ────────────────────────────────────────────────────────────────
@callback(
    Output("spectral-kpi-row", "children"),
    Output("eigenvalue-graph", "figure"),
    Output("diffusion-heatmap","figure"),
    Input("active-page",       "data"),
    Input("sim-history-store", "data"),
)
def update_spectral(page, _h):
    if page != "spectral":
        return no_update, no_update, no_update
    s = sim_state.get_spectral_data()
    n = len(s.get("bank_ids", []))
    err = np.exp(-float(s.get("gap_size", 0)))
    cards = html.Div([
        metric_card("Algebraic Conn. λ₂", f"{s['algebraic_connectivity']:.4f}",
                    TEAL2, "λ₂ of norm. Laplacian"),
        metric_card("Spectral Gap δ",    f"{s['gap_size']:.4f}",
                    AMBER2, f"at k = {s['gap_index']}"),
        metric_card("Spectral Radius",   f"{s['spectral_radius']:.4f}",
                    PURP2, "largest eigenvalue"),
        metric_card("SCG Clusters",      str(s["gap_index"] + 1),
                    GREEN2, f"of {n} banks"),
        metric_card("Error Bound ε(1)",  f"{err:.4f}",
                    BLUE2, "e^{−δ} worst case"),
    ], style={"display": "flex", "gap": "1px"})
    return cards, eigenvalue_figure(s), diffusion_heatmap(s)


# ─── SCG comparison ──────────────────────────────────────────────────────────
@callback(
    Output("scg-eigenvalue-compare", "figure"),
    Output("scg-diffusion-error",    "figure"),
    Input("active-page",             "data"),
    Input("sim-history-store",       "data"),
)
def update_scg_comparison(page, _h):
    if page != "spectral":
        return no_update, no_update
    try:
        cg = sim_state.get_coarse_grained_data()
    except Exception:
        empty = _blank_fig("SCG unavailable")
        return empty, empty

    orig = cg["original_eigenvalues"]
    cg_ev = cg["cg_eigenvalues"]
    n = len(orig)

    # Eigenvalue comparison bar chart
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=list(range(n)), y=orig, name="Original",
                          marker_color=BLUE2, opacity=0.8))
    fig1.add_trace(go.Bar(x=list(range(n)), y=cg_ev, name="Coarse-grained",
                          marker_color=AMBER2, opacity=0.8))
    fig1.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT3),
        margin=dict(l=30, r=10, t=10, b=30), height=200,
        barmode="group", legend=dict(x=0.7, y=0.95, font=dict(size=8)),
        xaxis=dict(title="Index", gridcolor=BORD2),
        yaxis=dict(title="λ", gridcolor=BORD2),
    )

    # Diffusion error decay
    errors = cg["diffusion_errors"]
    steps = list(range(1, len(errors) + 1))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=steps, y=errors, mode="lines+markers",
                              name="ε(t) actual", line=dict(color=BLUE2, width=2),
                              marker=dict(size=4)))
    # Theoretical bound: e^{-t * lambda_{k+1}}
    gap_idx = cg.get("n_clusters", 2)
    if gap_idx < len(orig):
        lam_k1 = orig[gap_idx]
        bound = [float(np.exp(-t * lam_k1)) for t in steps]
        fig2.add_trace(go.Scatter(x=steps, y=bound, mode="lines",
                                  name="Bound e^{−tλₖ₊₁}",
                                  line=dict(color=AMBER2, width=1.5, dash="dash")))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT3),
        margin=dict(l=30, r=10, t=10, b=30), height=200,
        legend=dict(x=0.6, y=0.95, font=dict(size=8)),
        xaxis=dict(title="Time step t", gridcolor=BORD2),
        yaxis=dict(title="Relative error", gridcolor=BORD2, type="log"),
    )

    return fig1, fig2


# ─── AI / Data ────────────────────────────────────────────────────────────────
@callback(
    Output("llm-output", "children"),
    Output("ai-status",  "children"),
    Input("btn-llm", "n_clicks"),
    State("llm-model-dropdown", "value"),
    prevent_initial_call=True,
)
def llm_analysis(_, model):
    sim  = sim_state.get_simulation()
    gd   = sim_state.get_network_graph_data()
    sm   = sim.get_system_metrics(); sm["time"] = sim.time
    snap = build_snapshot(gd, sm, sim_state.get_spectral_data())
    out  = analyze_system_state(snap, model=model)
    return out, f"✓ Analysis complete ({model})"


@callback(
    Output("ai-status", "children", allow_duplicate=True),
    Input("btn-gnn-export-ai", "n_clicks"),
    State("llm-model-dropdown", "value"),
    prevent_initial_call=True,
)
def gnn_export_ai(_, model):
    model = model or "qwen-3-235b-a22b-instruct-2507"
    try:
        gnn = fetch_bank_features_for_gnn(ALL_BANKS, model=model)
        if "error" in gnn:
            return f"✗ {gnn['error']}"
        info = build_and_export(gnn, sim_state.get_network_graph_data(),
                                output_dir=os.path.join(_REPO_ROOT, "data", "gnn_datasets"))
        return (f"✓ {info['n_nodes']}n · {info['n_edges']}e · "
                f"{info['n_features']}f\n→ {info['output_dir']}")
    except Exception as e:
        return f"✗ {e}"


# ─── Evolution page ──────────────────────────────────────────────────────────

# Module-level state for evolution data (populated by callbacks)
_evolve_features: list = []
_evolve_predictor = None
_evolve_train_metrics: dict = {}
_evolve_test_metrics: dict = {}

_DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=TEXT3),
)


@callback(
    Output("evolve-spectral-ts",    "figure"),
    Output("evolve-cet1-ts",        "figure"),
    Output("evolve-recon-accuracy", "figure"),
    Output("evolve-kpi-row",        "children"),
    Output("evolve-status",         "children"),
    Input("btn-run-evolve",         "n_clicks"),
    State("evolve-steps",           "value"),
    State("evolve-source",          "value"),
    State("evolve-corr-window",     "value"),
    prevent_initial_call=True,
)
def run_evolution(_, steps, source, corr_window):
    global _evolve_features
    source = source or "market"
    corr_window = corr_window or 60
    try:
        _evolve_features = generate_evolution_data(
            n_steps=steps, source=source, corr_window=corr_window,
        )
    except Exception as e:
        empty = _blank_fig(f"Error: {e}")
        return empty, empty, empty, no_update, f"✗ {e}"

    n_snaps = len(_evolve_features)
    times = [r["time"] for r in _evolve_features]

    # Spectral metrics
    fig1 = go.Figure()
    for name, color in [("lambda_2", BLUE2), ("spectral_gap", AMBER2), ("spectral_radius", PURP2)]:
        fig1.add_trace(go.Scatter(
            x=times, y=[r[name] for r in _evolve_features],
            mode="lines", name=name, line=dict(color=color, width=1.5),
        ))
    fig1.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=10, t=10, b=30), height=220,
                       legend=dict(x=0.01, y=0.99, font=dict(size=8)),
                       xaxis=dict(title="Day" if source == "market" else "Step", gridcolor=BORD2),
                       yaxis=dict(gridcolor=BORD2))

    # Spectral feature distribution (replaces CET1 chart for market data)
    fig2 = go.Figure()
    for name, color in [("lambda_2", BLUE2), ("spectral_gap", AMBER2), ("spectral_radius", PURP2)]:
        vals = [r[name] for r in _evolve_features]
        fig2.add_trace(go.Histogram(x=vals, name=name, marker_color=color, opacity=0.6, nbinsx=30))
    fig2.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=10, t=10, b=30), height=180,
                       legend=dict(x=0.01, y=0.99, font=dict(size=8)),
                       barmode="overlay",
                       xaxis=dict(title="Value", gridcolor=BORD2),
                       yaxis=dict(title="Count", gridcolor=BORD2))

    # SCG reconstruction accuracy
    try:
        recon = compute_scg_reconstruction_accuracy()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=recon["time"], y=recon["r2"], mode="lines+markers",
                                  name="R²", line=dict(color=TEAL2, width=2), marker=dict(size=4)))
        fig3.add_trace(go.Scatter(x=recon["time"], y=recon["rmse"], mode="lines+markers",
                                  name="RMSE", line=dict(color=AMBER2, width=1.5, dash="dash"),
                                  marker=dict(size=3), yaxis="y2"))
        fig3.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=45, t=10, b=30), height=220,
                           legend=dict(x=0.6, y=0.95, font=dict(size=8)),
                           xaxis=dict(title="Diffusion time t", gridcolor=BORD2),
                           yaxis=dict(title="R²", gridcolor=BORD2, range=[0, 1.05]),
                           yaxis2=dict(title="RMSE", overlaying="y", side="right",
                                       gridcolor="rgba(0,0,0,0)"))
        avg_r2 = f"{np.mean(recon['r2']):.4f}"
    except Exception:
        fig3 = _blank_fig("SCG reconstruction unavailable")
        avg_r2 = "—"

    src_label = "Market" if source == "market" else "ABM"
    kpis = html.Div([
        metric_card("Snapshots", str(n_snaps), BLUE2, f"{src_label} data"),
        metric_card("SCG Recon R²", avg_r2, TEAL2, "avg over diffusion t"),
        metric_card("GNN Params", "—", PURP2, "train GNN first"),
        metric_card("Train MSE", "—", AMBER2, "train GNN first"),
        metric_card("Test MSE", "—", GREEN2, "train GNN first"),
        metric_card("Test R²", "—", BLUE2, "train GNN first"),
    ], style={"display": "flex", "gap": "1px"})

    return fig1, fig2, fig3, kpis, f"✓ {n_snaps} {src_label.lower()} snapshots loaded"


def _run_training_thread(features, seq_len, hidden_dim, gcn_layers, lstm_layers, epochs, lr, dropout):
    """Background thread that trains the GNN and updates _training_state."""
    import psutil, os
    global _evolve_predictor, _evolve_train_metrics, _evolve_test_metrics

    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024 / 1024  # MB

    def _progress_cb(epoch, total, train_loss, test_loss):
        with _training_lock:
            ram_now = proc.memory_info().rss / 1024 / 1024
            _training_state.update({
                "epoch": epoch,
                "total_epochs": total,
                "train_loss": train_loss,
                "test_loss": test_loss or 0.0,
                "pct": 100.0 * epoch / total,
                "ram_mb": ram_now,
                "ram_delta_mb": ram_now - ram_before,
            })

    try:
        with _training_lock:
            _training_state.update({
                "running": True, "done": False, "error": None,
                "epoch": 0, "pct": 0.0, "start_time": _time.time(),
                "total_epochs": epochs, "ram_mb": ram_before, "ram_delta_mb": 0.0,
            })

        predictor, tr, te = train_predictor(
            features,
            seq_len=seq_len, hidden_dim=hidden_dim,
            num_gat_layers=gcn_layers, num_lstm_layers=lstm_layers,
            epochs=epochs, lr=lr, dropout=dropout,
            progress_callback=_progress_cb,
        )
        preds = predictor.predict(features, steps=30)

        _evolve_predictor = predictor
        _evolve_train_metrics = tr
        _evolve_test_metrics = te

        ram_after = proc.memory_info().rss / 1024 / 1024
        with _training_lock:
            _training_state.update({
                "running": False, "done": True, "pct": 100.0,
                "ram_mb": ram_after, "ram_delta_mb": ram_after - ram_before,
            })
    except Exception as e:
        with _training_lock:
            _training_state.update({"running": False, "done": True, "error": str(e)})


# Store hyperparams for the polling callback to access
_train_hparams: dict = {}


@callback(
    Output("evolve-train-banner", "children", allow_duplicate=True),
    Output("evolve-train-banner", "style", allow_duplicate=True),
    Output("evolve-progress-track", "style", allow_duplicate=True),
    Output("evolve-train-interval", "disabled"),
    Output("evolve-status", "children", allow_duplicate=True),
    Input("btn-train-pred", "n_clicks"),
    State("gnn-hidden-dim",  "value"),
    State("gnn-gcn-layers",  "value"),
    State("gnn-lstm-layers", "value"),
    State("gnn-seq-len",     "value"),
    State("gnn-epochs",      "value"),
    State("gnn-lr",          "value"),
    State("gnn-dropout",     "value"),
    prevent_initial_call=True,
)
def start_training(_, hidden_dim, gcn_layers, lstm_layers, seq_len, epochs, lr, dropout):
    """Kick off training in a background thread and enable the progress interval."""
    global _train_hparams
    if len(_evolve_features) < 25:
        return (no_update, no_update, no_update, True,
                "✗ Load data first (need >= 25 snapshots)")

    # Already running?
    with _training_lock:
        if _training_state["running"]:
            return (no_update, no_update, no_update, False,
                    "⏳ Training already in progress…")

    hidden_dim = hidden_dim or 64
    gcn_layers = gcn_layers or 3
    lstm_layers = lstm_layers or 2
    seq_len = seq_len or 10
    epochs = epochs or 200
    lr = lr or 3e-3
    dropout = dropout or 0.1

    _train_hparams = dict(hidden_dim=hidden_dim, gcn_layers=gcn_layers,
                          lstm_layers=lstm_layers, seq_len=seq_len,
                          epochs=epochs, lr=lr, dropout=dropout)

    t = threading.Thread(target=_run_training_thread, daemon=True,
                         args=(_evolve_features, seq_len, hidden_dim,
                               gcn_layers, lstm_layers, epochs, lr, dropout))
    t.start()

    banner_style = {"display": "block", "padding": "8px 14px",
                    "background": f"{PURP2}22", "borderBottom": f"1px solid {PURP2}",
                    "fontSize": "11px", "fontFamily": MONO, "color": PURP2}
    progress_style = {"display": "block", "height": "3px",
                      "background": f"{PURP2}15", "width": "100%"}

    return (f"⏳ Training GNN — 0/{epochs} epochs (0%)",
            banner_style, progress_style,
            False,  # enable interval
            "⏳ Training started…")


@callback(
    Output("evolve-train-banner", "children", allow_duplicate=True),
    Output("evolve-progress-bar", "style"),
    Output("evolve-spectral-ts",  "figure", allow_duplicate=True),
    Output("evolve-test-scatter", "figure"),
    Output("evolve-loss-curve",   "figure"),
    Output("evolve-kpi-row",      "children", allow_duplicate=True),
    Output("evolve-train-interval", "disabled", allow_duplicate=True),
    Output("evolve-progress-track", "style", allow_duplicate=True),
    Output("evolve-train-banner", "style", allow_duplicate=True),
    Output("evolve-status", "children", allow_duplicate=True),
    Input("evolve-train-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_training_progress(_):
    """Polled every 500ms while training runs. Updates banner + progress bar.
    When done, renders all result charts."""
    with _training_lock:
        state = dict(_training_state)

    bar_style = {"height": "3px", "background": PURP2,
                 "width": f"{state['pct']:.1f}%", "transition": "width 0.3s ease",
                 "borderRadius": "2px"}

    if not state["done"]:
        # Still training — update banner only
        elapsed = _time.time() - state["start_time"] if state["start_time"] else 0
        eta = ""
        if state["epoch"] > 0 and state["pct"] < 100:
            remaining = elapsed / state["pct"] * (100 - state["pct"])
            eta = f" — ETA {remaining:.0f}s"
        banner = (f"⏳ Training GNN — {state['epoch']}/{state['total_epochs']} epochs "
                  f"({state['pct']:.0f}%) — loss: {state['train_loss']:.6f} — "
                  f"RAM: {state['ram_mb']:.0f} MB (+{state['ram_delta_mb']:.0f} MB){eta}")
        return (banner, bar_style,
                no_update, no_update, no_update, no_update,
                False,  # keep interval running
                no_update, no_update,
                f"⏳ Epoch {state['epoch']}/{state['total_epochs']} ({state['pct']:.0f}%)")

    # Training done — disable interval and render results
    banner_style = {"display": "block", "padding": "8px 14px",
                    "background": f"{PURP2}22", "borderBottom": f"1px solid {PURP2}",
                    "fontSize": "11px", "fontFamily": MONO, "color": PURP2}
    progress_hide = {"display": "none", "height": "3px"}

    if state["error"]:
        return (f"✗ Training failed: {state['error']}", bar_style,
                no_update, _blank_fig(f"Error: {state['error']}"),
                _blank_fig("Training failed"), no_update,
                True, progress_hide, banner_style,
                f"✗ Training failed: {state['error']}")

    hp = _train_hparams
    seq_len = hp.get("seq_len", 10)
    hidden_dim = hp.get("hidden_dim", 64)
    gcn_layers = hp.get("gcn_layers", 3)
    lstm_layers = hp.get("lstm_layers", 2)
    epochs = hp.get("epochs", 200)

    # Build spectral fig with predictions
    preds = _evolve_predictor.predict(_evolve_features, steps=30)
    times = [r["time"] for r in _evolve_features]
    last_t = times[-1]
    n_total = len(_evolve_features) - seq_len
    split_idx = max(5, int(n_total * 0.8))
    train_boundary_t = _evolve_features[min(split_idx + seq_len, len(_evolve_features) - 1)]["time"]
    pred_times = [last_t + i + 1 for i in range(len(preds))]

    fig = go.Figure()
    for name, color in [("lambda_2", BLUE2), ("spectral_gap", AMBER2), ("spectral_radius", PURP2)]:
        fig.add_trace(go.Scatter(x=times, y=[r[name] for r in _evolve_features],
                                 mode="lines", name=name, line=dict(color=color, width=1.5)))
        if preds and name in preds[0]:
            fig.add_trace(go.Scatter(x=pred_times, y=[p[name] for p in preds],
                                     mode="lines", name=f"{name} (GNN pred)",
                                     line=dict(color=color, width=1.5, dash="dash")))
    fig.add_vline(x=train_boundary_t, line_dash="dot", line_color=AMBER2, opacity=0.6,
                  annotation_text="train|test", annotation_font_size=8, annotation_font_color=AMBER2)
    fig.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=10, t=10, b=30), height=220,
                      legend=dict(x=0.01, y=0.99, font=dict(size=8)),
                      xaxis=dict(title="Day", gridcolor=BORD2), yaxis=dict(gridcolor=BORD2))

    # Test scatter plot
    fig_scatter = go.Figure()
    if hasattr(_evolve_predictor, 'test_actuals') and len(_evolve_predictor.test_actuals) > 0:
        colors_map = {"lambda_2": BLUE2, "spectral_gap": AMBER2, "spectral_radius": PURP2}
        for i, name in enumerate(TARGET_NAMES):
            actuals = _evolve_predictor.test_actuals[:, i]
            predicted = _evolve_predictor.test_predictions[:, i]
            fig_scatter.add_trace(go.Scatter(
                x=actuals, y=predicted, mode="markers", name=name,
                marker=dict(color=colors_map.get(name, BLUE2), size=5, opacity=0.7),
            ))
        all_vals = np.concatenate([_evolve_predictor.test_actuals.flatten(),
                                   _evolve_predictor.test_predictions.flatten()])
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        fig_scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         line=dict(color=TEXT3, width=1, dash="dash"),
                                         showlegend=False))
    fig_scatter.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=10, t=10, b=30), height=220,
                              legend=dict(x=0.01, y=0.99, font=dict(size=8)),
                              xaxis=dict(title="Actual", gridcolor=BORD2),
                              yaxis=dict(title="Predicted", gridcolor=BORD2))

    # Training loss curve
    fig_loss = go.Figure()
    if hasattr(_evolve_predictor, 'training_history') and _evolve_predictor.training_history:
        hist = _evolve_predictor.training_history
        ep = [h["epoch"] for h in hist]
        fig_loss.add_trace(go.Scatter(x=ep, y=[h["train_loss"] for h in hist],
                                      mode="lines+markers", name="Train Loss",
                                      line=dict(color=PURP2, width=2), marker=dict(size=3)))
        test_losses = [h["test_loss"] for h in hist if h["test_loss"] is not None]
        if test_losses:
            test_ep = [h["epoch"] for h in hist if h["test_loss"] is not None]
            fig_loss.add_trace(go.Scatter(x=test_ep, y=test_losses,
                                          mode="lines+markers", name="Test Loss",
                                          line=dict(color=TEAL2, width=2), marker=dict(size=3)))
    fig_loss.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=10, t=10, b=30), height=220,
                           legend=dict(x=0.7, y=0.99, font=dict(size=8)),
                           xaxis=dict(title="Epoch", gridcolor=BORD2),
                           yaxis=dict(title="MSE Loss", gridcolor=BORD2, type="log"))

    # KPIs
    tr = _evolve_train_metrics
    te = _evolve_test_metrics
    n_params = _evolve_predictor.model.count_parameters() if _evolve_predictor.model else 0
    try:
        recon = compute_scg_reconstruction_accuracy()
        avg_r2 = f"{np.mean(recon['r2']):.4f}"
    except Exception:
        avg_r2 = "—"

    kpis = html.Div([
        metric_card("Snapshots", str(len(_evolve_features)), BLUE2, f"seq_len={seq_len}"),
        metric_card("SCG Recon R²", avg_r2, TEAL2, "avg over diffusion t"),
        metric_card("GNN Params", f"{n_params:,}", PURP2, f"{gcn_layers}×GCN + {lstm_layers}×LSTM"),
        metric_card("Train MSE", f"{tr['mse']:.6f}", AMBER2, f"R²={tr['r2']:.3f}"),
        metric_card("Test MSE", f"{te['mse']:.6f}", GREEN2, f"R²={te['r2']:.3f}"),
        metric_card("Test R²", f"{te['r2']:.4f}", BLUE2,
                    " / ".join(f"{v:.2f}" for v in te.get("r2_per_target", {}).values())),
    ], style={"display": "flex", "gap": "1px"})

    banner = (f"✓ GNN trained: {n_params:,} params, {epochs} epochs, "
              f"hidden={hidden_dim}, {gcn_layers}×GCN + {lstm_layers}×LSTM — "
              f"Test R²={te['r2']:.4f}, Test MSE={te['mse']:.6f} — "
              f"RAM: {state['ram_mb']:.0f} MB (+{state['ram_delta_mb']:.0f} MB)")

    return (banner, bar_style,
            fig, fig_scatter, fig_loss, kpis,
            True,  # disable interval
            progress_hide, banner_style,
            f"✓ GNN trained — test R²={te['r2']:.4f}, {len(preds)} steps predicted")


@callback(
    Output("evolve-scg-vs-basel", "figure"),
    Output("evolve-status",       "children", allow_duplicate=True),
    Input("btn-scg-vs-basel",     "n_clicks"),
    prevent_initial_call=True,
)
def compare_scg_basel(_):
    if len(_evolve_features) < 5:
        return _blank_fig("Run evolution first"), "✗ Need evolution data"
    comp = build_scg_vs_basel_comparison(_evolve_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp["time"], y=comp["scg_risk"],
                             mode="lines", name="SCG Risk (1−λ₂/ρ)", line=dict(color=BLUE2, width=2)))
    fig.add_trace(go.Bar(x=comp["time"], y=comp["basel_stress"],
                         name="Basel Stress Count", marker_color=RED, opacity=0.4, yaxis="y2"))
    # Add CoVaR and MES if available
    if "delta_covar" in comp:
        fig.add_trace(go.Scatter(x=comp["time"], y=comp["delta_covar"],
                                 mode="lines", name="ΔCoVaR", line=dict(color=PURP2, width=1.5)))
    if "mes" in comp:
        fig.add_trace(go.Scatter(x=comp["time"], y=comp["mes"],
                                 mode="lines", name="MES", line=dict(color=TEAL2, width=1.5)))
    fig.update_layout(**_DARK_LAYOUT, margin=dict(l=35, r=45, t=10, b=30), height=220,
                      legend=dict(x=0.01, y=0.99, font=dict(size=8)),
                      xaxis=dict(title="Step", gridcolor=BORD2),
                      yaxis=dict(title="Risk Score", gridcolor=BORD2, side="left"),
                      yaxis2=dict(title="Stressed Banks", overlaying="y", side="right",
                                  gridcolor="rgba(0,0,0,0)", range=[0, 12]))

    return fig, "✓ Comparison complete"


if __name__ == "__main__":
    port  = int(os.environ.get("DASHBOARD_PORT", 8050))
    debug = os.environ.get("DASHBOARD_DEBUG", "true").lower() == "true"
    print(f"  SCR Dashboard  →  http://localhost:{port}")
    app.run(debug=debug, port=port, use_reloader=False)
