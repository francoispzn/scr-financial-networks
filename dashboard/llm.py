"""
LLM-powered financial analyst using Cerebras API with web search tool.

Uses the Cerebras API (OpenAI-compatible function calling) to generate
narrative analysis of the current simulation state.  The LLM has access
to a ``web_search`` tool backed by the DuckDuckGo Instant Answer API
(no extra API key required) so it can fetch live market data or news.

Set CEREBRAS_API_KEY in your environment before starting the dashboard.

Usage::

    from dashboard.llm import analyze_system_state
    narrative = analyze_system_state(state_snapshot)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507"
_MAX_TOOL_ROUNDS = 5  # prevent infinite loops

_SYSTEM_PROMPT = """You are an expert financial stability analyst specialising in
European interbank networks and systemic risk. You interpret quantitative outputs
from a spectral coarse-graining (SCR) model and an agent-based model (ABM) of the
interbank market. Your audience is a senior risk officer or researcher who wants
concise, actionable insights — not generic disclaimers.

You have access to a web_search tool. Use it when you need current market data,
recent regulatory news, or up-to-date sovereign spread information to contextualise
the simulation results. Search selectively — only when real-world context genuinely
adds value.

When given a snapshot of the banking system, you will:
1. Summarise the current health of the system (2-3 sentences).
2. Identify the top 2-3 vulnerabilities or risk concentrations.
3. Flag any banks that are close to regulatory thresholds (CET1 < 8 %, LCR < 100 %).
4. Comment on network topology (density, key hubs) and what it implies for contagion.
5. Give a brief forward-looking assessment enriched by any relevant real-world context.

Be direct and quantitative. Use the numbers provided. Do not hedge excessively."""

# ── Tool definitions ─────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current financial news, market data, or regulatory "
                "updates. Returns a brief summary of the most relevant results. "
                "Use for: live spreads, recent ECB/EBA announcements, bank headlines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (English, concise).",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ── Web search implementation (DuckDuckGo Instant Answer API) ────────────────

def _web_search(query: str, max_results: int = 5) -> str:
    """
    Query the DuckDuckGo Instant Answer API and return a short text summary.
    Falls back to a descriptive error string on failure (never raises).
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=8,
            headers={"User-Agent": "scr-financial-networks-dashboard/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()

        parts: list[str] = []

        abstract = data.get("AbstractText", "").strip()
        if abstract:
            parts.append(abstract)

        for topic in data.get("RelatedTopics", [])[:max_results]:
            text = topic.get("Text", "").strip()
            if text:
                parts.append(f"• {text}")

        if parts:
            return "\n".join(parts)
        return f"No instant-answer results found for: {query}"

    except Exception as exc:
        logger.warning("web_search failed for query %r: %s", query, exc)
        return f"Search unavailable: {exc}"


# ── Tool dispatcher ──────────────────────────────────────────────────────────

def _dispatch_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the string result."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return "Error: could not parse tool arguments."

    if name == "web_search":
        return _web_search(args.get("query", ""))
    return f"Unknown tool: {name}"


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_user_message(snapshot: Dict[str, Any]) -> str:
    """Format a simulation snapshot into a user message for the LLM."""
    banks = snapshot.get("banks", {})
    system = snapshot.get("system_metrics", {})
    spectral = snapshot.get("spectral", {})
    time = snapshot.get("time", 0)

    lines = [f"## System snapshot — simulation step {time}\n"]

    lines.append("### System-level indicators")
    lines.append(f"- CISS (systemic stress): {system.get('CISS', 'N/A'):.3f}")
    lines.append(f"- Funding stress: {system.get('funding_stress', 'N/A'):.3f}")
    lines.append(f"- Average CET1 ratio: {system.get('avg_CET1_ratio', 0):.2f}%")
    lines.append(f"- Average LCR: {system.get('avg_LCR', 0):.1f}%")
    lines.append(f"- Network density: {system.get('network_density', 0):.3f}")
    lines.append("")

    lines.append("### Individual bank metrics")
    lines.append("| Bank | CET1 (%) | LCR (%) | Total Assets (€bn) | Solvent | Liquid |")
    lines.append("|------|----------|---------|-------------------|---------|--------|")
    for bid, bdata in banks.items():
        ta = bdata.get("total_assets", 0) / 1e9
        lines.append(
            f"| {bid} | {bdata.get('CET1_ratio', 0):.2f} "
            f"| {bdata.get('LCR', 0):.1f} "
            f"| {ta:.0f} "
            f"| {'✓' if bdata.get('solvent', True) else '✗'} "
            f"| {'✓' if bdata.get('liquid', True) else '✗'} |"
        )
    lines.append("")

    if spectral:
        lines.append("### Spectral properties")
        lines.append(
            f"- Algebraic connectivity (Fiedler value): "
            f"{spectral.get('algebraic_connectivity', 0):.4f}"
        )
        lines.append(
            f"- Spectral gap: {spectral.get('gap_size', 0):.4f} "
            f"(at index {spectral.get('gap_index', 0)})"
        )
        lines.append(
            f"- Spectral radius: {spectral.get('spectral_radius', 0):.4f}"
        )

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze_system_state(
    snapshot: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a narrative analysis of the current simulation state.

    The LLM may call ``web_search`` autonomously to fetch live context.

    Parameters
    ----------
    snapshot : dict
        Combined simulation state (banks, system metrics, spectral data).
    model : str, optional
        Cerebras model ID. Defaults to ``llama-4-scout-17b-16e-instruct``.
    api_key : str, optional
        Cerebras API key. Falls back to ``CEREBRAS_API_KEY`` env var.

    Returns
    -------
    str
        Markdown-formatted narrative analysis.
    """
    key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        return (
            "**Cerebras API key not configured.**\n\n"
            "Set the `CEREBRAS_API_KEY` environment variable and restart the dashboard."
        )

    try:
        from openai import OpenAI  # openai >= 1.0 required
    except ImportError:
        return (
            "**`openai` package not installed.**\n\n"
            "Run `pip install openai>=1.0` to enable LLM analysis."
        )

    client = OpenAI(api_key=key, base_url=_CEREBRAS_BASE_URL)
    chosen_model = model or _DEFAULT_MODEL

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(snapshot)},
    ]

    try:
        for _round in range(_MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.3,
            )
            msg = response.choices[0].message

            # No tool calls — we have the final answer
            if not msg.tool_calls:
                return msg.content or "No response from model."

            # Append assistant's message (with tool calls)
            messages.append(msg.model_dump(exclude_unset=True))

            # Execute each tool call and append results
            for tc in msg.tool_calls:
                result = _dispatch_tool(tc.function.name, tc.function.arguments)
                logger.debug(
                    "Tool call: %s(%s) → %s…",
                    tc.function.name,
                    tc.function.arguments[:80],
                    result[:120],
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # If we exhausted rounds, ask for a final summary without tools
        messages.append({
            "role": "user",
            "content": "Please provide your final analysis now.",
        })
        final = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return final.choices[0].message.content or "No final response."

    except Exception as exc:
        logger.exception("LLM analysis failed")
        return f"**LLM analysis failed:** {exc}"


# ── Real data fetcher ────────────────────────────────────────────────────────

_FETCH_SYSTEM = """You are a financial data extraction agent. You have access to web_search.
Search for the most recent publicly available regulatory data for each requested bank.
Focus on official sources: annual reports, EBA transparency exercise, investor presentations,
Q4 or H1 results press releases. Return ONLY valid JSON — no prose, no code fences."""

_FETCH_PROMPT_TEMPLATE = """
Search for the most recent CET1 ratio (%), LCR (%), NSFR (%), and total assets (EUR)
for each of the following banks:

{bank_list}

Use web_search for each bank to find their latest reported figures.
Return a single JSON object — keys are the bank IDs below, values are dicts with keys:
  CET1_ratio  (float, percent)
  LCR         (float, percent)
  NSFR        (float, percent, use null if unavailable)
  total_assets (float, EUR value — e.g. 1.32e12 for €1.32 trillion)

Bank IDs and names:
{id_name_map}

Example output (do NOT include this, find real values):
{{"DE_DBK": {{"CET1_ratio": 13.7, "LCR": 148, "NSFR": 119, "total_assets": 1.32e12}}}}

Return ONLY the JSON object. No explanation, no markdown fences.
"""

# ── GNN feature fetcher ──────────────────────────────────────────────────────

_GNN_FETCH_SYSTEM = """You are a financial data extraction agent specialising in
European banking regulatory and market data. You have web_search access.
Search official sources: EBA transparency exercises, annual reports, Pillar 3 disclosures,
investor presentations, Q4/H1 press releases, ECB banking supervision data.
Return ONLY valid JSON — no prose, no markdown fences."""

_GNN_FETCH_PROMPT = """
For each bank listed below, search and return the MOST RECENT available values for ALL
of the following fields. Use web_search for each bank. If a field is genuinely unavailable
after searching, use null.

Fields required (all numeric):
  CET1_ratio        – CET1 capital ratio (%, e.g. 13.7)
  tier1_ratio       – Tier 1 capital ratio (%)
  total_capital_ratio – Total capital ratio (%)
  LCR               – Liquidity Coverage Ratio (%)
  NSFR              – Net Stable Funding Ratio (%)
  leverage_ratio    – Basel III leverage ratio (%)
  npl_ratio         – Non-performing loan ratio (%)
  roe               – Return on equity (%)
  roa               – Return on assets (%)
  cost_income_ratio – Cost-to-income ratio (%)
  total_assets      – Total assets (EUR, e.g. 1.32e12)
  tier1_capital     – Tier 1 capital amount (EUR)
  rwa               – Risk-weighted assets (EUR)
  net_income        – Net income (EUR, last annual)
  deposits          – Total customer deposits (EUR)

Bank IDs and names:
{id_name_map}

Return a single JSON object. Keys are the bank IDs above, values are dicts with the
fields listed. Example structure (use REAL values, not these):
{{"DE_DBK": {{"CET1_ratio": 13.7, "tier1_ratio": 14.9, "total_capital_ratio": 17.2,
  "LCR": 148, "NSFR": 119, "leverage_ratio": 4.6, "npl_ratio": 1.8,
  "roe": 9.1, "roa": 0.4, "cost_income_ratio": 62, "total_assets": 1.32e12,
  "tier1_capital": 5.2e10, "rwa": 3.5e11, "net_income": 4.2e9,
  "deposits": 6.1e11}}}}

Return ONLY the JSON object.
"""

# Numeric fields we accept from the GNN fetch (in this order → feature vector)
GNN_NODE_FEATURES = [
    "CET1_ratio", "tier1_ratio", "total_capital_ratio",
    "LCR", "NSFR", "leverage_ratio",
    "npl_ratio", "roe", "roa", "cost_income_ratio",
    "total_assets", "tier1_capital", "rwa", "net_income", "deposits",
]


def fetch_bank_features_for_gnn(
    bank_ids: list,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Cerebras LLM + web_search to fetch a rich feature set for GNN training.

    Returns ``{bank_id: {feature: value, ...}}`` for all GNN_NODE_FEATURES,
    or ``{"error": "..."}`` on failure.  Missing fields are set to ``None``.
    """
    from dashboard.data_loader import BANK_LABELS  # avoid circular at module level

    key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        return {"error": "CEREBRAS_API_KEY not set."}

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "`openai` package not installed."}

    client = OpenAI(api_key=key, base_url=_CEREBRAS_BASE_URL)
    chosen_model = model or _DEFAULT_MODEL

    id_name_map = {bid: BANK_LABELS.get(bid, bid) for bid in bank_ids}
    prompt = _GNN_FETCH_PROMPT.format(id_name_map=json.dumps(id_name_map, indent=2))

    messages: list[dict] = [
        {"role": "system", "content": _GNN_FETCH_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    try:
        for _round in range(_MAX_TOOL_ROUNDS * 3):  # more rounds for more banks/features
            resp = client.chat.completions.create(
                model=chosen_model, messages=messages,
                tools=_TOOLS, tool_choice="auto",
                max_tokens=4000, temperature=0.1,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                content = (msg.content or "").strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                try:
                    raw = json.loads(content)
                    result: Dict[str, Any] = {}
                    for bid in bank_ids:
                        fields = raw.get(bid, {})
                        node: Dict[str, Any] = {}
                        for feat in GNN_NODE_FEATURES:
                            v = fields.get(feat)
                            if v is not None:
                                try:
                                    node[feat] = float(v)
                                except (TypeError, ValueError):
                                    node[feat] = None
                            else:
                                node[feat] = None
                        result[bid] = node
                    logger.info(
                        "GNN fetch: %d banks, features: %s",
                        len(result), GNN_NODE_FEATURES,
                    )
                    return result
                except json.JSONDecodeError as exc:
                    logger.warning("GNN JSON parse failed: %s", exc)
                    return {"error": f"Could not parse LLM response: {exc}"}

            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                result_str = _dispatch_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result_str,
                })

        return {"error": "LLM exhausted tool rounds without returning GNN data."}

    except Exception as exc:
        logger.exception("fetch_bank_features_for_gnn failed")
        return {"error": str(exc)}


def fetch_bank_data_via_llm(
    bank_ids: list,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Cerebras LLM + web_search to fetch real bank regulatory data.

    Returns a dict of {bank_id: {CET1_ratio, LCR, NSFR, total_assets}}
    or {"error": "..."} on failure.
    """
    from dashboard.data_loader import BANK_LABELS  # avoid circular at module level

    key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        return {"error": "CEREBRAS_API_KEY not set."}

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "`openai` package not installed."}

    client = OpenAI(api_key=key, base_url=_CEREBRAS_BASE_URL)
    chosen_model = model or _DEFAULT_MODEL

    id_name_map = {bid: BANK_LABELS.get(bid, bid) for bid in bank_ids}
    bank_list_str = "\n".join(f"- {BANK_LABELS.get(b,b)} ({b})" for b in bank_ids)

    prompt = _FETCH_PROMPT_TEMPLATE.format(
        bank_list=bank_list_str,
        id_name_map=json.dumps(id_name_map, indent=2),
    )

    messages: list[dict] = [
        {"role": "system", "content": _FETCH_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    try:
        for _round in range(_MAX_TOOL_ROUNDS):
            resp = client.chat.completions.create(
                model=chosen_model, messages=messages,
                tools=_TOOLS, tool_choice="auto",
                max_tokens=2000, temperature=0.1,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                # Parse JSON from the final response
                content = (msg.content or "").strip()
                # Strip accidental markdown fences
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                try:
                    data = json.loads(content)
                    # Sanitise: keep only numeric fields we expect
                    result = {}
                    for bid, fields in data.items():
                        if bid not in bank_ids:
                            continue
                        clean = {}
                        for k in ("CET1_ratio", "LCR", "NSFR", "total_assets"):
                            v = fields.get(k)
                            if v is not None:
                                try:
                                    clean[k] = float(v)
                                except (TypeError, ValueError):
                                    pass
                        if clean:
                            result[bid] = clean
                    logger.info("LLM fetched data for %d banks", len(result))
                    return result
                except json.JSONDecodeError as exc:
                    logger.warning("JSON parse failed: %s\nContent: %.200s", exc, content)
                    return {"error": f"Could not parse LLM JSON response: {exc}"}

            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                result_str = _dispatch_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result_str,
                })

        return {"error": "LLM exhausted tool rounds without returning data."}

    except Exception as exc:
        logger.exception("fetch_bank_data_via_llm failed")
        return {"error": str(exc)}


# ── Snapshot builder ─────────────────────────────────────────────────────────

def build_snapshot(
    sim_state: Dict[str, Any],
    system_metrics: Dict[str, Any],
    spectral_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine raw simulation data into a snapshot dict."""
    banks_snapshot = {}
    for node in sim_state.get("nodes", []):
        bid = node["id"]
        banks_snapshot[bid] = {
            "CET1_ratio": node.get("CET1_ratio", 0),
            "LCR": node.get("LCR", 0),
            "total_assets": node.get("total_assets", 0),
            "solvent": node.get("solvent", True),
            "liquid": node.get("liquid", True),
        }

    return {
        "time": system_metrics.get("time", 0),
        "banks": banks_snapshot,
        "system_metrics": system_metrics,
        "spectral": {
            "algebraic_connectivity": spectral_data.get("algebraic_connectivity", 0),
            "gap_size": spectral_data.get("gap_size", 0),
            "gap_index": spectral_data.get("gap_index", 0),
            "spectral_radius": spectral_data.get("spectral_radius", 0),
        },
    }
