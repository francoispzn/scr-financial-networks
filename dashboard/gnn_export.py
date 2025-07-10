"""
GNN dataset exporter for the SCR Financial Networks dashboard.

Builds a graph dataset from the current simulation state + LLM-fetched bank
features and writes it to disk in multiple formats:

  nodes.csv          — node feature matrix (one row per bank)
  edges.csv          — directed edge list with weights
  graph_data.json    — full graph as JSON (PyG-loadable via custom loader)
  pyg_data.pt        — torch_geometric.data.Data object (if PyG installed)
  metadata.json      — feature names, bank labels, dataset provenance

Usage::

    from dashboard.gnn_export import build_and_export
    info = build_and_export(gnn_features, output_dir="data/gnn_datasets")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dashboard.llm import GNN_NODE_FEATURES
from dashboard.data_loader import BANK_LABELS, BANK_COUNTRIES

logger = logging.getLogger(__name__)

# Features derived from the simulation (not from the LLM web-fetch).
# These are always available and supplement the LLM-fetched features.
_SIM_FEATURES = ["CET1_ratio", "LCR", "NSFR", "total_assets"]

# Binary node labels (for supervised GNN tasks).
_LABEL_FIELDS = ["solvent", "liquid"]


def _country_encoding(bank_ids: List[str]) -> Dict[str, int]:
    """Map unique country codes to integer labels."""
    countries = sorted({BANK_COUNTRIES.get(b, "XX") for b in bank_ids})
    return {b: countries.index(BANK_COUNTRIES.get(b, "XX")) for b in bank_ids}


def build_graph_tensors(
    bank_ids: List[str],
    node_data: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Convert node/edge dicts into numpy arrays ready for GNN consumption.

    Returns
    -------
    X          : float32 [N, F]  — node feature matrix (NaN-imputed with column mean)
    edge_index : int64   [2, E]  — source/target index pairs
    edge_attr  : float32 [E, 1]  — edge weight (normalised 0-1)
    y          : int64   [N, 2]  — binary labels [solvent, liquid]
    feat_names : list[str]       — feature column names (matches X columns)
    """
    if feature_cols is None:
        feature_cols = GNN_NODE_FEATURES + ["country_code"]

    country_enc = _country_encoding(bank_ids)
    n = len(bank_ids)
    idx = {b: i for i, b in enumerate(bank_ids)}

    # ── Node features ────────────────────────────────────────────────────────
    X_raw = np.full((n, len(feature_cols)), np.nan, dtype=np.float32)
    for i, bid in enumerate(bank_ids):
        row = node_data.get(bid, {})
        for j, feat in enumerate(feature_cols):
            if feat == "country_code":
                X_raw[i, j] = float(country_enc.get(bid, 0))
            else:
                v = row.get(feat)
                if v is not None:
                    try:
                        X_raw[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass

    # Impute missing values with column median (robust to outliers)
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        valid = col[~np.isnan(col)]
        fill = float(np.median(valid)) if len(valid) > 0 else 0.0
        X_raw[np.isnan(col), j] = fill

    # ── Labels ───────────────────────────────────────────────────────────────
    y = np.zeros((n, len(_LABEL_FIELDS)), dtype=np.int64)
    for i, bid in enumerate(bank_ids):
        row = node_data.get(bid, {})
        for j, lf in enumerate(_LABEL_FIELDS):
            y[i, j] = int(bool(row.get(lf, True)))

    # ── Edges ────────────────────────────────────────────────────────────────
    valid_edges = [e for e in edges
                   if e["source"] in idx and e["target"] in idx and e["weight"] > 0]
    if valid_edges:
        max_w = max(e["weight"] for e in valid_edges) or 1.0
        srcs = np.array([idx[e["source"]] for e in valid_edges], dtype=np.int64)
        dsts = np.array([idx[e["target"]] for e in valid_edges], dtype=np.int64)
        edge_index = np.stack([srcs, dsts], axis=0)  # [2, E]
        edge_attr = np.array(
            [[e["weight"] / max_w] for e in valid_edges], dtype=np.float32
        )  # [E, 1]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, 1), dtype=np.float32)

    return X_raw, edge_index, edge_attr, y, feature_cols


def build_and_export(
    gnn_features: Dict[str, Dict[str, Any]],
    sim_graph: Dict[str, Any],
    output_dir: str = "data/gnn_datasets",
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a GNN dataset from LLM-fetched features + simulation graph and save
    it to *output_dir*.

    Parameters
    ----------
    gnn_features : {bank_id: {feature: value}}
        Output of ``fetch_bank_features_for_gnn()``.
    sim_graph : {nodes: [...], edges: [...]}
        Output of ``simulation_state.get_network_graph_data()``.
    output_dir : str
        Directory to write dataset files into.
    tag : str, optional
        Short label for the export (used in filenames). Defaults to a timestamp.

    Returns
    -------
    dict with keys: output_dir, files, n_nodes, n_edges, n_features, timestamp
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = tag or timestamp
    out = os.path.join(output_dir, tag)
    os.makedirs(out, exist_ok=True)

    nodes_raw = sim_graph.get("nodes", [])
    edges_raw = sim_graph.get("edges", [])
    bank_ids  = [n["id"] for n in nodes_raw]

    # Merge simulation node data with LLM-fetched features
    # LLM features take precedence for shared fields (except solvent/liquid which
    # come from the ABM).
    node_data: Dict[str, Dict[str, Any]] = {}
    for nd in nodes_raw:
        bid = nd["id"]
        merged = dict(nd)  # sim fields: CET1_ratio, LCR, NSFR, total_assets, solvent, liquid
        llm_fields = gnn_features.get(bid, {})
        for k, v in llm_fields.items():
            if v is not None:
                merged[k] = v          # LLM overrides sim for financial ratios
        node_data[bid] = merged

    feature_cols = GNN_NODE_FEATURES + ["country_code"]
    X, edge_index, edge_attr, y, feat_names = build_graph_tensors(
        bank_ids, node_data, edges_raw, feature_cols
    )

    # ── Save CSVs ────────────────────────────────────────────────────────────
    import csv

    nodes_csv = os.path.join(out, "nodes.csv")
    with open(nodes_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bank_id", "label", "country"] + feat_names
                        + ["solvent", "liquid"])
        for i, bid in enumerate(bank_ids):
            writer.writerow(
                [bid, BANK_LABELS.get(bid, bid), BANK_COUNTRIES.get(bid, "")]
                + X[i].tolist()
                + [int(y[i, 0]), int(y[i, 1])]
            )

    edges_csv = os.path.join(out, "edges.csv")
    with open(edges_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_idx", "target_idx", "source_id", "target_id",
                         "weight_norm"])
        for k in range(edge_index.shape[1]):
            s, t = int(edge_index[0, k]), int(edge_index[1, k])
            writer.writerow([s, t, bank_ids[s], bank_ids[t],
                             float(edge_attr[k, 0])])

    # ── Save graph JSON ───────────────────────────────────────────────────────
    graph_json = {
        "bank_ids": bank_ids,
        "bank_labels": {b: BANK_LABELS.get(b, b) for b in bank_ids},
        "bank_countries": {b: BANK_COUNTRIES.get(b, "") for b in bank_ids},
        "feature_names": feat_names,
        "label_names": _LABEL_FIELDS,
        "node_features": X.tolist(),      # [N, F]
        "edge_index": edge_index.tolist(),  # [2, E]
        "edge_attr": edge_attr.tolist(),    # [E, 1]
        "labels": y.tolist(),              # [N, 2]
        "raw_node_data": node_data,        # full unprocessed fields per bank
    }
    graph_json_path = os.path.join(out, "graph_data.json")
    with open(graph_json_path, "w") as f:
        json.dump(graph_json, f, indent=2, default=str)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "timestamp": timestamp,
        "tag": tag,
        "n_nodes": len(bank_ids),
        "n_edges": int(edge_index.shape[1]),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "label_names": _LABEL_FIELDS,
        "llm_coverage": {
            bid: sum(1 for v in gnn_features.get(bid, {}).values() if v is not None)
            for bid in bank_ids
        },
        "source": "Cerebras LLM + DuckDuckGo web search + SCR ABM simulation",
    }
    meta_path = os.path.join(out, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    files = {
        "nodes_csv": nodes_csv,
        "edges_csv": edges_csv,
        "graph_json": graph_json_path,
        "metadata": meta_path,
    }

    # ── PyTorch Geometric .pt (optional) ─────────────────────────────────────
    try:
        import torch
        from torch_geometric.data import Data  # type: ignore

        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
        )
        # Store metadata as graph-level attributes
        data.bank_ids     = bank_ids
        data.feat_names   = feat_names
        data.label_names  = _LABEL_FIELDS
        pt_path = os.path.join(out, "pyg_data.pt")
        torch.save(data, pt_path)
        files["pyg_pt"] = pt_path
        logger.info("Saved PyTorch Geometric Data object → %s", pt_path)
    except ImportError:
        logger.info("torch_geometric not installed — skipping .pt export")

    # ── Also save numpy arrays ────────────────────────────────────────────────
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "edge_index.npy"), edge_index)
    np.save(os.path.join(out, "edge_attr.npy"), edge_attr)
    np.save(os.path.join(out, "y.npy"), y)
    files["numpy"] = out

    logger.info(
        "GNN dataset exported: %d nodes, %d edges, %d features → %s",
        len(bank_ids), int(edge_index.shape[1]), len(feat_names), out,
    )

    return {
        "output_dir": out,
        "files": files,
        "n_nodes": len(bank_ids),
        "n_edges": int(edge_index.shape[1]),
        "n_features": len(feat_names),
        "timestamp": timestamp,
    }
