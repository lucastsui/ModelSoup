"""Load panels, labels, and feature lookup for the research loop."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x: Any, default: float | None = None) -> float | None:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_config(config_path: Path) -> dict:
    import json

    with open(config_path) as f:
        cfg = json.load(f)
    base = config_path.parent
    data_root = (base / cfg["data_root"]).resolve()
    cfg["_data_root"] = data_root
    cfg["_panel_path"] = data_root / cfg["panel_path"]
    cfg["_label_path"] = data_root / cfg["label_path"]
    cfg["_split_path"] = data_root / cfg["split_path"]
    return cfg


def build_feature_index(panel_rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(area, date) -> panel row with numeric fields."""
    idx = {}
    for r in panel_rows:
        key = (r["area"], r["date"][:10])
        idx[key] = r
    return idx


def filter_labels(labels: list[dict], cfg: dict) -> list[dict]:
    smoke = cfg.get("smoke", {})
    price_types = set(smoke.get("price_types") or [])
    horizons = {str(h) for h in (smoke.get("horizons") or [])}
    areas = set(smoke.get("areas_include") or [])
    min_as_of = smoke.get("min_as_of") or "1900-01-01"
    out = []
    for r in labels:
        if price_types and r.get("price_type") not in price_types:
            continue
        if horizons and str(r.get("horizon_months")) not in horizons:
            continue
        if areas and r.get("area") not in areas:
            continue
        if (r.get("as_of_date") or "") < min_as_of:
            continue
        if r.get("actual_change_pct") in ("", None):
            continue
        out.append(r)
    return out


def attach_splits(labels: list[dict], split_rows: list[dict]) -> list[dict]:
    """Prefer external split file; else 80% time cut on as_of."""
    split_map = {}
    for s in split_rows:
        k = (
            s.get("target"),
            s.get("aggregation"),
            s.get("area"),
            s.get("as_of_date"),
            s.get("start_date"),
            s.get("end_date"),
            str(s.get("horizon_months")),
            s.get("price_type"),
        )
        split_map[k] = s.get("split", "train")

    dates = sorted({r["as_of_date"] for r in labels})
    cut = dates[int(len(dates) * 0.8)] if dates else ""

    out = []
    for r in labels:
        k = (
            r.get("target"),
            r.get("aggregation"),
            r.get("area"),
            r.get("as_of_date"),
            r.get("start_date"),
            r.get("end_date"),
            str(r.get("horizon_months")),
            r.get("price_type"),
        )
        split = split_map.get(k)
        if not split:
            split = "train" if r["as_of_date"] < cut else "validation"
        rr = dict(r)
        rr["split"] = split
        out.append(rr)
    return out


def feature_vector(panel_row: dict | None, horizon: int) -> dict[str, float]:
    """Numeric features available at as_of (no forward fields)."""
    if not panel_row:
        return {}
    keys = [
        "avg_price_gbp",
        "mom_change_pct",
        "yoy_change_pct",
        "sales_volume",
        "bank_rate_pct",
        "price_lag1",
        "mom_lag1",
        "price_lag2",
        "mom_lag2",
        "price_lag3",
        "mom_lag3",
        "price_lag6",
        "mom_lag6",
        "price_lag12",
        "mom_lag12",
        "ftb_rent_pcm",
        "ftb_mortgage_pcm",
        "ftb_rent_minus_mortgage",
    ]
    feats = {"horizon_months": float(horizon)}
    for k in keys:
        v = fnum(panel_row.get(k))
        if v is not None:
            feats[k] = v
    # derived
    if "mom_change_pct" in feats:
        feats["mom_x_horizon"] = feats["mom_change_pct"] * horizon
    if "yoy_change_pct" in feats:
        feats["yoy_scaled"] = feats["yoy_change_pct"] * (horizon / 12.0)
    if "bank_rate_pct" in feats:
        feats["bank_rate"] = feats["bank_rate_pct"]
    return feats
