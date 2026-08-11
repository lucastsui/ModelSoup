"""
Score hypotheses with:

S = exp(-MAE/s) * (1 + α·h) * (1 + β·max(0, -ΔMAE)) * (1 - exp(-n/n0))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Prediction:
    target: str
    aggregation: str
    area: str
    as_of_date: str
    start_date: str
    end_date: str
    change_pct: float
    tolerated_error_pp: float
    model_id: str
    notes: str = ""
    horizon_months: int | None = None
    price_type: str | None = None


@dataclass
class ScoreBreakdown:
    model_id: str
    pool: str
    S: float
    MAE: float
    hit_rate: float
    n: int
    delta_MAE: float = 0.0
    exp_mae: float = 0.0
    hit_factor: float = 0.0
    improve_factor: float = 0.0
    maturity: float = 0.0
    n_train: int = 0
    n_validation: int = 0
    params: dict = field(default_factory=dict)
    lineage: list = field(default_factory=list)
    notes: str = ""


def label_index(labels: list[dict]) -> dict[tuple, dict]:
    idx = {}
    for r in labels:
        key = (
            r["target"],
            r["aggregation"],
            r["area"],
            r["as_of_date"],
            r["start_date"],
            r["end_date"],
        )
        idx[key] = r
    return idx


def compute_score(
    predictions: list[Prediction],
    labels_by_key: dict[tuple, dict],
    scorer_cfg: dict,
    prev_MAE: float | None = None,
) -> tuple[ScoreBreakdown | None, list[dict]]:
    """Match predictions to labels; compute MAE, hit rate, S."""
    s = float(scorer_cfg.get("s", 2.0))
    alpha = float(scorer_cfg.get("alpha", 1.0))
    beta = float(scorer_cfg.get("beta", 0.5))
    n0 = float(scorer_cfg.get("n0", 30.0))

    abs_errs: list[float] = []
    hits = 0
    residuals: list[dict] = []
    for p in predictions:
        key = (
            p.target,
            p.aggregation,
            p.area,
            p.as_of_date,
            p.start_date,
            p.end_date,
        )
        lab = labels_by_key.get(key)
        if lab is None:
            continue
        actual = float(lab["actual_change_pct"])
        tol = float(
            p.tolerated_error_pp
            if p.tolerated_error_pp is not None
            else (lab.get("tolerated_error_pp") or 1.5)
        )
        err = abs(p.change_pct - actual)
        abs_errs.append(err)
        hit = err <= tol
        if hit:
            hits += 1
        residuals.append(
            {
                "model_id": p.model_id,
                "area": p.area,
                "as_of_date": p.as_of_date,
                "start_date": p.start_date,
                "end_date": p.end_date,
                "predicted_change_pct": p.change_pct,
                "actual_change_pct": actual,
                "abs_error_pp": err,
                "tolerated_error_pp": tol,
                "hit": int(hit),
                "split": lab.get("split", ""),
                "horizon_months": lab.get("horizon_months", p.horizon_months),
            }
        )

    n = len(abs_errs)
    if n == 0:
        return None, []

    mae = sum(abs_errs) / n
    h = hits / n
    delta_mae = 0.0 if prev_MAE is None else mae - prev_MAE

    exp_mae = math.exp(-mae / s)
    hit_factor = 1.0 + alpha * h
    # Relative improvement only (absolute -ΔMAE exploded for 10y / bad parents).
    # Cap contribution so S stays on a comparable scale across cycles.
    if prev_MAE is None or prev_MAE <= 0:
        improve_factor = 1.0
    else:
        rel = max(0.0, (prev_MAE - mae) / prev_MAE)
        improve_factor = 1.0 + beta * min(1.0, rel)
    maturity = 1.0 - math.exp(-n / n0)
    S = exp_mae * hit_factor * improve_factor * maturity

    n_train = sum(1 for r in residuals if r["split"] == "train")
    n_val = sum(1 for r in residuals if r["split"] == "validation")

    breakdown = ScoreBreakdown(
        model_id=predictions[0].model_id if predictions else "",
        pool="",
        S=S,
        MAE=mae,
        hit_rate=h,
        n=n,
        delta_MAE=delta_mae,
        exp_mae=exp_mae,
        hit_factor=hit_factor,
        improve_factor=improve_factor,
        maturity=maturity,
        n_train=n_train,
        n_validation=n_val,
    )
    return breakdown, residuals


def score_on_split(
    predictions: list[Prediction],
    labels_by_key: dict[tuple, dict],
    scorer_cfg: dict,
    split: str | None = "validation",
    prev_MAE: float | None = None,
) -> tuple[ScoreBreakdown | None, list[dict]]:
    if split:
        filtered = []
        for p in predictions:
            key = (
                p.target,
                p.aggregation,
                p.area,
                p.as_of_date,
                p.start_date,
                p.end_date,
            )
            lab = labels_by_key.get(key)
            if lab and lab.get("split") == split:
                filtered.append(p)
        predictions = filtered
    return compute_score(predictions, labels_by_key, scorer_cfg, prev_MAE=prev_MAE)
