"""
Human-readable summaries for hypotheses.

- mutation: how a child differs from its direct parent(s)
- prediction: description of *actual* forecast outputs (from run predictions)
"""
from __future__ import annotations

import statistics
from typing import Any

from hypotheses import Hypothesis
from scorer import Prediction


def _sc(x: Any, dig: int = 1) -> str:
    try:
        n = float(x)
    except Exception:
        return str(x)
    t = f"{n:.{dig}f}"
    return t.rstrip("0").rstrip(".") if "." in t else t


def _fmt_kind(kind: str) -> str:
    names = {
        "zero": "no price change",
        "constant": "fixed monthly drift",
        "last_mom": "last-month price momentum",
        "last_yoy_scaled": "last-year price growth",
        "blend_mom_yoy": "blend of monthly and yearly growth",
        "mean_reversion": "mean-reverting growth",
        "rate_penalty": "growth with bank-rate drag",
        "momentum_lag": "multi-month momentum mix",
        "ridge_linear": "linear fit on market features",
        "linear_nn": "linear neural head",
        "mlp": "neural network",
    }
    return names.get(kind, kind or "unknown rule")


def summarize_mutation(child: Hypothesis, parents: list[Hypothesis] | None = None) -> str:
    """One line: how this hypothesis differs from its direct parent(s)."""
    parents = parents or []
    # Prefer live parent objects matching parent_ids
    if not parents and not child.parent_ids:
        return "Seed hypothesis (no parents)"

    if "elite_copy" in (child.model_id or "") or "exact carry-forward" in (child.notes or "").lower():
        pid = child.parent_ids[0] if child.parent_ids else "parent"
        return f"Exact copy of parent {pid} (elite carry-forward; no mutation)"

    if not parents:
        pnames = ", ".join(child.parent_ids)
        return f"Evolved from {pnames} (parent details unavailable)"

    if len(parents) == 1:
        p = parents[0]
        # Detect exact copy by kind+core params
        if child.kind == p.kind:
            ck = {k: v for k, v in (child.params or {}).items() if not str(k).endswith("_summary") and not str(k).startswith("_")}
            pk = {k: v for k, v in (p.params or {}).items() if not str(k).endswith("_summary") and not str(k).startswith("_")}
            # ignore train diagnostics
            for drop in ("train_n", "train_mse", "train_backend", "architecture"):
                ck.pop(drop, None)
                pk.pop(drop, None)
            if ck == pk:
                return f"Exact copy of parent {p.model_id} (elite carry-forward; no mutation)"
        bits = []
        if child.kind != p.kind:
            bits.append(f"changed method from {_fmt_kind(p.kind)} to {_fmt_kind(child.kind)}")
        # param diffs
        ck, pk = child.params or {}, p.params or {}
        keys = sorted(set(ck) | set(pk))
        changed = []
        for k in keys:
            if k.startswith("_"):
                continue
            if k in ("train_n", "train_mse", "train_backend", "architecture", "feature_names"):
                continue
            va, vb = ck.get(k), pk.get(k)
            if va != vb and va is not None:
                if isinstance(va, float) or isinstance(vb, float):
                    try:
                        changed.append(f"{k} {_sc(vb)}→{_sc(va)}")
                    except Exception:
                        changed.append(f"{k} updated")
                else:
                    changed.append(f"{k}={va}")
        if child.kind == "mlp" and p.kind == "mlp":
            h1 = p.params.get("hidden_layers") or p.params.get("architecture")
            h2 = child.params.get("hidden_layers") or child.params.get("architecture")
            if h1 != h2:
                bits.append(f"network shape {h1}→{h2}")
            a1, a2 = p.params.get("activation"), child.params.get("activation")
            if a1 and a2 and a1 != a2:
                bits.append(f"activation {a1}→{a2}")
        if changed and child.kind == p.kind:
            bits.append("tweaked " + ", ".join(changed[:4]))
        if not bits:
            bits.append(f"re-proposed same {_fmt_kind(child.kind)} rule as {p.model_id}")
        return f"From {p.model_id}: " + "; ".join(bits)

    # two parents (crossover / in_pool / cross_pool)
    a, b = parents[0], parents[1]
    if child.kind == "mlp" and (a.kind == "mlp" or b.kind == "mlp"):
        return (
            f"Merged ideas from {a.model_id} and {b.model_id} into a neural net "
            f"({_fmt_kind(child.kind)}) for house-price change"
        )
    if child.kind != a.kind and child.kind != b.kind:
        return (
            f"Combined {a.model_id} ({_fmt_kind(a.kind)}) and {b.model_id} ({_fmt_kind(b.kind)}) "
            f"into {_fmt_kind(child.kind)}"
        )
    return (
        f"Blended parents {a.model_id} and {b.model_id} into {_fmt_kind(child.kind)} "
        f"for the price forecast"
    )


def summarize_prediction(
    preds: list[Prediction],
    *,
    prefer_area: str = "London",
    horizon_months: int | None = None,
) -> str:
    """
    One line from *actual* model outputs (not a kind guess).
    Prefers London if present; otherwise all validation preds passed in.
    """
    if not preds:
        return "No forecasts produced when the hypothesis was run"

    # Prefer target area; drop non-finite predictions (exploded linear fits, etc.)
    def _finite_vals(ps: list[Prediction]) -> list[float]:
        out = []
        for p in ps:
            try:
                v = float(p.change_pct)
            except Exception:
                continue
            if v != v or abs(v) == float("inf"):  # NaN / Inf
                continue
            # cap absurd values for summary display only
            if abs(v) > 1e6:
                continue
            out.append(v)
        return out

    area_preds = [p for p in preds if (p.area or "") == prefer_area]
    used_area = bool(_finite_vals(area_preds))
    source_preds = area_preds if used_area else preds
    vals = _finite_vals(source_preds)
    if not vals:
        return "When run, the hypothesis produced no usable price forecasts (degenerate / non-finite outputs)"

    mean = statistics.mean(vals)
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    try:
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    except Exception:
        sd = 0.0

    # horizon label from preds that contributed finite values
    hs = set()
    for p in source_preds:
        try:
            v = float(p.change_pct)
            if v == v and abs(v) != float("inf") and abs(v) <= 1e6 and p.horizon_months:
                hs.add(p.horizon_months)
        except Exception:
            continue
    if horizon_months:
        htxt = f"{horizon_months // 12}-year" if horizon_months >= 12 and horizon_months % 12 == 0 else f"{horizon_months}-month"
    elif len(hs) == 1:
        h = next(iter(hs))
        htxt = f"{h // 12}-year" if h and h >= 12 and h % 12 == 0 else f"{h}-month"
    else:
        htxt = "horizon"

    where = prefer_area if used_area else "the validation areas"
    # flat vs varied
    if sd < 0.5 and abs(hi - lo) < 2.0:
        return (
            f"When run on {where}, forecasts house prices change by about {_sc(mean)}% "
            f"over the {htxt} window (almost the same for every case)"
        )
    return (
        f"When run on {where}, forecasts house prices change by about {_sc(mean)}% "
        f"over the {htxt} window (median {_sc(med)}%, typically from {_sc(lo)}% to {_sc(hi)}%)"
    )


def attach_summaries(
    hyp: Hypothesis,
    preds: list[Prediction],
    parent_hyps: list[Hypothesis] | None = None,
    *,
    prefer_area: str = "London",
) -> Hypothesis:
    """Write mutation_summary and prediction_summary onto hyp.params (and attributes)."""
    mut = summarize_mutation(hyp, parent_hyps)
    pred = summarize_prediction(preds, prefer_area=prefer_area)
    hyp.params = dict(hyp.params or {})
    hyp.params["mutation_summary"] = mut
    hyp.params["prediction_summary"] = pred
    # also plain attributes for easy access
    hyp.mutation_summary = mut  # type: ignore[attr-defined]
    hyp.prediction_summary = pred  # type: ignore[attr-defined]
    return hyp
