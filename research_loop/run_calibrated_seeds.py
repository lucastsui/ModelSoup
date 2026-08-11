#!/usr/bin/env python3
"""
Wipe run data, seed 10 param + 10 nn hypotheses, tune constant-level
predictors so scores are roughly evenly spaced on [0, 0.5], write cycle
artifacts, and rebuild lineage_graph.json for the viz.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_access import (
    attach_splits,
    build_feature_index,
    filter_labels,
    load_config,
    load_csv,
)
from hypotheses import (
    FEATURE_SET,
    Hypothesis,
    generate_predictions,
    prepare_model,
)
from scorer import label_index, score_on_split
from summarize import attach_summaries
from evolver import ensure_dirs, write_csv


# Evenly spaced targets on [0, 0.5] (10 points)
TARGET_S = [round(i * 0.5 / 9, 4) for i in range(10)]  # 0.0 … 0.5


def wipe_run_data(loop_root: Path) -> None:
    """
    Clear run artifacts but keep the previous lineage_graph.json until new
    scores are written, so the viz never flashes an empty chart.
    """
    paths = ensure_dirs(loop_root)
    runs = paths["runs"]
    if runs.exists():
        for child in runs.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    archive = paths["archive"]
    if archive.exists():
        for child in archive.iterdir():
            if child.is_file():
                child.unlink()
    for pool in ("param", "nn"):
        p = paths[pool] / "current_pool.csv"
        if p.exists():
            p.unlink()
    latest = loop_root / "LATEST_RUN.txt"
    if latest.exists():
        latest.unlink()
    # Intentionally do NOT delete viz/lineage_graph.json here — rebuild after
    # new scores exist. build_lineage_data also refuses empty overwrites.
    print("Wiped runs/, archive/, pool CSVs, LATEST_RUN.txt (kept last chart until rebuild)")


def load_labels_and_features(cfg: dict):
    panel = load_csv(cfg["_panel_path"])
    labels_raw = load_csv(cfg["_label_path"])
    labels = filter_labels(labels_raw, cfg)
    labels = attach_splits(labels, split_rows=[])
    feature_index = build_feature_index(panel)
    if cfg.get("smoke", {}).get("require_lags"):
        filtered = []
        for r in labels:
            prow = feature_index.get((r["area"], r["as_of_date"][:10]))
            if not prow:
                continue
            if prow.get("mom_change_pct") in ("", None) and prow.get("yoy_change_pct") in ("", None):
                continue
            filtered.append(r)
        labels = filtered
    labels_by_key = label_index(labels)
    train = [r for r in labels if r.get("split") == "train"]
    return labels, labels_by_key, feature_index, train


def score_hyp(hyp, labels, feature_index, labels_by_key, scorer_cfg):
    preds = generate_predictions(hyp, labels, feature_index)
    bd, residuals = score_on_split(
        preds, labels_by_key, scorer_cfg, split="validation", prev_MAE=None
    )
    if bd is None:
        bd, residuals = score_on_split(
            preds, labels_by_key, scorer_cfg, split=None, prev_MAE=None
        )
    return bd, residuals, preds


def make_param_constant(idx: int, total_return_pct: float) -> Hypothesis:
    """Param constant: change_pct = c * horizon; c = total/120 for 10y."""
    # predict_label constant: return c * horizon
    c = total_return_pct / 120.0
    return Hypothesis(
        model_id=f"param_const_cal_{idx:02d}",
        pool="param",
        kind="constant",
        params={"c": c, "target_total_return_pct": total_return_pct},
        ready=True,
        notes=f"calibrated constant total≈{total_return_pct:.1f}% over 10y",
    )


def make_nn_constant(idx: int, total_return_pct: float) -> Hypothesis:
    """NN linear head that outputs a constant (bias only, zero feature weights)."""
    names = list(FEATURE_SET)
    weights = [float(total_return_pct)] + [0.0] * len(names)
    return Hypothesis(
        model_id=f"nn_const_cal_{idx:02d}",
        pool="nn",
        kind="linear_nn",
        params={
            "l2": 0.0,
            "target_total_return_pct": total_return_pct,
            "calibrated": True,
        },
        weights=weights,
        feature_names=names,
        ready=True,
        notes=f"calibrated linear_nn constant≈{total_return_pct:.1f}% over 10y",
    )


def score_for_level(
    pool: str,
    idx: int,
    total_return_pct: float,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
):
    if pool == "param":
        hyp = make_param_constant(idx, total_return_pct)
    else:
        hyp = make_nn_constant(idx, total_return_pct)
    bd, _, _ = score_hyp(hyp, labels, feature_index, labels_by_key, scorer_cfg)
    return hyp, bd


def find_level_for_target_S(
    pool: str,
    idx: int,
    target_S: float,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    lo: float = -80.0,
    hi: float = 400.0,
    iters: int = 28,
) -> tuple[Hypothesis, object, float]:
    """
    Search total-return level so S is near target.
    S vs constant level is unimodal (best near true mean return);
    we sample a grid then refine, and also check extremes for low S.
    """
    # Coarse grid over prediction levels
    grid = []
    n_grid = 41
    for i in range(n_grid):
        level = lo + (hi - lo) * i / (n_grid - 1)
        hyp, bd = score_for_level(
            pool, idx, level, labels, feature_index, labels_by_key, scorer_cfg
        )
        if bd is None:
            continue
        grid.append((abs(bd.S - target_S), bd.S, level, hyp, bd))

    if not grid:
        raise RuntimeError(f"no scores for {pool} idx={idx}")

    # Prefer candidates with S close to target; among ties prefer lower level variance
    grid.sort(key=lambda x: (x[0], abs(x[2])))
    best_err, best_S, best_level, best_hyp, best_bd = grid[0]

    # Local refine around best_level
    span = (hi - lo) / (n_grid - 1)
    for _ in range(iters):
        candidates = []
        for delta in (-span, 0.0, span, -span / 2, span / 2):
            level = best_level + delta
            hyp, bd = score_for_level(
                pool, idx, level, labels, feature_index, labels_by_key, scorer_cfg
            )
            if bd is None:
                continue
            candidates.append((abs(bd.S - target_S), bd.S, level, hyp, bd))
        candidates.sort(key=lambda x: x[0])
        err, S, level, hyp, bd = candidates[0]
        if err < best_err - 1e-6:
            best_err, best_S, best_level, best_hyp, best_bd = err, S, level, hyp, bd
            span *= 0.6
        else:
            span *= 0.5
        if span < 0.05:
            break

    # Rebuild hyp with final id/notes
    if pool == "param":
        best_hyp = make_param_constant(idx, best_level)
    else:
        best_hyp = make_nn_constant(idx, best_level)
    best_hyp.params["target_S"] = target_S
    best_hyp.params["achieved_S"] = best_bd.S
    best_hyp.params["calibrated_level_pct"] = best_level
    best_bd, residuals, preds = score_hyp(
        best_hyp, labels, feature_index, labels_by_key, scorer_cfg
    )
    return best_hyp, best_bd, residuals, preds


def diversify_nn_with_real_mlps(
    calibrated: list[tuple[Hypothesis, object, list, list]],
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    train,
    n_real: int = 3,
) -> list[tuple[Hypothesis, object, list, list]]:
    """
    Replace a few mid/high-S constant NNs with real trained MLPs if their S
    lands near a free target slot (optional diversity). Keep exact count of 10.
    For distribution control we keep calibrated constants as primary.
    """
    return calibrated  # keep pure calibration for even spacing


def main():
    cfg = load_config(ROOT / "config.json")
    loop_root = ROOT
    paths = ensure_dirs(loop_root)

    print("=== 1) Wipe run data ===")
    wipe_run_data(loop_root)

    print("=== 2) Load labels ===")
    labels, labels_by_key, feature_index, train = load_labels_and_features(cfg)
    train_n = sum(1 for r in labels if r["split"] == "train")
    val_n = sum(1 for r in labels if r["split"] == "validation")
    print(f"labels={len(labels)} train={train_n} val={val_n}")
    print(f"target S grid: {TARGET_S}")

    scorer_cfg = cfg["scorer"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths["runs"] / f"run_{ts}"
    cycle_dir = run_dir / "cycle_01"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    print("=== 3) Calibrate 10 param + 10 nn constants to target S ===")
    param_results = []
    nn_results = []

    for i, target in enumerate(TARGET_S):
        print(f"  param[{i}] target_S={target:.3f} ...", flush=True)
        hyp, bd, res, preds = find_level_for_target_S(
            "param",
            i,
            target,
            labels,
            feature_index,
            labels_by_key,
            scorer_cfg,
        )
        attach_summaries(hyp, preds, None, prefer_area="London")
        hyp.params["mutation_summary"] = "Seed hypothesis (calibrated constant; no parents)"
        hyp.mutation_summary = hyp.params["mutation_summary"]  # type: ignore
        hyp.lineage = [
            {
                "hypothesis_id": hyp.model_id,
                "model_id": hyp.model_id,
                "S": bd.S,
                "MAE": bd.MAE,
                "hit_rate": bd.hit_rate,
                "n": bd.n,
            }
        ]
        print(
            f"    → S={bd.S:.4f} MAE={bd.MAE:.2f} hit={bd.hit_rate:.3f} "
            f"level={hyp.params.get('calibrated_level_pct'):.1f}%",
            flush=True,
        )
        param_results.append((hyp, bd, res, preds))

        print(f"  nn[{i}] target_S={target:.3f} ...", flush=True)
        hyp_n, bd_n, res_n, preds_n = find_level_for_target_S(
            "nn",
            i,
            target,
            labels,
            feature_index,
            labels_by_key,
            scorer_cfg,
        )
        attach_summaries(hyp_n, preds_n, None, prefer_area="London")
        hyp_n.params["mutation_summary"] = (
            "Seed hypothesis (calibrated linear_nn constant; no parents)"
        )
        hyp_n.mutation_summary = hyp_n.params["mutation_summary"]  # type: ignore
        hyp_n.lineage = [
            {
                "hypothesis_id": hyp_n.model_id,
                "model_id": hyp_n.model_id,
                "S": bd_n.S,
                "MAE": bd_n.MAE,
                "hit_rate": bd_n.hit_rate,
                "n": bd_n.n,
            }
        ]
        print(
            f"    → S={bd_n.S:.4f} MAE={bd_n.MAE:.2f} hit={bd_n.hit_rate:.3f} "
            f"level={hyp_n.params.get('calibrated_level_pct'):.1f}%",
            flush=True,
        )
        nn_results.append((hyp_n, bd_n, res_n, preds_n))

    # Keep calibrated linear_nn heads for even S spacing (true trained MLPs
    # often cluster or exceed 0.5 on this 10y task and break the grid).
    print("=== 3b) Skip real-MLP swap (preserve even 0–0.5 grid for both pools) ===")

    def row_from(hyp, bd, cycle=1):
        mut = getattr(hyp, "mutation_summary", None) or hyp.params.get("mutation_summary", "")
        pred_s = getattr(hyp, "prediction_summary", None) or hyp.params.get(
            "prediction_summary", ""
        )
        return {
            "cycle": cycle,
            "pool": hyp.pool,
            "hypothesis_id": hyp.model_id,
            "kind": hyp.kind,
            "S": round(bd.S, 6),
            "MAE": round(bd.MAE, 6),
            "hit_rate": round(bd.hit_rate, 6),
            "n": bd.n,
            "delta_MAE": 0.0,
            "maturity": round(bd.maturity, 6),
            "exp_mae": round(bd.exp_mae, 6),
            "hit_factor": round(bd.hit_factor, 6),
            "improve_factor": round(bd.improve_factor, 6),
            "params_json": json.dumps(hyp.params),
            "parent_ids": "",
            "lineage_len": len(hyp.lineage),
            "notes": hyp.notes,
            "mutation_summary": mut,
            "prediction_summary": pred_s,
        }

    score_rows = [row_from(h, b) for h, b, _, _ in param_results]
    score_rows += [row_from(h, b) for h, b, _, _ in nn_results]
    # sort by S desc within write is fine
    score_rows.sort(key=lambda r: (-1 if r["pool"] == "param" else 0, -r["S"]))

    write_csv(cycle_dir / "scores.csv", score_rows)

    # residuals samples
    param_res = []
    nn_res = []
    for hyp, bd, res, preds in param_results:
        param_res.extend(res[:200])
    for hyp, bd, res, preds in nn_results:
        nn_res.extend(res[:200])
    write_csv(cycle_dir / "residuals_param_sample.csv", param_res[:5000])
    write_csv(cycle_dir / "residuals_nn_sample.csv", nn_res[:5000])
    write_csv(cycle_dir / "proposals.csv", [])  # empty — seeds only
    write_csv(cycle_dir / "archived.csv", [])

    # empty proposals file still needs headers for viz — write empty properly
    # write_csv no-ops on empty; create minimal headers
    with open(cycle_dir / "proposals.csv", "w") as f:
        f.write(
            "cycle,hypothesis_id,llm_model_id,mode,used_api,error,kind,pool,"
            "params_json,parents,rationale,mutation_summary,prediction_summary\n"
        )
    with open(cycle_dir / "archived.csv", "w") as f:
        f.write("cycle,pool,hypothesis_id,S,MAE,hit_rate,n,reason\n")

    param_hyps = [h for h, _, _, _ in param_results]
    nn_hyps = [h for h, _, _, _ in nn_results]

    write_csv(
        paths["param"] / "current_pool.csv",
        [
            {
                "hypothesis_id": h.model_id,
                "kind": h.kind,
                "params_json": json.dumps(h.params),
                "ready": h.ready,
                "notes": h.notes,
                "lineage_len": len(h.lineage),
                "parent_ids": "",
            }
            for h in param_hyps
        ],
    )
    write_csv(
        paths["nn"] / "current_pool.csv",
        [
            {
                "hypothesis_id": h.model_id,
                "kind": h.kind,
                "params_json": json.dumps(h.params),
                "ready": h.ready,
                "notes": h.notes,
                "lineage_len": len(h.lineage),
                "parent_ids": "",
            }
            for h in nn_hyps
        ],
    )

    history = [
        {
            "cycle": 1,
            "n_labels": len(labels),
            "n_train": train_n,
            "n_validation": val_n,
            "param_pool_scored": len(param_results),
            "nn_pool_scored": len(nn_results),
            "proposals": 0,
            "proposals_via_llm_api": 0,
            "best_param_id": max(param_results, key=lambda x: x[1].S)[0].model_id,
            "best_param_S": max(param_results, key=lambda x: x[1].S)[1].S,
            "best_param_MAE": max(param_results, key=lambda x: x[1].S)[1].MAE,
            "best_param_hit_rate": max(param_results, key=lambda x: x[1].S)[1].hit_rate,
            "best_nn_id": max(nn_results, key=lambda x: x[1].S)[0].model_id,
            "best_nn_S": max(nn_results, key=lambda x: x[1].S)[1].S,
            "best_nn_MAE": max(nn_results, key=lambda x: x[1].S)[1].MAE,
            "best_nn_hit_rate": max(nn_results, key=lambda x: x[1].S)[1].hit_rate,
            "mode": "calibrated_seed",
        }
    ]
    write_csv(run_dir / "cycle_history.csv", history)
    write_csv(run_dir / "all_proposals.csv", [])
    with open(run_dir / "all_proposals.csv", "w") as f:
        f.write(
            "cycle,hypothesis_id,llm_model_id,mode,used_api,error,kind,pool,"
            "params_json,parents,rationale,mutation_summary,prediction_summary\n"
        )

    summary = {
        "run_dir": str(run_dir),
        "cycles": 1,
        "labels_used": len(labels),
        "train": train_n,
        "validation": val_n,
        "history": history,
        "scorer": cfg["scorer"],
        "mode": "calibrated_seed",
        "target_S": TARGET_S,
        "param_scores": [round(b.S, 4) for _, b, _, _ in param_results],
        "nn_scores": [round(b.S, 4) for _, b, _, _ in nn_results],
    }
    (run_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    (cycle_dir / "summary.json").write_text(json.dumps(history[0], indent=2, default=str))

    report = [
        "# Calibrated seed run",
        "",
        f"- Labels: **{len(labels)}** (train={train_n}, val={val_n})",
        f"- Target S grid: `{TARGET_S}`",
        f"- Param S: `{summary['param_scores']}`",
        f"- NN S: `{summary['nn_scores']}`",
        "",
        "Seeds are constant-level predictors (param `constant`, nn `linear_nn` bias-only)",
        "optionally mixed with real trained MLPs when S lands near a target slot.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(report))
    (loop_root / "LATEST_RUN.txt").write_text(str(run_dir) + "\n")

    print("=== 4) Score summary ===")
    print("PARAM:")
    for hyp, bd, _, _ in sorted(param_results, key=lambda x: x[1].S):
        print(f"  {hyp.model_id:28s} S={bd.S:.4f} MAE={bd.MAE:7.2f}")
    print("NN:")
    for hyp, bd, _, _ in sorted(nn_results, key=lambda x: x[1].S):
        print(f"  {hyp.model_id:28s} S={bd.S:.4f} MAE={bd.MAE:7.2f} kind={hyp.kind}")

    # Distribution check
    def spread_report(name, scores):
        scores = sorted(scores)
        gaps = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        print(
            f"{name}: min={scores[0]:.3f} max={scores[-1]:.3f} "
            f"mean_gap={sum(gaps)/len(gaps):.3f} gaps={[round(g,3) for g in gaps]}"
        )

    spread_report("param", [b.S for _, b, _, _ in param_results])
    spread_report("nn", [b.S for _, b, _, _ in nn_results])

    print("=== 5) Rebuild lineage_graph.json ===")
    from viz.build_lineage_data import main as build_viz

    build_viz()
    print(f"run_dir: {run_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
