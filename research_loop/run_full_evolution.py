#!/usr/bin/env python3
"""
Reset to initial calibrated hypotheses, then run the full evolver for N cycles
with hybrid live animation:

  score generation → top-n →
    PARALLEL: propose + train + score many offspring
    SEQUENTIAL: as each finishes, yellow parents → place child → clear

No batch "spawn everyone then flash highlights" after all compute is done.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from evolver import ensure_dirs, score_pool, write_csv
from hypotheses import Hypothesis, generate_predictions, prepare_model
from live_events import clear as clear_live
from live_events import write_event
from llm_models import default_registry, select_available_models
from proposer import (
    _coerce_proposal_to_pool,
    proposal_to_hypothesis,
    propose_hypothesis,
)
from run_calibrated_seeds import TARGET_S, find_level_for_target_S
from scorer import label_index, score_on_split
from summarize import attach_summaries, summarize_mutation

CONSIDER_HOLD_S = 1.2  # yellow parents visible before child lands
PLACED_HOLD_S = 0.6    # brief pause after child appears


def wipe_all(loop_root: Path) -> None:
    """
    Clear run artifacts but keep lineage_graph.json until new seeds are scored
    so the browser never polls an empty chart mid-reset.
    """
    paths = ensure_dirs(loop_root)
    for child in paths["runs"].iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    for child in paths["archive"].iterdir():
        if child.is_file():
            child.unlink()
    for pool in ("param", "nn"):
        p = paths[pool] / "current_pool.csv"
        p.unlink(missing_ok=True)
    (loop_root / "LATEST_RUN.txt").unlink(missing_ok=True)
    clear_live("reset")
    # Keep viz/lineage_graph.json; rebuild only after cycle scores exist.
    print("Wiped runs/, archive/, pools, LATEST_RUN (kept last chart until rebuild)")


def load_labels(cfg):
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
            if prow.get("mom_change_pct") in ("", None) and prow.get("yoy_change_pct") in (
                "",
                None,
            ):
                continue
            filtered.append(r)
        labels = filtered
    return labels, label_index(labels), feature_index


def build_initial_hypotheses(labels, feature_index, labels_by_key, scorer_cfg):
    param_hyps: list[Hypothesis] = []
    nn_hyps: list[Hypothesis] = []
    print(f"Building initial calibrated seeds for S grid {TARGET_S}")
    for i, target in enumerate(TARGET_S):
        hyp, bd, _, preds = find_level_for_target_S(
            "param", i, target, labels, feature_index, labels_by_key, scorer_cfg
        )
        attach_summaries(hyp, preds, None, prefer_area="London")
        hyp.params["mutation_summary"] = "Seed hypothesis (calibrated constant; no parents)"
        hyp.mutation_summary = hyp.params["mutation_summary"]  # type: ignore
        hyp.lineage = [
            {
                "hypothesis_id": hyp.model_id,
                "S": bd.S,
                "MAE": bd.MAE,
                "hit_rate": bd.hit_rate,
                "n": bd.n,
            }
        ]
        print(f"  param[{i}] target={target:.3f} → S={bd.S:.4f}")
        param_hyps.append(hyp)

        hyp_n, bd_n, _, preds_n = find_level_for_target_S(
            "nn", i, target, labels, feature_index, labels_by_key, scorer_cfg
        )
        attach_summaries(hyp_n, preds_n, None, prefer_area="London")
        hyp_n.params["mutation_summary"] = (
            "Seed hypothesis (calibrated linear_nn constant; no parents)"
        )
        hyp_n.mutation_summary = hyp_n.params["mutation_summary"]  # type: ignore
        hyp_n.lineage = [
            {
                "hypothesis_id": hyp_n.model_id,
                "S": bd_n.S,
                "MAE": bd_n.MAE,
                "hit_rate": bd_n.hit_rate,
                "n": bd_n.n,
            }
        ]
        print(f"  nn[{i}]    target={target:.3f} → S={bd_n.S:.4f}")
        nn_hyps.append(hyp_n)
    return param_hyps, nn_hyps


def score_pools_parallel(
    param_pool,
    nn_pool,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    prev_mae,
    known_hyps,
):
    def _score(name, pool):
        return name, score_pool(
            name,
            pool,
            labels,
            feature_index,
            labels_by_key,
            scorer_cfg,
            prev_mae,
            known_hyps=known_hyps,
        )

    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(_score, "param", param_pool),
            ex.submit(_score, "nn", nn_pool),
        ]
        for fut in as_completed(futs):
            name, (scored, res) = fut.result()
            results[name] = (scored, res)
    return results["param"], results["nn"]


def score_row(hyp: Hypothesis, bd, cycle: int) -> dict:
    mut = getattr(hyp, "mutation_summary", None) or (hyp.params or {}).get(
        "mutation_summary", ""
    )
    pred_s = getattr(hyp, "prediction_summary", None) or (hyp.params or {}).get(
        "prediction_summary", ""
    )
    S = bd.S if bd and bd.S == bd.S else None
    MAE = bd.MAE if bd and bd.MAE == bd.MAE else None
    hit = bd.hit_rate if bd and bd.hit_rate == bd.hit_rate else None
    return {
        "cycle": cycle,
        "pool": hyp.pool,
        "hypothesis_id": hyp.model_id,
        "kind": hyp.kind,
        "S": "" if S is None else round(S, 6),
        "MAE": "" if MAE is None else round(MAE, 6),
        "hit_rate": "" if hit is None else round(hit, 6),
        "n": bd.n if bd else 0,
        "delta_MAE": round(bd.delta_MAE, 6) if bd else 0.0,
        "maturity": round(bd.maturity, 6) if bd else "",
        "exp_mae": round(bd.exp_mae, 6) if bd and bd.exp_mae == bd.exp_mae else "",
        "hit_factor": round(bd.hit_factor, 6) if bd else "",
        "improve_factor": round(bd.improve_factor, 6) if bd else "",
        "params_json": json.dumps(hyp.params),
        "parent_ids": "|".join(hyp.parent_ids),
        "lineage_len": len(hyp.lineage),
        "notes": hyp.notes,
        "mutation_summary": mut,
        "prediction_summary": pred_s,
    }


def rows_from_scored(scored, cycle: int):
    return [score_row(h, s, cycle) for h, s in scored]


def append_csv_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_pool_csv(path: Path, hyps: list[Hypothesis]) -> None:
    write_csv(
        path,
        [
            {
                "hypothesis_id": h.model_id,
                "kind": h.kind,
                "params_json": json.dumps(h.params),
                "ready": h.ready,
                "notes": h.notes,
                "lineage_len": len(h.lineage),
                "parent_ids": "|".join(h.parent_ids),
            }
            for h in hyps
        ],
    )


def rebuild_lineage() -> None:
    from viz.build_lineage_data import main as build_viz

    build_viz()


def score_one(
    hyp: Hypothesis,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    train,
    known: dict[str, Hypothesis],
    prev_mae: dict[str, float] | None = None,
):
    pool_name = hyp.pool
    if pool_name == "param" and hyp.kind in ("mlp", "linear_nn"):
        hyp.kind = "ridge_linear"
        hyp.pool = "param"
        hyp.ready = False
        hyp.mlp_state = None
        hyp.weights = None
    if pool_name == "nn" and hyp.kind not in ("mlp", "linear_nn", "ridge_linear"):
        hyp.kind = "linear_nn"
        hyp.pool = "nn"
        hyp.ready = False

    hyp = prepare_model(hyp, train, feature_index)
    if hyp.kind in ("mlp", "linear_nn"):
        hyp.pool = "nn"
    elif hyp.kind == "ridge_linear":
        hyp.pool = "param"

    preds = generate_predictions(hyp, labels, feature_index)
    val_preds = []
    for p in preds:
        key = (
            p.target,
            p.aggregation,
            p.area,
            p.as_of_date,
            p.start_date,
            p.end_date,
        )
        lab = labels_by_key.get(key)
        if lab and lab.get("split") == "validation":
            val_preds.append(p)
    parents = [known[pid] for pid in hyp.parent_ids if pid in known]
    attach_summaries(hyp, val_preds or preds, parents or None, prefer_area="London")
    mut = summarize_mutation(hyp, parents or None)
    if not hyp.params.get("mutation_summary"):
        hyp.params["mutation_summary"] = mut
        hyp.mutation_summary = mut  # type: ignore

    prev = None
    if prev_mae is not None:
        prev = prev_mae.get(hyp.model_id)
    if prev is None and hyp.lineage:
        prev = hyp.lineage[-1].get("MAE")

    bd, res = score_on_split(
        preds, labels_by_key, scorer_cfg, split="validation", prev_MAE=prev
    )
    if bd is None:
        bd, res = score_on_split(
            preds, labels_by_key, scorer_cfg, split=None, prev_MAE=prev
        )
    if bd is not None:
        bd.pool = hyp.pool
        bd.model_id = hyp.model_id
        hyp.lineage = list(hyp.lineage) + [
            {
                "hypothesis_id": hyp.model_id,
                "S": bd.S,
                "MAE": bd.MAE,
                "hit_rate": bd.hit_rate,
                "n": bd.n,
            }
        ]
    return hyp, bd, res


def pick_parents_and_mode(
    survivors: list[tuple[Hypothesis, object]],
    other: list[tuple[Hypothesis, object]],
    rng: random.Random,
    i: int,
):
    modes = ["lineage", "in_pool", "cross_pool"]
    mode = modes[i % len(modes)]
    hyp_list = [h for h, _ in survivors]
    other_hyps = [h for h, _ in other]
    primary = secondary = None

    if mode == "lineage":
        if not hyp_list:
            return None, None, mode
        primary = rng.choice(hyp_list)
    elif mode == "in_pool":
        if len(hyp_list) < 2:
            if not hyp_list:
                return None, None, mode
            primary = hyp_list[0]
            mode = "lineage"
        else:
            primary, secondary = rng.sample(hyp_list, 2)
    else:
        if not hyp_list or not other_hyps:
            if not hyp_list:
                return None, None, mode
            primary = rng.choice(hyp_list)
            mode = "lineage"
        else:
            primary = rng.choice(hyp_list)
            secondary = rng.choice(other_hyps)
    return primary, secondary, mode


def _n_proposals_for_pool(n_proposals: int, max_size: int, n_survivors: int) -> int:
    slots = max(0, max_size - n_survivors)
    n = min(n_proposals, slots)
    n = max(n, min(slots, max_size - n_survivors))
    return min(n, slots)


def _compute_one_offspring(
    *,
    job_id: int,
    target_pool: str,
    primary: Hypothesis,
    secondary: Hypothesis | None,
    mode: str,
    cycle: int,
    serial: int,
    llm,
    score_map: dict,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    train,
    seed: int,
) -> dict:
    """
    Heavy work only (no viz). Thread-safe if each call uses its own rng and
    does not mutate shared dicts until the main thread merges results.
    """
    rng = random.Random(seed)
    parents = [p for p in (primary, secondary) if p is not None]
    parent_ids = [p.model_id for p in parents]
    # Local known for attach_summaries only
    known_local = {p.model_id: p for p in parents}

    prop, meta = propose_hypothesis(
        llm, mode, target_pool, primary, secondary, rng, scores=score_map
    )
    prop = _coerce_proposal_to_pool(prop, target_pool)
    hyp = proposal_to_hypothesis(prop, meta, cycle, serial, parents)
    hyp.pool = target_pool

    hyp, bd, _ = score_one(
        hyp,
        labels,
        feature_index,
        labels_by_key,
        scorer_cfg,
        train,
        known_local,
        prev_mae=None,
    )
    if bd is None:
        return {
            "ok": False,
            "job_id": job_id,
            "target_pool": target_pool,
            "parent_ids": parent_ids,
            "mode": mode,
            "error": "score failed",
        }

    mut = hyp.params.get("mutation_summary") or prop.get("rationale") or ""
    hyp.params["mutation_summary"] = mut
    hyp.mutation_summary = mut  # type: ignore

    log = {
        "cycle": cycle,
        "hypothesis_id": hyp.model_id,
        "llm_model_id": meta["llm_model_id"],
        "mode": meta["mode"],
        "used_api": meta["used_api"],
        "error": meta.get("error", ""),
        "kind": hyp.kind,
        "pool": hyp.pool,
        "params_json": json.dumps(hyp.params),
        "parents": "|".join(hyp.parent_ids),
        "rationale": prop.get("rationale", ""),
        "mutation_summary": mut,
        "prediction_summary": getattr(hyp, "prediction_summary", ""),
    }
    return {
        "ok": True,
        "job_id": job_id,
        "target_pool": target_pool,
        "parent_ids": parent_ids,
        "mode": meta["mode"],
        "hyp": hyp,
        "bd": bd,
        "log": log,
        "llm_model_id": meta["llm_model_id"],
    }


def _animate_one_result(
    result: dict,
    *,
    cycle: int,
    next_cycle: int,
    scores_next_path: Path,
    proposals_path: Path,
    known: dict[str, Hypothesis],
    prev_mae: dict[str, float],
    next_by_pool: dict[str, list[Hypothesis]],
    max_by_pool: dict[str, int],
) -> None:
    """Sequential yellow playback for one finished offspring (main thread only)."""
    if not result.get("ok"):
        print(f"    skip job {result.get('job_id')}: {result.get('error')}", flush=True)
        return

    hyp = result["hyp"]
    bd = result["bd"]
    parent_ids = result["parent_ids"]
    pool = result["target_pool"]

    # Cap pool size
    if len(next_by_pool[pool]) >= max_by_pool[pool]:
        print(f"    skip {hyp.model_id}: pool {pool} full", flush=True)
        return

    # 1) Yellow parents on source run
    write_event(
        "considering",
        parent_ids,
        mode=result["mode"],
        pool=pool,
        child_id=None,
        parent_cycle=cycle,
        child_cycle=next_cycle,
        message=f"Considering {parent_ids} ({result['mode']}) → {hyp.model_id}",
    )
    time.sleep(CONSIDER_HOLD_S)

    # 2) Place child on next run
    known[hyp.model_id] = hyp
    prev_mae[hyp.model_id] = bd.MAE
    next_by_pool[pool].append(hyp)
    append_csv_row(proposals_path, result["log"])
    append_csv_row(scores_next_path, score_row(hyp, bd, next_cycle))

    write_event(
        "placed",
        parent_ids,
        mode=result["mode"],
        pool=pool,
        child_id=hyp.model_id,
        parent_cycle=cycle,
        child_cycle=next_cycle,
        message=f"Placed {hyp.model_id}",
    )
    rebuild_lineage()
    time.sleep(PLACED_HOLD_S)
    clear_live(f"placed {hyp.model_id}")

    print(
        f"    + {hyp.model_id} S={bd.S:.4f} mode={result['mode']} "
        f"parents={parent_ids} llm={result.get('llm_model_id')}",
        flush=True,
    )


def reproduce_hybrid(
    *,
    survivors_p: list[tuple[Hypothesis, object]],
    survivors_n: list[tuple[Hypothesis, object]],
    pcfg: dict,
    ncfg: dict,
    cycle: int,
    next_cycle: int,
    scores_next_path: Path,
    proposals_path: Path,
    labels,
    feature_index,
    labels_by_key,
    scorer_cfg,
    train,
    known: dict[str, Hypothesis],
    prev_mae: dict[str, float],
    llm_registry,
    prefer_llms,
    rng: random.Random,
    max_workers: int = 4,
) -> tuple[list[Hypothesis], list[Hypothesis], list[dict]]:
    """
    Hybrid reproduction:
      - PARALLEL: propose + train + score many offspring at once
      - SEQUENTIAL: yellow parent highlight → place child on chart → clear
        (playback as each job finishes, so wall-clock ≈ slowest job + short anims)
    """
    next_param: list[Hypothesis] = [h for h, _ in survivors_p]
    next_nn: list[Hypothesis] = [h for h, _ in survivors_n]
    for h in next_param:
        h.pool = "param"
    for h in next_nn:
        h.pool = "nn"

    available = select_available_models(llm_registry, prefer=prefer_llms)
    if not available:
        available = [m for m in llm_registry if m.model_id == "offline_heuristic"]

    score_map = {
        h.model_id: {"S": s.S, "MAE": s.MAE, "hit_rate": s.hit_rate, "n": s.n}
        for h, s in survivors_p + survivors_n
    }

    n_prop_p = _n_proposals_for_pool(
        int(pcfg.get("n_proposals", 8)),
        int(pcfg.get("max_pool_size", 12)),
        len(survivors_p),
    )
    n_prop_n = _n_proposals_for_pool(
        int(ncfg.get("n_proposals", 5)),
        int(ncfg.get("max_pool_size", 8)),
        len(survivors_n),
    )

    jobs: list[dict] = []
    serial_p, serial_n = 0, 100

    for i in range(n_prop_p):
        primary, secondary, mode = pick_parents_and_mode(
            survivors_p, survivors_n, rng, i
        )
        if primary is None:
            continue
        serial_p += 1
        jobs.append(
            {
                "job_id": len(jobs),
                "target_pool": "param",
                "primary": primary,
                "secondary": secondary,
                "mode": mode,
                "serial": serial_p,
                "llm": available[i % len(available)],
                "seed": rng.randint(1, 10_000_000),
            }
        )

    for i in range(n_prop_n):
        primary, secondary, mode = pick_parents_and_mode(
            survivors_n, survivors_p, rng, i
        )
        if primary is None:
            continue
        serial_n += 1
        jobs.append(
            {
                "job_id": len(jobs),
                "target_pool": "nn",
                "primary": primary,
                "secondary": secondary,
                "mode": mode,
                "serial": serial_n,
                "llm": available[(i + n_prop_p) % len(available)],
                "seed": rng.randint(1, 10_000_000),
            }
        )

    print(
        f"  hybrid: {len(jobs)} offspring in parallel "
        f"(workers={min(max_workers, max(1, len(jobs)))}), "
        f"then sequential yellow playback",
        flush=True,
    )
    write_event(
        "idle",
        [],
        message=f"Computing {len(jobs)} proposals in parallel…",
    )

    workers = min(max_workers, max(1, len(jobs)))
    logs: list[dict] = []
    next_by_pool = {"param": next_param, "nn": next_nn}
    max_by_pool = {
        "param": int(pcfg.get("max_pool_size", 12)),
        "nn": int(ncfg.get("max_pool_size", 8)),
    }

    if not jobs:
        return next_param, next_nn, logs

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _compute_one_offspring,
                job_id=j["job_id"],
                target_pool=j["target_pool"],
                primary=j["primary"],
                secondary=j["secondary"],
                mode=j["mode"],
                cycle=cycle,
                serial=j["serial"],
                llm=j["llm"],
                score_map=score_map,
                labels=labels,
                feature_index=feature_index,
                labels_by_key=labels_by_key,
                scorer_cfg=scorer_cfg,
                train=train,
                seed=j["seed"],
            )
            for j in jobs
        ]
        # Playback as each compute job finishes (hybrid)
        for fut in as_completed(futs):
            try:
                result = fut.result()
            except Exception as e:
                print(f"    worker failed: {e}", flush=True)
                continue
            if result.get("ok") and result.get("log"):
                logs.append(result["log"])
            _animate_one_result(
                result,
                cycle=cycle,
                next_cycle=next_cycle,
                scores_next_path=scores_next_path,
                proposals_path=proposals_path,
                known=known,
                prev_mae=prev_mae,
                next_by_pool=next_by_pool,
                max_by_pool=max_by_pool,
            )

    next_param = next_by_pool["param"][: max_by_pool["param"]]
    next_nn = next_by_pool["nn"][: max_by_pool["nn"]]
    return next_param, next_nn, logs


def main():
    n_cycles = 10
    if len(sys.argv) > 1:
        n_cycles = int(sys.argv[1])

    cfg = load_config(ROOT / "config.json")
    paths = ensure_dirs(ROOT)
    rng = random.Random(42)

    print("=== RESET: wipe + initial calibrated hypotheses ===")
    wipe_all(ROOT)
    labels, labels_by_key, feature_index = load_labels(cfg)
    train = [r for r in labels if r.get("split") == "train"]
    train_n = sum(1 for r in labels if r["split"] == "train")
    val_n = sum(1 for r in labels if r["split"] == "validation")
    print(f"labels={len(labels)} train={train_n} val={val_n}")

    param_pool, nn_pool = build_initial_hypotheses(
        labels, feature_index, labels_by_key, cfg["scorer"]
    )
    llm_registry = default_registry(cfg)
    prefer_llms = cfg.get("prefer_llms") or [
        "claude",
        "grok",
        "deepseek",
        "offline_heuristic",
    ]
    prev_mae: dict[str, float] = {}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths["runs"] / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "LATEST_RUN.txt").write_text(str(run_dir) + "\n")

    write_csv(
        run_dir / "llm_model_registry.csv",
        [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "provider": m.provider,
                "available": m.is_available(),
                "notes": m.notes,
            }
            for m in llm_registry
        ],
    )

    history = []
    all_proposal_logs: list[dict] = []
    # Pre-create empty next-cycle score files only as needed

    print(
        f"\n=== FULL EVOLVER (hybrid parallel compute + sequential yellow): "
        f"{n_cycles} cycles ===",
        flush=True,
    )
    for cycle in range(1, n_cycles + 1):
        t0 = time.time()
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(exist_ok=True)
        clear_live(f"scoring cycle {cycle}")
        print(f"\n--- cycle {cycle}/{n_cycles}: score pools ---", flush=True)

        known_hyps = {h.model_id: h for h in param_pool + nn_pool}
        (param_scored, param_res), (nn_scored, nn_res) = score_pools_parallel(
            param_pool,
            nn_pool,
            labels,
            feature_index,
            labels_by_key,
            cfg["scorer"],
            prev_mae,
            known_hyps,
        )
        for h, s in param_scored + nn_scored:
            prev_mae[h.model_id] = s.MAE
            known_hyps[h.model_id] = h

        # Overwrite this cycle's scores with the true scored generation
        score_rows = rows_from_scored(param_scored, cycle) + rows_from_scored(
            nn_scored, cycle
        )
        write_csv(cycle_dir / "scores.csv", score_rows)
        write_csv(cycle_dir / "residuals_param_sample.csv", param_res[:3000])
        write_csv(cycle_dir / "residuals_nn_sample.csv", nn_res[:3000])
        rebuild_lineage()

        best_p = param_scored[0] if param_scored else None
        best_n = nn_scored[0] if nn_scored else None
        print(
            f"  scored param={len(param_scored)} nn={len(nn_scored)} | "
            f"best_param={best_p[0].model_id if best_p else '-'} "
            f"S={best_p[1].S if best_p else float('nan'):.4f} | "
            f"best_nn={best_n[0].model_id if best_n else '-'} "
            f"S={best_n[1].S if best_n else float('nan'):.4f}",
            flush=True,
        )

        # Selection
        pcfg = cfg["pools"]["param"]
        ncfg = cfg["pools"]["nn"]
        top_p = int(pcfg.get("top_n", 4))
        top_n = int(ncfg.get("top_n", 3))
        survivors_p = param_scored[:top_p]
        survivors_n = nn_scored[:top_n]
        culled = param_scored[top_p:] + nn_scored[top_n:]

        archive_rows = []
        for h, s in culled:
            archive_rows.append(
                {
                    "cycle": cycle,
                    "pool": h.pool,
                    "hypothesis_id": h.model_id,
                    "S": s.S,
                    "MAE": s.MAE,
                    "hit_rate": s.hit_rate,
                    "n": s.n,
                    "reason": "below_top_n",
                }
            )
        write_csv(cycle_dir / "archived.csv", archive_rows)
        if archive_rows:
            write_csv(paths["archive"] / f"cycle_{cycle:02d}_archived.csv", archive_rows)

        # Prepare next-cycle scores file for live-placed children
        next_cycle = cycle + 1
        next_dir = run_dir / f"cycle_{next_cycle:02d}"
        next_dir.mkdir(exist_ok=True)
        scores_next = next_dir / "scores.csv"
        # Reset next-cycle scores so only this generation's kids accumulate there
        # until next cycle re-scores the full pool.
        if scores_next.exists():
            scores_next.unlink()
        proposals_path = cycle_dir / "proposals.csv"
        if proposals_path.exists():
            proposals_path.unlink()

        print(
            f"  reproduce → cycle {next_cycle} "
            f"(hybrid: parallel propose/score, sequential yellow)",
            flush=True,
        )
        # Do NOT pre-place survivors on the next run during animation.
        next_param, next_nn, proposal_logs = reproduce_hybrid(
            survivors_p=survivors_p,
            survivors_n=survivors_n,
            pcfg=pcfg,
            ncfg=ncfg,
            cycle=cycle,
            next_cycle=next_cycle,
            scores_next_path=scores_next,
            proposals_path=proposals_path,
            labels=labels,
            feature_index=feature_index,
            labels_by_key=labels_by_key,
            scorer_cfg=cfg["scorer"],
            train=train,
            known=known_hyps,
            prev_mae=prev_mae,
            llm_registry=llm_registry,
            prefer_llms=prefer_llms,
            rng=rng,
            max_workers=int(cfg.get("hybrid_max_workers", 4)),
        )
        all_proposal_logs.extend(proposal_logs)
        if not proposals_path.exists():
            proposals_path.write_text(
                "cycle,hypothesis_id,llm_model_id,mode,used_api,error,kind,pool,"
                "params_json,parents,rationale,mutation_summary,prediction_summary\n"
            )

        # After offspring animation: carry survivors onto next run (same id)
        for h, s in survivors_p + survivors_n:
            append_csv_row(scores_next, score_row(h, s, next_cycle))
        rebuild_lineage()

        write_pool_csv(paths["param"] / "current_pool.csv", next_param)
        write_pool_csv(paths["nn"] / "current_pool.csv", next_nn)

        hist = {
            "cycle": cycle,
            "n_labels": len(labels),
            "n_train": train_n,
            "n_validation": val_n,
            "param_pool_scored": len(param_scored),
            "nn_pool_scored": len(nn_scored),
            "proposals": len(proposal_logs),
            "proposals_via_llm_api": sum(1 for x in proposal_logs if x.get("used_api")),
            "best_param_id": best_p[0].model_id if best_p else "",
            "best_param_S": best_p[1].S if best_p else None,
            "best_param_MAE": best_p[1].MAE if best_p else None,
            "best_param_hit_rate": best_p[1].hit_rate if best_p else None,
            "best_nn_id": best_n[0].model_id if best_n else "",
            "best_nn_S": best_n[1].S if best_n else None,
            "best_nn_MAE": best_n[1].MAE if best_n else None,
            "best_nn_hit_rate": best_n[1].hit_rate if best_n else None,
            "elapsed_s": round(time.time() - t0, 1),
            "animation": "hybrid_parallel_compute_sequential_yellow",
        }
        history.append(hist)
        (cycle_dir / "summary.json").write_text(json.dumps(hist, indent=2, default=str))
        write_csv(run_dir / "cycle_history.csv", history)

        param_pool = next_param
        nn_pool = next_nn
        clear_live(f"cycle {cycle} complete")
        rebuild_lineage()
        print(f"  cycle {cycle} done in {hist['elapsed_s']}s", flush=True)

        # Last cycle: still scored current gen; kids sit on cycle+1 already.
        # If this was the last cycle, no need to re-score next_pool.

    write_csv(run_dir / "all_proposals.csv", all_proposal_logs)
    summary = {
        "run_dir": str(run_dir),
        "cycles": n_cycles,
        "labels_used": len(labels),
        "train": train_n,
        "validation": val_n,
        "history": history,
        "scorer": cfg["scorer"],
        "mode": "full_evolution_one_at_a_time",
        "elite_copies": False,
    }
    (run_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        f"# Full evolution ({n_cycles} cycles, hybrid parallel + sequential yellow)",
        "",
        "- Elite copies: **off**",
        "- Compute: propose+score many offspring in parallel",
        "- Animation: as each finishes, yellow parents → place child → clear",
        "",
        "| Cycle | Best param | S | Best nn | S | props | s |",
        "|------:|------------|--:|---------|--:|------:|--:|",
    ]
    for h in history:
        lines.append(
            f"| {h['cycle']} | {h['best_param_id']} | {h['best_param_S']:.4f} | "
            f"{h['best_nn_id']} | {h['best_nn_S']:.4f} | "
            f"{h.get('proposals_via_llm_api', 0)}/{h.get('proposals', 0)} | {h['elapsed_s']} |"
        )
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    clear_live("full evolution complete")
    rebuild_lineage()
    print(f"\nDONE. run_dir={run_dir}")
    print("Open http://127.0.0.1:8765/")


if __name__ == "__main__":
    main()
