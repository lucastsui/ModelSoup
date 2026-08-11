#!/usr/bin/env python3
"""
Continue evolution from the latest run's scored hypotheses, writing live_events.json
so the viz can highlight parent sets (yellow) while each new child is being proposed,
then clear when the child is scored and drawn.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import time
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
from evolver import ensure_dirs, write_csv
from hypotheses import FEATURE_SET, Hypothesis, generate_predictions, prepare_model
from live_events import clear as clear_live
from live_events import write_event
from llm_models import default_registry, select_available_models
from proposer import (
    _coerce_proposal_to_pool,
    proposal_to_hypothesis,
    propose_hypothesis,
)
from scorer import label_index, score_on_split
from summarize import attach_summaries, summarize_mutation

# How long to leave yellow parent links visible before/while proposing
CONSIDER_HOLD_S = 2.0
# Pause after child lands so the chart update is readable
PLACED_HOLD_S = 0.8


def rebuild_lineage() -> None:
    from viz.build_lineage_data import main as build_viz

    build_viz()


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


def hyp_from_score_row(row: dict) -> Hypothesis:
    hid = row.get("hypothesis_id") or row.get("model_id")
    pool = row.get("pool") or "param"
    kind = row.get("kind") or "constant"
    try:
        params = json.loads(row.get("params_json") or "{}")
    except Exception:
        params = {}
    parents = [p for p in (row.get("parent_ids") or "").split("|") if p.strip()]
    hyp = Hypothesis(
        model_id=hid,
        pool=pool,
        kind=kind,
        params=params,
        ready=True,
        parent_ids=parents,
        notes=row.get("notes") or "",
    )
    # Rebuild constant / bias-only linear_nn from calibrated level if present
    level = params.get("calibrated_level_pct")
    if level is not None and kind == "constant":
        hyp.params["c"] = float(level) / 120.0
        hyp.ready = True
    if kind == "linear_nn" and level is not None:
        names = list(FEATURE_SET)
        hyp.weights = [float(level)] + [0.0] * len(names)
        hyp.feature_names = names
        hyp.ready = True
    if kind in ("ridge_linear", "linear_nn", "mlp") and not hyp.weights and not hyp.mlp_state:
        # Will retrain in prepare_model
        hyp.ready = False
    # lineage seed from score
    try:
        S = float(row["S"])
        MAE = float(row.get("MAE") or 0)
        hit = float(row.get("hit_rate") or 0)
        n = int(float(row.get("n") or 0))
        hyp.lineage = [
            {
                "hypothesis_id": hid,
                "S": S,
                "MAE": MAE,
                "hit_rate": hit,
                "n": n,
            }
        ]
    except Exception:
        pass
    mut = row.get("mutation_summary") or params.get("mutation_summary")
    pred = row.get("prediction_summary") or params.get("prediction_summary")
    if mut:
        hyp.mutation_summary = mut  # type: ignore
        hyp.params["mutation_summary"] = mut
    if pred:
        hyp.prediction_summary = pred  # type: ignore
        hyp.params["prediction_summary"] = pred
    return hyp


def score_one(hyp, labels, feature_index, labels_by_key, scorer_cfg, train, known=None):
    hyp = prepare_model(hyp, train, feature_index)
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
    parents = []
    if known:
        parents = [known[pid] for pid in hyp.parent_ids if pid in known]
    attach_summaries(hyp, val_preds or preds, parents or None, prefer_area="London")
    mut = summarize_mutation(hyp, parents or None)
    if not hyp.params.get("mutation_summary"):
        hyp.params["mutation_summary"] = mut
        hyp.mutation_summary = mut  # type: ignore
    bd, res = score_on_split(
        preds, labels_by_key, scorer_cfg, split="validation", prev_MAE=None
    )
    if bd is None:
        bd, res = score_on_split(preds, labels_by_key, scorer_cfg, split=None, prev_MAE=None)
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


def score_row(hyp, bd, cycle: int) -> dict:
    mut = getattr(hyp, "mutation_summary", None) or hyp.params.get("mutation_summary", "")
    pred_s = getattr(hyp, "prediction_summary", None) or hyp.params.get(
        "prediction_summary", ""
    )
    return {
        "cycle": cycle,
        "pool": hyp.pool,
        "hypothesis_id": hyp.model_id,
        "kind": hyp.kind,
        "S": round(bd.S, 6) if bd else "",
        "MAE": round(bd.MAE, 6) if bd else "",
        "hit_rate": round(bd.hit_rate, 6) if bd else "",
        "n": bd.n if bd else 0,
        "delta_MAE": 0.0,
        "maturity": round(bd.maturity, 6) if bd else "",
        "exp_mae": round(bd.exp_mae, 6) if bd else "",
        "hit_factor": round(bd.hit_factor, 6) if bd else "",
        "improve_factor": round(bd.improve_factor, 6) if bd else "",
        "params_json": json.dumps(hyp.params),
        "parent_ids": "|".join(hyp.parent_ids),
        "lineage_len": len(hyp.lineage),
        "notes": hyp.notes,
        "mutation_summary": mut,
        "prediction_summary": pred_s,
    }


def append_score_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_proposal_log(path: Path, log: dict) -> None:
    fields = [
        "cycle",
        "hypothesis_id",
        "llm_model_id",
        "mode",
        "used_api",
        "error",
        "kind",
        "pool",
        "params_json",
        "parents",
        "rationale",
        "mutation_summary",
        "prediction_summary",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(log)


def main():
    cfg = load_config(ROOT / "config.json")
    paths = ensure_dirs(ROOT)
    labels, labels_by_key, feature_index = load_labels(cfg)
    train = [r for r in labels if r.get("split") == "train"]
    scorer_cfg = cfg["scorer"]
    rng = random.Random(7)

    latest = (ROOT / "LATEST_RUN.txt").read_text().strip() if (ROOT / "LATEST_RUN.txt").exists() else ""
    run_dir = Path(latest) if latest else None
    if not run_dir or not run_dir.exists():
        # fall back to newest run_*
        runs = sorted(paths["runs"].glob("run_*"))
        if not runs:
            print("No existing run to continue from. Seed a run first.")
            sys.exit(1)
        run_dir = runs[-1]

    c1 = run_dir / "cycle_01" / "scores.csv"
    if not c1.exists():
        print(f"No cycle_01 scores in {run_dir}")
        sys.exit(1)

    print(f"Continuing from {run_dir}")
    rows = list(csv.DictReader(open(c1)))
    hyps = [hyp_from_score_row(r) for r in rows]
    known = {h.model_id: h for h in hyps}

    # Re-score seeds so we have live ScoreBreakdowns
    print("Re-scoring current pool…")
    scored: list[tuple[Hypothesis, object]] = []
    for hyp in hyps:
        h, bd, _ = score_one(hyp, labels, feature_index, labels_by_key, scorer_cfg, train, known)
        if bd is None:
            continue
        known[h.model_id] = h
        scored.append((h, bd))
        print(f"  {h.model_id:28s} pool={h.pool:5s} S={bd.S:.4f}")

    param_scored = sorted(
        [(h, s) for h, s in scored if h.pool == "param"],
        key=lambda x: x[1].S,
        reverse=True,
    )
    nn_scored = sorted(
        [(h, s) for h, s in scored if h.pool == "nn"],
        key=lambda x: x[1].S,
        reverse=True,
    )

    cycle = 2
    cycle_dir = run_dir / f"cycle_{cycle:02d}"
    if cycle_dir.exists():
        shutil.rmtree(cycle_dir)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    scores_path = cycle_dir / "scores.csv"
    proposals_path = cycle_dir / "proposals.csv"
    # empty archive / residuals stubs
    (cycle_dir / "archived.csv").write_text(
        "cycle,pool,hypothesis_id,S,MAE,hit_rate,n,reason\n"
    )

    clear_live("starting live evolution")
    rebuild_lineage()

    param_cfg = {**cfg["pools"]["param"], "name": "param"}
    nn_cfg = {**cfg["pools"]["nn"], "name": "nn"}
    top_param = int(param_cfg.get("top_n", 4))
    top_nn = int(nn_cfg.get("top_n", 3))
    n_prop_param = int(param_cfg.get("n_proposals", 6))
    n_prop_nn = int(nn_cfg.get("n_proposals", 4))

    survivors_p = param_scored[:top_param]
    survivors_n = nn_scored[:top_nn]
    culled = param_scored[top_param:] + nn_scored[top_nn:]

    # Archive culled (cycle 1 losers)
    arch_rows = []
    for h, s in culled:
        arch_rows.append(
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
    if arch_rows:
        write_csv(cycle_dir / "archived.csv", arch_rows)

    # Place survivors on cycle 2 (carry-forward) so the chart shows run 2 column
    print(f"\n=== Cycle {cycle}: place survivors ===")
    for h, s in survivors_p + survivors_n:
        # re-score under same id for cycle 2
        h2, bd, _ = score_one(h, labels, feature_index, labels_by_key, scorer_cfg, train, known)
        if bd is None:
            continue
        known[h2.model_id] = h2
        append_score_row(scores_path, score_row(h2, bd, cycle))
        print(f"  survivor {h2.model_id} S={bd.S:.4f}")
        rebuild_lineage()
        time.sleep(0.35)

    llm_registry = default_registry(cfg)
    prefer = cfg.get("prefer_llms") or [
        "offline_heuristic",
        "claude",
        "grok",
        "deepseek",
    ]
    available = select_available_models(llm_registry, prefer=prefer)
    if not available:
        available = [m for m in llm_registry if m.model_id == "offline_heuristic"]

    def emit_consider(parent_ids: list[str], mode: str, pool: str, msg: str):
        write_event(
            "considering",
            parent_ids,
            mode=mode,
            pool=pool,
            message=msg,
        )
        # Hold so the user can see yellow links (viz polls 2/s)
        time.sleep(CONSIDER_HOLD_S)

    def propose_and_place(
        target_pool: str,
        survivors: list[tuple[Hypothesis, object]],
        other: list[tuple[Hypothesis, object]],
        n_proposals: int,
        start_serial: int,
    ) -> int:
        hyp_list = [h for h, _ in survivors]
        other_hyps = [h for h, _ in other]
        score_map = {
            h.model_id: {"S": s.S, "MAE": s.MAE, "hit_rate": s.hit_rate, "n": s.n}
            for h, s in survivors + other
        }
        modes = ["lineage", "in_pool", "cross_pool"]
        serial = start_serial

        # No elite copies — only exploratory proposals fill free slots
        for i in range(n_proposals):
            mode = modes[i % len(modes)]
            llm = available[i % len(available)]
            primary = secondary = None
            if mode == "lineage":
                if not hyp_list:
                    continue
                primary = rng.choice(hyp_list)
            elif mode == "in_pool":
                if len(hyp_list) < 2:
                    if not hyp_list:
                        continue
                    primary = hyp_list[0]
                    mode = "lineage"
                else:
                    primary, secondary = rng.sample(hyp_list, 2)
            else:
                if not hyp_list or not other_hyps:
                    if not hyp_list:
                        continue
                    primary = rng.choice(hyp_list)
                    mode = "lineage"
                else:
                    primary = rng.choice(hyp_list)
                    secondary = rng.choice(other_hyps)

            parent_ids = [p.model_id for p in (primary, secondary) if p is not None]
            emit_consider(
                parent_ids,
                mode,
                target_pool,
                f"Considering {parent_ids} ({mode}) for new {target_pool} hypothesis",
            )

            prop, meta = propose_hypothesis(
                llm, mode, target_pool, primary, secondary, rng, scores=score_map
            )
            prop = _coerce_proposal_to_pool(prop, target_pool)
            serial += 1
            parents = [p for p in (primary, secondary) if p is not None]
            hyp = proposal_to_hypothesis(prop, meta, cycle, serial, parents)
            hyp.pool = target_pool

            hyp, bd, _ = score_one(
                hyp, labels, feature_index, labels_by_key, scorer_cfg, train, known
            )
            if bd is None:
                clear_live("score failed")
                continue
            known[hyp.model_id] = hyp
            mut = hyp.params.get("mutation_summary") or prop.get("rationale") or ""
            hyp.params["mutation_summary"] = mut
            append_score_row(scores_path, score_row(hyp, bd, cycle))
            append_proposal_log(
                proposals_path,
                {
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
                },
            )
            write_event(
                "placed",
                parent_ids,
                mode=meta["mode"],
                pool=target_pool,
                child_id=hyp.model_id,
                message=f"Placed {hyp.model_id}",
            )
            rebuild_lineage()
            time.sleep(PLACED_HOLD_S)
            clear_live(f"placed {hyp.model_id}")
            print(
                f"  child {hyp.model_id} S={bd.S:.4f} mode={meta['mode']} "
                f"parents={parent_ids} llm={meta['llm_model_id']}"
            )
        return serial

    print(f"\n=== Cycle {cycle}: param pool reproduce ===")
    propose_and_place("param", survivors_p, survivors_n, n_prop_param, 0)
    print(f"\n=== Cycle {cycle}: nn pool reproduce ===")
    propose_and_place("nn", survivors_n, survivors_p, n_prop_nn, 100)

    clear_live("evolution cycle complete")
    rebuild_lineage()

    summary = {
        "run_dir": str(run_dir),
        "cycle": cycle,
        "mode": "live_evolution",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (cycle_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "LATEST_RUN.txt").write_text(str(run_dir) + "\n")
    print("\nDONE live evolution")
    print(f"scores: {scores_path}")
    print("Watch http://127.0.0.1:8765/ for yellow parent links → new dots")


if __name__ == "__main__":
    main()
