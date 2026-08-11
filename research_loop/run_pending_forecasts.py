#!/usr/bin/env python3
"""
Train + run any hypotheses that only appear as proposals (not yet scored),
write proposal_forecasts.csv per cycle, and rebuild lineage_graph.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_access import (  # noqa: E402
    attach_splits,
    build_feature_index,
    filter_labels,
    load_config,
    load_csv,
)
from hypotheses import Hypothesis, generate_predictions, prepare_model  # noqa: E402
from scorer import label_index, score_on_split  # noqa: E402
from summarize import attach_summaries, summarize_mutation  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def hyp_from_proposal(r: dict) -> Hypothesis:
    try:
        params = json.loads(r.get("params_json") or "{}")
    except Exception:
        params = {}
    # strip summary fields from params if present
    params = {k: v for k, v in params.items() if not str(k).endswith("_summary")}
    kind = r.get("kind") or "zero"
    pool = r.get("pool") or ("nn" if kind in ("mlp", "linear_nn") else "param")
    parents = [p for p in (r.get("parents") or "").split("|") if p.strip()]
    ready = kind not in ("ridge_linear", "linear_nn", "mlp")
    return Hypothesis(
        model_id=r.get("hypothesis_id") or r.get("model_id") or "unknown",
        pool=pool,
        kind=kind,
        params=params,
        ready=ready,
        parent_ids=parents,
        notes=r.get("rationale") or r.get("notes") or "",
    )


def run_for_run(run_dir: Path, cfg: dict, labels: list, feature_index: dict, labels_by_key: dict) -> int:
    train = [r for r in labels if r.get("split") == "train"]
    n_done = 0
    for cyc in sorted(run_dir.glob("cycle_*")):
        prop_path = cyc / "proposals.csv"
        if not prop_path.exists():
            continue
        props = list(csv.DictReader(open(prop_path)))
        # already scored ids this cycle
        scored_ids = set()
        sc = cyc / "scores.csv"
        if sc.exists():
            for r in csv.DictReader(open(sc)):
                scored_ids.add(r.get("hypothesis_id") or r.get("model_id"))

        out_rows = []
        for r in props:
            hid = r.get("hypothesis_id") or r.get("model_id")
            if not hid:
                continue
            # Always forecast proposals; even if id appears later scored under another cycle
            hyp = hyp_from_proposal(r)
            try:
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
                attach_summaries(hyp, val_preds or preds, None, prefer_area="London")
                # mutation from parent ids only (parents not loaded as objects)
                mut = summarize_mutation(hyp, None)
                if r.get("rationale") and (not hyp.parent_ids or "Seed" in mut):
                    # keep LLM rationale as mutation hint when richer
                    mut = r.get("rationale", mut)[:300]
                hyp.mutation_summary = mut  # type: ignore[attr-defined]
                hyp.params["mutation_summary"] = mut

                bd, _ = score_on_split(
                    preds, labels_by_key, cfg["scorer"], split="validation", prev_MAE=None
                )
                if bd is None:
                    bd, _ = score_on_split(
                        preds, labels_by_key, cfg["scorer"], split=None, prev_MAE=None
                    )
                S = bd.S if bd else None
                MAE = bd.MAE if bd else None
                hit = bd.hit_rate if bd else None
                n = bd.n if bd else 0
                if S is not None and (math.isnan(S) or math.isinf(S)):
                    S = None
                out_rows.append(
                    {
                        "cycle": r.get("cycle") or cyc.name.split("_")[1],
                        "hypothesis_id": hid,
                        "pool": hyp.pool,
                        "kind": hyp.kind,
                        "S": "" if S is None else round(S, 6),
                        "MAE": "" if MAE is None else round(MAE, 6),
                        "hit_rate": "" if hit is None else round(hit, 6),
                        "n": n,
                        "params_json": json.dumps(hyp.params),
                        "parents": r.get("parents") or "|".join(hyp.parent_ids),
                        "mutation_summary": getattr(hyp, "mutation_summary", mut),
                        "prediction_summary": getattr(hyp, "prediction_summary", ""),
                        "llm_model_id": r.get("llm_model_id", ""),
                        "mode": r.get("mode", ""),
                        "rationale": r.get("rationale", ""),
                        "was_already_in_scores": hid in scored_ids,
                    }
                )
                n_done += 1
                print(f"  ran {hid}: S={S} pred={(getattr(hyp,'prediction_summary','') or '')[:80]}")
            except Exception as e:
                print(f"  FAIL {hid}: {e}")
                out_rows.append(
                    {
                        "cycle": r.get("cycle") or "",
                        "hypothesis_id": hid,
                        "pool": r.get("pool", ""),
                        "kind": r.get("kind", ""),
                        "S": "",
                        "MAE": "",
                        "hit_rate": "",
                        "n": 0,
                        "params_json": r.get("params_json", ""),
                        "parents": r.get("parents", ""),
                        "mutation_summary": r.get("rationale", "")[:300],
                        "prediction_summary": f"Run failed: {e}",
                        "llm_model_id": r.get("llm_model_id", ""),
                        "mode": r.get("mode", ""),
                        "rationale": r.get("rationale", ""),
                        "was_already_in_scores": hid in scored_ids,
                    }
                )
        _write_csv(cyc / "proposal_forecasts.csv", out_rows)
    return n_done


def main() -> None:
    cfg = load_config(ROOT / "config.json")
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

    runs = sorted((ROOT / "runs").glob("run_*"))
    if not runs:
        print("no runs")
        return
    total = 0
    for run in runs:
        print("run", run.name)
        total += run_for_run(run, cfg, labels, feature_index, labels_by_key)
    print(f"forecasted {total} proposal hypotheses")

    # rebuild lineage
    from viz.build_lineage_data import main as build_main  # type: ignore

    # build_lineage_data is a script; call via path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_lineage_data", ROOT / "viz" / "build_lineage_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
