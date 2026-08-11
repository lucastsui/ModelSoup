#!/usr/bin/env python3
"""Aggregate research_loop run CSVs into lineage_graph.json for the web viz."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent


def _clean(obj):
    """Recursively replace NaN/Inf with None so json.dumps is valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def main() -> None:
    nodes: list[dict] = []
    edges: list[dict] = []
    node_index: dict[tuple, dict] = {}
    proposal_parents: dict[str, list[str]] = {}
    proposal_meta: dict[str, dict] = {}
    proposals_by_run: list[dict] = []

    for run in sorted(RUNS.glob("run_*")):
        for cyc_dir in sorted(run.glob("cycle_*")):
            pr = cyc_dir / "proposals.csv"
            if not pr.exists():
                continue
            for r in csv.DictReader(open(pr)):
                hid = r.get("hypothesis_id") or r.get("model_id")
                if not hid:
                    continue
                parents = [p.strip() for p in (r.get("parents") or r.get("parent_ids") or "").split("|") if p.strip()]
                proposal_parents[hid] = parents
                try:
                    pcyc = int(float(r.get("cycle") or cyc_dir.name.split("_")[1]))
                except Exception:
                    pcyc = 1
                meta = {
                    "llm_model_id": r.get("llm_model_id", ""),
                    "mode": r.get("mode", ""),
                    "rationale": r.get("rationale", ""),
                    "used_api": r.get("used_api", ""),
                    "proposed_in_cycle": pcyc,
                    "run": run.name,
                    "pool": r.get("pool", ""),
                    "kind": r.get("kind", ""),
                    "params_json": r.get("params_json", ""),
                    "parents": parents,
                    "hypothesis_id": hid,
                }
                proposal_meta[hid] = meta
                proposals_by_run.append(meta)

    for run in sorted(RUNS.glob("run_*")):
        for cyc_dir in sorted(run.glob("cycle_*")):
            sc = cyc_dir / "scores.csv"
            if not sc.exists():
                continue
            for r in csv.DictReader(open(sc)):
                try:
                    cycle = int(float(r.get("cycle") or cyc_dir.name.split("_")[1]))
                except Exception:
                    continue
                hid = r.get("hypothesis_id") or r.get("model_id")
                if not hid:
                    continue
                try:
                    S = float(r["S"])
                    if math.isnan(S) or math.isinf(S):
                        continue  # unusable score point
                except Exception:
                    continue
                parents = [p.strip() for p in (r.get("parent_ids") or "").split("|") if p.strip()]
                if not parents and hid in proposal_parents:
                    parents = proposal_parents[hid]
                key = (run.name, cycle, hid)
                if key in node_index:
                    continue
                meta = proposal_meta.get(hid, {})
                mut = r.get("mutation_summary") or ""
                pred_s = r.get("prediction_summary") or ""
                if not mut or not pred_s:
                    try:
                        pj = json.loads(r.get("params_json") or meta.get("params_json") or "{}")
                        mut = mut or pj.get("mutation_summary") or ""
                        pred_s = pred_s or pj.get("prediction_summary") or ""
                    except Exception:
                        pass
                node = {
                    "id": f"{run.name}|{cycle}|{hid}",
                    "run": run.name,
                    "cycle": cycle,
                    "hypothesis_id": hid,
                    "pool": r.get("pool") or meta.get("pool", ""),
                    "kind": r.get("kind") or meta.get("kind", ""),
                    "S": S,
                    "MAE": _f(r.get("MAE")),
                    "hit_rate": _f(r.get("hit_rate")),
                    "n": _i(r.get("n")),
                    "parents": parents,
                    "notes": r.get("notes", ""),
                    "params_json": r.get("params_json") or meta.get("params_json", ""),
                    "llm_model_id": meta.get("llm_model_id", ""),
                    "mode": meta.get("mode", ""),
                    "rationale": meta.get("rationale", ""),
                    "mutation_summary": mut,
                    "prediction_summary": pred_s,
                    "scored": True,
                }
                node_index[key] = node
                nodes.append(node)

    # Proposal forecasts (trained+run even if not in scores.csv — e.g. final-cycle proposals)
    proposal_forecasts: dict[tuple[str, str], dict] = {}
    for run in sorted(RUNS.glob("run_*")):
        for cyc_dir in sorted(run.glob("cycle_*")):
            pf = cyc_dir / "proposal_forecasts.csv"
            if not pf.exists():
                continue
            for r in csv.DictReader(open(pf)):
                hid = r.get("hypothesis_id") or r.get("model_id")
                if hid:
                    proposal_forecasts[(run.name, hid)] = r

    # Unscored proposals: place at cycle+1; prefer real forecast file over "not run yet"
    run_hyp_nodes: dict[tuple, dict[int, str]] = defaultdict(dict)
    for n in nodes:
        run_hyp_nodes[(n["run"], n["hypothesis_id"])][n["cycle"]] = n["id"]

    for prop in proposals_by_run:
        hid = prop["hypothesis_id"]
        run = prop["run"]
        entry_cycle = int(prop["proposed_in_cycle"]) + 1
        key = (run, entry_cycle, hid)
        if key in node_index:
            continue  # already scored when it entered
        fc = proposal_forecasts.get((run, hid), {})
        # parent scores for provisional y (skip NaN/Inf)
        parent_Ss = []
        for p in prop["parents"]:
            cmap = run_hyp_nodes.get((run, p), {})
            if cmap:
                pc = max(c for c in cmap if c <= prop["proposed_in_cycle"])
                pn = node_index.get((run, pc, p))
                if pn and pn.get("S") is not None:
                    try:
                        sv = float(pn["S"])
                        if not math.isnan(sv) and not math.isinf(sv):
                            parent_Ss.append(sv)
                    except Exception:
                        pass
        prov = sum(parent_Ss) / len(parent_Ss) if parent_Ss else None
        if fc.get("S") not in ("", None):
            try:
                fs = float(fc["S"])
                if not math.isnan(fs) and not math.isinf(fs):
                    prov = fs
            except Exception:
                pass
        if prov is not None and (math.isnan(prov) or math.isinf(prov)):
            prov = None
        # Need a finite S to plot; skip otherwise
        if prov is None:
            continue
        parent_txt = ", ".join(prop["parents"]) if prop.get("parents") else "unknown parents"
        mut = (
            fc.get("mutation_summary")
            or prop.get("rationale")
            or f"Proposed from {parent_txt}"
        )
        if len(mut) > 180:
            mut = mut[:177] + "…"
        pred_s = fc.get("prediction_summary") or "Not run yet — proposal only (no scored forecasts)"
        ran = bool(fc.get("prediction_summary")) and not str(pred_s).startswith("Not run")
        node = {
            "id": f"{run}|{entry_cycle}|{hid}",
            "run": run,
            "cycle": entry_cycle,
            "hypothesis_id": hid,
            "pool": fc.get("pool") or prop.get("pool", ""),
            "kind": fc.get("kind") or prop.get("kind", ""),
            "S": prov,
            "provisional_S": not ran,
            "MAE": _f(fc.get("MAE")),
            "hit_rate": _f(fc.get("hit_rate")),
            "n": _i(fc.get("n")),
            "parents": prop["parents"],
            "notes": "proposal forecast run" if ran else "proposed, not yet scored in a later cycle",
            "params_json": fc.get("params_json") or prop.get("params_json", ""),
            "llm_model_id": prop.get("llm_model_id", ""),
            "mode": prop.get("mode", ""),
            "rationale": prop.get("rationale", ""),
            "mutation_summary": mut,
            "prediction_summary": pred_s,
            "scored": ran,
        }
        node_index[key] = node
        nodes.append(node)
        run_hyp_nodes[(run, hid)][entry_cycle] = node["id"]

    # Rebuild cycle index after all nodes (scored + proposals) are present
    run_hyp_nodes = defaultdict(dict)
    for n in nodes:
        run_hyp_nodes[(n["run"], n["hypothesis_id"])][n["cycle"]] = n["id"]

    # Edges from explicit parent_ids (mutation / elite copy / LLM proposal)
    for n in nodes:
        for p in n["parents"]:
            cmap = run_hyp_nodes.get((n["run"], p), {})
            if not cmap:
                continue
            prior = [c for c in cmap if c < n["cycle"]]
            if prior:
                pc = max(prior)
            else:
                le = [c for c in cmap if c <= n["cycle"] and cmap[c] != n["id"]]
                if not le:
                    continue
                pc = max(le)
            sid = cmap[pc]
            if sid == n["id"]:
                continue
            edges.append(
                {
                    "source": sid,
                    "target": n["id"],
                    "parent_hypothesis_id": p,
                    "child_hypothesis_id": n["hypothesis_id"],
                    "run": n["run"],
                    "edge_type": "parent",
                }
            )

    # Carry-forward edges: same hypothesis_id re-scored in a later cycle
    # (survivor kept in the pool — not a new spawn, but continuous lineage on the chart)
    for (run, hid), cmap in run_hyp_nodes.items():
        cycles = sorted(cmap.keys())
        for i in range(1, len(cycles)):
            c_prev, c_cur = cycles[i - 1], cycles[i]
            sid, tid = cmap[c_prev], cmap[c_cur]
            if sid == tid:
                continue
            edges.append(
                {
                    "source": sid,
                    "target": tid,
                    "parent_hypothesis_id": hid,
                    "child_hypothesis_id": hid,
                    "run": run,
                    "edge_type": "carry_forward",
                }
            )

    seen = set()
    uniq = []
    for e in edges:
        k = (e["source"], e["target"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    edges = uniq

    runs_meta = []
    for run in sorted(RUNS.glob("run_*")):
        ns = [n for n in nodes if n["run"] == run.name]
        es = [e for e in edges if e["run"] == run.name]
        cycles = sorted({n["cycle"] for n in ns})
        runs_meta.append(
            {
                "run": run.name,
                "cycles": cycles,
                "n_nodes": len(ns),
                "n_edges": len(es),
            }
        )

    graph = _clean({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs_meta,
        "nodes": nodes,
        "edges": edges,
    })
    out_path = OUT / "lineage_graph.json"

    # Never clobber a good chart with an empty graph (wipe / mid-run races).
    # Callers that intentionally clear must delete the file first.
    if not nodes and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            prev_n = len(prev.get("nodes") or [])
            if prev_n > 0:
                print(
                    f"skip empty rebuild (kept previous lineage_graph.json with {prev_n} nodes)"
                )
                return
        except Exception:
            pass

    # Atomic write so the browser never polls a half-written JSON file
    payload = json.dumps(graph, indent=2, allow_nan=False)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(out_path)
    print(f"wrote {out_path} nodes={len(nodes)} edges={len(edges)}")


def _f(x):
    try:
        if x in ("", None):
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _i(x):
    try:
        return int(float(x)) if x not in ("", None) else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
