"""
Evolver: manage hypothesis pools (param vs nn).

Terminology (PropView.md):
  - LLM **model** (Opus/DeepSeek/Grok/…) proposes hypotheses — see llm_models.py + proposer.py
  - **Hypothesis** is the scorable prediction algorithm — see hypotheses.py
  - This module scores hypotheses, keeps top-n per pool, archives the rest, and asks
    LLM models to propose the next generation (lineage / in-pool / cross-pool).
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from data_access import (
    attach_splits,
    build_feature_index,
    filter_labels,
    load_config,
    load_csv,
)
from hypotheses import (
    Hypothesis,
    generate_predictions,
    prepare_model,
    seed_nn_pool,
    seed_param_pool,
)
from llm_models import default_registry
from proposer import propose_for_pool
from scorer import ScoreBreakdown, label_index, score_on_split
from summarize import attach_summaries, summarize_mutation


def ensure_dirs(loop_root: Path) -> dict[str, Path]:
    paths = {
        "param": loop_root / "pools" / "param",
        "nn": loop_root / "pools" / "nn",
        "archive": loop_root / "archive",
        "runs": loop_root / "runs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def score_pool(
    pool_name: str,
    hypotheses: list[Hypothesis],
    labels: list[dict],
    feature_index: dict,
    labels_by_key: dict,
    scorer_cfg: dict,
    prev_scores: dict[str, float],
    known_hyps: dict[str, Hypothesis] | None = None,
) -> tuple[list[tuple[Hypothesis, ScoreBreakdown]], list[dict]]:
    """Score a pool of *hypotheses* (not LLM models)."""
    train = [r for r in labels if r.get("split") == "train"]
    results: list[tuple[Hypothesis, ScoreBreakdown]] = []
    all_residuals: list[dict] = []
    known = dict(known_hyps or {})
    for h in hypotheses:
        known[h.model_id] = h

    for hyp in hypotheses:
        # Keep pools pure: MLP / linear_nn only in nn; pure rules only in param.
        if pool_name == "param" and hyp.kind in ("mlp", "linear_nn"):
            # Demote accidental nets to ridge so they stay parametric
            hyp.kind = "ridge_linear"
            hyp.pool = "param"
            hyp.ready = False
            hyp.mlp_state = None
            hyp.weights = None
            hyp.params.pop("hidden_layers", None)
            hyp.params.pop("hidden", None)
        if pool_name == "nn" and hyp.kind not in ("mlp", "linear_nn", "ridge_linear"):
            # Rule-like kinds that landed in nn → train as linear head
            if hyp.kind not in ("linear_nn", "mlp"):
                hyp.kind = "linear_nn"
                hyp.pool = "nn"
                hyp.ready = False
        hyp = prepare_model(hyp, train, feature_index)
        # After training, align reported pool with kind
        if hyp.kind in ("mlp", "linear_nn"):
            hyp.pool = "nn"
        elif hyp.kind == "ridge_linear":
            hyp.pool = "param"
        preds = generate_predictions(hyp, labels, feature_index)
        # Validation-only preds for the human summary (matches scoring split)
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
        parent_objs = [known[pid] for pid in hyp.parent_ids if pid in known]
        attach_summaries(
            hyp,
            val_preds if val_preds else preds,
            parent_objs or None,
            prefer_area="London",
        )

        prev_mae = prev_scores.get(hyp.model_id)
        # lineage carries prior MAE for ΔMAE term
        if prev_mae is None and hyp.lineage:
            prev_mae = hyp.lineage[-1].get("MAE")

        breakdown, residuals = score_on_split(
            preds,
            labels_by_key,
            scorer_cfg,
            split="validation",
            prev_MAE=prev_mae,
        )
        if breakdown is None:
            breakdown, residuals = score_on_split(
                preds, labels_by_key, scorer_cfg, split=None, prev_MAE=prev_mae
            )
        if breakdown is None:
            continue
        # Prefer hypothesis's own pool after coercion; fall back to which list we scored
        breakdown.pool = hyp.pool or pool_name
        breakdown.model_id = hyp.model_id  # hypothesis id (legacy field name)
        breakdown.params = dict(hyp.params)
        breakdown.lineage = list(hyp.lineage)
        breakdown.notes = hyp.notes
        # append hypothesis score into lineage (model-score history for proposers)
        hyp.lineage = list(hyp.lineage) + [
            {
                "hypothesis_id": hyp.model_id,
                "model_id": hyp.model_id,  # compat
                "S": breakdown.S,
                "MAE": breakdown.MAE,
                "hit_rate": breakdown.hit_rate,
                "n": breakdown.n,
            }
        ]
        results.append((hyp, breakdown))
        all_residuals.extend(residuals)

    results.sort(key=lambda x: x[1].S, reverse=True)
    return results, all_residuals


def select_and_reproduce(
    scored: list[tuple[Hypothesis, ScoreBreakdown]],
    pool_cfg: dict,
    rng: random.Random,
    cycle: int,
    other_pool_scored: list[tuple[Hypothesis, ScoreBreakdown]] | None,
    llm_registry: list,
    prefer_llms: list[str] | None = None,
) -> tuple[list[Hypothesis], list[tuple[Hypothesis, ScoreBreakdown]], list[dict]]:
    """
    Keep top_n hypotheses; archive rest; LLM models propose replacements
    (lineage / in-pool / cross-pool).
    """
    top_n = int(pool_cfg.get("top_n", 3))
    max_size = int(pool_cfg.get("max_pool_size", 10))
    n_proposals = int(pool_cfg.get("n_proposals", max(3, max_size - top_n)))
    survivors = scored[:top_n]
    culled = scored[top_n:]

    next_hyps = [h for h, _ in survivors]
    target_pool = pool_cfg.get("name") or (
        survivors[0][0].pool if survivors else "param"
    )
    # Survivors must stay in this pool (no elite clones — explorers fill free slots)
    for h in next_hyps:
        h.pool = target_pool
        if target_pool == "param" and h.kind in ("mlp", "linear_nn"):
            h.kind = "ridge_linear"
            h.ready = False
            h.mlp_state = None
            h.weights = None
        if target_pool == "nn" and h.kind not in ("mlp", "linear_nn"):
            # keep as-is until next prepare; mark pool correctly
            h.pool = "nn"

    # Fill remaining slots with exploratory LLM / offline proposals only.
    # Exact elite copies were removed: they refilled top-n with identical S and
    # blocked diversity (see plateau on calibrated constants).
    slots_left = max(0, max_size - len(next_hyps))
    n_proposals = min(n_proposals, slots_left)
    # Prefer filling the pool when config n_proposals is smaller than free slots
    n_proposals = max(n_proposals, min(slots_left, max_size - top_n))

    proposals, proposal_logs = propose_for_pool(
        target_pool=target_pool,
        survivors=survivors,
        other_pool_scored=other_pool_scored or [],
        llm_registry=llm_registry,
        rng=rng,
        cycle=cycle,
        n_proposals=n_proposals,
        prefer_llms=prefer_llms,
    )
    # Order: survivors, then explorers (no elite copies)
    for h in proposals:
        if h.pool != target_pool:
            h.pool = target_pool
        if target_pool == "param" and h.kind in ("mlp", "linear_nn"):
            continue  # drop mis-tagged nets from param refill
        if target_pool == "nn" and h.kind not in ("mlp", "linear_nn", "ridge_linear"):
            h.kind = "mlp"
            h.pool = "nn"
            h.ready = False
        next_hyps.append(h)
    next_hyps = next_hyps[:max_size]
    return next_hyps, culled, proposal_logs


def run_cycles(config_path: Path, cycles: int | None = None, seed: int = 42) -> dict:
    cfg = load_config(config_path)
    loop_root = config_path.parent
    paths = ensure_dirs(loop_root)
    rng = random.Random(seed)
    n_cycles = cycles if cycles is not None else int(cfg.get("cycles", 2))

    panel = load_csv(cfg["_panel_path"])
    labels_raw = load_csv(cfg["_label_path"])
    splits = load_csv(cfg["_split_path"]) if cfg["_split_path"].exists() else []

    labels = filter_labels(labels_raw, cfg)
    # Always recompute time-based split on the filtered smoke set
    # (precomputed split file is for the full ledger and misaligns after filters).
    labels = attach_splits(labels, split_rows=[])
    # require some lag presence for fair compare
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
    train_n = sum(1 for r in labels if r["split"] == "train")
    val_n = sum(1 for r in labels if r["split"] == "validation")

    # Hypothesis pools (algorithms). LLM models live in the registry and only propose.
    param_pool = seed_param_pool()
    nn_pool = seed_nn_pool()
    llm_registry = default_registry(cfg)
    prefer_llms = cfg.get("prefer_llms") or ["deepseek", "grok", "opus", "local_vllm", "offline_heuristic"]
    prev_mae: dict[str, float] = {}

    history = []
    all_proposal_logs: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths["runs"] / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # persist LLM registry snapshot
    write_csv(
        run_dir / "llm_model_registry.csv",
        [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "provider": m.provider,
                "cli_bin": getattr(m, "cli_bin", ""),
                "ssh_host": getattr(m, "ssh_host", ""),
                "remote_base_url": getattr(m, "remote_base_url", ""),
                "api_model": getattr(m, "api_model", ""),
                "available": m.is_available(),
                "notes": m.notes,
            }
            for m in llm_registry
        ],
    )

    for cycle in range(1, n_cycles + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(exist_ok=True)

        known_hyps = {h.model_id: h for h in param_pool + nn_pool}
        param_scored, param_res = score_pool(
            "param",
            param_pool,
            labels,
            feature_index,
            labels_by_key,
            cfg["scorer"],
            prev_mae,
            known_hyps=known_hyps,
        )
        for h, _ in param_scored:
            known_hyps[h.model_id] = h
        nn_scored, nn_res = score_pool(
            "nn",
            nn_pool,
            labels,
            feature_index,
            labels_by_key,
            cfg["scorer"],
            prev_mae,
            known_hyps=known_hyps,
        )

        for h, s in param_scored + nn_scored:
            prev_mae[h.model_id] = s.MAE
            known_hyps[h.model_id] = h

        def rows_from_scored(scored):
            out = []
            for h, s in scored:
                mut = getattr(h, "mutation_summary", None) or (h.params or {}).get("mutation_summary", "")
                pred_s = getattr(h, "prediction_summary", None) or (h.params or {}).get(
                    "prediction_summary", ""
                )
                out.append(
                    {
                        "cycle": cycle,
                        "pool": s.pool,
                        "hypothesis_id": s.model_id,
                        "kind": h.kind,
                        "S": round(s.S, 6),
                        "MAE": round(s.MAE, 6),
                        "hit_rate": round(s.hit_rate, 6),
                        "n": s.n,
                        "delta_MAE": round(s.delta_MAE, 6),
                        "maturity": round(s.maturity, 6),
                        "exp_mae": round(s.exp_mae, 6),
                        "hit_factor": round(s.hit_factor, 6),
                        "improve_factor": round(s.improve_factor, 6),
                        "params_json": json.dumps(h.params),
                        "parent_ids": "|".join(h.parent_ids),
                        "lineage_len": len(h.lineage),
                        "notes": h.notes,
                        "mutation_summary": mut,
                        "prediction_summary": pred_s,
                    }
                )
            return out

        score_rows = rows_from_scored(param_scored) + rows_from_scored(nn_scored)
        write_csv(cycle_dir / "scores.csv", score_rows)
        write_csv(cycle_dir / "residuals_param_sample.csv", param_res[:5000])
        write_csv(cycle_dir / "residuals_nn_sample.csv", nn_res[:5000])

        # LLM proposers refill pools after top-n selection
        next_param, culled_p, logs_p = select_and_reproduce(
            param_scored,
            {**cfg["pools"]["param"], "name": "param"},
            rng,
            cycle,
            nn_scored,
            llm_registry,
            prefer_llms=prefer_llms,
        )
        next_nn, culled_n, logs_n = select_and_reproduce(
            nn_scored,
            {**cfg["pools"]["nn"], "name": "nn"},
            rng,
            cycle,
            param_scored,
            llm_registry,
            prefer_llms=prefer_llms,
        )
        proposal_logs = logs_p + logs_n
        all_proposal_logs.extend(proposal_logs)
        write_csv(cycle_dir / "proposals.csv", proposal_logs)

        # Immediately train+run new proposals so viz never shows "not run yet"
        # (final cycle especially never gets a follow-up scoring pass).
        try:
            from run_pending_forecasts import hyp_from_proposal, _write_csv as _wf

            train = [r for r in labels if r.get("split") == "train"]
            fc_rows = []
            for log in proposal_logs:
                try:
                    hyp = hyp_from_proposal(
                        {
                            **log,
                            "params_json": log.get("params_json") or json.dumps({}),
                            "parents": log.get("parents")
                            or "|".join(
                                []
                                if not log.get("parents")
                                else str(log.get("parents")).split("|")
                            ),
                        }
                    )
                    # parents field in logs may already be pipe-joined
                    if not hyp.parent_ids and log.get("parents"):
                        hyp.parent_ids = [
                            p for p in str(log["parents"]).split("|") if p.strip()
                        ]
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
                    mut = summarize_mutation(hyp, None)
                    if log.get("rationale"):
                        mut = str(log["rationale"])[:300]
                    hyp.params["mutation_summary"] = mut
                    hyp.mutation_summary = mut  # type: ignore[attr-defined]
                    bd, _ = score_on_split(
                        preds, labels_by_key, cfg["scorer"], split="validation", prev_MAE=None
                    )
                    if bd is None:
                        bd, _ = score_on_split(
                            preds, labels_by_key, cfg["scorer"], split=None, prev_MAE=None
                        )
                    S = bd.S if bd else None
                    if S is not None and (S != S):  # NaN
                        S = None
                    fc_rows.append(
                        {
                            "cycle": cycle,
                            "hypothesis_id": hyp.model_id,
                            "pool": hyp.pool,
                            "kind": hyp.kind,
                            "S": "" if S is None else round(S, 6),
                            "MAE": "" if not bd else round(bd.MAE, 6),
                            "hit_rate": "" if not bd else round(bd.hit_rate, 6),
                            "n": 0 if not bd else bd.n,
                            "params_json": json.dumps(hyp.params),
                            "parents": log.get("parents") or "|".join(hyp.parent_ids),
                            "mutation_summary": mut,
                            "prediction_summary": getattr(hyp, "prediction_summary", ""),
                            "llm_model_id": log.get("llm_model_id", ""),
                            "mode": log.get("mode", ""),
                            "rationale": log.get("rationale", ""),
                            "was_already_in_scores": False,
                        }
                    )
                except Exception as e:
                    fc_rows.append(
                        {
                            "cycle": cycle,
                            "hypothesis_id": log.get("hypothesis_id", ""),
                            "pool": log.get("pool", ""),
                            "kind": log.get("kind", ""),
                            "S": "",
                            "MAE": "",
                            "hit_rate": "",
                            "n": 0,
                            "params_json": log.get("params_json", ""),
                            "parents": log.get("parents", ""),
                            "mutation_summary": (log.get("rationale") or "")[:300],
                            "prediction_summary": f"Run failed: {e}",
                            "llm_model_id": log.get("llm_model_id", ""),
                            "mode": log.get("mode", ""),
                            "rationale": log.get("rationale", ""),
                            "was_already_in_scores": False,
                        }
                    )
            if fc_rows:
                write_csv(cycle_dir / "proposal_forecasts.csv", fc_rows)
        except Exception as e:
            print(f"warning: proposal forecast pass failed: {e}", flush=True)

        archive_rows = []
        for h, s in culled_p + culled_n:
            archive_rows.append(
                {
                    "cycle": cycle,
                    "pool": s.pool,
                    "hypothesis_id": s.model_id,
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
                    "parent_ids": "|".join(h.parent_ids),
                }
                for h in next_param
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
                    "parent_ids": "|".join(h.parent_ids),
                }
                for h in next_nn
            ],
        )

        best_p = param_scored[0] if param_scored else None
        best_n = nn_scored[0] if nn_scored else None
        api_props = sum(1 for x in proposal_logs if x.get("used_api"))
        history.append(
            {
                "cycle": cycle,
                "n_labels": len(labels),
                "n_train": train_n,
                "n_validation": val_n,
                "param_pool_scored": len(param_scored),
                "nn_pool_scored": len(nn_scored),
                "proposals": len(proposal_logs),
                "proposals_via_llm_api": api_props,
                "best_param_id": best_p[0].model_id if best_p else "",
                "best_param_S": best_p[1].S if best_p else None,
                "best_param_MAE": best_p[1].MAE if best_p else None,
                "best_param_hit_rate": best_p[1].hit_rate if best_p else None,
                "best_nn_id": best_n[0].model_id if best_n else "",
                "best_nn_S": best_n[1].S if best_n else None,
                "best_nn_MAE": best_n[1].MAE if best_n else None,
                "best_nn_hit_rate": best_n[1].hit_rate if best_n else None,
            }
        )

        param_pool = next_param
        nn_pool = next_nn

        (cycle_dir / "summary.json").write_text(
            json.dumps(history[-1], indent=2, default=str)
        )

    write_csv(run_dir / "cycle_history.csv", history)
    write_csv(run_dir / "all_proposals.csv", all_proposal_logs)
    summary = {
        "run_dir": str(run_dir),
        "cycles": n_cycles,
        "labels_used": len(labels),
        "train": train_n,
        "validation": val_n,
        "history": history,
        "scorer": cfg["scorer"],
        "smoke": cfg.get("smoke"),
        "terminology": {
            "model": "LLM (Opus/DeepSeek/Grok/local/offline) that proposes hypotheses",
            "hypothesis": "prediction algorithm scored by S",
        },
        "llm_registry": [
            {
                "model_id": m.model_id,
                "available": m.is_available(),
                "proposal_count": m.proposal_count,
                "last_error": m.last_error,
            }
            for m in llm_registry
        ],
    }
    (run_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Research loop run (LLM models → hypotheses)",
        "",
        f"- Labels used: **{len(labels)}** (train={train_n}, validation={val_n})",
        f"- Cycles: **{n_cycles}**",
        f"- Scorer: `s={cfg['scorer']['s']}`, `α={cfg['scorer']['alpha']}`, `β={cfg['scorer']['beta']}`, `n0={cfg['scorer']['n0']}`",
        "",
        "## Terminology",
        "- **Model** = LLM proposer (registry below)",
        "- **Hypothesis** = prediction algorithm in param/nn pools",
        "",
        "## LLM model registry",
        "",
        "| model_id | available | proposals | last_error |",
        "|----------|-----------|----------:|------------|",
    ]
    for m in llm_registry:
        err = (m.last_error or "").replace("|", "/")[:60]
        lines.append(f"| {m.model_id} | {m.is_available()} | {m.proposal_count} | {err} |")
    lines += [
        "",
        "## Cycle results (best hypothesis per pool)",
        "",
        "| Cycle | Best param hyp | S | MAE | hit | Best nn hyp | S | MAE | hit | LLM proposals |",
        "|------:|----------------|--:|----:|----:|-------------|--:|----:|----:|--------------:|",
    ]
    for h in history:
        lines.append(
            f"| {h['cycle']} | {h['best_param_id']} | {h['best_param_S']:.4f} | {h['best_param_MAE']:.4f} | {h['best_param_hit_rate']:.3f} | "
            f"{h['best_nn_id']} | {h['best_nn_S']:.4f} | {h['best_nn_MAE']:.4f} | {h['best_nn_hit_rate']:.3f} | "
            f"{h.get('proposals_via_llm_api', 0)}/{h.get('proposals', 0)} |"
        )
    lines.append("")
    lines.append(f"Artifacts: `{run_dir}`")
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    (loop_root / "LATEST_RUN.txt").write_text(str(run_dir) + "\n")
    return summary
