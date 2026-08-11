"""
Proposer: LLM models generate new **hypotheses** (prediction algorithms).

Modes (PropView.md):
  1. lineage  — evolve from one hypothesis's model-score lineage history
  2. in_pool  — combine two random hypotheses in the same pool
  3. cross_pool — combine two random hypotheses from different pools

Output is a constrained hypothesis *config* (kind + params + pool), not free Python,
so proposals stay sandbox-safe and scorable immediately.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any

from hypotheses import Hypothesis
from llm_models import (
    LLMModel,
    chat_completion,
    extract_json_object,
    select_available_models,
)

ALLOWED_KINDS_PARAM = {
    "zero",
    "constant",
    "last_mom",
    "last_yoy_scaled",
    "blend_mom_yoy",
    "mean_reversion",
    "rate_penalty",
    "momentum_lag",
    "ridge_linear",
}
ALLOWED_KINDS_NN = {
    "linear_nn",
    "mlp",  # real multilayer neural network (trained)
}

SYSTEM_PROMPT = """You are a PropView hypothesis proposer.
You invent prediction algorithms (hypotheses) that map UK housing features to a forecasted change_pct over a horizon.

A hypothesis is NOT an LLM. It is a small algorithm with a kind and parameters.
You MAY propose a real neural network (kind=mlp) when nonlinearity seems justified given parent scores/lineage.

Allowed kinds and params:
PARAM pool only:
- zero: {}
- constant: {c: float}  # monthly drift × horizon
- last_mom: {scale: float}
- last_yoy_scaled: {scale: float}
- blend_mom_yoy: {w_mom: float 0-1, scale: float}
- mean_reversion: {strength: float}
- rate_penalty: {rate_coef: float, scale: float}
- momentum_lag: {w1,w2,w3: float, scale: float}
- ridge_linear: {l2: float, lr: float, steps: int}  # fitted linear

NN pool:
- linear_nn: {l2: float, lr: float, steps: int}  # fitted linear head
- mlp: REAL multilayer net with hidden layers, trained by SGD
  params: {
    "hidden_layers": [int, ...] e.g. [16,8] or [32,16] or [8],
    "activation": "relu" | "tanh",
    "lr": float,
    "steps": int (training iterations),
    "l2": float,
    "batch_size": int,
    "seed": int
  }
  Prefer modest nets (hidden widths ≤ 64, depth ≤ 3) unless lineage shows linear methods stuck/high MAE.
  With PyTorch training available, wider nets ([64,32], [128,64]) and more steps (up to 3000) are OK when justified.

When parents are linear and MAE is still high, consider kind=mlp.
When parents already use mlp and improve, refine architecture (width/depth/activation/lr).
When simple baselines nearly match neural MAE, prefer simpler kinds.

Pool must be "param" or "nn". Never put mlp in param pool.

Respond with ONLY a JSON object:
{
  "kind": "...",
  "pool": "param"|"nn",
  "params": {...},
  "rationale": "one sentence why this architecture/rule given the parents"
}
"""


def _hyp_summary(h: Hypothesis, score: dict | None = None) -> dict:
    d = {
        "hypothesis_id": h.model_id,
        "pool": h.pool,
        "kind": h.kind,
        "params": h.params,
        "lineage": h.lineage[-5:],  # recent scores
        "notes": h.notes,
    }
    if score:
        d["last_score"] = score
    return d


def _normalize_kind(kind: str) -> str:
    kind = str(kind or "").strip()
    aliases = {
        "neural_network": "mlp",
        "neural": "mlp",
        "nn": "mlp",
        "network": "mlp",
        "deep": "mlp",
        "feedforward": "mlp",
        "mlp_net": "mlp",
        "linear": "linear_nn",
        "ridge": "ridge_linear",
    }
    return aliases.get(kind, kind)


def _coerce_proposal_to_pool(prop: dict, target_pool: str) -> dict:
    """
    Hard rule: pool is determined by the evolver call; kind must be legal there.
    - param pool: only parametric / ridge kinds (never mlp)
    - nn pool: only linear_nn / mlp
    """
    kind = _normalize_kind(prop.get("kind", ""))
    params = dict(prop.get("params") or {}) if isinstance(prop.get("params"), dict) else {}

    if target_pool == "param":
        # Neural architectures are not allowed in the parametric pool.
        if kind in ("mlp", "linear_nn") or params.get("hidden_layers") or params.get("hidden"):
            # Keep a trainable linear parametric stand-in, not an MLP.
            kind = "ridge_linear"
            params.pop("hidden_layers", None)
            params.pop("hidden", None)
            params.pop("activation", None)
            params.setdefault("l2", 1e-3)
            params.setdefault("lr", 0.02)
            params.setdefault("steps", 400)
        elif kind not in ALLOWED_KINDS_PARAM:
            kind = "blend_mom_yoy"
            params = {"w_mom": 0.5, "scale": 1.0}
        # strip any smuggled net hyperparams from pure rules
        if kind not in ("ridge_linear",):
            params.pop("hidden_layers", None)
            params.pop("hidden", None)
            params.pop("activation", None)
        pool = "param"
    else:
        # nn pool
        if kind in ALLOWED_KINDS_PARAM and kind not in ("ridge_linear",):
            # free-form rules → real net
            kind = "mlp"
        if kind == "ridge_linear":
            kind = "linear_nn"
        if kind not in ALLOWED_KINDS_NN:
            kind = "mlp"
        if kind == "mlp":
            params.setdefault("hidden_layers", [16, 8])
            params.setdefault("activation", "relu")
            params.setdefault("lr", 0.01)
            params.setdefault("steps", 800)
            params.setdefault("l2", 1e-4)
        pool = "nn"

    prop = dict(prop)
    prop["kind"] = kind
    prop["pool"] = pool
    prop["params"] = params
    return prop


def _validate_proposal(obj: dict, target_pool: str) -> dict:
    kind = _normalize_kind(obj.get("kind") or "")
    pool = str(obj.get("pool") or target_pool).strip()
    if pool not in ("param", "nn"):
        pool = target_pool
    # kind dictates pool for nets; never leave mlp tagged as param
    if kind == "mlp" or kind == "linear_nn":
        pool = "nn"
    if kind == "ridge_linear":
        pool = "param"
    allowed = ALLOWED_KINDS_NN if pool == "nn" else ALLOWED_KINDS_PARAM
    if kind not in allowed:
        if kind in ALLOWED_KINDS_PARAM and pool == "nn":
            kind = "linear_nn" if kind == "ridge_linear" else "mlp"
        elif kind == "linear_nn" and pool == "param":
            kind = "ridge_linear"
        elif kind == "mlp":
            pool = "nn"
        else:
            kind = "blend_mom_yoy" if pool == "param" else "mlp"
    if kind in ("mlp", "linear_nn"):
        pool = "nn"
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    # clamp common params
    if "scale" in params:
        try:
            params["scale"] = float(max(0.05, min(3.0, float(params["scale"]))))
        except Exception:
            params["scale"] = 1.0
    if "w_mom" in params:
        try:
            params["w_mom"] = float(max(0.05, min(0.95, float(params["w_mom"]))))
        except Exception:
            params["w_mom"] = 0.5
    if "strength" in params:
        try:
            params["strength"] = float(max(0.01, min(2.0, float(params["strength"]))))
        except Exception:
            params["strength"] = 0.3
    if "l2" in params:
        try:
            params["l2"] = float(max(1e-6, min(1.0, float(params["l2"]))))
        except Exception:
            params["l2"] = 1e-3
    if "lr" in params:
        try:
            params["lr"] = float(max(1e-4, min(0.2, float(params["lr"]))))
        except Exception:
            params["lr"] = 0.02
    if "steps" in params:
        try:
            params["steps"] = int(max(50, min(5000, int(params["steps"]))))
        except Exception:
            params["steps"] = 400
    if "batch_size" in params:
        try:
            params["batch_size"] = int(max(8, min(512, int(params["batch_size"]))))
        except Exception:
            params["batch_size"] = 64
    if "hidden_layers" in params or "hidden" in params:
        hl = params.get("hidden_layers", params.get("hidden"))
        if isinstance(hl, str):
            hl = [int(x) for x in hl.replace("[", "").replace("]", "").split(",") if x.strip()]
        try:
            hl = [int(max(2, min(256, int(h)))) for h in list(hl)[:6]]
        except Exception:
            hl = [16, 8]
        if not hl:
            hl = [16, 8]
        params["hidden_layers"] = hl
    if kind == "mlp" and "hidden_layers" not in params:
        params["hidden_layers"] = [16, 8]
    if kind == "mlp" and "activation" in params:
        act = str(params["activation"]).lower()
        params["activation"] = act if act in ("relu", "tanh") else "relu"
    # Final hard invariant before return
    if kind in ("mlp", "linear_nn"):
        pool = "nn"
    if kind in ALLOWED_KINDS_PARAM and kind != "ridge_linear" and pool == "nn":
        # shouldn't happen; keep nn pool pure
        kind = "mlp"
        pool = "nn"
    return {
        "kind": kind,
        "pool": pool,
        "params": params,
        "rationale": str(obj.get("rationale") or "")[:300],
    }


def _offline_propose(
    mode: str,
    target_pool: str,
    primary: Hypothesis | None,
    secondary: Hypothesis | None,
    rng: random.Random,
) -> dict:
    """Lineage-aware heuristic when no LLM API works."""
    if mode == "lineage" and primary is not None:
        # If MAE improving in lineage, strengthen same kind; else diversify
        lin = primary.lineage
        improving = False
        if len(lin) >= 2:
            try:
                improving = float(lin[-1].get("MAE", 9)) < float(lin[-2].get("MAE", 9))
            except Exception:
                improving = False
        if improving and primary.kind in ("last_mom", "blend_mom_yoy", "last_yoy_scaled", "momentum_lag", "mlp"):
            params = dict(primary.params)
            if primary.kind == "mlp":
                # refine architecture slightly
                hl = list(params.get("hidden_layers") or [16, 8])
                if rng.random() < 0.5 and hl:
                    i = rng.randrange(len(hl))
                    hl[i] = int(max(4, min(256, hl[i] + rng.choice([-8, -4, -2, 2, 4, 8, 16]))))
                params["hidden_layers"] = hl
                params["lr"] = float(max(1e-4, min(0.05, float(params.get("lr", 0.01)) * rng.uniform(0.7, 1.3))))
                params["steps"] = int(max(200, min(3000, int(params.get("steps", 800)) * rng.uniform(0.8, 1.2))))
                return {
                    "kind": "mlp",
                    "pool": "nn",
                    "params": params,
                    "rationale": "offline: refine improving MLP architecture",
                }
            scale = float(params.get("scale", 1.0))
            params["scale"] = max(0.05, min(2.5, scale * rng.uniform(0.9, 1.15)))
            return {
                "kind": primary.kind,
                "pool": target_pool,
                "params": params,
                "rationale": "offline: reinforce improving lineage",
            }
        # diversify: if linear/stuck parametric, try a real MLP in nn pool
        if primary.kind in ("last_mom", "momentum_lag", "linear_nn", "ridge_linear", "zero"):
            if target_pool == "nn":
                return {
                    "kind": "mlp",
                    "pool": "nn",
                    "params": {
                        "hidden_layers": rng.choice([[8], [16, 8], [32, 16], [64, 32], [16], [24, 12], [48, 24, 12]]),
                        "activation": rng.choice(["relu", "tanh"]),
                        "lr": rng.uniform(0.003, 0.02),
                        "steps": int(rng.uniform(600, 1500)),
                        "l2": 10 ** rng.uniform(-5, -3),
                        "batch_size": rng.choice([32, 64, 128]),
                        "seed": rng.randint(1, 9999),
                    },
                    "rationale": "offline: try real MLP after linear/simple parents stuck",
                }
            return {
                "kind": "mean_reversion",
                "pool": "param",
                "params": {"strength": rng.uniform(0.15, 0.45)},
                "rationale": "offline: diversify parametric after flat lineage",
            }
        if target_pool == "nn":
            if primary.kind == "mlp" or rng.random() < 0.6:
                return {
                    "kind": "mlp",
                    "pool": "nn",
                    "params": {
                        "hidden_layers": rng.choice([[12, 6], [20, 10], [16, 8, 4], [32, 8], [64, 32], [128, 64]]),
                        "activation": rng.choice(["relu", "tanh"]),
                        "lr": rng.uniform(0.003, 0.025),
                        "steps": int(rng.uniform(600, 2000)),
                        "l2": 10 ** rng.uniform(-5, -3),
                        "batch_size": rng.choice([64, 128]),
                        "seed": rng.randint(1, 9999),
                    },
                    "rationale": "offline: nn pool proposes MLP architecture search",
                }
            return {
                "kind": "linear_nn",
                "pool": "nn",
                "params": {
                    "l2": 10 ** rng.uniform(-4, -1.5),
                    "lr": rng.uniform(0.008, 0.04),
                    "steps": int(rng.uniform(350, 700)),
                },
                "rationale": "offline: nn linear hyperparam search from lineage parent",
            }
        return {
            "kind": "blend_mom_yoy",
            "pool": "param",
            "params": {"w_mom": rng.uniform(0.3, 0.7), "scale": rng.uniform(0.5, 1.2)},
            "rationale": "offline: blend after lineage review",
        }

    # in_pool / cross_pool: blend two parents
    if primary and secondary:
        if target_pool == "nn" or primary.pool == "nn" or secondary.pool == "nn":
            # Prefer real MLP when merging into nn pool, especially if either parent is weak linear
            if target_pool == "nn" and (
                primary.kind == "mlp"
                or secondary.kind == "mlp"
                or primary.kind in ("linear_nn", "ridge_linear", "zero")
                or secondary.kind in ("linear_nn", "ridge_linear", "zero")
            ):
                h1 = list(primary.params.get("hidden_layers") or [16, 8])
                h2 = list(secondary.params.get("hidden_layers") or [16, 8])
                # union-ish width average
                width = int(max(4, min(128, 0.5 * (sum(h1) / max(1, len(h1)) + sum(h2) / max(1, len(h2))))))
                depth = 2 if (len(h1) + len(h2)) >= 2 else 1
                if max(len(h1), len(h2)) >= 3:
                    depth = 3
                if depth == 1:
                    hidden = [width]
                elif depth == 2:
                    hidden = [width, max(4, width // 2)]
                else:
                    hidden = [width, max(8, width // 2), max(4, width // 4)]
                return {
                    "kind": "mlp",
                    "pool": "nn",
                    "params": {
                        "hidden_layers": hidden,
                        "activation": rng.choice(["relu", "tanh"]),
                        "lr": rng.uniform(0.003, 0.02),
                        "steps": int(rng.uniform(700, 1500)),
                        "l2": 10 ** rng.uniform(-5, -3.5),
                        "batch_size": rng.choice([64, 128]),
                        "seed": rng.randint(1, 9999),
                    },
                    "rationale": f"offline: merge {primary.model_id}+{secondary.model_id} into trained MLP",
                }
            return {
                "kind": "linear_nn",
                "pool": "nn",
                "params": {
                    "l2": float(
                        0.5
                        * (
                            float(primary.params.get("l2", 1e-3))
                            + float(secondary.params.get("l2", 1e-3))
                        )
                    )
                    if primary.params.get("l2") or secondary.params.get("l2")
                    else 10 ** rng.uniform(-3.5, -2),
                    "lr": rng.uniform(0.01, 0.03),
                    "steps": int(rng.uniform(400, 650)),
                },
                "rationale": f"offline: merge {primary.model_id} + {secondary.model_id} into linear_nn",
            }
        # parametric merge
        if primary.kind == secondary.kind:
            params = {}
            for k in set(primary.params) | set(secondary.params):
                va, vb = primary.params.get(k), secondary.params.get(k)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    params[k] = 0.5 * float(va) + 0.5 * float(vb)
                elif va is not None:
                    params[k] = va
                else:
                    params[k] = vb
            return {
                "kind": primary.kind,
                "pool": target_pool,
                "params": params,
                "rationale": f"offline: same-kind merge {primary.kind}",
            }
        return {
            "kind": "blend_mom_yoy",
            "pool": target_pool,
            "params": {"w_mom": rng.uniform(0.4, 0.6), "scale": rng.uniform(0.6, 1.1)},
            "rationale": f"offline: cross-kind {primary.kind}+{secondary.kind}",
        }

    return {
        "kind": "ridge_linear" if target_pool == "param" else "linear_nn",
        "pool": target_pool,
        "params": {"l2": 1e-3, "lr": 0.02, "steps": 400},
        "rationale": "offline: default fitted linear",
    }


def propose_hypothesis(
    llm: LLMModel,
    mode: str,
    target_pool: str,
    primary: Hypothesis | None,
    secondary: Hypothesis | None,
    rng: random.Random,
    scores: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    """
    Returns (validated_proposal, meta).
    meta includes llm_model_id, mode, used_api, error.
    """
    scores = scores or {}
    meta = {
        "llm_model_id": llm.model_id,
        "llm_display_name": llm.display_name,
        "mode": mode,
        "target_pool": target_pool,
        "used_api": False,
        "error": "",
        "parents": [x.model_id for x in (primary, secondary) if x is not None],
    }

    user_payload = {
        "mode": mode,
        "target_pool": target_pool,
        "primary": _hyp_summary(primary, scores.get(primary.model_id)) if primary else None,
        "secondary": _hyp_summary(secondary, scores.get(secondary.model_id)) if secondary else None,
        "instruction": {
            "lineage": "Improve or diversify based on the primary hypothesis lineage scores (S, MAE, hit_rate).",
            "in_pool": "Combine strengths of two hypotheses from the same pool.",
            "cross_pool": "Combine ideas from two hypotheses across param vs nn pools.",
        }.get(mode, ""),
    }
    user = json.dumps(user_payload, indent=2)

    if llm.provider != "offline_heuristic" and llm.is_available():
        try:
            raw = chat_completion(llm, SYSTEM_PROMPT, user)
            obj = extract_json_object(raw)
            prop = _validate_proposal(obj, target_pool)
            meta["used_api"] = True
            llm.proposal_count += 1
            llm.last_error = ""
            return prop, meta
        except Exception as e:
            meta["error"] = str(e)[:300]
            llm.last_error = meta["error"]

    # fallback
    obj = _offline_propose(mode, target_pool, primary, secondary, rng)
    prop = _validate_proposal(obj, target_pool)
    meta["used_api"] = False
    if not meta["error"]:
        meta["error"] = "used offline_heuristic"
    llm.proposal_count += 1
    return prop, meta


def proposal_to_hypothesis(
    prop: dict,
    meta: dict,
    cycle: int,
    serial: int,
    parents: list[Hypothesis],
) -> Hypothesis:
    pool = prop["pool"]
    kind = prop["kind"]
    hid = f"{pool}_llm_{meta['llm_model_id']}_c{cycle}_{serial}"
    ready = kind not in ("ridge_linear", "linear_nn", "mlp")
    hyp = Hypothesis(
        model_id=hid,
        pool=pool,
        kind=kind,
        params=dict(prop.get("params") or {}),
        ready=ready,
        parent_ids=[p.model_id for p in parents],
        notes=f"proposed_by={meta['llm_model_id']}; mode={meta['mode']}; {prop.get('rationale','')}",
    )
    # carry lineage from best parent
    if parents:
        best = max(parents, key=lambda p: (p.lineage[-1]["S"] if p.lineage else -1))
        hyp.lineage = list(best.lineage)
    return hyp


def propose_for_pool(
    target_pool: str,
    survivors: list[tuple[Hypothesis, Any]],
    other_pool_scored: list[tuple[Hypothesis, Any]],
    llm_registry: list[LLMModel],
    rng: random.Random,
    cycle: int,
    n_proposals: int = 3,
    prefer_llms: list[str] | None = None,
) -> tuple[list[Hypothesis], list[dict]]:
    """
    Generate n_proposals for target_pool using rotating modes and LLMs.
    """
    available = select_available_models(llm_registry, prefer=prefer_llms)
    if not available:
        available = [m for m in llm_registry if m.model_id == "offline_heuristic"]
    if not available or n_proposals <= 0:
        return [], []

    hyp_list = [h for h, _ in survivors]
    score_map = {
        h.model_id: {"S": s.S, "MAE": s.MAE, "hit_rate": s.hit_rate, "n": s.n}
        for h, s in survivors
    }
    other_hyps = [h for h, _ in other_pool_scored]
    for h, s in other_pool_scored:
        score_map[h.model_id] = {"S": s.S, "MAE": s.MAE, "hit_rate": s.hit_rate, "n": s.n}

    modes = ["lineage", "in_pool", "cross_pool"]
    out_hyps: list[Hypothesis] = []
    logs: list[dict] = []
    serial = 0

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
                if hyp_list:
                    primary = hyp_list[0]
                    mode = "lineage"
                else:
                    continue
            else:
                primary, secondary = rng.sample(hyp_list, 2)
        else:  # cross_pool
            if not hyp_list or not other_hyps:
                if hyp_list:
                    primary = rng.choice(hyp_list)
                    mode = "lineage"
                else:
                    continue
            else:
                primary = rng.choice(hyp_list)
                secondary = rng.choice(other_hyps)

        prop, meta = propose_hypothesis(
            llm, mode, target_pool, primary, secondary, rng, scores=score_map
        )
        # Enforce: kind must match the pool this call is refilling.
        # Neural nets (mlp / linear_nn) never enter the param pool; simple rules
        # never enter the nn pool as "mlp" under a param id.
        prop = _coerce_proposal_to_pool(prop, target_pool)

        serial += 1
        parents = [p for p in (primary, secondary) if p is not None]
        hyp = proposal_to_hypothesis(prop, meta, cycle, serial, parents)
        out_hyps.append(hyp)
        logs.append(
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
            }
        )

    return out_hyps, logs
