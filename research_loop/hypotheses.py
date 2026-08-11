"""
Hypotheses = prediction algorithms (PropView.md).

A hypothesis is ready when params/weights are set, then maps raw features → prediction.
LLM **models** (Opus/DeepSeek/…) live in llm_models.py and only *propose* hypotheses.
`model_id` on Hypothesis is the hypothesis id (legacy field name kept for scorer CSV compat).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from data_access import feature_vector, fnum
from mlp import MLP, build_mlp_from_params
from scorer import Prediction


@dataclass
class Hypothesis:
    model_id: str  # hypothesis_id (legacy name)
    pool: str  # "param" | "nn"
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    weights: list[float] | None = None  # for linear ridge
    feature_names: list[str] | None = None
    mlp_state: dict[str, Any] | None = None  # serialized MLP for kind=mlp
    ready: bool = True
    lineage: list[dict] = field(default_factory=list)  # score history for proposers
    parent_ids: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def hypothesis_id(self) -> str:
        return self.model_id

    def clone(self, new_id: str, **param_updates) -> "Hypothesis":
        h = copy.deepcopy(self)
        h.model_id = new_id
        h.params.update(param_updates)
        h.parent_ids = [self.model_id]
        return h


def _horizon(label: dict) -> int:
    try:
        return int(float(label.get("horizon_months") or 1))
    except Exception:
        return 1


def _tol(label: dict, default: float = 1.5) -> float:
    v = fnum(label.get("tolerated_error_pp"), default)
    return float(v if v is not None else default)


def predict_label(
    hyp: Hypothesis,
    label: dict,
    panel_row: dict | None,
) -> Prediction | None:
    """Generate one prediction for a label row using features at as_of."""
    if not hyp.ready:
        return None
    # Window must not precede as-of (PropView.md)
    as_of = (label.get("as_of_date") or "")[:10]
    start = (label.get("start_date") or "")[:10]
    end = (label.get("end_date") or "")[:10]
    if as_of and start and start < as_of:
        return None
    if start and end and end < start:
        return None

    h = _horizon(label)
    feats = feature_vector(panel_row, h)
    if not feats and hyp.kind not in ("zero", "constant"):
        if hyp.kind not in ("zero", "constant"):
            return None

    change = _predict_change(hyp, feats, h)
    if change is None:
        return None
    return Prediction(
        target=label["target"],
        aggregation=label["aggregation"],
        area=label["area"],
        as_of_date=label["as_of_date"],
        start_date=label["start_date"],
        end_date=label["end_date"],
        change_pct=float(change),
        tolerated_error_pp=_tol(label),
        model_id=hyp.model_id,  # hypothesis id
        notes=hyp.kind,
        horizon_months=h,
        price_type=label.get("price_type"),
    )


def _predict_change(hyp: Hypothesis, feats: dict[str, float], horizon: int) -> float | None:
    k = hyp.kind
    p = hyp.params

    if k == "zero":
        return 0.0

    if k == "constant":
        return float(p.get("c", 0.0)) * horizon

    if k == "last_mom":
        # compound last MoM over horizon (works for short and 10y horizons)
        mom = feats.get("mom_change_pct")
        if mom is None:
            return 0.0
        scale = float(p.get("scale", 1.0))
        m = max(-50.0, min(50.0, float(mom))) / 100.0
        return scale * ((1.0 + m) ** horizon - 1.0) * 100.0

    if k == "last_yoy_scaled":
        # persist YoY rate over horizon/12 years
        yoy = feats.get("yoy_change_pct")
        if yoy is None:
            return 0.0
        scale = float(p.get("scale", 1.0))
        years = horizon / 12.0
        y = max(-80.0, min(80.0, float(yoy))) / 100.0
        return scale * ((1.0 + y) ** years - 1.0) * 100.0

    if k == "blend_mom_yoy":
        mom = feats.get("mom_change_pct", 0.0) or 0.0
        yoy = feats.get("yoy_change_pct", 0.0) or 0.0
        w = float(p.get("w_mom", 0.5))
        scale = float(p.get("scale", 1.0))
        m = max(-50.0, min(50.0, float(mom))) / 100.0
        y = max(-80.0, min(80.0, float(yoy))) / 100.0
        years = horizon / 12.0
        from_mom = ((1.0 + m) ** horizon - 1.0) * 100.0
        from_yoy = ((1.0 + y) ** years - 1.0) * 100.0
        return scale * (w * from_mom + (1.0 - w) * from_yoy)

    if k == "mean_reversion":
        # if recent yoy high, predict lower multi-year total (partial fade)
        yoy = feats.get("yoy_change_pct", feats.get("mom_change_pct"))
        if yoy is None:
            return 0.0
        strength = float(p.get("strength", 0.3))
        years = horizon / 12.0
        y = max(-80.0, min(80.0, float(yoy))) / 100.0
        # fade annual rate toward 0 by strength, then compound
        faded = y * (1.0 - strength)
        return ((1.0 + faded) ** years - 1.0) * 100.0

    if k == "rate_penalty":
        # higher bank rate → lower expected multi-year appreciation
        yoy = feats.get("yoy_change_pct", 0.0) or 0.0
        rate = feats.get("bank_rate_pct", feats.get("bank_rate"))
        years = horizon / 12.0
        y = max(-80.0, min(80.0, float(yoy))) / 100.0
        base = ((1.0 + y) ** years - 1.0) * 100.0
        if rate is None:
            return base * float(p.get("scale", 1.0))
        coef = float(p.get("rate_coef", 0.15))
        # ~coef pp of total return per 1pp rate above 3% per year of horizon
        return (base - coef * (float(rate) - 3.0) * years * 5.0) * float(p.get("scale", 1.0))

    if k == "momentum_lag":
        m1 = feats.get("mom_lag1", feats.get("mom_change_pct", 0.0)) or 0.0
        m2 = feats.get("mom_lag2", 0.0) or 0.0
        m3 = feats.get("mom_lag3", 0.0) or 0.0
        w1, w2, w3 = float(p.get("w1", 0.5)), float(p.get("w2", 0.3)), float(p.get("w3", 0.2))
        s = w1 + w2 + w3
        avg = (w1 * m1 + w2 * m2 + w3 * m3) / s
        m = max(-50.0, min(50.0, float(avg))) / 100.0
        return float(p.get("scale", 1.0)) * ((1.0 + m) ** horizon - 1.0) * 100.0

    if k in ("linear_nn", "ridge_linear"):
        if not hyp.weights or not hyp.feature_names:
            return 0.0
        names = hyp.feature_names
        w = hyp.weights
        if len(w) != len(names) + 1:
            return 0.0
        acc = w[0]
        for i, name in enumerate(names):
            acc += w[i + 1] * feats.get(name, 0.0)
        return acc

    if k == "mlp":
        if not hyp.mlp_state or not hyp.feature_names:
            return 0.0
        names = hyp.feature_names
        x = [max(-_FEAT_CLIP, min(_FEAT_CLIP, feats.get(n, 0.0))) for n in names]
        # apply train-time normalization if stored
        mu = hyp.params.get("_feat_mean") or []
        sd = hyp.params.get("_feat_std") or []
        if mu and sd and len(mu) == len(x):
            x = [(x[i] - mu[i]) / sd[i] for i in range(len(x))]
        mlp = MLP.from_state(hyp.mlp_state)
        return mlp.predict(x)

    return None


# ---------------------------------------------------------------------------
# Training for linear / MLP on train split
# ---------------------------------------------------------------------------

FEATURE_SET = [
    "mom_change_pct",
    "yoy_change_pct",
    "mom_lag1",
    "mom_lag2",
    "mom_lag3",
    "yoy_scaled",  # yoy * (horizon/12) — total-return style scale
    "mom_x_horizon",
    "bank_rate",
    "horizon_months",
]

# Wide clip so multi-year total-return features (e.g. yoy*10, mom*120) survive.
_FEAT_CLIP = 500.0


def _build_xy(
    train_labels: list[dict],
    feature_index: dict[tuple[str, str], dict],
    names: list[str],
) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    for lab in train_labels:
        prow = feature_index.get((lab["area"], lab["as_of_date"][:10]))
        h = _horizon(lab)
        feats = feature_vector(prow, h)
        if "mom_change_pct" not in feats and "yoy_change_pct" not in feats:
            continue
        x = [max(-_FEAT_CLIP, min(_FEAT_CLIP, feats.get(n, 0.0))) for n in names]
        X.append(x)
        y.append(float(lab["actual_change_pct"]))
    return X, y


def _normalize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not X:
        return X, [], []
    d = len(X[0])
    mu = [0.0] * d
    for row in X:
        for j, v in enumerate(row):
            mu[j] += v
    n = len(X)
    mu = [m / n for m in mu]
    var = [0.0] * d
    for row in X:
        for j, v in enumerate(row):
            var[j] += (v - mu[j]) ** 2
    sd = [max(1e-6, (v / n) ** 0.5) for v in var]
    Xn = [[(row[j] - mu[j]) / sd[j] for j in range(d)] for row in X]
    return Xn, mu, sd


def fit_linear(
    hyp: Hypothesis,
    train_labels: list[dict],
    feature_index: dict[tuple[str, str], dict],
    l2: float = 1e-3,
    steps: int = 400,
    lr: float = 0.02,
) -> Hypothesis:
    """Fit bias+linear weights with SGD + L2; pure Python."""
    names = list(FEATURE_SET)
    X, y = _build_xy(train_labels, feature_index, names)
    if len(X) < 20:
        hyp.ready = True
        hyp.weights = [0.0] + [0.0] * len(names)
        hyp.feature_names = names
        hyp.notes = f"underfit_n={len(X)}"
        return hyp

    w = [0.0] + [0.0] * len(names)
    n = len(X)
    for step in range(steps):
        g = [0.0] * len(w)
        for x, yi in zip(X, y):
            pred = w[0] + sum(w[i + 1] * x[i] for i in range(len(names)))
            err = pred - yi
            g[0] += err
            for i in range(len(names)):
                g[i + 1] += err * x[i]
        for i in range(len(w)):
            g[i] = g[i] / n + l2 * (0.0 if i == 0 else w[i])
            w[i] -= lr * g[i]
        if step in (100, 200, 300):
            lr *= 0.5

    hyp.weights = w
    hyp.feature_names = names
    hyp.mlp_state = None
    hyp.ready = True
    if hyp.kind not in ("ridge_linear", "linear_nn"):
        hyp.kind = "ridge_linear" if hyp.pool == "param" else "linear_nn"
    hyp.params["l2"] = l2
    hyp.params["train_n"] = len(X)
    hyp.notes = f"linear_fitted_n={len(X)}"
    return hyp


def fit_mlp(
    hyp: Hypothesis,
    train_labels: list[dict],
    feature_index: dict[tuple[str, str], dict],
) -> Hypothesis:
    """Build and train a multilayer perceptron with PyTorch."""
    names = list(FEATURE_SET)
    X, y = _build_xy(train_labels, feature_index, names)
    if len(X) < 30:
        # fall back to linear if too little data
        hyp.kind = "linear_nn"
        return fit_linear(
            hyp,
            train_labels,
            feature_index,
            l2=float(hyp.params.get("l2", 1e-3)),
            steps=int(hyp.params.get("steps", 400)),
            lr=float(hyp.params.get("lr", 0.02)),
        )

    Xn, mu, sd = _normalize(X)
    seed = int(hyp.params.get("seed", 42))
    mlp = build_mlp_from_params(len(names), hyp.params, seed=seed)
    lr = float(hyp.params.get("lr", 0.01))
    steps = min(max(50, int(hyp.params.get("steps", 800))), 5000)
    l2 = float(hyp.params.get("l2", 1e-4))
    batch = int(hyp.params.get("batch_size", 64))
    diag = mlp.train(Xn, y, lr=lr, steps=steps, l2=l2, batch_size=batch, seed=seed)

    hyp.kind = "mlp"
    hyp.pool = "nn"
    hyp.feature_names = names
    hyp.weights = None
    hyp.mlp_state = mlp.to_state()
    hyp.params["hidden_layers"] = mlp.layer_sizes[1:-1]
    hyp.params["activation"] = mlp.activation
    hyp.params["train_n"] = diag.get("train_n")
    hyp.params["train_mse"] = diag.get("final_loss")
    hyp.params["train_backend"] = "torch"
    hyp.params["steps"] = steps
    hyp.params["_feat_mean"] = mu
    hyp.params["_feat_std"] = sd
    hyp.params["architecture"] = "→".join(str(s) for s in mlp.layer_sizes)
    hyp.ready = True
    hyp.notes = (
        f"mlp_arch={hyp.params['architecture']} act={mlp.activation} "
        f"backend=torch n={diag.get('train_n')} mse={diag.get('final_loss')}"
    )
    return hyp


def seed_param_pool() -> list[Hypothesis]:
    seeds = [
        Hypothesis("param_zero_v0", "param", "zero", notes="always 0% change"),
        Hypothesis("param_last_mom_v0", "param", "last_mom", {"scale": 1.0}),
        Hypothesis("param_last_mom_damp_v0", "param", "last_mom", {"scale": 0.5}),
        Hypothesis("param_yoy_scaled_v0", "param", "last_yoy_scaled", {"scale": 1.0}),
        Hypothesis("param_blend_50_v0", "param", "blend_mom_yoy", {"w_mom": 0.5, "scale": 1.0}),
        Hypothesis("param_blend_70mom_v0", "param", "blend_mom_yoy", {"w_mom": 0.7, "scale": 1.0}),
        Hypothesis("param_mean_rev_v0", "param", "mean_reversion", {"strength": 0.25}),
        Hypothesis("param_rate_penalty_v0", "param", "rate_penalty", {"rate_coef": 0.15, "scale": 1.0}),
        Hypothesis("param_momentum_lag_v0", "param", "momentum_lag", {"w1": 0.5, "w2": 0.3, "w3": 0.2, "scale": 1.0}),
        Hypothesis("param_ridge_v0", "param", "ridge_linear", {"l2": 1e-3}, ready=False),
    ]
    return seeds


def seed_nn_pool() -> list[Hypothesis]:
    """NN pool seeds: linear heads + real MLPs."""
    return [
        Hypothesis("nn_linear_l2_1e3_v0", "nn", "linear_nn", {"l2": 1e-3, "lr": 0.02, "steps": 400}, ready=False),
        Hypothesis("nn_linear_slow_v0", "nn", "linear_nn", {"l2": 5e-3, "lr": 0.01, "steps": 600}, ready=False),
        Hypothesis(
            "nn_mlp_16_8_v0",
            "nn",
            "mlp",
            {
                "hidden_layers": [16, 8],
                "activation": "relu",
                "lr": 0.01,
                "steps": 700,
                "l2": 1e-4,
                "batch_size": 64,
                "seed": 1,
            },
            ready=False,
        ),
        Hypothesis(
            "nn_mlp_32_16_v0",
            "nn",
            "mlp",
            {
                "hidden_layers": [32, 16],
                "activation": "relu",
                "lr": 0.008,
                "steps": 800,
                "l2": 5e-5,
                "batch_size": 64,
                "seed": 2,
            },
            ready=False,
        ),
        Hypothesis(
            "nn_mlp_8_tanh_v0",
            "nn",
            "mlp",
            {
                "hidden_layers": [8],
                "activation": "tanh",
                "lr": 0.015,
                "steps": 600,
                "l2": 1e-4,
                "batch_size": 32,
                "seed": 3,
            },
            ready=False,
        ),
    ]


def mutate_param(parent: Hypothesis, rng: random.Random, cycle: int, serial: int) -> Hypothesis:
    """Spawn a child by perturbing params or mixing kinds."""
    new_id = f"{parent.pool}_mut_c{cycle}_{serial}"
    child = parent.clone(new_id)
    child.lineage = list(parent.lineage)

    if parent.kind in ("last_mom", "last_yoy_scaled"):
        scale = float(parent.params.get("scale", 1.0))
        child.params["scale"] = max(0.05, scale * rng.uniform(0.6, 1.4))
    elif parent.kind == "blend_mom_yoy":
        child.params["w_mom"] = min(0.95, max(0.05, float(parent.params.get("w_mom", 0.5)) + rng.uniform(-0.2, 0.2)))
        child.params["scale"] = max(0.05, float(parent.params.get("scale", 1.0)) * rng.uniform(0.7, 1.3))
    elif parent.kind == "mean_reversion":
        child.params["strength"] = max(0.05, float(parent.params.get("strength", 0.3)) * rng.uniform(0.5, 1.5))
    elif parent.kind == "rate_penalty":
        child.params["rate_coef"] = max(0.01, float(parent.params.get("rate_coef", 0.15)) * rng.uniform(0.5, 1.5))
        child.params["scale"] = max(0.05, float(parent.params.get("scale", 1.0)) * rng.uniform(0.7, 1.3))
    elif parent.kind == "momentum_lag":
        child.params["w1"] = rng.uniform(0.3, 0.7)
        child.params["w2"] = rng.uniform(0.1, 0.4)
        child.params["w3"] = rng.uniform(0.05, 0.3)
        child.params["scale"] = max(0.05, float(parent.params.get("scale", 1.0)) * rng.uniform(0.7, 1.3))
    elif parent.kind in ("ridge_linear", "linear_nn"):
        child.params["l2"] = 10 ** rng.uniform(-4, -1.5)
        child.params["lr"] = rng.uniform(0.005, 0.05)
        child.params["steps"] = int(rng.uniform(300, 700))
        child.ready = False
        child.weights = None
    elif parent.kind == "mlp":
        hl = list(parent.params.get("hidden_layers") or [16, 8])
        if hl and rng.random() < 0.7:
            i = rng.randrange(len(hl))
            hl[i] = int(max(4, min(256, hl[i] + rng.choice([-16, -8, -4, 4, 8, 16]))))
        elif rng.random() < 0.3 and len(hl) < 4:
            hl.append(max(4, hl[-1] // 2))
        child.kind = "mlp"
        child.params["hidden_layers"] = hl
        child.params["activation"] = parent.params.get("activation", "relu")
        if rng.random() < 0.25:
            child.params["activation"] = "tanh" if child.params["activation"] == "relu" else "relu"
        child.params["lr"] = float(parent.params.get("lr", 0.01)) * rng.uniform(0.6, 1.4)
        child.params["steps"] = int(max(200, min(5000, int(parent.params.get("steps", 800)) * rng.uniform(0.8, 1.3))))
        child.params["l2"] = float(parent.params.get("l2", 1e-4)) * rng.uniform(0.5, 2.0)
        child.params["batch_size"] = int(parent.params.get("batch_size", 64))
        child.ready = False
        child.weights = None
        child.mlp_state = None
    elif parent.kind == "zero":
        child.kind = "constant"
        child.params["c"] = rng.uniform(-0.2, 0.2)
    else:
        child.params["scale"] = rng.uniform(0.3, 1.2)
    child.notes = f"mutate from {parent.model_id}"
    return child


def crossover(a: Hypothesis, b: Hypothesis, rng: random.Random, cycle: int, serial: int) -> Hypothesis:
    """Mix two hypotheses (same or cross pool → child stays in a.pool)."""
    new_id = f"{a.pool}_x_c{cycle}_{serial}"
    # prefer kind from fitter parent if lineage scores exist
    pick = a if rng.random() < 0.5 else b
    child = pick.clone(new_id)
    child.parent_ids = [a.model_id, b.model_id]
    child.pool = a.pool
    # blend numeric params
    params = {}
    keys = set(a.params) | set(b.params)
    for k in keys:
        va, vb = a.params.get(k), b.params.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            params[k] = 0.5 * float(va) + 0.5 * float(vb)
        else:
            params[k] = va if rng.random() < 0.5 else vb
    child.params = {k: v for k, v in params.items() if v is not None}
    if child.kind in ("ridge_linear", "linear_nn", "mlp"):
        child.ready = False
        child.weights = None
        if child.kind == "mlp":
            child.mlp_state = None
    child.notes = f"crossover {a.model_id} x {b.model_id}"
    return child


def prepare_model(
    hyp: Hypothesis,
    train_labels: list[dict],
    feature_index: dict[tuple[str, str], dict],
) -> Hypothesis:
    """Ensure hypothesis is ready for inference (train linear/MLP if needed)."""
    has_hidden = bool(hyp.params.get("hidden_layers") or hyp.params.get("hidden"))

    # Param pool must never train an MLP — demote to ridge linear.
    if hyp.pool == "param" and (hyp.kind in ("mlp", "linear_nn") or has_hidden):
        hyp.kind = "ridge_linear"
        hyp.pool = "param"
        hyp.mlp_state = None
        hyp.params.pop("hidden_layers", None)
        hyp.params.pop("hidden", None)
        hyp.params.pop("activation", None)
        return fit_linear(
            hyp,
            train_labels,
            feature_index,
            l2=float(hyp.params.get("l2", 1e-3)),
            steps=int(hyp.params.get("steps", 400)),
            lr=float(hyp.params.get("lr", 0.02)),
        )

    # Already trained exact copy / carried-forward model — do not re-fit.
    if hyp.ready and hyp.kind == "mlp" and hyp.mlp_state:
        hyp.pool = "nn"
        return hyp
    if hyp.ready and hyp.kind in ("ridge_linear", "linear_nn") and hyp.weights and hyp.feature_names:
        return hyp

    # NN pool: real MLP when requested (or architecture params present)
    if hyp.pool == "nn" and (hyp.kind == "mlp" or (not hyp.ready and has_hidden)):
        hyp.kind = "mlp"
        hyp.pool = "nn"
        return fit_mlp(hyp, train_labels, feature_index)

    if hyp.kind == "mlp":
        # Untagged pool but kind=mlp → nn
        hyp.pool = "nn"
        return fit_mlp(hyp, train_labels, feature_index)

    if hyp.kind in ("ridge_linear", "linear_nn") or not hyp.ready:
        if hyp.kind == "linear_nn":
            hyp.pool = "nn"
        elif hyp.kind == "ridge_linear":
            hyp.pool = "param"
        l2 = float(hyp.params.get("l2", 1e-3))
        steps = int(hyp.params.get("steps", 400))
        lr = float(hyp.params.get("lr", 0.02))
        return fit_linear(hyp, train_labels, feature_index, l2=l2, steps=steps, lr=lr)
    hyp.ready = True
    return hyp


def generate_predictions(
    hyp: Hypothesis,
    labels: list[dict],
    feature_index: dict[tuple[str, str], dict],
) -> list[Prediction]:
    preds = []
    for lab in labels:
        prow = feature_index.get((lab["area"], lab["as_of_date"][:10]))
        p = predict_label(hyp, lab, prow)
        if p is not None:
            preds.append(p)
    return preds
