"""
PyTorch multilayer perceptron for the nn hypothesis pool.

Weights serialize to nested Python lists so pools/CSV stay JSON-friendly.
Requires: torch (see PropView/requirements.txt and .venv).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

BACKEND = "torch"


class TorchMLP(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: str = "relu"):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name == "tanh":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def export_lists(self) -> tuple[list[list[list[float]]], list[list[float]]]:
        weights: list[list[list[float]]] = []
        biases: list[list[float]] = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.detach().cpu().tolist())
                biases.append(m.bias.detach().cpu().tolist())
        return weights, biases

    def load_lists(self, weights: list, biases: list) -> None:
        li = 0
        for m in self.net:
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.copy_(torch.tensor(weights[li], dtype=torch.float32))
                    m.bias.copy_(torch.tensor(biases[li], dtype=torch.float32))
                li += 1


@dataclass
class MLP:
    """Fully connected MLP: input -> hidden... -> 1 output (regression)."""

    layer_sizes: list[int]  # e.g. [F, 16, 8, 1]
    activation: str = "relu"
    weights: list[list[list[float]]] = field(default_factory=list)  # W[layer][out][in]
    biases: list[list[float]] = field(default_factory=list)
    seed: int = 0
    backend: str = BACKEND

    def _module(self) -> TorchMLP:
        model = TorchMLP(self.layer_sizes, self.activation)
        if self.weights and self.biases:
            model.load_lists(self.weights, self.biases)
        return model

    def init_weights(self, rng: random.Random | None = None) -> None:
        del rng  # torch seeding below
        torch.manual_seed(self.seed)
        model = TorchMLP(self.layer_sizes, self.activation)
        self.weights, self.biases = model.export_lists()
        self.backend = BACKEND

    def predict(self, x: list[float]) -> float:
        model = self._module()
        model.eval()
        with torch.no_grad():
            t = torch.tensor([x], dtype=torch.float32)
            return float(model(t).item())

    def train(
        self,
        X: list[list[float]],
        y: list[float],
        lr: float = 0.01,
        steps: int = 500,
        l2: float = 1e-4,
        batch_size: int = 64,
        seed: int = 0,
        clip: float = 5.0,
    ) -> dict[str, Any]:
        n = len(X)
        if n == 0:
            return {"train_n": 0, "final_loss": None, "backend": BACKEND}

        torch.manual_seed(seed)
        model = TorchMLP(self.layer_sizes, self.activation)
        if self.weights:
            model.load_lists(self.weights, self.biases)

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        bs = max(1, min(batch_size, n))
        hist: list[dict[str, Any]] = []
        model.train()
        cur_lr = lr

        for step in range(steps):
            idx = torch.randint(0, n, (bs,))
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if step % max(1, steps // 5) == 0 or step == steps - 1:
                hist.append({"step": step, "mse": float(loss.item())})
            if step > 0 and step % max(100, steps // 4) == 0:
                cur_lr *= 0.7
                for g in opt.param_groups:
                    g["lr"] = cur_lr

        self.weights, self.biases = model.export_lists()
        self.backend = BACKEND
        return {
            "train_n": n,
            "final_loss": hist[-1]["mse"] if hist else None,
            "history": hist,
            "backend": BACKEND,
        }

    def to_state(self) -> dict:
        return {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "weights": self.weights,
            "biases": self.biases,
            "seed": self.seed,
            "backend": self.backend,
        }

    @classmethod
    def from_state(cls, state: dict) -> "MLP":
        m = cls(
            layer_sizes=list(state["layer_sizes"]),
            activation=state.get("activation", "relu"),
            seed=int(state.get("seed", 0)),
            backend=state.get("backend", BACKEND),
        )
        m.weights = state["weights"]
        m.biases = state["biases"]
        return m


def build_mlp_from_params(n_features: int, params: dict, seed: int = 0) -> MLP:
    hidden = params.get("hidden_layers") or params.get("hidden") or [16, 8]
    if isinstance(hidden, str):
        hidden = [int(x) for x in hidden.replace("[", "").replace("]", "").split(",") if x.strip()]
    hidden = [int(h) for h in hidden if int(h) > 0]
    if not hidden:
        hidden = [16, 8]
    # practical caps for proposal search
    hidden = [min(256, max(2, h)) for h in hidden[:6]]
    act = str(params.get("activation", "relu")).lower()
    if act not in ("relu", "tanh"):
        act = "relu"
    sizes = [n_features] + hidden + [1]
    mlp = MLP(layer_sizes=sizes, activation=act, seed=seed)
    mlp.init_weights()
    return mlp
