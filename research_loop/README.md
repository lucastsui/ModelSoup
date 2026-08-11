# PropView research loop

Aligned with Obsidian `PropView.md` terminology:

| Term | Meaning | Code |
|------|---------|------|
| **Model** | LLM (Opus, DeepSeek, Grok, local vLLM, offline heuristic) | `llm_models.py` registry |
| **Hypothesis** | Prediction algorithm (params / linear / **MLP** → change %) | `hypotheses.py`, `mlp.py` |
| **Proposer** | Asks an LLM model for a new hypothesis config | `proposer.py` modes: lineage / in_pool / cross_pool |
| **Scorer** | \(S = e^{-\mathrm{MAE}/s}(1+\alpha h)(1+\beta\max(0,-\Delta\mathrm{MAE}))(1-e^{-n/n_0})\) | `scorer.py` |
| **Evolver** | Dual pools (param vs nn), top‑n keep, archive rest, LLM refill | `evolver.py` |

## Setup

From the PropView root (creates `.venv` if needed):

```bash
cd /Users/tsuimingleong/Desktop/PropView
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # numpy + torch for real MLP training
```

MLP training is **PyTorch-only** (Adam + mini-batches). Install deps via `requirements.txt`.

## Run

```bash
cd research_loop
# use project venv so torch/numpy resolve
../.venv/bin/python run_smoke_cycle.py
../.venv/bin/python run_smoke_cycle.py 3
```

### LLM backends (no paid cloud APIs)

| model_id | Backend |
|----------|---------|
| `claude` | Claude Code CLI: `claude -p` (subscription OAuth) |
| `grok` | Grok Build CLI: `grok --single` (session auth) |
| `deepseek` | SSH `anaclast@100.73.106.98` → local llama-server `http://127.0.0.1:8080/v1` |
| `offline_heuristic` | Fallback if CLI/SSH fails |

Probe backends:
```bash
python3 -c "from llm_models import probe_backends; import json; print(json.dumps(probe_backends(), indent=2))"
```

Config overrides: `deepseek_ssh_host`, `deepseek_model`, `prefer_llms` in `config.json`.

## Proposal modes

1. **lineage** — improve/diversify from one survivor’s score history  
2. **in_pool** — two random hypotheses in the same pool  
3. **cross_pool** — one from param + one from nn  

## Cycle artifacts

`runs/run_*/`

- `llm_model_registry.csv` — which LLMs were registered/available  
- `cycle_XX/scores.csv` — hypothesis scores  
- `cycle_XX/proposals.csv` — who proposed what (llm_model_id, mode, used_api)  
- `cycle_XX/archived.csv` — culled hypotheses  
- `REPORT.md` — summary  

## Data

Uses `../research_data` panels + backtest labels from the collector.
