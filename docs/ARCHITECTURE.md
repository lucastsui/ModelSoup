# PropView architecture (Model Soup)

Terminology (important):

| Term | Meaning |
|------|---------|
| **Model** | LLM proposer (Claude, Grok, DeepSeek, offline heuristic) |
| **Hypothesis** | Prediction algorithm (param rules, linear head, or MLP) |

## System overview

```mermaid
flowchart TB
  subgraph External["External sources"]
    UKHPI["UKHPI / Land Registry"]
    BOE["Bank of England rates"]
    RM["Rightmove pages"]
    NEWS["BBC / Guardian politics RSS"]
  end

  subgraph Collect["Data collector"]
    COL["scripts/collect_full_research_data.py"]
    RD["research_data/\npanels · labels · rates · news"]
  end

  subgraph Loop["Research loop  research_loop/"]
    CFG["config.json"]

    subgraph Pools["Hypothesis pools"]
      PP["param pool\nrules / ridge"]
      NP["nn pool\nlinear_nn / MLP"]
    end

    SC["Scorer\nscorer.py"]
    EV["Evolver\nevolver.py"]
    PR["Proposer\nproposer.py"]

    subgraph LLMs["Models = LLM choosers"]
      CL["Claude CLI"]
      GR["Grok CLI"]
      DS["DeepSeek via SSH"]
      OH["offline heuristic"]
    end

    HYP["Hypotheses\nhypotheses.py · mlp.py"]
    RUNS["runs/run_*/cycle_*/\nscores · proposals · archive"]
  end

  subgraph Viz["Live lineage viz"]
    BL["build_lineage_data.py"]
    LG["lineage_graph.json"]
    WEB["index.html :8765\nD3 chart + live_events"]
  end

  UKHPI --> COL
  BOE --> COL
  RM --> COL
  NEWS --> COL
  COL --> RD
  RD --> SC
  CFG --> EV
  CFG --> SC

  PP --> SC
  NP --> SC
  SC -->|"S = f(MAE, hit, ΔMAE, n)"| EV
  EV -->|"top-n survivors"| PP
  EV -->|"top-n survivors"| NP
  EV --> PR
  PR --> CL & GR & DS & OH
  CL & GR & DS & OH -->|"kind + params JSON"| HYP
  HYP -->|"new offspring"| PP
  HYP -->|"new offspring"| NP
  SC --> RUNS
  PR --> RUNS
  RUNS --> BL --> LG --> WEB
  EV -.->|"live_events.json\nparent highlight"| WEB
```

## One evolution cycle

```mermaid
sequenceDiagram
  participant D as research_data
  participant S as Scorer
  participant E as Evolver
  participant L as LLM models
  participant P as Pools
  participant V as Viz :8765

  Note over P: Generation t on chart as run t
  P->>S: score all hypotheses vs labels
  S->>D: features + actual change %
  S-->>E: ranked S per pool
  E->>E: keep top-n, archive rest
  par Hybrid parallel compute
    E->>L: propose child from parents<br/>(lineage / in_pool / cross_pool)
    L-->>E: hypothesis config
    E->>S: train if needed + score child
  end
  loop Sequential yellow animation
    E->>V: highlight parents on run t
    E->>V: place child on run t+1
    E->>V: clear highlight
  end
  Note over P: Survivors + children → generation t+1
```

## Dual pools and selection

```mermaid
flowchart LR
  subgraph Param["param pool"]
    A1["rules: zero, mom, yoy, blend…"]
    A2["ridge_linear"]
  end
  subgraph NN["nn pool"]
    B1["linear_nn"]
    B2["mlp PyTorch"]
  end

  Param -->|"score · top_n=4"| SP["survivors"]
  NN -->|"score · top_n=3"| SN["survivors"]
  SP --> R1["lineage"]
  SP --> R2["in_pool"]
  SP --> R3["cross_pool"]
  SN --> R1
  SN --> R2
  SN --> R3
  R1 & R2 & R3 --> LLM["Claude / Grok / DeepSeek / offline"]
  LLM --> Kids["new hypotheses"]
  Kids --> Param
  Kids --> NN
```

## Score formula

\[
S = e^{-\mathrm{MAE}/s}\,(1+\alpha h)\,(1+\beta\max(0,-\Delta\mathrm{MAE}))\,(1-e^{-n/n_0})
\]

Falsifiable prediction: 9-tuple (target, aggregation, area, as-of, start, end, change %, tolerance, …).

## Key paths

| Piece | Path |
|-------|------|
| Collector | `scripts/collect_full_research_data.py` |
| Data | `research_data/` |
| Loop | `research_loop/` |
| Full run (hybrid) | `research_loop/run_full_evolution.py` |
| Viz | `research_loop/viz/` → http://127.0.0.1:8765/ |
| Design notes | Obsidian `PropView.md` |
