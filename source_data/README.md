# Source data

All input / collected datasets for Model Soup live under this folder.

| Path | Contents |
|------|----------|
| `*.csv` (root of this folder) | Sample Rightmove-derived CSVs for schema intuition |
| `research_data/` | Full research pack (panels, labels, rates, news) used by the research loop |
| `agent_runs*` / `london_verified/` | Earlier multi-agent collection experiments (local; not all tracked) |

The research loop reads panels via `research_loop/config.json` → `data_root: "../source_data/research_data"`.
