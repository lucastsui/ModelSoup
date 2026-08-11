#!/usr/bin/env python3
"""Run scorer + dual-pool evolution smoke cycle on research_data panels."""
from __future__ import annotations

import sys
from pathlib import Path

# allow local imports when run as script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolver import run_cycles


def main():
    config = ROOT / "config.json"
    cycles = None
    if len(sys.argv) > 1:
        cycles = int(sys.argv[1])
    summary = run_cycles(config, cycles=cycles)
    print("=== SMOKE CYCLE COMPLETE ===")
    print(f"run_dir: {summary['run_dir']}")
    print(f"labels: {summary['labels_used']} (train={summary['train']}, val={summary['validation']})")
    for h in summary["history"]:
        print(
            f"cycle {h['cycle']}: "
            f"param={h['best_param_id']} S={h['best_param_S']:.4f} MAE={h['best_param_MAE']:.4f} hit={h['best_param_hit_rate']:.3f} | "
            f"nn={h['best_nn_id']} S={h['best_nn_S']:.4f} MAE={h['best_nn_MAE']:.4f} hit={h['best_nn_hit_rate']:.3f}"
        )
    print(f"report: {summary['run_dir']}/REPORT.md")


if __name__ == "__main__":
    main()
