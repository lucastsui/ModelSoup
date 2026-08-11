#!/usr/bin/env python3
"""Probe Claude CLI, Grok CLI, and DeepSeek-via-SSH backends."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_access import load_config
from llm_models import default_registry, probe_backends


def main():
    cfg = {}
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    reg = default_registry(cfg)
    rows = probe_backends(reg)
    print(json.dumps(rows, indent=2))
    ok = sum(1 for r in rows if r["ok"])
    print(f"\n{ok}/{len(rows)} backends OK", file=sys.stderr)
    sys.exit(0 if ok >= 1 else 1)


if __name__ == "__main__":
    main()
