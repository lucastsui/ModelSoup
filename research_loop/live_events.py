"""Write live evolution events for the lineage viz (consider parents → place child)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent / "viz"
EVENTS_PATH = VIZ_DIR / "live_events.json"

_seq = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_event(
    phase: str,
    hypothesis_ids: list[str] | None = None,
    *,
    mode: str = "",
    pool: str = "",
    child_id: str | None = None,
    message: str = "",
    parent_cycle: int | None = None,
    child_cycle: int | None = None,
    path: Path | None = None,
) -> dict:
    """
    phase:
      - considering: framework is looking at these parents to spawn a child
      - idle: clear highlights
      - placed: child just landed (brief; usually followed by idle)

    parent_cycle: which run column the parents live on (highlight those dots,
      not same-id carry-forwards on a later run).
    child_cycle: which run column the new offspring was written to.
    """
    global _seq
    _seq += 1
    payload = {
        "seq": _seq,
        "phase": phase,
        "hypothesis_ids": list(hypothesis_ids or []),
        "mode": mode,
        "pool": pool,
        "child_id": child_id,
        "parent_cycle": parent_cycle,
        "child_cycle": child_cycle,
        "message": message,
        "updated_at": _now(),
    }
    out = path or EVENTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def clear(message: str = "idle") -> dict:
    return write_event("idle", [], message=message)
