#!/usr/bin/env bash
# Serve the lineage viz and rebuild lineage_graph.json when run data changes.
set -euo pipefail
cd "$(dirname "$0")"

python3 build_lineage_data.py
echo "Open http://127.0.0.1:8765/  (page auto-refreshes data; lineage rebuilds every ~3s)"

# Background: rebuild graph whenever scores/proposals under ../runs change
(
  prev=""
  while true; do
    # Fingerprint mtimes + sizes of run artifacts
    sig=$(find ../runs -type f \( -name 'scores.csv' -o -name 'proposals.csv' -o -name 'proposal_forecasts.csv' -o -name 'RUN_SUMMARY.json' \) -print0 2>/dev/null \
      | xargs -0 stat -f '%N %m %z' 2>/dev/null | sort | shasum -a 256 | awk '{print $1}')
    if [[ "$sig" != "$prev" ]]; then
      # build_lineage_data refuses to overwrite a good graph with empty
      python3 build_lineage_data.py >/dev/null 2>&1 || true
      prev="$sig"
    fi
    sleep 3
  done
) &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null || true' EXIT

python3 -m http.server 8765
