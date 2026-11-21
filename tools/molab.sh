#!/bin/bash
# Run python on the MoLab kernel. Needs .molab (gitignored): MOLAB_URL=... and MOLAB_TOKEN_FILE=...
# usage: tools/molab.sh - <<'PY' ... PY      (stdin python)
cd "$(dirname "$0")/.." && source .molab
MARIMO_TOKEN=$(cat "$MOLAB_TOKEN_FILE") bash ~/.claude/skills/marimo-pair/scripts/execute-code.sh --url "$MOLAB_URL" "$@" 2>&1 | grep -v "non-local server"
