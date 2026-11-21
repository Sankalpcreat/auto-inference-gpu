#!/bin/bash
# Run a profiling/scratch script on the GPU with the current engine/:  tools/gpu_script.sh scratch/profile.py [timeout_s]
# Output -> stdout (also logs/gpu_script.log). Script runs from /tmp/aig (import engine.*, ref.*; weights at /tmp/inkling).
set -u
cd "$(dirname "$0")/.."
SCRIPT=$1; TMO=${2:-900}
PAYLOAD=$(COPYFILE_DISABLE=1 tar czf - --exclude='__pycache__' engine ref data scratch | base64)
tools/molab.sh - > logs/gpu_start.log <<PY
import base64, io, tarfile, subprocess, shutil
shutil.rmtree('/tmp/aig/engine', ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("""$PAYLOAD""")), mode='r:gz').extractall('/tmp/aig', filter='data')
subprocess.Popen("cd /tmp/aig && setsid timeout $TMO python $SCRIPT > /tmp/aig/script.log 2>&1; echo SCRIPT_EXIT \$? >> /tmp/aig/script.log", shell=True)
print("STARTED")
PY
grep -q STARTED logs/gpu_start.log || { cat logs/gpu_start.log; exit 1; }
while true; do
  sleep 20
  tools/molab.sh - > logs/gpu_script.log <<'PY'
print(open('/tmp/aig/script.log', errors='replace').read()[-40000:])
PY
  grep -q SCRIPT_EXIT logs/gpu_script.log && break
done
cat logs/gpu_script.log
