#!/bin/bash
# Agent entry point: ./run_remote.sh "<one-line description>"   (or: ./run_remote.sh --golden)
# Syncs engine/ ref/ data/ golden/ prepare_gpu.py to MoLab, runs the frozen harness on the GPU, polls, then decide.py.
set -u
cd "$(dirname "$0")"
DESC=${1:-"(no description)"}; MODE=bench; [ "$DESC" = "--golden" ] && MODE=golden
mkdir -p logs runs golden
PAYLOAD=$(COPYFILE_DISABLE=1 tar czf - --exclude='__pycache__' engine ref data golden prepare_gpu.py | base64)
tools/molab.sh - > logs/remote_start.log <<PY
import base64, io, tarfile, subprocess, shutil, os
shutil.rmtree('/tmp/aig/engine', ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("""$PAYLOAD""")), mode='r:gz').extractall('/tmp/aig', filter='data')
subprocess.run("pkill -f prepare_gpu.py", shell=True)
subprocess.Popen("cd /tmp/aig && setsid python prepare_gpu.py $MODE > /tmp/aig/run.log 2>&1; echo HARNESS_EXIT \$? >> /tmp/aig/run.log", shell=True)
print("STARTED")
PY
grep -q STARTED logs/remote_start.log || { echo "REMOTE START FAILED:"; cat logs/remote_start.log; exit 1; }
START=$(date +%s)
while true; do
  sleep 30
  tools/molab.sh - > logs/remote_poll.log <<'PY'
print(open('/tmp/aig/run.log', errors='replace').read()[-60000:])
PY
  grep -q "HARNESS_EXIT" logs/remote_poll.log && break
  [ $(( $(date +%s) - START )) -gt 2400 ] && { echo "TIMEOUT after 40 min"; exit 1; }
done
cp logs/remote_poll.log logs/last_run.log
if [ "$MODE" = golden ]; then
  grep "^RESULT " logs/last_run.log | sed 's/^RESULT //' > golden/golden.json && echo "golden written: $(wc -c < golden/golden.json) bytes"; exit 0
fi
grep "^RESULT " logs/last_run.log | sed 's/^RESULT //' > logs/last_result.json
[ -s logs/last_result.json ] || { echo "CRASH — no RESULT. Tail:"; tail -30 logs/last_run.log; exit 1; }
python3 decide.py "$DESC" logs/last_result.json
