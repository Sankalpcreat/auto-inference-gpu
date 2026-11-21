#!/bin/bash
# ./loop.sh N  — N fresh agents, one experiment each. Live log: tail -f logs/agent_<n>.log
set -u
N=${1:-5}; cd "$(dirname "$0")"; mkdir -p logs
MODEL_FLAG="--model ${MODEL:-opus} --effort ${EFFORT:-low}"
for i in $(seq 1 "$N"); do
  echo "=== experiment $i/$N  $(date +%H:%M:%S)" | tee -a logs/loop.log
  claude -p "Follow program.md exactly: read the knowledge files, results.tsv and notes.md first, run exactly ONE experiment (or one profiling study), update notes.md, then stop." $MODEL_FLAG \
    --permission-mode acceptEdits --disallowedTools "Monitor" \
    --allowedTools "Read" "Edit" "Write" "WebSearch" "WebFetch" "Bash(./run_remote.sh*)" "Bash(tools/gpu_script.sh*)" \
                   "Bash(git add*)" "Bash(git commit*)" "Bash(git reset*)" "Bash(git revert*)" "Bash(git stash*)" "Bash(git log*)" "Bash(git diff*)" "Bash(git status*)" \
                   "Bash(tail*)" "Bash(cat*)" "Bash(grep*)" "Bash(ls*)" "Bash(head*)" "Bash(sed -n*)" "Bash(wc*)" "Bash(python3 -c*)" "Bash(mkdir -p scratch)" \
    --output-format stream-json --verbose 2>&1 | tee "logs/agent_$i.jsonl" | python3 tools/live_log.py > "logs/agent_$i.log"
  tail -6 "logs/agent_$i.log" | tee -a logs/loop.log
  # enforce KEEP/DISCARD on engine/ only: it must equal engine/ at the last KEEP/BASELINE commit in results.tsv
  KEEP=$(awk -F'\t' '$3=="KEEP"||$3=="BASELINE"{c=$2} END{print c}' results.tsv)
  if [ -n "$KEEP" ] && ! git diff --quiet "$KEEP" -- engine; then
    git add -A engine && git commit -q -m "wip (unkept)" 2>/dev/null; git branch -f "discarded/$(date +%m%d-%H%M%S)" HEAD
    git checkout "$KEEP" -- engine && git add -A engine && git commit -q -m "loop: revert engine/ to last KEEP $KEEP"
    echo "loop: engine/ reverted to last KEEP $KEEP (unkept work saved on a discarded/ branch)" | tee -a logs/loop.log
  fi
  if grep '"type":"result"' "logs/agent_$i.jsonl" | tail -1 | grep -qE '"api_error_status":429|"api_error":"usage_limit_reached"'; then
    echo "!!! usage limit — stopping; restart with ./loop.sh $((N - i + 1))" | tee -a logs/loop.log; exit 2; fi
  if grep -qE "REMOTE START FAILED|TIMEOUT after" logs/last_out.log 2>/dev/null; then
    echo "!!! MoLab sandbox unreachable — stopping" | tee -a logs/loop.log; exit 3; fi
  if ! git diff --quiet HEAD -- prepare_gpu.py decide.py run_remote.sh tools ref golden data knowledge program.md reference; then
    echo "!!! protected file modified — stopping" | tee -a logs/loop.log; git checkout -- prepare_gpu.py decide.py run_remote.sh tools ref golden data knowledge program.md reference; exit 1; fi
done
echo "=== done $(date +%H:%M:%S)" | tee -a logs/loop.log
