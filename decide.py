"""FROZEN. Mac-side decision: reads a RESULT json, compares with results.tsv history, prints DECISION."""
import json, sys, time
from pathlib import Path
MIN_GAIN = 0.015
desc, res = sys.argv[1], json.load(open(sys.argv[2]))
root = Path(__file__).parent
tsv = root / "results.tsv"
hdr = "id\tcommit\tstatus\tscore_tps\tnoise\teval\tnll_ratio\tlong_ttft_ms\tpeak_gb\tdescription\n"
rows = [l.split("\t") for l in tsv.read_text().splitlines()[1:]] if tsv.exists() else []
best = max((float(r[3]) for r in rows if r[2] in ("BASELINE", "KEEP")), default=None)
gates_ok = all(res["gates"].values())
import subprocess
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
s, noise = res["score_tps"], res["noise"]
if not gates_ok:
    status = "FAIL-GATE"
elif best is None:
    status = "BASELINE"
else:
    status = "KEEP" if s > best * (1 + max(MIN_GAIN, 2 * noise)) else "DISCARD"
rid = time.strftime("%Y%m%d-%H%M%S")
res.update(id=rid, commit=commit, status=status, description=desc, best_before=best)
(root / "runs").mkdir(exist_ok=True)
(root / "runs" / f"{rid}.json").write_text(json.dumps(res, indent=1))
if not tsv.exists():
    tsv.write_text(hdr)
with tsv.open("a") as f:
    f.write(f"{rid}\t{commit}\t{status}\t{s}\t{noise}\t{res['eval_ok']}\t{res['nll_ratio']}\t{res['workloads']['long']['ttft_ms']}\t{res['peak_gb']}\t{desc}\n")
for k, v in res["workloads"].items():
    print(f"  {k:10s} prompt {v['prompt_tokens']:5d} | TTFT {v['ttft_ms']:8.1f} ms | decode {v['decode_tps']:7.2f} tok/s | golden-prefix {v['matches_golden_prefix']}")
for g, ok in res["gates"].items():
    print(f"  gate {'PASS' if ok else 'FAIL'}  {g}")
print(f"score {s} tok/s  noise ±{noise*100:.1f}%  eval {res['eval_ok']}  nll_ratio {res['nll_ratio']}  peak {res['peak_gb']} GB")
if best is not None:
    print(f"  best kept so far: {best} tok/s; keep bar {best * (1 + max(MIN_GAIN, 2 * noise)):.2f}")
print(f"DECISION: {status}")
