"""Confirm a decode change on one GPU in control/candidate/candidate/control order."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def referee_files(root):
    files = [root / "prepare_gpu.py"]
    for directory in ("golden", "ref", "data"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): digest(p) for p in files}


def measure(root, log):
    with log.open("x") as stream:
        subprocess.run([sys.executable, "-u", "prepare_gpu.py"], cwd=root, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
    rows = [json.loads(line[7:]) for line in log.read_text().splitlines() if line.startswith("RESULT ")]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one benchmark result in {log}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior-best", type=float, required=True)
    args = parser.parse_args()
    if referee_files(args.control) != referee_files(args.candidate):
        raise ValueError("Referee inputs differ between control and candidate")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {"control": [], "candidate": []}
    for index, role in enumerate(("control", "candidate", "candidate", "control")):
        result = measure(getattr(args, role), args.output / f"{index}-{role}.log")
        runs[role].append(result)
        print(role, result["score_tps"], result["eval_ok"], flush=True)
    best = max(args.prior_best, statistics.median(r["score_tps"] for r in runs["control"]))
    quality = all(r["eval_ok"] == "12/12" and all(r["gates"].values())
                  for group in runs.values() for r in group)
    keep = quality and all(r["score_tps"] > best * (1 + max(0.015, 2 * r["noise"]))
                           for r in runs["candidate"])
    hashes = {role: {p.name: digest(p) for p in sorted((getattr(args, role) / "engine").glob("*.py"))}
              for role in runs}
    record = dict(runs=runs, control_bar_base=best, keep=keep, engine_sha256=hashes)
    (args.output / "result.json").write_text(json.dumps(record, indent=2))
    print("KEEP" if keep else "DISCARD", flush=True)


if __name__ == "__main__":
    main()
