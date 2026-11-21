"""FROZEN HARNESS — runs ON THE GPU. Measures engine/infer.py, checks quality, prints one JSON line (RESULT {...}).
The KEEP/DISCARD decision is made on the Mac by decide.py (history lives there). Agents never edit this file.

Speed: per workload 1 untimed warm-up, then TTFT (max_tokens=1) and full runs, median of REPEATS, timed outside the
engine with cuda.synchronize. decode tok/s = (n-1) / (T_full - T_ttft). Score = geomean over workloads.
Quality (vs golden.json, made once by the frozen reference engine in ref/): eval items, NLL on fixed texts, fixed work,
determinism.
"""
import importlib, json, math, os, statistics, sys, time
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
TOK = Tokenizer.from_file("/tmp/inkling/tokenizer.json")
ARTICLE = open(os.path.join(ROOT, "data/article.txt")).read()
REPEATS = int(os.environ.get("REPEATS", 3))

WORKLOADS = {  # name: (prompt, max_tokens)
    "chat": ("Q: Give three practical tips for staying focused while studying.\nA:", 64),
    "long": (ARTICLE + "\n\nSummary of the article in plain words:", 64),
    "code": ("# Python: parse a CSV file and print the average of the 'price' column\nimport csv\n", 64),
    "reasoning": ("Q: A train leaves at 9:40 and arrives at 13:05. How long is the trip? Think step by step.\nA:", 128),
}
EVAL = [  # (prompt, max_tokens, check)
    ("Q: What is the capital of Japan?\nA:", 8, lambda s: "Tokyo" in s),
    ("Q: What is 17 + 25?\nA:", 8, lambda s: "42" in s),
    ("Q: What is the chemical symbol for gold?\nA:", 8, lambda s: "Au" in s),
    ("Q: Who wrote Romeo and Juliet?\nA:", 8, lambda s: "Shakespeare" in s),
    ("Q: How many days are in a leap year?\nA:", 8, lambda s: "366" in s),
    ("Q: What is the largest planet in the solar system?\nA:", 8, lambda s: "Jupiter" in s),
    ("Q: What is 12 times 12?\nA:", 8, lambda s: "144" in s),
    ("Q: On which continent is Kenya?\nA:", 8, lambda s: "Africa" in s),
    ("def add(a, b):\n    return", 6, lambda s: "a + b" in s or "a+b" in s),
    ("Q: Which gas do plants absorb from the air for photosynthesis?\nA:", 10, lambda s: "carbon dioxide" in s.lower() or "CO2" in s),
    (ARTICLE + "\n\nQ: What was the codename of the test lab in the article?\nA:", 10, lambda s: "BLUE HERON" in s.upper()),
    (ARTICLE + "\n\nQ: According to the article, what was the average acceptance length for code?\nA:", 12, lambda s: "2.7" in s),
]
NLL_TEXTS = [
    "The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods were produced, replacing hand tools with machines powered by water and steam.",
    "def binary_search(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x:\n            return mid\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "Photosynthesis converts light energy into chemical energy. Plants take in carbon dioxide and water and, using sunlight, produce glucose and release oxygen into the atmosphere.",
    ARTICLE[:3000],
]
enc = lambda s: TOK.encode(s).ids
dec = lambda ids: TOK.decode(ids)


def sync():
    torch.cuda.synchronize()


def timed(eng, st, ids, n):
    sync(); t = time.perf_counter(); out = eng.generate(st, ids, n); sync()
    return time.perf_counter() - t, out


def run_eval(eng, st):
    res = []
    for p, n, check in EVAL:
        s = dec(eng.generate(st, enc(p), n))
        res.append({"q": p[-70:], "a": s, "ok": bool(check(s))})
    return res


def main(mode):
    eng = importlib.import_module("ref.infer_ref" if mode == "golden" else "engine.infer")
    t0 = time.time(); st = eng.load(); sync(); load_s = time.time() - t0
    out = {"load_s": round(load_s, 1), "workloads": {}}
    if mode == "golden":
        out["eval"] = run_eval(eng, st)
        out["nll"] = [eng.nll(st, enc(t)) for t in NLL_TEXTS]
        out["outputs"] = {k: eng.generate(st, enc(p), n) for k, (p, n) in WORKLOADS.items()}
        print("RESULT " + json.dumps(out)); return
    golden = json.load(open(os.path.join(ROOT, "golden/golden.json")))
    gates, outs = {}, {}
    for name, (p, n) in WORKLOADS.items():
        ids = enc(p)
        eng.generate(st, ids, n)                                   # warm-up (untimed)
        ttft = statistics.median(timed(eng, st, ids, 1)[0] for _ in range(REPEATS))
        fulls, gens = [], []
        for _ in range(REPEATS):
            dt, o = timed(eng, st, ids, n); fulls.append(dt); gens.append(list(o))
        tps = [(n - 1) / max(f - ttft, 1e-9) for f in fulls]
        outs[name] = gens[0]
        out["workloads"][name] = {"prompt_tokens": len(ids), "ttft_ms": round(ttft * 1000, 1),
                                  "decode_tps": round(statistics.median(tps), 2),
                                  "spread": round(statistics.pstdev(tps) / statistics.mean(tps), 4),
                                  "deterministic": all(g == gens[0] for g in gens), "gen_len": len(gens[0]),
                                  "matches_golden_prefix": next((i for i, (a, b) in enumerate(zip(gens[0], golden["outputs"][name])) if a != b), len(gens[0]))}
    w = out["workloads"]
    out["score_tps"] = round(math.exp(statistics.mean(math.log(v["decode_tps"]) for v in w.values())), 2)
    out["noise"] = round(statistics.median(v["spread"] for v in w.values()), 4)
    ev = run_eval(eng, st)
    nll = [eng.nll(st, enc(t)) for t in NLL_TEXTS]
    g_eval = sum(e["ok"] for e in golden["eval"])
    nll_ratio = statistics.mean(a / b for a, b in zip(nll, golden["nll"]))
    gates["deterministic (greedy)"] = all(v["deterministic"] for v in w.values())
    gates["fixed work (exact max_tokens)"] = all(v["gen_len"] == WORKLOADS[k][1] for k, v in w.items())
    gates[f"eval >= golden - 1 ({sum(e['ok'] for e in ev)}/{len(ev)} vs {g_eval})"] = sum(e["ok"] for e in ev) >= g_eval - 1
    gates[f"NLL within +2% of golden (ratio {nll_ratio:.4f})"] = nll_ratio <= 1.02
    out.update(gates=gates, eval=ev, nll=nll, nll_ratio=round(nll_ratio, 4),
               eval_ok=f"{sum(e['ok'] for e in ev)}/{len(ev)}", peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench")
