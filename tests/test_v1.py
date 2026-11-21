"""v1 (cached decode) must match v0 (full recompute); then time eager decode."""
import sys, time, torch
sys.path.insert(0, "/tmp/aig")
from tokenizers import Tokenizer
from engine.model_v0 import InklingV0
from engine.model_v1 import InklingV1
tok = Tokenizer.from_file("/tmp/inkling/tokenizer.json")
m = InklingV1()
prompt = "Explain step by step why the sky is blue, in simple words."
ids = torch.tensor(tok.encode(prompt).ids, device="cuda")
# 1) correctness: incremental vs full recompute, 24 greedy steps
logits, cache = m.forward(ids)
seq = ids.clone(); maxdiff = 0.0; agree = 0; N = 24
for s in range(N):
    ref = InklingV0.forward(m, seq)[-1]
    maxdiff = max(maxdiff, (logits[-1] - ref).abs().max().item())
    a, b = logits[-1].argmax(), ref.argmax()
    agree += int(a == b)
    seq = torch.cat([seq, b[None]])
    logits, cache = m.forward(b[None], cache)
print(f"cached vs full: argmax agree {agree}/{N}, max |logit diff| {maxdiff:.4f}", flush=True)
print("text:", repr(tok.decode(seq[len(ids):].tolist())), flush=True)
# 2) eager decode speed (batch 1)
for n in (64,):
    torch.cuda.synchronize(); t = time.time()
    out = m.generate(ids, n); torch.cuda.synchronize()
    dt = time.time() - t
    print(f"eager generate {n} tokens: {dt:.2f}s = {n/dt:.1f} tok/s (incl. prefill of {len(ids)})", flush=True)
logits, cache = m.forward(ids); torch.cuda.synchronize(); t = time.time()
x = logits[-1].argmax()
for _ in range(32):
    logits, cache = m.forward(x[None], cache); x = logits[-1].argmax()
torch.cuda.synchronize(); dt = (time.time() - t) / 32
print(f"pure decode step: {dt*1000:.1f} ms = {1/dt:.1f} tok/s", flush=True)
