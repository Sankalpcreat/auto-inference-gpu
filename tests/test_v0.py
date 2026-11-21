import sys, time, math, torch
sys.path.insert(0, "/tmp/aig")
from tokenizers import Tokenizer
from engine.model_v0 import InklingV0
tok = Tokenizer.from_file("/tmp/inkling/tokenizer.json")
t0 = time.time(); m = InklingV0(); print(f"loaded in {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
text = ("The Industrial Revolution began in Great Britain in the late eighteenth century. It transformed the way goods "
        "were produced, replacing hand tools with machines powered by water and steam. Factories grew rapidly in cities "
        "such as Manchester, and the population moved from the countryside to urban areas in search of work.")
ids = torch.tensor(tok.encode(text).ids, device="cuda")
t0 = time.time(); lg = m.forward(ids); torch.cuda.synchronize()
nll = torch.nn.functional.cross_entropy(lg[:-1], ids[1:])
print(f"tokens {len(ids)}  ppl {math.exp(nll.item()):.2f}  forward {time.time()-t0:.1f}s", flush=True)
for p in ["The capital of France is", "def fibonacci(n):\n    "]:
    ids = torch.tensor(tok.encode(p).ids, device="cuda")
    out = m.generate(ids, 20)
    print(repr(p), "->", repr(tok.decode(out[len(ids):].tolist())), flush=True)
