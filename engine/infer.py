"""THE AGENT'S FILE (plus anything else under engine/ it creates).

Contract with prepare_gpu.py (do not change the signatures):
    load() -> state                                   called once
    generate(state, prompt_ids: list[int], max_tokens: int) -> list[int]
        greedy; must return exactly max_tokens ids (ignore EOS); deterministic across repeats
    nll(state, ids: list[int]) -> float               mean next-token NLL (teacher forcing) over ids, for the quality gate
Decode = graphed speculative step (engine/spec.py): DSpark draft K tokens + T=K+1 verify + on-device accept/commit.
"""
import torch
import torch.nn.functional as F

from engine.model_v1 import InklingV1
from engine.dspark import DSpark
from engine.spec import Spec

K = 2


def load():
    m = InklingV1()
    return {"m": m, "s": Spec(m, DSpark(m))}


@torch.inference_mode()
def generate(state, prompt_ids, max_tokens):
    ids = torch.tensor(prompt_ids, device="cuda")
    return [int(t) for t in state["s"].generate(ids, max_tokens, K)]


@torch.inference_mode()
def nll(state, ids):
    m = state["m"]
    t = torch.tensor(ids, device="cuda")
    logits, _ = m.forward(t)
    return F.cross_entropy(logits[:-1].float(), t[1:]).item()
