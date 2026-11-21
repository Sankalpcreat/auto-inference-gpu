"""FROZEN reference wrapper for golden outputs (ref/ engine). Agents never edit ref/.

Contract with prepare_gpu.py (do not change the signatures):
    load() -> state                                   called once
    generate(state, prompt_ids: list[int], max_tokens: int) -> list[int]
        greedy; must return exactly max_tokens ids (ignore EOS); deterministic across repeats
    nll(state, ids: list[int]) -> float               mean next-token NLL (teacher forcing) over ids, for the quality gate
Baseline: engine v1 (plain PyTorch, eager, per-pass caches) = 6.0 tok/s.
"""
import torch
import torch.nn.functional as F

from ref.model_v1 import InklingV1


def load():
    return {"m": InklingV1()}


@torch.inference_mode()
def generate(state, prompt_ids, max_tokens):
    m = state["m"]
    ids = torch.tensor(prompt_ids, device="cuda")
    return [int(t) for t in m.generate(ids, max_tokens).tolist()]


@torch.inference_mode()
def nll(state, ids):
    m = state["m"]
    t = torch.tensor(ids, device="cuda")
    logits, _ = m.forward(t)
    return F.cross_entropy(logits[:-1].float(), t[1:]).item()
