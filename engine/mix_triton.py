"""Fuse attention-residual mixing and optional output RMS normalization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mix(SOURCES, QUERY, NORM, OUT, D: tl.constexpr, T: tl.constexpr,
         Q: tl.constexpr, NORM_ROWS: tl.constexpr, COUNT: tl.constexpr, WIDTH: tl.constexpr):
    query_id = tl.program_id(0)
    token = tl.program_id(1)
    lane = tl.arange(0, WIDTH)
    query = tl.load(QUERY + (query_id % Q) * D + lane, lane < D, other=0.)
    logits = ()
    inverses = ()
    largest = -float("inf")
    for i in tl.static_range(COUNT):
        x = tl.load(SOURCES[i] + token * D + lane, lane < D, other=0.).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1.e-6)
        score = tl.sum(x * query, 0) * inv
        logits += (score,)
        inverses += (inv,)
        largest = tl.maximum(largest, score)
    denom = 0.
    weights = ()
    for i in tl.static_range(COUNT):
        weight = tl.exp(logits[i] - largest)
        weights += (weight,)
        denom += weight
    mixed = tl.full((WIDTH,), 0, tl.float32)
    for i in tl.static_range(COUNT):
        x = tl.load(SOURCES[i] + token * D + lane, lane < D, other=0.).to(tl.float32)
        mixed += x * ((weights[i] / denom) * inverses[i])
    if NORM_ROWS:
        norm = tl.load(NORM + (query_id % NORM_ROWS) * D + lane, lane < D, other=0.).to(tl.float32)
        mixed = mixed * tl.rsqrt(tl.sum(mixed * mixed, 0) / D + 1.e-6) * norm
    tl.store(OUT + (query_id * T + token) * D + lane, mixed, lane < D)


def mix(sources, queries, norm=None):
    tokens, width = sources[0].shape
    norm_rows = 0 if norm is None else norm.shape[0]
    outputs = max(queries.shape[0], norm_rows)
    dtype = torch.float32 if norm is None else torch.bfloat16
    out = torch.empty(outputs, tokens, width, dtype=dtype, device=queries.device)
    _mix[(outputs, tokens)](tuple(sources), queries, queries if norm is None else norm, out,
                           width, tokens, queries.shape[0], norm_rows, len(sources),
                           triton.next_power_of_2(width), num_warps=8, enable_fp_fusion=False)
    return out
