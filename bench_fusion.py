"""Compare the old and new speculative cache commits on a CUDA graph without model weights."""

import json
import statistics

import torch
import triton

from engine import fused_t1 as ft
from engine.model_v0 import D, HD, HKV, WINDOW
from engine.spec_commit import CommitBuffers, _commit_histories, _commit_rings


def capture(fn, calls):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return graph


def time_graph(fn, calls=64, trials=9):
    graph = capture(fn, calls)
    for _ in range(3):
        graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return round(statistics.median(times), 3)


def report(name, old, new, **checks):
    print("MICRO " + json.dumps(dict(kernel=name, old_us=time_graph(old),
                                    new_us=time_graph(new), **checks)), flush=True)


def benchmark_commit(rows):
    dev = "cuda"
    widths = [D, HKV * HD, HKV * HD, D] * 48
    histories = [(torch.randn(rows + 3, c, device=dev, dtype=ft.BF),
                  torch.empty(3, c, device=dev, dtype=ft.BF)) for c in widths]
    rings = [(torch.randn(HKV, rows, HD, device=dev),
              torch.randn(HKV, WINDOW, HD, device=dev)) for _ in range(2 * 40)]
    ref_histories = [(src, torch.empty_like(dst)) for src, dst in histories]
    ref_rings = [(src, dst.clone()) for src, dst in rings]
    plan = CommitBuffers(histories, rings)
    accepted = torch.tensor(rows, device=dev)
    position = torch.tensor(WINDOW - 1, device=dev)
    arange3 = torch.arange(3, device=dev)
    arange_rows = torch.arange(rows, device=dev)

    def old_hist():
        idx = accepted + arange3
        for full, hist in ref_histories:
            hist.copy_(full.index_select(0, idx))

    def old_ring():
        slots = torch.remainder(position + arange_rows, WINDOW)
        keep = (arange_rows < accepted)[None, :, None]
        for new, ring in ref_rings:
            ring.index_copy_(1, slots, torch.where(keep, new, ring.index_select(1, slots)))

    def new_hist():
        _commit_histories[(plan.history_tiles, len(histories))](
            plan.history_table, accepted, 256, num_warps=4,
        )

    def new_ring():
        _commit_rings[(triton.cdiv(rows * HD, 128), len(rings), HKV)](
            plan.ring_table, accepted, position, rows, WINDOW, HD, 128, num_warps=4,
        )

    def reference_commit():
        old_hist()
        old_ring()

    baseline_graph = capture(reference_commit, 1)
    fused_graph = capture(lambda: plan.commit(accepted, position), 1)
    for pos in (0, WINDOW - 1, 2 * WINDOW + 5):
        position.fill_(pos)
        for count in range(1, rows + 1):
            accepted.fill_(count)
            for _, ring in rings:
                ring.fill_(-123.0)
            for _, ring in ref_rings:
                ring.fill_(-123.0)
            baseline_graph.replay()
            fused_graph.replay()
            torch._assert_async(torch.stack([torch.all(a == b) for (_, a), (_, b)
                                             in zip(histories, ref_histories)]).all())
            torch._assert_async(torch.stack([torch.all(a == b) for (_, a), (_, b)
                                             in zip(rings, ref_rings)]).all())
    torch.cuda.synchronize()
    if rows == 3:
        report("spec history commit (all visits)", old_hist, new_hist, max_abs_error=0.0)
        report("spec ring commit (all visits)", old_ring, new_ring, max_abs_error=0.0)

if __name__ == "__main__":
    with torch.inference_mode():
        torch.manual_seed(11)
        for rows in (1, 3, 8):
            benchmark_commit(rows)
