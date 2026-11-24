"""Run both FP8 key and value projections in a single decode kernel."""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _project(XK, XV, WK, WV, SK, SV, OUT, N: tl.constexpr, K: tl.constexpr,
             M: tl.constexpr, MP: tl.constexpr, BK: tl.constexpr):
    group = tl.program_id(0)
    is_value = group % 2
    col = (group // 2) * 16 + tl.arange(0, 16)
    row = tl.arange(0, MP)
    xptr = tl.where(is_value != 0, XV, XK)
    wptr = tl.where(is_value != 0, WV, WK)
    sptr = tl.where(is_value != 0, SV, SK)
    acc = tl.full((MP, 16, BK), 0, tl.float32)
    for block in range(K // BK):
        inner = block * BK + tl.arange(0, BK)
        x = tl.load(xptr + row[:, None] * K + inner[None, :], row[:, None] < M, other=0.).to(tl.float32)
        w = tl.load(wptr + col[:, None] * K + inner[None, :], col[:, None] < N, other=0.).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    y = tl.sum(acc, 2) * tl.load(sptr + col, col < N, other=0.)[None, :]
    tl.store(OUT + (is_value*M+row[:, None])*N+col[None, :], y, (row[:, None]<M)&(col[None, :]<N))


@triton_op("inkling::kv_dense", mutates_args={})
def kv_dense(xk: torch.Tensor, xv: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
             sk: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
    n, k = wk.shape
    m = xk.shape[0]
    mp = triton.next_power_of_2(m)
    bk = min(256, 512 // mp)
    while k % bk:
        bk //= 2
    out = torch.empty((2,m,n), device=xk.device, dtype=xk.dtype)
    wrap_triton(_project)[(2*triton.cdiv(n,16),)](xk.contiguous(), xv.contiguous(), wk, wv, sk, sv,
                                                  out,n,k,m,mp,bk,num_warps=4)
    return out
