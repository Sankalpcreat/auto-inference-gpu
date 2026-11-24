"""Grouped GEMV over the checkpoint's packed experts (dequantized in registers), for a single decode token.

out[j, n] = sum_k deq(W[ids[j]])[n, k] * x[j, k], fp32 accumulate.  deq() reproduces weights.py exactly:
  NVFP4: bf16((E2M1[nib] * fp8_scale[k // 16]) * (amax[e] / den))   (u8 packs k even = low nibble, k odd = high)
  MXFP8: bf16(fp8[k] * 2^(e8m0[k // 32] - 127))
"""
import torch
import triton
import triton.language as tl

from .weights import open_checkpoint


@triton.jit
def _e2m1(n):
    """4-bit E2M1 code (int32) -> fp32 value * 2^-14 (exact): the 3 magnitude bits placed as fp16 exponent[1:0] +
    mantissa msb give 2^(e-15)(1+m/2) (e>0) or m*2^-15 (e=0, subnormal); caller folds 2^14 into the scale."""
    h = ((n & 8) << 12) | ((n & 7) << 9)
    return h.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gemv_nvfp4(W, S, OUTER, X, IDS, OUT, N, K: tl.constexpr, stride_we, stride_wn, stride_se, stride_sn, stride_xj,
                BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr):
    # BLOCK_B packed bytes per iteration = 2 * BLOCK_B values
    j = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(IDS + j).to(tl.int64)
    outer = tl.load(OUTER + e)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
    sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
    xbase = X + j * stride_xj
    acc = tl.zeros([BLOCK_N, BLOCK_B], dtype=tl.float32)
    for b0 in range(0, K // 2, BLOCK_B):
        offs_b = b0 + tl.arange(0, BLOCK_B)
        xe = tl.load(xbase + 2 * offs_b).to(tl.float32)
        xo = tl.load(xbase + 2 * offs_b + 1).to(tl.float32)
        w = tl.load(wbase + offs_b[None, :]).to(tl.int32)
        s = tl.load(sbase + (offs_b // 8)[None, :]).to(tl.float32)
        low, high = tl.inline_asm_elementwise(
            "{ .reg .b8 byte0; .reg .b32 halves; mov.b32 {byte0, _, _, _}, $2; "
            "cvt.rn.f16x2.e2m1x2 halves, byte0; mov.b32 {$0, $1}, halves; }",
            constraints="=h,=h,r", args=[w], dtype=(tl.float16, tl.float16), is_pure=True, pack=1)
        lo = ((low.to(tl.float32) * s) * outer).to(tl.bfloat16).to(tl.float32)
        hi = ((high.to(tl.float32) * s) * outer).to(tl.bfloat16).to(tl.float32)
        acc += lo * xe[None, :] + hi * xo[None, :]
    tl.store(OUT + j * N + offs_n, tl.sum(acc, 1))


@triton.jit
def _gemv_mxfp8(W, S, X, IDS, OUT, N, K, stride_we, stride_wn, stride_se, stride_sn, stride_xj,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    j = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(IDS + j).to(tl.int64)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
    sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
    xbase = X + j * stride_xj
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(xbase + offs_k).to(tl.float32)
        w = tl.load(wbase + offs_k[None, :]).to(tl.float32)
        s = tl.exp2(tl.load(sbase + (offs_k // 32)[None, :]).to(tl.float32) - 127.0)
        w = (w * s).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(OUT + j * N + offs_n, acc)


CFG = {"nvfp4": (16, 256, 4), "mxfp8": (16, 128, 4)}   # (BLOCK_N, BLOCK_B|BLOCK_K, num_warps), cold-L2 sweep


def packed_gemv(P, x, ids, shared_x, cfg=None):
    """P: packed expert dict (kind, w, s[, outer]); x (K,) if shared_x else (S, K) bf16; ids (S,) -> (S, N) fp32."""
    w, s = P["w"], P["s"]
    N = w.shape[1]
    S = ids.shape[0]
    out = torch.empty(S, N, dtype=torch.float32, device=w.device)
    BN, BK, nw = cfg or CFG[P["kind"]]
    if P["kind"] == "nvfp4":
        K = w.shape[2] * 2
        while (K // 2) % BK:
            BK //= 2
        assert N % BN == 0 and BK >= 8
        _gemv_nvfp4[(S, N // BN)](w, s, P["outer"], x, ids, out, N, K, w.stride(0), w.stride(1),
                                  s.stride(0), s.stride(1), 0 if shared_x else K,
                                  BLOCK_N=BN, BLOCK_B=BK, num_warps=nw)
    else:
        K = w.shape[2]
        while K % BK:
            BK //= 2
        assert N % BN == 0 and BK >= 32
        _gemv_mxfp8[(S, N // BN)](w, s, x, ids, out, N, K, w.stride(0), w.stride(1),
                                  s.stride(0), s.stride(1), 0 if shared_x else K,
                                  BLOCK_N=BN, BLOCK_K=BK, num_warps=nw)
    return out


def load_packed(path="/tmp/inkling/model-00001-of-00001.safetensors", n_layers=24, device="cuda"):
    """Per layer {"w13": P, "w2": P} with the raw packed tensors (no copy of the dequantized BF16 weights)."""
    f = open_checkpoint(path, device)
    g = lambda k: f.get_tensor("model.llm." + k)
    out = []
    for i in range(n_layers):
        e = f"layers.{i}.mlp.experts."
        lp = {}
        for n in ("w13", "w2"):
            if i < 20:
                amax = g(e + n + "_weight_amax").float().reshape(-1)
                den = g(e + n + "_weight_outer_scale_denominator").float().reshape(-1)
                w = g(e + n + "_weight").contiguous()
                outer = (amax / den).reshape(-1)
                if outer.numel() == 1:
                    outer = outer.expand(w.shape[0])
                assert outer.numel() == w.shape[0]
                lp[n] = {"kind": "nvfp4", "w": w, "s": g(e + n + "_weight_scale").contiguous(),
                         "outer": outer.contiguous()}
            else:
                lp[n] = {"kind": "mxfp8", "w": g(e + n + "_weight").contiguous(),
                         "s": g(e + n + "_weight_scale").contiguous()}
        out.append(lp)
    return out
