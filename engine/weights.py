"""Phase 0: Inkling 2 Mini checkpoint loader + dequant (verified on RTX PRO 6000, 2026-11-21).

NVFP4 experts (layers 0-19): U8 packs 2 E2M1 values (low nibble first), FP8-E4M3 scale per 16 values,
outer scale = amax / outer_scale_denominator (2688 = 6 * 448). Check: dequant absmax == amax exactly.
MXFP8 experts (layers 20-23): FP8-E4M3 values, E8M0 scale (2^(s-127)) per 32 values.
"""
import torch
from safetensors import safe_open

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant_nvfp4(w: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """w (..., N, K/2) uint8, scale (..., N, K/16) fp8, amax (...,) fp32, den (1,) -> (..., N, K) bf16."""
    lut = torch.tensor(E2M1, dtype=torch.float32, device=w.device)
    vals = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    outer = (amax / den).reshape(*amax.shape, 1, 1, 1)
    out = vals.unflatten(-1, (-1, 16)) * scale.float().unsqueeze(-1) * outer
    return out.flatten(-2).to(torch.bfloat16)


def dequant_mxfp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """w (..., N, K) fp8_e4m3, scale (..., N, K/32) uint8 E8M0 -> (..., N, K) bf16."""
    sc = torch.pow(2.0, scale.float() - 127.0)
    return (w.float().unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def open_checkpoint(path="/tmp/inkling/model-00001-of-00001.safetensors", device="cuda"):
    return safe_open(path, "pt", device=device)
