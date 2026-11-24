import torch
from engine import moe_packed
import argparse
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare packed expert kernels to a baseline checkout.")
parser.add_argument("baseline", type=Path)
path = parser.parse_args().baseline / "engine/moe_packed.py"
spec = spec_from_file_location("engine.reference_moe", path)
moe_original = module_from_spec(spec)
spec.loader.exec_module(moe_original)
import triton
torch.manual_seed(42)
p = moe_packed.load_packed(n_layers=1)[0]
for name in ("w13","w2"):
    w=p[name]
    k=w["w"].shape[-1]*2
    for rows in (8,24):
        ids=torch.randperm(128,device="cuda")[:rows]
        x=torch.randn(rows,k,device="cuda",dtype=torch.bfloat16)
        reference=moe_original.packed_gemv(w,x,ids,False)
        actual=moe_packed.packed_gemv(w,x,ids,False)
        print(name,rows,"max_error",(reference-actual).abs().max().item(),flush=True)
        torch.testing.assert_close(actual,reference,rtol=2e-5,atol=1e-4)
        with torch.cuda.stream(torch.cuda.Stream()):
            for module in (moe_original,moe_packed):
                print(name,rows,module.__name__,triton.testing.do_bench_cudagraph(
                    lambda: module.packed_gemv(w,x,ids,False),rep=25)*1000,"us",flush=True)
