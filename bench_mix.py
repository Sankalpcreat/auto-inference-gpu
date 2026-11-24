import torch
from engine.fused_t1 import compile_
from engine.model_v0 import attn_res
from engine.mix_triton import mix
import triton

@compile_
def reference(sources,q,w):
    y = attn_res(sources,q,w)
    return y if w is None else y.bfloat16()

torch.manual_seed(13)
with torch.cuda.stream(torch.cuda.Stream()):
    for t in (1,3,8):
        for count, qrows, nrows in ((2,4,0),(5,1,1),(3,3,3),(3,1,3)):
            src=[torch.randn(t,2304,device="cuda",dtype=torch.bfloat16) for _ in range(count)]
            q=torch.randn(qrows,2304,device="cuda")/2304**0.5
            n=torch.ones(nrows,2304,device="cuda") if nrows else None
            expected=reference(src,q,n)
            actual=mix(src,q,n)
            err=(actual.float()-expected.float()).abs().max().item()
            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=.008)
            us=[triton.testing.do_bench_cudagraph(lambda: fn(src,q,n),rep=15)*1000 for fn in (reference,mix)]
            print("MIX",t,count,qrows,nrows,"max_error",err,"old/new us",us,flush=True)
