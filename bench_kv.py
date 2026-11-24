import torch
import triton
from engine.kv_dense import kv_dense
from engine.fp8_dense import fp8_rows, fp8_gemv
torch.manual_seed(3)
with torch.cuda.stream(torch.cuda.Stream()):
    w=[torch.randn(1024,2304,device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
    s=[torch.rand(1024,device="cuda") for _ in range(2)]
    for m in (1,2,3,8):
        x=[torch.randn(m,2304,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
        def old():
            fn=fp8_gemv if m==1 else fp8_rows
            return [fn(a,b,c).view(m,-1) for a,b,c in zip(x,w,s)]
        def new():
            return kv_dense(*x,*w,*s)
        ref=torch.stack(old()); actual=new()
        print("KV",m,"max_error",(ref-actual).abs().max().item(),flush=True)
        torch.testing.assert_close(actual,ref,rtol=0,atol=0)
        print("KV",m,"old/new us",[triton.testing.do_bench_cudagraph(fn,rep=25)*1000 for fn in (old,new)],flush=True)
