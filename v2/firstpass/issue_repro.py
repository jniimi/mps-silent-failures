import torch

torch.manual_seed(0)
L, D = 256, 64
for B in (65536 - 16, 65536 + 16):  # output numel crosses 2**32 at B = 65536
    q = torch.randn(B, L, D, device="mps")
    k = torch.randn(B, L, D, device="mps")
    out = torch.bmm(q, k.transpose(1, 2))  # batch2 is a transposed (non-contiguous) view
    for i in (0, B - 1):
        ref = q[i].cpu() @ k[i].cpu().T
        err = ((out[i].cpu() - ref).abs().max() / ref.abs().max()).item()
        print(f"B={B} numel/2**32={B * L * L / 2**32:.4f} batch={i} rel_err={err:.2e}")
    del q, k, out
    torch.mps.empty_cache()
