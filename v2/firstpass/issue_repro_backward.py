import torch

B, M, K, N = 65537, 256, 64, 256          # output (B, 256, 256) has 2**32 + 65536 elements
torch.manual_seed(0)
a = torch.rand(B, M, K, device="mps", requires_grad=True)
b = torch.rand(B, K, N, device="mps", requires_grad=True)
G = torch.rand(B, M, N, device="mps")      # upstream gradient, contiguous

y = torch.bmm(a, b)
grad_a, grad_b = torch.autograd.grad(y, (a, b), G)

def rel_err(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max()).item()

for i in (0, B - 2, B - 1):                # batch B-1 = 65536 lies beyond element 2**32 of y and G
    ai, bi, Gi = a[i].detach().cpu().double(), b[i].detach().cpu().double(), G[i].cpu().double()
    print(f"batch {i:5d}  forward {rel_err(y[i].detach().cpu(), ai @ bi):.2e}"
          f"  grad_a {rel_err(grad_a[i].cpu(), Gi @ bi.T):.2e}  grad_b {rel_err(grad_b[i].cpu(), ai.T @ Gi):.2e}")

# what the last batch of grad_a equals instead: the product formed with batch 0 of G
G0, bl = G[0].cpu().double(), b[B - 1].detach().cpu().double()
print("grad_a[65536] vs G[0] @ b[65536].T :", f"{rel_err(grad_a[B - 1].cpu(), G0 @ bl.T):.2e}")
print(torch.__version__)
