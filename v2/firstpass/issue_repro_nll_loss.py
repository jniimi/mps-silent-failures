import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, C = 2**23 + 2**16, 256                  # input has 2**31 + 2**24 elements
x = torch.rand(N, C) * 8 - 4
t = torch.randint(0, C, (N,))
xm, tm = x.to("mps"), t.to("mps")

loss = F.cross_entropy(xm, tm, reduction="none").cpu()
ref = F.cross_entropy(x, t, reduction="none")          # CPU
bad = (loss - ref).abs() > 1e-4
beyond = torch.arange(N) * C + t >= 2**31              # flat index of the target element
print("wrong rows:", bad.sum().item(), "| rows whose target lies at flat index >= 2**31:", beyond.sum().item(),
      "| same rows:", torch.equal(bad, beyond), "| wrong values that are 0:", (loss[bad] == 0).sum().item())
print("sum on MPS:", F.cross_entropy(xm, tm, reduction="sum").item(), "| CPU:", ref.double().sum().item())
print("-log_softmax(x).gather(...).sum() on MPS:", (-F.log_softmax(xm, -1).gather(1, tm[:, None])).sum().item())
