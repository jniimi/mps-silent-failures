"""MPS で F.scaled_dot_product_attention がどの aten 演算に解決されるかを TorchDispatchMode で記録する(小さい tensor、数秒)。

    uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python sdpa_ops.py

2.12.1 と 2.14.0 のどちらも aten._scaled_dot_product_attention_math_for_mps.default の 1 演算(2026-09-19、Mac Studio)。
"""
import torch, torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("  ", func); return func(*args, **(kwargs or {}))
q = torch.randn(2, 4, 64, 32, device="mps")
print(torch.__version__)
with Log(): F.scaled_dot_product_attention(q, q, q)
print([n for n in dir(torch.ops.aten) if "scaled_dot" in n or "flash" in n])
