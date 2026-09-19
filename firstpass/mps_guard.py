"""MPS で 2**32 要素を超えるテンソルを扱う演算を検出して止める。

    import mps_guard
    mps_guard.install()          # 以後、MPS 上の演算の入力か出力が 2**32 要素を超えたら RuntimeError

macOS 15 以上 + torch 2.5 以上の MPS では、2**32 要素を超えるテンソルを扱う演算の一部が
エラーを出さずに誤った値を返す(pytorch#197636 ほか。一覧は README)。
どの演算が壊れるかは torch の版によって変わり、全部を個別に回避するのは現実的でないため、
「MPS 上で 2**32 要素を超えるテンソルを作らない」を規則にし、破ったら止める。
呼び出し側はバッチをチャンクに分けて対応する。

TorchDispatchMode で全 aten 演算を見るので、Python 側のオーバーヘッドが乗る。
小さいテンソルを大量に回すループでは遅くなる(計測値は README)。
環境変数 MPS_GUARD=0 で install() を無効化できる。
"""

from __future__ import annotations

import os

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

LIMIT = 2**32


class Over2Pow32Error(RuntimeError):
    pass


def _check(tensors, op, where):
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device.type == "mps" and t.numel() > LIMIT:
            raise Over2Pow32Error(
                f"{op}: {where} tensor of shape {tuple(t.shape)} has {t.numel():,} elements (> 2**32) on MPS. "
                "Results of such ops can be silently wrong on macOS 15+; split the batch into chunks."
            )


_aten = torch.ops.aten
# 実行前に出力サイズを見積もる演算。torch 2.11 など、出力が大きすぎると演算の中で
# segfault する版があり、実行後の検査では間に合わないため。
_PRE = {
    _aten.bmm.default: lambda a, b, *_: a.shape[0] * a.shape[1] * b.shape[2],
    _aten.baddbmm.default: lambda c, a, b, *_: a.shape[0] * a.shape[1] * b.shape[2],
    _aten.mm.default: lambda a, b, *_: a.shape[0] * b.shape[1],
    _aten.addmm.default: lambda c, a, b, *_: a.shape[0] * b.shape[1],
}
# fused SDPA は内部の attention score (…, Lq, Lk) が表に出ないので、その要素数で判定する。
# torch 2.11 / 2.12 はこれが 2**32 を超えると誤答する(2.13 以降は正しいが、一律に止める)。
if hasattr(_aten, "_scaled_dot_product_attention_math_for_mps"):
    _PRE[_aten._scaled_dot_product_attention_math_for_mps.default] = (
        lambda q, k, *_: q.numel() // q.shape[-1] * k.shape[-2]
    )


class MPSSizeGuard(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat_in, _ = tree_flatten((args, kwargs))
        _check(flat_in, func, "input")
        est = _PRE.get(func)
        if est is not None and args[0].device.type == "mps":
            n = est(*args)
            if n > LIMIT:
                raise Over2Pow32Error(
                    f"{func}: would need a tensor with {n:,} elements (> 2**32) on MPS. "
                    "Results of such ops can be silently wrong on macOS 15+; split the batch into chunks."
                )
        out = func(*args, **kwargs)
        flat_out, _ = tree_flatten(out)
        _check(flat_out, func, "output")
        return out


_mode: MPSSizeGuard | None = None


def install() -> None:
    """プロセス全体でガードを有効にする。冪等。MPS_GUARD=0 なら何もしない。"""
    global _mode
    if _mode is not None or os.environ.get("MPS_GUARD") == "0":
        return
    _mode = MPSSizeGuard()
    _mode.__enter__()


def uninstall() -> None:
    global _mode
    if _mode is not None:
        _mode.__exit__(None, None, None)
        _mode = None
