"""MPS 上の nll_loss(cross_entropy を含む)で、入力が 2**31 要素を超えるものを止める。

    import mps_nll_guard
    mps_nll_guard.install()      # 以後、MPS 上の nll_loss の入力が 2**31 要素を超えたら RuntimeError

MPS の nll_loss は、正解クラスの要素が入力の flat index 2**31 以降にある行について、
エラーを出さずに誤った損失(ほとんどが 0)を返す(pytorch#198471。2.4.1 / 2.9.1 / 2.13.0 / 2.14.0 で確認)。
F.cross_entropy は log_softmax の後に nll_loss を呼ぶので同じく壊れる。log_softmax 自体と、
入力に対する勾配(nll_loss_backward)は正しかった。
flat index 2**31 以降の要素があるのは入力が 2**31 要素を超えるときなので、判定は「より大きい」。
nll_loss2d(入力が 3 次元以上)は確認していないが、同じ判定で止める。

これは mps_guard(2**32 要素以上の tensor を止める)とは別の不具合なので、別の guard にしてある。
両方を install() して併用できる。呼び出し側は行をチャンクに分けて損失を計算する。
環境変数 MPS_NLL_GUARD=0 で install() を無効化できる。
"""

from __future__ import annotations

import os

import torch
from torch.utils._python_dispatch import TorchDispatchMode

LIMIT = 2**31

_aten = torch.ops.aten
_OPS = (_aten.nll_loss_forward.default, _aten.nll_loss2d_forward.default)


class NllOver2Pow31Error(RuntimeError):
    pass


class MPSNllGuard(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _OPS:
            x = args[0]
            if x.device.type == "mps" and x.numel() > LIMIT:
                raise NllOver2Pow31Error(
                    f"{func}: input of shape {tuple(x.shape)} has {x.numel():,} elements (> 2**31) on MPS. "
                    "nll_loss / cross_entropy can silently return wrong losses for rows whose target element lies at "
                    f"flat index >= 2**31 (pytorch#198471); split the rows into chunks of at most {LIMIT // (x.numel() // x.shape[0]):,}."
                )
        return func(*args, **kwargs)


_mode: MPSNllGuard | None = None


def install() -> None:
    """プロセス全体でガードを有効にする。冪等。MPS_NLL_GUARD=0 なら何もしない。"""
    global _mode
    if _mode is not None or os.environ.get("MPS_NLL_GUARD") == "0":
        return
    _mode = MPSNllGuard()
    _mode.__enter__()


def uninstall() -> None:
    global _mode
    if _mode is not None:
        _mode.__exit__(None, None, None)
        _mode = None
