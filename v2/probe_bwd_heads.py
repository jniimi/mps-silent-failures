"""出力側の構造の積(小さな連続な入力、出力 > 2^32 要素)の出力をそのまま典型的な loss head に入れた学習 1 ステップで、MPS が誤答・例外を出すかを見る。

    uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python probe_bwd_heads.py

    (demo_train_step.py と同じく版を固定した使い捨て環境で動かす。使うのは torch と numpy だけで、transformers / datasets は要らない。
    動作確認は --tiny(小さな形状で全体が数秒)で、例: --tiny --device cpu、--tiny --device mps)

目的: 1 個の bmm の backward の掃引(出力側の形 (B,256,64)@(B,64,256)、連続なオペランド)では、B=65537 で forward は正しく、入力の
2 つの勾配が、出力の勾配 G の要素 2^32 を越えるバッチ(B=65537 では最後のバッチだけ)で黙って誤った(勾配の積 G·bᵀ と aᵀ·G が
G(連続、> 2^32 要素)をオペランドに取るので規則 3)。RoBERTa の学習 1 ステップ(demo_train_step.py)では attention の backward が
常に先に例外を出した(Pᵀ の view が 2^31 要素以上、規則 2)。ここでは「出力側」の構造の積の出力が、間に bmm を挟まずに典型的な
loss head に入るとき、学習ステップが (a) loss は正しく勾配だけが黙って誤る(勾配の積に規則 3)、(b) どこかで例外を出す(例: loss
head の > 2^32 要素の reduction)、(c) それ以外、のどれになるかを確かめる。

積(葉は fp32 で requires_grad、渡す時点で全て連続):
  bmm       a (B,256,64)、b (B,64,256)(各要素 N(0,1)·64^(-1/4)、Y の要素の分散が 1)、Y = torch.bmm(a, b) (B,256,256)。
            B = 65536(Y がちょうど 2^32 要素、対照)と 65537(すぐ上)。
  relation  MiniLMv2 の self-attention relation(Wang et al. 2021: Q-Q / K-K / V-V を R 個の relation head に分けた scaled
            dot-product の softmax。R = 48(base の教師)/ 64(large の教師)、最大長 512、batch 256)。x (n, L, R·d) が葉(N(0,1))、
            q = x.view(n,L,R,d).transpose(1,2)、Y = matmul(q, qᵀ)/√d (n,R,L,L)、R=64, L=512, d=16。n = 256(ちょうど 2^32、
            MiniLMv2 の large の教師の設定、対照)と 257(すぐ上)。
loss head(Y の最後の次元をクラスの次元 C とする。どれも行ごとの項の和なので、単位ごとに分けた loss と勾配の和に等しい):
  ce       F.cross_entropy(Y.reshape(-1, C), t, reduction="sum")、t は固定の乱数のクラス
  kl_rows  (T·(log T − log_softmax(Y, −1))).sum(−1).sum()(行ごとの和が先。log T = T.clamp_min(tiny).log())
  kl_sum   F.kl_div(F.log_softmax(Y, −1), T, reduction="sum")(MiniLMv2 の形)
  lse      torch.logsumexp(Y, −1).sum()
  amax     Y.amax(−1).sum()
  sum      Y.sum()(backward の G は ones を expand した stride 0 の view)
  T = softmax(Z, −1)、Z は Y と同じ形の固定の N(0,1)(勾配なし)。

乱数: 葉・t・Z は「単位」(bmm はバッチの添字、relation は系列)ごとに (seed, 種類, 添字) から決めた CPU の Generator で作る。値は B にも
分割にも依存しないので、一括の条件と基準でビット一致する。一括の条件では CPU で全体を作って device へ 1 回でコピーし、ブロックごとに
読み戻して CPU の元とビット一致することを確かめる(inputs_verified。> 2^32 要素の T では、下の指紋で使う大きな offset のスライスの
読み戻しの確認も兼ねる)。

予測(PyTorch 2.14.0 で見つけた bmm の規則から。この順に適用):
  1. 出力 > 2^32 要素でオペランドが転置の view → 全バッチで誤答、それ以外の出力 > 2^32 は正しい。
  2. view のオペランドが 2^31 要素以上 → 例外(INT_MAX)。3. 連続なオペランドが 2^32 要素超 → 要素 2^32 を越えるバッチだけ誤答。
  bmm の backward は grad_a = G·bᵀ、grad_b = aᵀ·G(ᵀ は転置の view)、matmul はバッチの次元を畳むときに畳めない q, qᵀ を連続な 3 次元の
  テンソルにコピーすると仮定する(dispatch の記録で確かめる)。
  対照(Y がちょうど 2^32 要素): forward も勾配も正しい。ただし bmm + sum は、backward の G = expand(ones) が 2^32 要素の stride 0 の
      view なので、規則 2 が stride 0 の view の見かけの要素数で効くなら対照でも BmmBackward0 で例外。
  すぐ上: forward の積は正しい(規則 1 の「それ以外」)。勾配の積は G(連続、> 2^32)を取るので、要素 2^32 を越える単位だけ誤る(bmm:
      バッチ 65536、relation: 要素 2^32 は畳んだバッチ 16384 = 系列 256 の head 0 なので系列 256)。ただし loss head が > 2^32 要素の
      reduction を持つと先に例外になりうる。MPS の reduction は 64-bit の添字が要るテンソルで NotImplementedError を出す(観測: 全体の
      sum で numel = 2^32 + 2^20、_safe_softmax の中の次元ごとの all で numel = 2^32 + 2^21。numel が 2^31〜2^32 の間では出なかった)。
      ちょうど 2^32 は未確認(判定が numel > UINT32_MAX なら対照でも出る)。
      ce: log_softmax(softmax と同じ系統。推論では > 2^32 要素で正しかった)と nll_loss(行の数 < 2^32 の reduction)だけなので forward
          は通る見込み。backward(nll_loss_backward、_log_softmax_backward_data)は不明
      kl_rows / lse / amax: > 2^32 要素の次元ごとの reduction(sum / amax、logsumexp の中の amax と sum)→ 例外かもしれない
      kl_sum / sum: > 2^32 要素の全体の sum → 例外の見込み。sum の backward に達すれば、bmm の G は stride 0 の view(> 2^31 要素)なので
          規則 2 で例外、relation では /√d の backward が G を実体化するので規則 3

基準(同じ device、同じ head): 単位の方向に chunk(既定 bmm 8192 バッチ、relation 32 系列。各 chunk の Y は 2^29 要素)ずつ forward +
backward し、loss を float64 で足し、葉の勾配を CPU で全体の形に並べる(勾配の各要素はちょうど 1 つの chunk から来るので累積はなく、
float32 のままで厳密)。半分の chunk でもう 1 回行い、2 つの基準の差を丸め誤差の目安(noise)とする。基準は一括の条件のあと同じ子
プロセスで計算する(一括が例外で止まっても基準と Y の確認は残る。一括が abort したら "crashed" で、比べるものはない)。MPS では基準の
先頭 2 単位を CPU の計算と比べる(ref.vs_cpu、基準の演算の確認)。

Y と G の確認(指紋、--no-fingerprint で省く): 一括の条件と基準で、head に渡す Y と、積の出力に付けた hook で受け取る G(bmm の
backward に入る勾配)をブロックごとに CPU へ読み戻し、単位ごとに (Σx², Σw·x)(w は固定の ±1)を float64 で取る。単位ごとの相対誤差の
目安 max(|ΔΣx²|/(2Σx²_ref), |ΔΣw·x|/‖x_ref‖) で、forward の誤り(規則 1)と head の backward の誤りを、勾配の積の誤り(規則 3)から
切り分ける。
許容誤差: 単位ごとの相対誤差(勾配は相対 L2、Y と G は上の目安)が max(10·noise_u, 1e-5) を超えたら誤り(noise_u は 2 つの基準の同じ
単位の差。非有限は誤り)。勾配全体の相対 L2 は τ = max(10·noise, 1e-6)、loss の相対差は max(10·noise, 1e-5) で判定する(誤った
単位が 65537 個中 1 個なら loss は高々その単位の相対誤差の 1.5e-5 倍しか動かないので、forward の単位ごとの判定は Y の確認で行う)。

構成: 親プロセスは子プロセスの起動と集計だけを行い、MPS に触れない。各条件(積 × size × head)は、このスクリプト自身を隠しオプション
--child で起動した別の子プロセスで動かす(demo_train_step.py と同じ。例外・abort・メモリ不足が他の条件を巻き込まない)。forward(積 →
指紋 → head)と backward を torch.autograd.set_detect_anomaly(True, check_nan=False) の中で行い、status(ok / raised_forward /
raised_backward / crashed / timeout)、例外の型とメッセージ、forward の例外の traceback、backward の node と forward の場所(anomaly の
警告)、時間、device のメモリ(数か所と 0.1 秒ごとの標本の最大)を記録する。子は段階ごとに途中の結果を書くので、abort しても forward
までの結果は残る。dispatch: 最初に別の子プロセスで、2 単位の入力について積・head・backward で dispatch された aten 演算を数え、行列積の
引数の shape / stride / 連続かを記録する(1 単位では relation の畳み込みが view になりうるので 2 単位)。
順序: 対照(Y <= 2^32)が先、すぐ上が後。head は ce, kl_rows, kl_sum, lse, amax, sum の順。既定では何も飛ばさない(--skip-over-gb)。

出力(--out、既定 results/bwd_heads): bwd_heads.json(環境、引数、仮説、dispatch、条件ごとの結果)、bwd_heads.md(要約表、基準、
dispatch、例外の詳細)。条件ごとに書き直すので途中で止めても残る。

メモリ(fp32、見積りは mem_est。|Y| = 17.2 GB): 一括の device のピーク ≈ 葉と勾配 + target + k·|Y|。k は同時に生きる |Y| 大の
テンソルの数の見積り(HEAD_MEM: ce 3、kl_rows と kl_sum 4 に T が加わる、lse 4、amax 3、sum 1、relation の積は forward で 2)。
bmm B=65537 で ce 約 69 GB、kl_* 約 103 GB、lse 約 86 GB、amax 約 69 GB、sum 約 34 GB。relation n=257 で ce 約 53 GB、kl_* 約 87 GB、
lse 約 70 GB、amax 約 53 GB、sum 約 36 GB。準備では CPU の元と device のコピーが同時にある(bmm の kl_* で約 52 GB)。基準の間は CPU に
一括と基準の勾配を持つ(bmm で約 17 GB)。各条件の前に見積りを表示する。
時間の目安(MBP で単位数個から外挿した CPU の乱数生成): bmm の葉 約 20 秒、T 約 45 秒、relation の T 約 40 秒(それぞれ一括と基準 2 つで
3 回)。指紋は CPU の計算が 1 テンソル約 8 秒(bmm)/ 7 秒(relation)に device からの読み戻しが加わる。
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from demo_train_step import ANOMALY, DebugRaise, exc_info, f3, mem, parse_anomaly, rel_of
from demo_workload import ROOT, empty, runs_of, sh, sync

P31, P32 = 2**31, 2**32
REAL = {"bmm": dict(M=256, K=64, N=256), "relation": dict(R=64, L=512, d=16)}
TINY = {"bmm": dict(M=8, K=4, N=8), "relation": dict(R=4, L=8, d=4)}
SIZES = {"bmm": [65536, 65537], "relation": [256, 257]}
TINY_SIZES = {"bmm": [6, 7], "relation": [3, 4]}
CHUNKS = {"bmm": 8192, "relation": 32}
TINY_CHUNKS = {"bmm": 2, "relation": 2}
HEADS = ["ce", "kl_rows", "kl_sum", "lse", "amax", "sum"]
TARGET = {"ce": "t", "kl_rows": "T", "kl_sum": "T", "lse": None, "amax": None, "sum": None}
# 一括の条件で同時に生きる |Y| 大のテンソルの数の見積り(葉・勾配・target を除く): (head の forward, backward)
HEAD_MEM = {"ce": (2, 3), "kl_rows": (3, 4), "kl_sum": (4, 4), "lse": (2, 4), "amax": (1, 3), "sum": (1, 1)}
NOISE_MULT, TAU_UNIT_MIN, TAU_MIN = 10.0, 1e-5, 1e-6
BLOCK_BYTES = 2**30  # 読み戻し(入力の確認、指紋)のブロックの大きさ
CPU_CHECK_UNITS = 2
STREAMS = {"a": 1, "b": 2, "x": 3, "t": 4, "Z": 5, "W": 6}
MM_OPS = re.compile(r"aten\.(bmm|mm|baddbmm|addmm|addbmm)\.")
HEAD_NOTE = {  # 予測の status(? は不確か)と根拠
    "ce": ("ok", "log_softmax (softmax family; softmax on > 2^32 elements was correct in inference) and nll_loss (reduces "
                 "over rows = numel/C < 2^32): no reduction over > 2^32 elements in forward; backward (nll_loss_backward, "
                 "_log_softmax_backward_data) unknown"),
    "kl_rows": ("raised_forward?", "dim-wise sum over > 2^32 elements (the dim-wise all in _safe_softmax raised at "
                                   "numel 2^32 + 2^21)"),
    "kl_sum": ("raised_forward", "full sum over > 2^32 elements inside kl_div (MPS sum_reduction raised for a full sum at "
                                 "numel 2^32 + 2^20)"),
    "lse": ("raised_forward?", "logsumexp: dim-wise amax and sum over > 2^32 elements"),
    "amax": ("raised_forward?", "dim-wise amax over > 2^32 elements; its backward counts ties with a dim-wise sum"),
    "sum": ("raised_forward", "full sum over > 2^32 elements; if passed, the backward G is expand(ones), a stride-0 view "
                              "with > 2^31 elements: bmm -> rule 2 (raise); relation: the /sqrt(d) backward materializes G "
                              "-> rule 3"),
}


# ---------------------------------------------------------------- 共通


def env_info() -> dict:
    return {
        "macos": sh("sw_vers", "-productVersion"), "macos_build": sh("sw_vers", "-buildVersion"),
        "hw_model": sh("sysctl", "-n", "hw.model"), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "mem_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "git_rev": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "git_dirty_v2": bool(sh("git", "-C", str(ROOT), "status", "--porcelain", "--", ".")),
    }


def spec(product: str, size: int, tiny: bool) -> dict:
    """積の形。単位 = bmm のバッチ / relation の系列。boundary_unit は要素 2^32 を含む単位(Y > 2^32 要素のときだけ)。"""
    s = (TINY if tiny else REAL)[product]
    if product == "bmm":
        leaves, y_unit, C, scale = {"a": (s["M"], s["K"]), "b": (s["K"], s["N"])}, (s["M"], s["N"]), s["N"], s["K"] ** -0.25
    else:
        leaves, y_unit, C, scale = {"x": (s["L"], s["R"] * s["d"])}, (s["R"], s["L"], s["L"]), s["L"], 1.0
    ue = math.prod(y_unit)
    return {"product": product, "size": size, **s, "leaves": leaves, "y_unit": y_unit, "C": C, "unit_elems": ue,
            "rows_per_unit": ue // C, "numel": size * ue, "over_2p32": size * ue / P32, "scale": scale,
            "unit": "batch" if product == "bmm" else "sequence", "boundary_unit": P32 // ue if size * ue > P32 else None}


def mem_est(sp: dict, head: str) -> dict:
    """一括の条件の device のピークの見積り(HEAD_MEM による概算)と、準備・基準の間の目安。"""
    S = sp["numel"] * 4
    P = sp["size"] * sum(math.prod(s) for s in sp["leaves"].values()) * 4
    tg = {"t": sp["size"] * sp["rows_per_unit"] * 8, "T": S, None: 0}[TARGET[head]]
    k = max(2 if sp["product"] == "relation" else 1, *HEAD_MEM[head])
    return {"Y_gb": S / 1e9, "leaves_gb": P / 1e9, "target_gb": tg / 1e9, "k_Y": k,
            "device_peak_gb": (2 * P + tg + k * S) / 1e9, "setup_peak_gb": 2 * (P + tg) / 1e9, "host_grads_gb": 2 * P / 1e9}


def predict(sp: dict, head: str) -> dict:
    n, u0 = sp["size"], sp["boundary_unit"]
    exact = " (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32)" \
        if sp["numel"] == P32 and head != "ce" else ""
    if u0 is None and head == "sum" and sp["product"] == "bmm" and sp["numel"] >= P31:
        # 対照でも G = expand(ones) は 2^31 要素以上の stride 0 の view
        return {"status": "raised_backward", "certain": False, "grads": "—",
                "text": "bwd raises (rule 2 on the stride-0 G)" + exact, "note": HEAD_NOTE[head][1]}
    if u0 is None:
        return {"status": "ok", "certain": True, "grads": "correct", "text": "ok: fwd and grads correct" + exact}
    us = f"{sp['unit']} {u0}" if u0 == n - 1 else f"{sp['unit']}s {u0}-{n - 1}"
    st, note = HEAD_NOTE[head]
    g = f"grads wrong in {us} only (rule 3 on G)"
    t = {"ok": f"fwd + loss correct; {g}", "raised_forward?": f"head fwd may raise; else {g}",
         "raised_forward": f"head fwd raises; else {g}"}[st]
    if head == "sum" and sp["product"] == "bmm":
        t = "head fwd raises; else bwd raises (rule 2 on the stride-0 G)"
    return {"status": st.rstrip("?"), "certain": not st.endswith("?"), "grads": g, "text": t, "note": note}


def hypotheses() -> dict:
    return {
        "rules": ["1. output > 2^32 elements and an operand is a transposed view -> silently wrong in every batch; "
                  "output > 2^32 otherwise -> correct",
                  "2. an operand that is a view has >= 2^31 elements -> raises (INT_MAX)",
                  "3. a contiguous operand has > 2^32 elements -> silently wrong only in batches beyond element 2^32"],
        "assumptions": "bmm backward: grad_a = G @ b^T, grad_b = a^T @ G (transposed views); matmul folds (n, R) into one "
                       "batch dim and copies the non-mergeable q and q^T into contiguous 3-D tensors (see dispatch)",
        "control": "Y exactly 2^32 elements (bmm B=65536, relation n=256): forward and gradients correct; except bmm + "
                   "sum, whose backward G = expand(ones) is a stride-0 view with 2^32 >= 2^31 elements -> rule 2 (raise "
                   "in BmmBackward0), if rule 2 counts the logical numel of a stride-0 view",
        "above": "Y just above 2^32 (bmm B=65537, relation n=257): forward product correct (rule 1, contiguous operands); "
                 "the gradient products take G (contiguous, > 2^32) -> rule 3 -> leaf gradients wrong only in the units "
                 "beyond element 2^32 of G (bmm: batch 65536; relation: element 2^32 is folded batch 16384 = sequence 256, "
                 "head 0 -> sequence 256); the loss head may raise first if it reduces over > 2^32 elements",
        "reduction_check": "MPS reductions over tensors requiring 64-bit indexing raise NotImplementedError (observed: full "
                           "sum at numel 2^32 + 2^20; dim-wise all inside _safe_softmax at 2^32 + 2^21; not at numel "
                           "between 2^31 and 2^32). Exactly 2^32 is untested (raises if the check is numel > UINT32_MAX)",
        "heads": {h: HEAD_NOTE[h][1] for h in HEADS},
    }


# ---------------------------------------------------------------- 入力・積・head


def gen(seed: int, kind: str, i: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 2**40 + STREAMS[kind] * 2**32 + i)


def make(sp: dict, kind: str, lo: int, hi: int, seed: int) -> torch.Tensor:
    """単位 lo..hi-1 の乱数(葉 / t / T)を CPU で作る。単位ごとに Generator を作り直すので、値は範囲の取り方に依存しない。"""
    if kind in sp["leaves"]:
        shape, dt = sp["leaves"][kind], torch.float32
    else:
        shape, dt = ((sp["rows_per_unit"],), torch.int64) if kind == "t" else (sp["y_unit"], torch.float32)
    out = torch.empty((hi - lo, *shape), dtype=dt)
    for i in range(lo, hi):
        if kind == "t":
            torch.randint(0, sp["C"], shape, generator=gen(seed, "t", i), out=out[i - lo])
        elif kind == "T":
            out[i - lo] = torch.softmax(torch.randn(shape, generator=gen(seed, "Z", i)), -1)
        else:
            torch.randn(shape, generator=gen(seed, kind, i), out=out[i - lo]).mul_(sp["scale"])
    return out


def to_target(t: torch.Tensor | None, kind: str | None) -> torch.Tensor | None:
    return t.reshape(-1) if kind == "t" else t


def product(sp: dict, lv: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(積の出力 Y0、head に渡す Y)。bmm では同じテンソル、relation では Y = Y0 / √d。"""
    if sp["product"] == "bmm":
        y = torch.bmm(lv["a"], lv["b"])
        return y, y
    x = lv["x"]
    q = x.view(x.shape[0], sp["L"], sp["R"], sp["d"]).transpose(1, 2)
    y0 = torch.matmul(q, q.transpose(-1, -2))
    return y0, y0 / math.sqrt(sp["d"])


# head は Y を list で受け取って pop する(呼び出し側に参照を残さず、log_softmax の後に Y を解放できるように)
def h_ce(ys: list, t):
    y = ys.pop()
    return F.cross_entropy(y.reshape(-1, y.shape[-1]), t, reduction="sum")


def h_kl_rows(ys: list, T):
    lp = F.log_softmax(ys.pop(), -1)
    return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).log() - lp)).sum(-1).sum()


def h_kl_sum(ys: list, T):
    return F.kl_div(F.log_softmax(ys.pop(), -1), T, reduction="sum")


def h_lse(ys: list, _):
    return torch.logsumexp(ys.pop(), -1).sum()


def h_amax(ys: list, _):
    return ys.pop().amax(-1).sum()


def h_sum(ys: list, _):
    return ys.pop().sum()


HEAD_FN = {"ce": h_ce, "kl_rows": h_kl_rows, "kl_sum": h_kl_sum, "lse": h_lse, "amax": h_amax, "sum": h_sum}


class DebugCorrupt(torch.autograd.Function):
    """試験用(--debug-corrupt-grad): forward は恒等、backward で最後の単位の勾配に 1 を足す(単位ごとの判定の確認用)。"""

    @staticmethod
    def forward(ctx, y):
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        g[-1] += 1.0
        return g


# ---------------------------------------------------------------- 指紋と誤差


def wvec(sp: dict, seed: int) -> torch.Tensor:
    return (torch.randint(0, 2, (sp["unit_elems"],), generator=gen(seed, "W", 0)) * 2 - 1).double()


def fingerprint(t: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    """単位ごとの (Σx², Σw·x, 非有限の数)(float64)。t はブロックごとに CPU へ読み戻す。"""
    n = t.shape[0]
    blk = max(1, BLOCK_BYTES // (w.numel() * t.element_size()))
    out = np.zeros((n, 3))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        x = t[s:e].cpu().double().reshape(e - s, -1)
        out[s:e, 2] = (~torch.isfinite(x)).sum(1).numpy()
        out[s:e, 0] = (x * x).sum(1).numpy()
        out[s:e, 1] = (x @ w).numpy()
    return out


def fp_rel(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """指紋の差から単位ごとの相対誤差の目安 max(|ΔΣx²|/(2Σx²_ref), |ΔΣw·x|/‖x_ref‖)。非有限は inf。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.maximum(np.abs(f[:, 0] - r[:, 0]) / (2 * r[:, 0]), np.abs(f[:, 1] - r[:, 1]) / np.sqrt(r[:, 0]))
    rel[(f[:, :2] == r[:, :2]).all(1)] = 0.0
    rel[(f[:, 2] > 0) | ~np.isfinite(f[:, :2]).all(1) | ~np.isfinite(rel)] = np.inf
    return rel


def unit_sums(g: torch.Tensor, r: torch.Tensor) -> np.ndarray:
    """単位ごとの (‖g−r‖², ‖r‖², ‖g‖², ⟨g,r⟩, g の非有限の数)(float64)。"""
    n = g.shape[0]
    blk = max(1, BLOCK_BYTES // (g[0].numel() * 4))
    out = np.zeros((n, 5))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        a, b = g[s:e].reshape(e - s, -1).double(), r[s:e].reshape(e - s, -1).double()
        out[s:e] = torch.stack([(a - b).square().sum(1), b.square().sum(1), a.square().sum(1), (a * b).sum(1),
                                (~torch.isfinite(a)).sum(1).double()], 1).numpy()
    return out


def urel(u: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(u[:, 0] / u[:, 1])
    rel[u[:, 0] == 0] = 0.0
    rel[~np.isfinite(rel) | (u[:, 4] > 0)] = np.inf
    return rel


def judge_units(rel: np.ndarray, noise: np.ndarray) -> dict:
    """単位ごとの誤り: rel > max(10·noise, 1e-5)(非有限の rel は inf なので誤り)。"""
    bad = ~(rel <= np.maximum(NOISE_MULT * noise, TAU_UNIT_MIN))
    ok = ~bad
    worst = np.argsort(-rel, kind="stable")[:5]
    return {"n_units": int(rel.size), "n_wrong": int(bad.sum()), "wrong_runs": runs_of(np.flatnonzero(bad))[:20],
            "max_rel": float(rel.max()), "max_rel_ok_units": float(rel[ok].max()) if ok.any() else None,
            "max_noise": float(noise.max()), "worst": [[int(i), float(rel[i]), float(noise[i])] for i in worst],
            "tail_rel": [float(x) for x in rel[-3:]]}


def grad_summary(u: np.ndarray, nz: np.ndarray) -> dict:
    d2, r2, g2, ab, nf = (float(x) for x in u.sum(0))
    rel, noise = rel_of(d2, r2), rel_of(float(nz[:, 0].sum()), float(nz[:, 1].sum()))
    tau = max(NOISE_MULT * noise, TAU_MIN)
    ok = math.isfinite(ab) and math.isfinite(g2) and g2 > 0 and r2 > 0
    units = judge_units(urel(u), urel(nz))
    return {"rel_l2": rel, "cos": ab / math.sqrt(g2 * r2) if ok else math.nan, "noise_rel_l2": noise, "tau": tau,
            "nonfinite": int(nf), "ref_norm": math.sqrt(r2),
            "class": "correct" if rel <= tau and units["n_wrong"] == 0 else "wrong", "units": units}


# ---------------------------------------------------------------- 子プロセス


class PeakSampler:
    """MPS の driver_allocated_memory を別スレッドで dt 秒ごとに読み、最大値を記録する(区間内のピークの目安)。"""

    def __init__(self, dev: str, dt: float = 0.1):
        self.on, self.dt, self.peak, self.stop = dev == "mps", dt, None, threading.Event()

    def __enter__(self):
        if self.on:
            self.peak = torch.mps.driver_allocated_memory()
            self.th = threading.Thread(target=self.run, daemon=True)
            self.th.start()
        return self

    def run(self):
        while not self.stop.wait(self.dt):
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())

    def __exit__(self, *exc):
        if self.on:
            self.stop.set()
            self.th.join(5)
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
        return False


class OpRec(TorchDispatchMode):
    """段階(product / head / backward)ごとに dispatch された aten 演算を数え、行列積の引数の shape / stride / 連続かを記録する。"""

    def __init__(self):
        super().__init__()
        self.phase, self.c, self.mm = "product", collections.defaultdict(collections.Counter), []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.c[self.phase][name] += 1
        if MM_OPS.search(name):
            self.mm.append({"phase": self.phase, "op": name, "args": [
                {"shape": list(x.shape), "stride": list(x.stride()), "contiguous": x.is_contiguous(), "numel": x.numel()}
                for x in args if isinstance(x, torch.Tensor)]})
        return func(*args, **(kwargs or {}))


def tb_frames(e: BaseException) -> list[str]:
    return [f"{Path(f.filename).name}:{f.lineno} in {f.name}: {(f.line or '').strip()}"
            for f in traceback.extract_tb(e.__traceback__)][-6:]


def forward(sp: dict, head: str, lv: dict, tgt, w, st: dict, dbg=frozenset()) -> torch.Tensor:
    """積 → (Y の指紋) → loss head。st に段階(product / fp_y / head)、時間、指紋を記録する。w があれば積の出力 Y0 に hook を付け、
    backward で G(bmm の backward に入る勾配)の指紋を取る。"""
    T = st["times"]
    st["stage"], t0 = "product", time.perf_counter()
    y0, y = product(sp, lv)
    if w is not None:
        def hook(g):
            t1 = time.perf_counter()
            st["fpG"] = fingerprint(g.detach(), w)
            T["g_fp_s"] = time.perf_counter() - t1
        y0.register_hook(hook)
    del y0
    dev = y.device.type
    sync(dev)
    T["product_s"] = time.perf_counter() - t0
    if w is not None:
        st["stage"], t0 = "fp_y", time.perf_counter()
        st["fpY"] = fingerprint(y.detach(), w)
        T["fp_y_s"] = time.perf_counter() - t0
    if "raise" in dbg:
        y = DebugRaise.apply(y)
    if "corrupt" in dbg:
        y = DebugCorrupt.apply(y)
    st["stage"], t0 = "head", time.perf_counter()
    ys = [y]
    del y
    loss = HEAD_FN[head](ys, tgt)
    sync(dev)
    T["head_s"] = time.perf_counter() - t0
    st["stage"] = "backward"
    return loss


def verify(d: torch.Tensor, c: torch.Tensor) -> dict:
    """device のコピー d をブロックごとに読み戻し、CPU の元 c とビット一致するか。"""
    n = c.shape[0]
    blk = max(1, BLOCK_BYTES // (c[0].numel() * c.element_size()))
    bad = [s for s in range(0, n, blk) if not torch.equal(d[s:s + blk].cpu(), c[s:s + blk])]
    return {"ok": not bad, "bad_block_starts": bad[:10], "block_units": blk}


def reference(sp: dict, head: str, dev: str, seed: int, chunk: int, w, base: dict | None = None) -> dict:
    """分割実行(chunk 単位ずつ forward + backward、同じ device)。base がなければ勾配を CPU の float32 で全体の形に並べて返し、
    あれば base との単位ごとの差(unit_sums)だけを返す(noise 用)。"""
    n, tk = sp["size"], TARGET[head]
    loss, fpy, fpg, pieces, t0 = 0.0, [], [], 0, time.perf_counter()
    grads = {k: torch.empty((n, *s)) for k, s in sp["leaves"].items()} if base is None else None
    noise = {k: np.zeros((n, 5)) for k in sp["leaves"]} if base is not None else None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lv = {k: make(sp, k, s, e, seed).to(dev).requires_grad_() for k in sp["leaves"]}
        tgt = to_target(make(sp, tk, s, e, seed).to(dev), tk) if tk else None
        st = {"times": {}}
        lo = forward(sp, head, lv, tgt, w, st)
        lo.backward()
        loss += float(lo.detach())
        fpy.append(st.get("fpY"))
        fpg.append(st.get("fpG"))
        for k, v in lv.items():
            g = v.grad.detach().cpu()
            if base is None:
                grads[k][s:e] = g
            else:
                noise[k][s:e] = unit_sums(g, base["grads"][k][s:e])
        pieces += 1
        del lv, tgt, lo, st
    empty(dev)
    cat = (lambda xs: np.concatenate(xs)) if w is not None else (lambda xs: None)
    return {"loss": loss, "fpY": cat(fpy), "fpG": cat(fpg), "grads": grads, "noise": noise, "pieces": pieces,
            "wall_s": time.perf_counter() - t0, "mem_end": mem(dev)}


def cpu_check(sp: dict, head: str, seed: int, w, ref: dict, k: int) -> dict:
    """基準の先頭 k 単位を CPU で 1 回に計算し、基準(device)と比べる(基準の演算の確認)。"""
    tk = TARGET[head]
    lv = {x: make(sp, x, 0, k, seed).requires_grad_() for x in sp["leaves"]}
    st = {"times": {}}
    forward(sp, head, lv, to_target(make(sp, tk, 0, k, seed), tk) if tk else None, w, st).backward()
    res = {"units": k, "grad_rel_l2": {x: rel_of(*unit_sums(ref["grads"][x][:k], v.grad)[:, :2].sum(0).tolist())
                                       for x, v in lv.items()}}
    if w is not None:
        res.update(y_rel_max=float(fp_rel(ref["fpY"][:k], st["fpY"]).max()),
                   g_rel_max=float(fp_rel(ref["fpG"][:k], st["fpG"]).max()))
    return res


def metrics(loss_full, ref: dict, ref2: dict, fpY, fpG, g_full) -> dict:
    lr, l2 = ref["loss"], ref2["loss"]
    noise = abs(l2 - lr) / abs(lr) if lr else math.inf
    m = {"loss": {"full": loss_full, "ref": lr, "ref2": l2, "noise": noise}, "y_check": None, "g_check": None, "grads": None}
    if loss_full is not None:
        d, tau = abs(loss_full - lr) / abs(lr) if lr else math.inf, max(NOISE_MULT * noise, TAU_UNIT_MIN)
        m["loss"].update({"rel_diff": d, "tau": tau, "class": "correct" if d <= tau else "wrong"})
    if fpY is not None and ref["fpY"] is not None:
        m["y_check"] = judge_units(fp_rel(fpY, ref["fpY"]), fp_rel(ref2["fpY"], ref["fpY"]))
    if fpG is not None and ref["fpG"] is not None:
        m["g_check"] = judge_units(fp_rel(fpG, ref["fpG"]), fp_rel(ref2["fpG"], ref["fpG"]))
    if g_full is not None:
        m["grads"] = {k: grad_summary(unit_sums(g, ref["grads"][k]), ref2["noise"][k]) for k, g in g_full.items()}
    return m


def put(job: dict, r: dict) -> None:
    Path(job["partial"]).write_text(json.dumps(r, ensure_ascii=False))


def job_case(job: dict) -> dict:
    """1 条件: 一括の forward + backward(anomaly の中)、基準 2 つ、指標。"""
    sp, head, dev, seed = spec(job["product"], job["size"], job["tiny"]), job["head"], job["device"], job["seed"]
    n, tk, tag = sp["size"], TARGET[head], f"[{job['name']}]"
    w = wvec(sp, seed) if job["fingerprint"] else None
    dbg = {k for k in ("raise", "crash", "corrupt") if job["name"] in job["debug"][k]}
    r = {"status": None, "stage": "setup", "debug": sorted(dbg) or None, "times": {}, "mem": {"start": mem(dev)},
         "recommended_max_memory": torch.mps.recommended_max_memory() if dev == "mps" else None}
    T = r["times"]
    put(job, r)
    # 入力: CPU で作り、device へ 1 回でコピーし、読み戻して確かめる
    t0 = time.perf_counter()
    cpu = {k: make(sp, k, 0, n, seed) for k in [*sp["leaves"], *([tk] if tk else [])]}
    T["make_s"], t0 = time.perf_counter() - t0, time.perf_counter()
    dv = {k: v.to(dev) for k, v in cpu.items()}
    sync(dev)
    T["copy_s"], t0 = time.perf_counter() - t0, time.perf_counter()
    r["inputs_verified"] = {k: verify(dv[k], v) for k, v in cpu.items()}
    T["verify_s"] = time.perf_counter() - t0
    del cpu
    lv = {k: dv.pop(k).requires_grad_() for k in sp["leaves"]}
    tgt = to_target(dv.pop(tk), tk) if tk else None
    r["mem"]["after_setup"] = mem(dev)
    ok_in = all(v["ok"] for v in r["inputs_verified"].values())
    print(f"{tag} setup make {T['make_s']:.1f}s copy {T['copy_s']:.1f}s verify {T['verify_s']:.1f}s "
          f"inputs_verified={ok_in}", flush=True)
    if not ok_in:
        print(f"{tag} WARNING: device copy of the inputs differs from the CPU original: {r['inputs_verified']}", flush=True)

    # 一括: forward と backward を anomaly mode の中で
    r["stage"] = "forward"
    put(job, r)
    st, box = {"times": T}, {}
    sampler = PeakSampler(dev)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        with torch.autograd.set_detect_anomaly(job["anomaly"], check_nan=False), sampler:
            t0 = time.perf_counter()
            try:
                sync(dev)
                box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
                r["loss_full"] = float(box["loss"].detach())
                T["forward_s"] = time.perf_counter() - t0
            except Exception as e:
                r.update(status="raised_forward", raised_in=st.get("stage"), forward_s_until_error=time.perf_counter() - t0,
                         frames=tb_frames(e), **exc_info(e))
            r["mem"]["after_forward"] = mem(dev)
            print(f"{tag} forward {r['status'] or 'ok'} "
                  f"{r.get('forward_s', r.get('forward_s_until_error', 0)):.1f}s loss={r.get('loss_full')}", flush=True)
            if "loss" in box:
                r["stage"] = "backward"
                put(job, r)
                if "crash" in dbg:
                    print(f"{tag} debug: os.abort() before backward (--debug-crash, test only)", file=sys.stderr, flush=True)
                    os.abort()
                t0 = time.perf_counter()
                try:
                    box["loss"].backward()
                    sync(dev)
                    r["status"], T["backward_s"] = "ok", time.perf_counter() - t0
                except Exception as e:
                    r.update(status="raised_backward", backward_s_until_error=time.perf_counter() - t0,
                             frames=tb_frames(e), **exc_info(e))
                r["mem"]["after_backward"] = mem(dev)
    r["mem"]["peak_sampled"] = sampler.peak
    msgs = [str(x.message) for x in W]
    an = [m for m in msgs if ANOMALY.search(m)]
    r["anomaly"] = parse_anomaly(an)
    r["other_warnings"] = [m[:300] for m in msgs if not ANOMALY.search(m)][:10]
    if r["status"] != "ok":
        a = r["anomaly"] or {}
        print(f"{tag} {r['status']} ({r.get('raised_in', 'backward')}): {r.get('exception_type')}: "
              f"{r.get('exception_msg', '')[:300]}\n{tag} backward node {a.get('backward_node')} / forward "
              f"{a.get('forward_location')}", flush=True)
    fpY, fpG = st.get("fpY"), st.get("fpG")
    g_full = {k: v.grad.detach().cpu() for k, v in lv.items()} if r["status"] == "ok" else None
    del lv, tgt, box, st
    gc.collect()
    empty(dev)
    r["mem"]["after_free"] = mem(dev)
    r["stage"] = "reference"
    put(job, r)

    # 基準(2 つ)と指標
    ref = None
    try:
        ref = reference(sp, head, dev, seed, job["chunk"], w)
        print(f"{tag} ref c{job['chunk']}: {ref['pieces']} pieces {ref['wall_s']:.1f}s loss={ref['loss']}", flush=True)
        ref2 = reference(sp, head, dev, seed, job["chunk2"], w, base=ref)
        print(f"{tag} ref c{job['chunk2']}: {ref2['pieces']} pieces {ref2['wall_s']:.1f}s loss={ref2['loss']}", flush=True)
        r["ref"] = {"status": "ok", "chunk": job["chunk"], "chunk2": job["chunk2"], "loss": ref["loss"],
                    "loss2": ref2["loss"], "pieces": [ref["pieces"], ref2["pieces"]],
                    "wall_s": [ref["wall_s"], ref2["wall_s"]], "mem_end": ref2["mem_end"]}
    except Exception as e:
        r["ref"] = {"status": "raised", "frames": tb_frames(e), **exc_info(e)}
        print(f"{tag} reference raised: {r['ref']['exception_type']}: {r['ref']['exception_msg'][:300]}", flush=True)
        ref = None
    if ref is not None:
        if dev != "cpu":
            try:
                r["ref"]["vs_cpu"] = cpu_check(sp, head, seed, w, ref, min(CPU_CHECK_UNITS, n))
            except Exception as e:
                r["ref"]["vs_cpu"] = {"error": repr(e)[:500]}
        r["metrics"] = metrics(r.get("loss_full"), ref, ref2, fpY, fpG, g_full)
    r["stage"] = "done"
    return r


def job_probe(job: dict) -> dict:
    """2 単位の入力で、積・head・backward で dispatch された aten 演算と行列積の引数を記録する。"""
    dev, res = job["device"], {}
    for p in job["products"]:
        sp, res[p] = spec(p, 2, job["tiny"]), {}
        for h in job["heads"]:
            tk = TARGET[h]
            try:
                lv = {k: make(sp, k, 0, 2, job["seed"]).to(dev).requires_grad_() for k in sp["leaves"]}
                tgt = to_target(make(sp, tk, 0, 2, job["seed"]).to(dev), tk) if tk else None
                rec = OpRec()
                with rec:
                    y0, y = product(sp, lv)
                    del y0
                    rec.phase = "head"
                    loss = HEAD_FN[h]([y], tgt)
                    del y
                    rec.phase = "backward"
                    loss.backward()
                res[p][h] = {**{ph: dict(rec.c[ph]) for ph in ("product", "head", "backward")}, "mm": rec.mm}
            except Exception as e:
                res[p][h] = {"error": repr(e)[:500]}
    print(f"[probe] {json.dumps({p: {h: v.get('mm', v.get('error')) for h, v in d.items() if h == 'ce'} for p, d in res.items()})}",
          flush=True)
    return {"status": "ok", "ops": res}


def child(path: Path) -> None:
    job = json.loads(path.read_text())
    torch.manual_seed(job["seed"])
    res = {"probe": job_probe, "case": job_case}[job["kind"]](job)
    Path(job["result"]).write_text(json.dumps(res, ensure_ascii=False))


# ---------------------------------------------------------------- 親プロセス


def run_child(work: Path, job: dict, timeout_min: float) -> dict:
    """子プロセスで 1 つのジョブを動かす。結果を書かずに終われば crashed / timeout(終了コード、stderr の末尾、途中の結果)。
    一括の条件が終わった後(基準の途中)で落ちたら、一括の status を残して ref.status を crashed / timeout にする。"""
    jd = work / "jobs"
    jd.mkdir(exist_ok=True)
    jf, rf, pf = jd / f"{job['name']}.json", jd / f"{job['name']}.result.json", jd / f"{job['name']}.partial.json"
    rf.unlink(missing_ok=True)
    pf.unlink(missing_ok=True)
    jf.write_text(json.dumps({**job, "result": str(rf), "partial": str(pf)}))
    print(f"== {job['name']}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(jf)],
                         stderr=subprocess.PIPE, text=True, errors="replace")
    tail: collections.deque[str] = collections.deque(maxlen=80)

    def pump():
        for line in p.stderr:
            sys.stderr.write(line)
            tail.append(line)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    status = None
    try:
        rc = p.wait(timeout=timeout_min * 60)
    except subprocess.TimeoutExpired:
        p.kill()
        rc, status = p.wait(), "timeout"
    th.join(10)
    if status is None and rf.exists():
        res = json.loads(rf.read_text())
    else:
        res = json.loads(pf.read_text()) if pf.exists() else {"stage": None}
        if res.get("stage") == "reference":
            res["ref"] = {"status": status or "crashed"}
        else:
            res.update(status=status or "crashed", crashed_stage=res.get("stage"))
    res.update(child_exit_code=rc, child_wall_s=time.perf_counter() - t0)
    if rc != 0 or status or res["status"] in ("crashed", "timeout"):
        res["stderr_tail"] = "".join(tail)[-4000:]
    return res


def kv(items: list[str], conv) -> dict:
    d = {}
    for it in items:
        k, _, v = it.partition(":")
        if k not in REAL or not v:
            raise SystemExit(f"PRODUCT:VALUE の形で指定する(PRODUCT は {list(REAL)}): {it}")
        d[k] = conv(v)
    return d


def brief(r: dict) -> str:
    m = r.get("metrics") or {}
    lo, y, g = m.get("loss") or {}, m.get("y_check"), m.get("g_check")
    gr = "; ".join(f"{k} rel {v['rel_l2']:.3g} wrong {v['units']['n_wrong']} {v['units']['wrong_runs'][:3]}"
                   for k, v in (m.get("grads") or {}).items())
    return (f"{r['status']} | loss rel {f3(lo.get('rel_diff'), '.2g')} (noise {f3(lo.get('noise'), '.2g')}) | "
            f"Y wrong {y and y['n_wrong']} {y and y['wrong_runs'][:3]} | G wrong {g and g['n_wrong']} "
            f"{g and g['wrong_runs'][:3]} | {gr or 'grads —'} | ref {(r.get('ref') or {}).get('status')}")


def save(out: Path, info: dict) -> None:
    (out / "bwd_heads.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    (out / "bwd_heads.md").write_text(render_md(info))


def parent(a) -> None:
    out = Path(a.out).resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    sizes = {**(TINY_SIZES if a.tiny else SIZES), **kv(a.sizes, lambda v: [int(x) for x in v.split(",")])}
    chunks = {**(TINY_CHUNKS if a.tiny else CHUNKS), **kv(a.chunks, int)}
    chunks = {p: (c, max(1, c // 2)) for p, c in chunks.items()}
    cases = []
    for p in a.products:
        for s in sizes[p]:
            sp = spec(p, s, a.tiny)
            for h in a.heads:
                cases.append({"name": f"{p}_{'B' if p == 'bmm' else 'n'}{s}_{h}", "product": p, "size": s, "head": h,
                              "numel": sp["numel"], "over_2p32": sp["over_2p32"], "unit": sp["unit"],
                              "boundary_unit": sp["boundary_unit"], "chunk": chunks[p][0], "chunk2": chunks[p][1],
                              "predicted": predict(sp, h), "mem_est": mem_est(sp, h)})
    cases.sort(key=lambda c: (c["numel"] > P32, a.products.index(c["product"]), c["size"], HEADS.index(c["head"])))
    names = {c["name"] for c in cases}
    for k in ("debug_raise_backward", "debug_crash", "debug_corrupt_grad"):
        if set(getattr(a, k)) - names:
            print(f"WARNING: --{k.replace('_', '-')} names unknown cases {sorted(set(getattr(a, k)) - names)}", flush=True)
    info = {
        "env": env_info(), "device": a.device, "tiny": a.tiny, "args": {k: v for k, v in vars(a).items() if k != "child"},
        "shapes": {p: (TINY if a.tiny else REAL)[p] for p in a.products}, "sizes": {p: sizes[p] for p in a.products},
        "chunks": {p: chunks[p] for p in a.products}, "dtype": "fp32", "seed": a.seed,
        "targets": "leaves, t and Z are generated per unit (bmm batch / relation sequence) on CPU from a Generator seeded "
                   "with (seed, kind, unit index), so they do not depend on the size or the chunking; T = softmax(Z, -1) "
                   "per unit. The full run copies the whole CPU tensors to the device once and verifies them by reading "
                   "back in blocks (inputs_verified); the references generate each chunk the same way",
        "reference": "same device and head, units in chunks of `chunk` (second reference: chunk2 = chunk // 2, noise "
                     "floor); loss summed in float64; leaf grads placed into full-size CPU float32 tensors (each element "
                     "comes from exactly one chunk); on MPS the first 2 units are also computed on CPU (ref.vs_cpu)",
        "fingerprint": None if a.no_fingerprint else
        "per unit (sum x^2, sum w*x) in float64 on CPU (w fixed +-1) of Y (the head input) and of G (grad at the product "
        "output, via a tensor hook), in the full run and both references; per-unit rel = max(|d sum x^2| / (2 sum x_ref^2), "
        "|d sum w*x| / ||x_ref||)",
        "tolerance": f"unit wrong if rel > max({NOISE_MULT:g} * noise_unit, {TAU_UNIT_MIN:g}) (grads: rel L2 per unit; Y, G: "
                     f"fingerprint rel); global grad rel L2: tau = max({NOISE_MULT:g} * noise, {TAU_MIN:g}); loss rel diff: "
                     f"max({NOISE_MULT:g} * noise, {TAU_UNIT_MIN:g}) (one wrong unit of 65537 moves the loss by only ~1.5e-5 "
                     "x its own relative error; the Y check is the per-unit forward test); noise = second reference vs first",
        "anomaly": None if a.no_anomaly else "forward+backward under set_detect_anomaly(True, check_nan=False)",
        "hypotheses": hypotheses(), "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "cases": [],
    }
    print(json.dumps({k: info[k] for k in ("device", "tiny", "shapes", "sizes", "chunks", "seed")}), flush=True)
    print("memory estimate per case (device peak of the full run; setup peak incl. the CPU original):", flush=True)
    for c in cases:
        e = c["mem_est"]
        print(f"  {c['name']:<24} Y {c['over_2p32']:.7g} x 2^32 ({e['Y_gb']:.2f} GB) -> device peak ~{e['device_peak_gb']:.1f} "
              f"GB (k={e['k_Y']}), setup ~{e['setup_peak_gb']:.1f} GB", flush=True)

    info["dispatch"] = run_child(work, {"kind": "probe", "name": "probe", "products": a.products, "heads": a.heads,
                                        "device": a.device, "tiny": a.tiny, "seed": a.seed}, a.timeout_min)
    save(out, info)
    base = {"kind": "case", "tiny": a.tiny, "device": a.device, "seed": a.seed, "anomaly": not a.no_anomaly,
            "fingerprint": not a.no_fingerprint,
            "debug": {"raise": a.debug_raise_backward, "crash": a.debug_crash, "corrupt": a.debug_corrupt_grad}}
    for c in cases:
        est = c["mem_est"]["device_peak_gb"]
        print(f"== {c['name']}: Y = {c['over_2p32']:.7g} x 2^32, est. device peak {est:.1f} GB; predicted: "
              f"{c['predicted']['text']}", flush=True)
        if a.skip_over_gb is not None and est > a.skip_over_gb:
            r = {"status": "skipped"}
        else:
            r = run_child(work, {**base, **{k: c[k] for k in ("name", "product", "size", "head", "chunk", "chunk2")}},
                          a.timeout_min)
        c = {**c, **r}
        info["cases"].append(c)
        save(out, info)
        print(f"== {c['name']}: {brief(c)}", flush=True)

    info["total_wall_s"] = time.perf_counter() - t_all
    save(out, info)
    if not a.keep_work:
        shutil.rmtree(work)
    print(f"done {info['total_wall_s']:.0f}s -> {out}", flush=True)


# ---------------------------------------------------------------- 要約(markdown)


def esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def runs_s(runs: list, k: int = 3) -> str:
    return ", ".join(f"{x}–{y}" if x != y else f"{x}" for x, y in runs[:k]) + (" …" if len(runs) > k else "")


def units_s(u: dict | None) -> str:
    if not u:
        return "—"
    return f"{u['n_wrong']}" + (f" ({runs_s(u['wrong_runs'])})" if u["wrong_runs"] else "")


def ops_s(d: dict | None) -> str:
    if not d:
        return "—"
    return ", ".join(f"{k.replace('aten.', '').replace('.default', '')}×{v}" for k, v in d.items())


def where(r: dict) -> str:
    st = r["status"]
    if st in ("crashed", "timeout"):
        last = [x for x in (r.get("stderr_tail") or "").splitlines() if x.strip()][-1:] or [""]
        return f"{st} in {r.get('crashed_stage')}, exit {r.get('child_exit_code')}: `{esc(last[0][:100])}`"
    if st == "raised_backward":
        a = r.get("anomaly")
        return (f"bwd: {a['backward_node']} @ `{esc(a['forward_location'] or '')}`" if a else "bwd: — (anomaly なし)") + \
            f"; {r.get('exception_type')}"
    if st == "raised_forward":
        return f"fwd ({r.get('raised_in')}): `{esc((r.get('frames') or ['?'])[-1][:90])}`; {r.get('exception_type')}"
    return "—"


def render_md(I: dict) -> str:
    e = I["env"]
    L = ["# 出力側の構造の積 + loss head: 学習 1 ステップの一括実行 vs 分割", "",
         "`probe_bwd_heads.py` の出力。数値は `bwd_heads.json`。", "", "## 環境と条件", "",
         f"- macOS {e['macos']} ({e['macos_build']})、{e['hw_model']} ({e['chip']}, {e['mem_bytes'] / 2**30:.0f} GiB)、"
         f"device {I['device']}" + ("、**tiny**(動作確認用の小さな形状)" if I["tiny"] else ""),
         f"- Python {e['python']}、torch {e['torch']}、numpy {e['numpy']}、コード git `{(e['git_rev'] or '')[:12]}`"
         f"{'(v2 に未 commit の変更あり)' if e.get('git_dirty_v2') else ''}",
         f"- 形状: {I['shapes']}、size {I['sizes']}、基準の chunk {I['chunks']}、fp32、seed {I['seed']}",
         f"- 乱数: {I['targets']}", f"- 基準: {I['reference']}", f"- 指紋: {I['fingerprint']}",
         f"- 許容誤差: {I['tolerance']}", f"- anomaly: {I['anomaly']}。PYTORCH_MPS_HIGH_WATERMARK_RATIO = "
                                         f"{I['mps_high_watermark_ratio']}", "", "## 予測(bmm の規則から)", ""]
    H = I["hypotheses"]
    L += [f"- {x}" for x in H["rules"]] + [f"- {k}: {H[k]}" for k in ("assumptions", "control", "above", "reduction_check")]
    L += [f"- head {k}: {v}" for k, v in H["heads"].items()]
    L += ["", "## 結果(一括の forward + backward)", "",
          "単位は bmm ではバッチ、relation では系列。Y / G の列は指紋で誤りと判定した単位の数(区間)。勾配は葉ごと。", "",
          "| 積 | size | Y / 2^32 | head | 予測 | status | 失敗の場所 | loss 相対差 (noise) | Y 誤り | G 誤り | "
          "勾配 rel L2 | 勾配の誤り単位 (区間) | fwd / bwd [s] | メモリ 推定 / 実測ピーク [GB] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in I["cases"]:
        m = r.get("metrics") or {}
        lo, gs = m.get("loss") or {}, m.get("grads") or {}
        ls = (f"{f3(lo.get('rel_diff'), '.2g')} ({f3(lo.get('noise'), '.1g')})" +
              (" **誤**" if lo.get("class") == "wrong" else "")) if lo else "—"
        grel = "<br>".join(f"{k}: {v['rel_l2']:.3g}" for k, v in gs.items()) or "—"
        gun = "<br>".join(f"{k}: {units_s(v['units'])}" for k, v in gs.items()) or "—"
        pk = (r.get("mem") or {}).get("peak_sampled")
        sz = f"{'B' if r['product'] == 'bmm' else 'n'}={r['size']}" + (" (≤ 2^32)" if r["numel"] <= P32 else "")
        L.append(f"| {r['product']} | {sz} | {r['over_2p32']:.7g} | {r['head']} | {esc(r['predicted']['text'])} | "
                 f"{r['status']} | {where(r)} | {ls} | {units_s(m.get('y_check'))} | {units_s(m.get('g_check'))} | "
                 f"{grel} | {gun} | {f3(r.get('times', {}).get('forward_s'), '.1f')} / "
                 f"{f3(r.get('times', {}).get('backward_s'), '.1f')} | {r['mem_est']['device_peak_gb']:.1f} / "
                 f"{f3(None if pk is None else pk / 1e9, '.1f')} |")
    L += ["", "## 基準と入力の確認", "",
          "| 条件 | 入力の確認 | 基準 | 分割数 | 所要 [s] | loss (基準) | 勾配の単位の noise 最大 | Y / G の noise 最大 | "
          "基準 vs CPU(先頭単位の勾配 rel L2、Y / G) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in I["cases"]:
        ref, m = r.get("ref") or {}, r.get("metrics") or {}
        iv = r.get("inputs_verified")
        ivs = "—" if iv is None else ("ok" if all(v["ok"] for v in iv.values()) else
                                      "**不一致** " + str({k: v["bad_block_starts"] for k, v in iv.items() if not v["ok"]}))
        nz = ", ".join(f"{k}: {v['units']['max_noise']:.2g}" for k, v in (m.get("grads") or {}).items()) or "—"
        yg = " / ".join(f3((m.get(k) or {}).get("max_noise"), ".2g") for k in ("y_check", "g_check"))
        vc = ref.get("vs_cpu") or {}
        vcs = "—" if not vc else (vc["error"][:80] if "error" in vc else
                                  ", ".join(f"{k}: {v:.2g}" for k, v in vc["grad_rel_l2"].items()) +
                                  f"; {f3(vc.get('y_rel_max'), '.2g')} / {f3(vc.get('g_rel_max'), '.2g')}")
        if ref.get("status") == "ok":
            L.append(f"| {r['name']} | {ivs} | ok (c{ref['chunk']}, c{ref['chunk2']}) | {ref['pieces'][0]}, "
                     f"{ref['pieces'][1]} | {ref['wall_s'][0]:.1f}, {ref['wall_s'][1]:.1f} | {ref['loss']:.9g} | {nz} | "
                     f"{yg} | {vcs} |")
        else:
            L.append(f"| {r['name']} | {ivs} | {ref.get('status', '—')} {esc(ref.get('exception_msg', '')[:120])} | | | | "
                     f"| | |")
    D = (I.get("dispatch") or {}).get("ops")
    if D:
        L += ["", "## dispatch された aten 演算(2 単位の入力)", "",
              "行列積の引数は shape / stride / 連続(c)か非連続(nc)か。backward の bmm の 1 つ目の引数が G のとき、その連続性と "
              "stride が規則 2 と 3 のどちらに当たるかを決める。", ""]
        for p, hs in D.items():
            first = next((v for v in hs.values() if "error" not in v), None)
            if first:
                L += [f"- {p} の積(forward): {ops_s(first['product'])}"]
            L += ["", "| 積 | head | head の forward | backward | 行列積の呼び出し |", "|---|---|---|---|---|"]
            for h, v in hs.items():
                if "error" in v:
                    L.append(f"| {p} | {h} | {esc(v['error'][:200])} | | |")
                    continue
                mm = "<br>".join(f"{x['phase']}: {x['op'].replace('aten.', '').replace('.default', '')}(" + ", ".join(
                    f"{tuple(y['shape'])}/{tuple(y['stride'])}/{'c' if y['contiguous'] else 'nc'}" for y in x["args"]) + ")"
                    for x in v["mm"])
                L.append(f"| {p} | {h} | {ops_s(v['head'])} | {ops_s(v['backward'])} | {mm} |")
    elif I.get("dispatch"):
        L += ["", f"dispatch の記録: {I['dispatch'].get('status')} (exit {I['dispatch'].get('child_exit_code')})"]
    bad = [r for r in I["cases"] if r["status"] != "ok" or (r.get("ref") or {}).get("status") not in ("ok", None)]
    if bad:
        L += ["", "## 例外・異常終了の詳細", ""]
        for r in bad:
            L += [f"### {r['name']}: {r['status']}" + (f"(基準: {r['ref']['status']})" if (r.get("ref") or {}).get(
                "status") not in ("ok", None) else ""), ""]
            if r.get("exception_type"):
                L += [f"- 例外({r.get('raised_in', 'backward')}): `{r['exception_type']}`: " +
                      esc(r["exception_msg"][:600])]
            if r.get("frames"):
                L += ["- traceback(末尾):", "", "```", *r["frames"], "```"]
            a = r.get("anomaly")
            if a:
                L += [f"- backward node: `{a['backward_node']}`、forward の演算: `{a['forward_location']}`",
                      "- forward の traceback(torch 以外のフレーム、末尾 8 個):", "", "```", *a["forward_frames"], "```"]
            ref = r.get("ref") or {}
            if ref.get("exception_type"):
                L += [f"- 基準の例外: `{ref['exception_type']}`: " + esc(ref["exception_msg"][:600])]
            if r.get("stderr_tail"):
                L += ["- stderr の末尾:", "", "```", *r["stderr_tail"].splitlines()[-15:], "```"]
            L.append("")
    if "total_wall_s" in I:
        L += ["", f"全体の所要時間 {I['total_wall_s']:.0f} 秒。", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="出力側の構造の積 + loss head の学習 1 ステップ(一括 vs 分割、MPS の事例)")
    ap.add_argument("--products", nargs="+", default=list(REAL), choices=list(REAL))
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    ap.add_argument("--sizes", nargs="+", default=[], metavar="PRODUCT:N[,N]",
                    help="積ごとの size(既定 bmm:65536,65537 relation:256,257)")
    ap.add_argument("--chunks", nargs="+", default=[], metavar="PRODUCT:N",
                    help="基準の chunk(既定 bmm:8192 relation:32)。2 つ目の基準はその半分")
    ap.add_argument("--device", default="mps", help="試験用に cpu も可")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-min", type=float, default=120, help="子プロセスごとの制限時間 [分]")
    ap.add_argument("--out", default=str(ROOT / "results" / "bwd_heads"))
    ap.add_argument("--tiny", "--scale-tiny", action="store_true",
                    help="動作確認用: bmm (B,8,4)@(B,4,8) B∈{6,7}、relation n∈{3,4} R=4 L=8 d=4、chunk 2 と 1")
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--no-fingerprint", action="store_true", help="Y と G の単位ごとの確認(CPU への読み戻し)を省く")
    ap.add_argument("--skip-over-gb", type=float, default=None,
                    help="見積りの device のピークがこれを超える条件を飛ばす(既定: 飛ばさない)")
    ap.add_argument("--keep-work", action="store_true", help="out/work/(ジョブと途中の結果の JSON)を消さない")
    ap.add_argument("--debug-raise-backward", nargs="*", default=[], metavar="CASE",
                    help="試験専用: この条件(例 bmm_B7_ce)の一括で backward の中で例外を出す")
    ap.add_argument("--debug-crash", nargs="*", default=[], metavar="CASE",
                    help="試験専用: この条件の子プロセスを backward の前に os.abort() で落とす")
    ap.add_argument("--debug-corrupt-grad", nargs="*", default=[], metavar="CASE",
                    help="試験専用: この条件の一括で head の backward の後に最後の単位の勾配に 1 を足す(判定の確認用)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(Path(a.child))
    else:
        parent(a)


if __name__ == "__main__":
    main()
