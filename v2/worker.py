"""1 回の実行 = 1 つの (演算 bmm | bmm_bwd, device, shape (M,K,N), dtype, レイアウト, 入力の種類, B, seed, 比較方式) を実行して判定する。

scan.py から 1 実行ごとに別プロセスで起動される(Metal 側の abort / segfault はプロセスごと落ちるため)。
結果は標準出力に `RESULT {json}` の 1 行で返す。単体でも動く:

    uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python worker.py [--device mps|cuda] \
        '{"M": 256, "K": 64, "N": 256, "cross": "out", "dtype": "fp32", "layout": "bT", "kind": "random",
          "B": 65537, "seed": 0, "compare": "sample", "tol": 1e-5}'

- 入力は論理 tensor a (B,M,K)、b (B,K,N) の値をレイアウトによらず同一に CPU で生成する(バッチごとに固定 seed)。
  そのうえで各レイアウトの格納を作る: 転置レイアウトは論理値を転置して連続化した格納 (B,c,r) を作って
  `.transpose(1,2)` の view を渡す。slice は先頭に余分な 1 バッチ(論理 index -1)を持つ格納の [1:] を渡す。
  したがって同じ seed なら全レイアウトで論理的に同一の行列積になる
- 正解は CPU fp64。比較時は同じバッチの論理値を CPU で生成し直して作る(デバイスから入力を読み戻さない)。
  全要素の正解を同時に持たない
- compare=sample はサンプルしたバッチだけ、full は全バッチを比較する
- レイアウト offset(v2): 小さい view が大きい storage_offset に居る場合。格納の先頭に pad_batches バッチの pad
  (バッチごとに違う定数。乱数は引かない)を a, b の両方に置き、[pad_batches:] を渡す。slice は pad_batches = 1 で中身が乱数の場合
- dtype は fp32 / fp16 / bf16
- 入力の種類 kind:
  - random: [-1,1] 一様(非ゼロ)
  - idx_{a|b}_{batch|rc}: index 符号化。符号化する側の各要素を (field mod P) + 1 で埋める。field は batch = バッチ index、
    rc = バッチ内の論理平坦 index (r*C + c)。反対側は選択行列(one-hot)で、idx_b なら a[m,k] = [k == m mod K]
    (out[i,m,n] = b[i, m mod K, n])、idx_a なら b[k,n] = [k == n mod K](out[i,m,n] = a[i, m, n mod K])。
    積は厳密なので、誤った出力要素の値から「実際に読まれた (バッチ | 行, 列)」を逆算できる
- 誤答したバッチについて、読み違いの仮説(転置 view の stride 無視、2**32 要素での巻き戻り、storage_offset 無視)
  ごとに出力を CPU で再現し、誤答要素のうち仮説と一致する割合を記録する
- op = bmm_bwd(v2): out = bmm(a, b) を a, b に requires_grad を付けて計算し、上流の勾配 G(出力と同じ形、連続、a・b と同じ作り方の
  乱数で seed の系列は別)から torch.autograd.grad で grad_a, grad_b を得る。正解は CPU fp64 でバッチごとに out_i = a_i b_i、
  grad_a_i = G_i b_i^T、grad_b_i = a_i^T G_i。3 つの tensor を同じ誤差の定義と許容誤差で比べ、cls は 3 つのうち最悪
  (wrong > truncated > ok)。tensor ごとの結果は cls_out / cls_grad_a / cls_grad_b と max_rel_err_<t>、bad_ranges_<t> など。
  v1 と同じ名前のキー(max_rel_err、bad_ranges、hist_* など)には最悪の tensor(worst_tensor)の値が入る。
  レイアウトは a, b に適用する。読み違いの仮説(hyp_match)は out についてだけ v1 と同じ検定をする。勾配については
  「autograd が G @ b^T と a^T @ G を転置 view で計算する」と仮定した簡易な検定だけ(hyp_match_grad_a / _grad_b:
  stride = その転置 view の stride 無視、wrap = G と相手のオペランドの巻き戻り)。index 符号化入力での勾配の逆算はしない
- device 依存の処理は Device クラスに分離(CUDA では TF32 を無効化し torch.cuda.synchronize で同期)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

T0 = time.perf_counter()
import torch  # noqa: E402

T_IMPORT = time.perf_counter() - T0

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# index 符号化の周期(素数)。fp32 は 2**24 未満で厳密に表せ、B・バッチ内の平坦 index の範囲で一意。
# fp16 は 2048 まで整数が厳密。2 の冪のずれ(例: 65536 バッチの巻き戻り)が delta ≡ 0 に隠れないよう素数にする。
# bf16 は仮数 7 bit なので整数が厳密なのは 256 まで(257 が最初に表せない整数。torch 2.14.0 で確認)→ 2**8 未満の最大の素数
IDX_PERIOD = {"fp32": 1048573, "fp16": 2039, "bf16": 251}
PAD_PERIOD = 61  # offset レイアウトの pad バッチの値の周期(素数)
CHUNK_ELEMS = 2**26  # 生成・転送のチャンク(要素数)
CMP_ELEMS = 2**25  # 比較のチャンク(fp64 1 テンソルあたり 256MB)
N_EVEN = 256  # サンプル比較の等間隔点の数
DECODE_MAX = 2**16  # index 符号化の逆算に使う誤答要素の上限(1 ブロックあたり)
N_HYP = 16  # 仮説検定に使う誤答バッチ数(先頭側・末尾側それぞれ)
WRAP = 2**32  # 巻き戻り仮説の周期(要素数)
POOL = ThreadPoolExecutor(8)  # バッチごとの乱数生成を並列化(結果はスレッド数によらない)


# ---------------------------------------------------------------------------- device 依存部分

class Device:
    """device 依存の処理(初期設定・同期・メモリ量・キャッシュ解放)をここにまとめる。"""

    def __init__(self, name: str):
        self.name = name
        self.dev = torch.device(name)
        if name == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()

    def sync(self):
        (torch.mps.synchronize if self.name == "mps" else torch.cuda.synchronize)()

    def mem(self) -> dict:
        if self.name == "mps":
            return {"dev_driver_alloc": torch.mps.driver_allocated_memory(),
                    "dev_current_alloc": torch.mps.current_allocated_memory()}
        return {"dev_driver_alloc": torch.cuda.memory_reserved(), "dev_current_alloc": torch.cuda.memory_allocated(),
                "dev_peak_alloc": torch.cuda.max_memory_allocated()}

    def empty_cache(self):
        (torch.mps.empty_cache if self.name == "mps" else torch.cuda.empty_cache)()

    def info(self) -> dict:
        d = {"device": self.name}
        if self.name == "cuda":
            d.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), tf32=False)
        return d


# ---------------------------------------------------------------------------- 入力の生成(device 非依存)

def parse_kind(kind: str):
    """'random' → None、'idx_b_rc' → ('b', 'rc')。"""
    if kind == "random":
        return None
    _, enc, field = kind.split("_")
    return enc, field


def op_shapes(spec):
    M, K, N = spec["M"], spec["K"], spec["N"]
    return {"a": (M, K), "b": (K, N), "g": (M, N)}  # g = bmm_bwd の上流の勾配 G(出力と同じ形)


def layout_of(spec, name):
    """pad = 格納の先頭に置く余分なバッチ数。slice は 1、offset は spec["pad_batches"](a, b とも同じ数)。
    レイアウトは a, b にだけ適用する(G は常に連続で pad なし)。"""
    if name == "g":
        return {"transposed": False, "pad": 0}
    lay = spec["layout"]
    pad = {"slice": 1, "offset": int(spec.get("pad_batches", 0))}.get(lay, 0)
    return {"transposed": (lay == "aT" and name == "a") or (lay == "bT" and name == "b"), "pad": pad}


def seed_key(seed, opnd, i):
    """CPU の Generator (mt19937) は seed の下位 32 bit しか使わないので 32 bit に詰める(torch 2.14.0 で確認)。
    a, b の i < 2**17 - 2 は v1 と同じキー(bit 31 = 0)。それ以外(--far の B > 2**17、3 本目の系列)は bit 31 = 1 の
    別の空間: seed 5 bit | opnd 2 bit | i 24 bit。"""
    if opnd < 2 and -1 <= i < 2**17 - 2:
        assert 0 <= seed < 2**13
        return ((seed * 2 + opnd) << 17) + (i + 2)
    assert 0 <= seed < 32 and 0 <= opnd < 4 and -1 <= i < 2**24 - 2
    return 2**31 | (seed << 26) | (opnd << 24) | (i + 2)


def gen_logical(spec, name, i0, i1):
    """論理バッチ index [i0, i1)(-1 は slice の余分バッチ)の論理値 (n, r, c) を CPU で作る。レイアウトに依存しない。"""
    r, c = op_shapes(spec)[name]
    dt = DTYPES[spec["dtype"]]
    n = i1 - i0
    enc = None if name == "g" else parse_kind(spec["kind"])  # G は入力の種類によらず乱数
    if enc is None:
        buf = torch.empty(n, r, c)
        opnd = {"a": 0, "b": 1, "g": 2}[name]

        def one(k):
            g = torch.Generator()
            g.manual_seed(seed_key(spec["seed"], opnd, i0 + k))
            torch.rand((r, c), generator=g, out=buf[k])

        list(POOL.map(one, range(n)))
        x = buf.mul_(2).sub_(1).to(dt)
        return x.masked_fill_(x == 0, 0.5)  # 全要素非ゼロ(切り捨て検出のため)
    e, field = enc
    P = IDX_PERIOD[spec["dtype"]]
    if name == e:
        if field == "batch":
            v = torch.arange(i0, i1, dtype=torch.int64).remainder(P).add_(1).to(dt)
            return v.view(n, 1, 1).expand(n, r, c).contiguous()
        v = torch.arange(r * c, dtype=torch.int64).remainder(P).add_(1).to(dt).view(1, r, c)
        return v.expand(n, r, c).contiguous()
    # 選択行列(one-hot)
    K = spec["K"]
    sel = torch.zeros(r, c, dtype=dt)
    if name == "a":  # a (M,K): a[m, m mod K] = 1
        sel[torch.arange(r), torch.arange(r) % K] = 1
    else:  # b (K,N): b[n mod K, n] = 1
        sel[torch.arange(c) % K, torch.arange(c)] = 1
    return sel.view(1, r, c).expand(n, r, c).contiguous()


def pad_values(j0, j1):
    """offset レイアウトの pad バッチ j(格納 index)の値: バッチ内は定数 2 + (1 + j mod 61) / 64。
    (2, 3) の非整数で、fp32 / fp16 / bf16 のどれでも厳密。乱数入力([-1, 1])とも index 符号(整数)とも選択行列とも重ならず、
    隣のバッチとも違う。周期が素数なので 2 の冪のずれでも同じ値に戻らない。乱数は引かない(数十万バッチでも安い)。"""
    return (torch.arange(j0, j1, dtype=torch.int64).remainder(PAD_PERIOD) + 1).double() / 64 + 2


def gen_storage(spec, name, j0, j1):
    """格納バッチ [j0, j1) の値(格納レイアウト)。offset レイアウトの pad 部分は定数、それ以外は論理値から作る。"""
    pad = layout_of(spec, name)["pad"]
    if spec["layout"] != "offset" or j0 >= pad:
        return to_storage(gen_logical(spec, name, j0 - pad, j1 - pad), spec, name)
    r, c = op_shapes(spec)[name]
    jm = min(j1, pad)
    parts = [pad_values(j0, jm).to(DTYPES[spec["dtype"]]).view(-1, 1, 1).expand(jm - j0, r, c)]
    if j1 > pad:
        parts.append(gen_logical(spec, name, 0, j1 - pad))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def storage_batch(spec, name, j):
    """格納バッチ j の論理値 (r, c) を fp64 で返す(仮説検定用)。pad は定数なので転置の有無によらない。"""
    pad = layout_of(spec, name)["pad"]
    if spec["layout"] == "offset" and j < pad:
        r, c = op_shapes(spec)[name]
        return pad_values(j, j + 1).to(DTYPES[spec["dtype"]]).double().view(1, 1).expand(r, c).contiguous()
    return gen_logical(spec, name, j - pad, j - pad + 1)[0].double()


def to_storage(x, spec, name):
    """論理値 (n,r,c) → 格納(転置レイアウトなら (n,c,r) の連続)。"""
    return x.transpose(1, 2).contiguous() if layout_of(spec, name)["transposed"] else x


def view_of(st, spec, name):
    """格納 → bmm に渡す論理 view。"""
    L = layout_of(spec, name)
    v = st[L["pad"]:]
    return v.transpose(1, 2) if L["transposed"] else v


# ---------------------------------------------------------------------------- サンプルと区間

def ranges(idxs):
    """昇順の index 列を連続区間 [(i0, i1), ...] にまとめる。"""
    out = []
    for i in idxs:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(x) for x in out]


def sample_indices(spec, B):
    """等間隔 N_EVEN 点 + 先頭 4 + 末尾 8 + 各オペランドの平坦 index が 2**31 / 2**32 要素 / 2**32 バイトの倍数を
    またぐバッチ(およびタイル分割のタイル境界 floor(2**32/(M*N)) の倍数)の前後 ±2。倍数は B の範囲にある全部(--far の
    2 × 2**32、3 × 2**32 も入る)。offset レイアウトでは a, b の平坦 index を格納の先頭から数える(pad の分だけずらす)。
    slice の 1 バッチのずれは v1 と同じく無視する(±2 の窓に入る)。"""
    M, K, N = spec["M"], spec["K"], spec["N"]
    s = DTYPES[spec["dtype"]].itemsize
    S = set(range(min(4, B))) | set(range(max(0, B - 8), B))
    S |= {round(x * (B - 1) / (N_EVEN - 1)) for x in range(N_EVEN)} if B > 1 else {0}
    marks = set()
    off = int(spec.get("pad_batches", 0)) if spec["layout"] == "offset" else 0
    for per, o in ((M * K, off), (K * N, off), (M * N, 0)):
        for q in (2**31, 2**32, 2**32 // s):
            k = max(1, (o * per) // q)
            while k * q < (o + B + 1) * per:
                marks.add((k * q) // per - o)
                k += 1
    tile = max(1, 2**32 // (M * N))
    marks |= {k * tile for k in range(1, B // tile + 1)}
    for m in marks:
        S |= {j for j in range(m - 2, m + 3) if 0 <= j < B}
    return sorted(S)


# ---------------------------------------------------------------------------- 誤差の集計

class Stats:
    def __init__(self, B, spec, decode=True):
        self.B, self.tol, self.spec = B, spec["tol"], spec
        self.enc = parse_kind(spec["kind"]) if decode else None  # 勾配は index 符号化の逆算をしない
        self.batch_err = {}  # 比較したバッチ i -> 正規化最大誤差 max|o-r| / max|r|
        self.n_elem = self.n_bad = self.n_bad_zero = self.n_nan = 0
        self.delta = Counter()  # idx: (読まれた field − 期待 field) mod P の分布(誤答要素)
        self.n_decoded = 0  # idx: 逆算した誤答要素の数(間引き後。delta・解釈不能の分母)
        self.n_invalid = 0  # idx: 符号として解釈できない誤答要素(非整数・範囲外。0 は zero に数える)
        self.rc_pairs = Counter()  # idx rc (fp32): ((期待 r,c), (読まれた r,c)) の例
        self.bad_pos = None

    def add(self, i0, o, r):
        """o: デバイスの出力(CPU, fp64)、r: fp64 正解。形 (n, M, N)。"""
        d = (o - r).abs()
        nan = torch.isnan(d)
        self.n_nan += int(nan.sum())
        d = d.masked_fill_(nan, math.inf)
        scale = r.abs().amax(dim=(1, 2)).clamp_min(1e-30)
        e = d.amax(dim=(1, 2)) / scale
        bad = d > (self.tol * scale).view(-1, 1, 1)
        self.n_elem += d.numel()
        nb = int(bad.sum())
        self.n_bad += nb
        for k, v in enumerate(e.tolist()):
            self.batch_err[i0 + k] = v
        if nb == 0:
            return
        self.n_bad_zero += int((bad & (o == 0)).sum())
        pos = bad.sum(dim=0)
        self.bad_pos = pos if self.bad_pos is None else self.bad_pos + pos
        if self.enc is not None:
            self.decode(o, r, bad)

    def decode(self, o, r, bad):
        P = IDX_PERIOD[self.spec["dtype"]]
        ob, rb = o[bad], r[bad]
        if ob.numel() > DECODE_MAX:  # 逆算は 1 ブロックあたり最大 DECODE_MAX 要素(等間隔に間引く)
            step = -(-ob.numel() // DECODE_MAX)
            ob, rb = ob[::step], rb[::step]
        self.n_decoded += ob.numel()
        rd = ob.round()
        valid = torch.isfinite(rd) & ((ob - rd).abs() <= 0.25) & (rd >= 1) & (rd <= P)
        self.n_invalid += int((~valid & (ob != 0)).sum())
        read = rd[valid].to(torch.int64) - 1
        exp_ = rb[valid].round().to(torch.int64) - 1  # 正解の値 = (期待 field mod P) + 1
        dl = (read - exp_) % P
        dl = torch.where(dl >= P // 2, dl - P, dl)
        u, cnt = torch.unique(dl, return_counts=True)
        self.delta.update(dict(zip(u.tolist(), cnt.tolist())))
        e, field = self.enc
        if field == "rc" and len(self.rc_pairs) < 64 and read.numel():
            R, C = op_shapes(self.spec)[e]
            if R * C < P:  # 一意に逆算できる(fp32)
                step = max(1, read.numel() // 16)
                for ex, rr in zip(exp_[::step][:16].tolist(), read[::step][:16].tolist()):
                    self.rc_pairs[((ex // C, ex % C), (rr // C, rr % C))] += 1

    def bad_batches(self):
        return sorted(i for i, v in self.batch_err.items() if not (v <= self.tol))

    def cls(self):
        if not self.bad_batches():
            return "ok"
        if self.enc is None and self.n_bad and self.n_bad_zero / self.n_bad >= 0.9:
            return "truncated"  # idx 入力は選択行列の読み違いでも 0 が出るので truncated にしない
        return "wrong"

    def summary(self, nbins=64):
        errs = self.batch_err
        bad = self.bad_batches()
        edges = [round(k * self.B / nbins) for k in range(nbins + 1)]
        hb, hc = [0] * nbins, [0] * nbins
        for i, v in errs.items():
            k = min(nbins - 1, i * nbins // self.B)
            hc[k] += 1
            if not (v <= self.tol):
                hb[k] += 1
        mx = max(errs.values()) if errs else None
        rg = ranges(bad)
        res = {
            "max_rel_err": mx if mx is None or math.isfinite(mx) else "inf",
            "n_batches_compared": len(errs), "n_batches_bad": len(bad),
            "n_elem_compared": self.n_elem, "n_elem_bad": self.n_bad,
            "frac_elem_bad": self.n_bad / max(self.n_elem, 1),
            "frac_bad_elem_zero": self.n_bad_zero / self.n_bad if self.n_bad else 0.0,
            "n_nan": self.n_nan,
            "bad_ranges": [list(x) for x in rg[:60]], "n_bad_ranges": len(rg),
            "first_bad_batch": bad[0] if bad else None,
            "hist_edges": edges, "hist_bad": hb, "hist_compared": hc,
        }
        if self.bad_pos is not None:
            res["bad_pos_rows"] = int((self.bad_pos.sum(dim=1) > 0).sum())
            res["bad_pos_cols"] = int((self.bad_pos.sum(dim=0) > 0).sum())
        if self.enc is not None:
            res["idx_period"] = IDX_PERIOD[self.spec["dtype"]]
            res["idx_delta_top"] = [[k, v] for k, v in self.delta.most_common(12)]
            res["idx_n_decoded"] = self.n_decoded
            res["idx_frac_invalid"] = self.n_invalid / self.n_decoded if self.n_decoded else 0.0
            res["idx_rc_examples"] = [[list(a), list(b), n] for (a, b), n in self.rc_pairs.most_common(12)]
        return res


# ---------------------------------------------------------------------------- 読み違いの仮説

def hyp_operand(spec, name, i, mode):
    """論理バッチ i の位置でカーネルが読んだと仮定するオペランド値 (r, c) を CPU で作る。mode は次の集合:
    stride: 転置 view の stride を無視し、格納 (c, r) の連続バッファを (r, c) として読む
    wrap: 格納の平坦 index が 2**32 要素で巻き戻る(バッチあたり要素数が 2**32 を割り切る場合のみ)。mod なので
          2 × 2**32 より先でも、offset レイアウトの大きな storage_offset でも同じ式(pad の領域に戻れば pad の定数を読む)
    nooffset: storage_offset を無視して格納の先頭から読む(slice なら 1 バッチ前、offset なら pad_batches バッチ前 = pad の定数)"""
    r, c = op_shapes(spec)[name]
    L = layout_of(spec, name)
    j = i + L["pad"]  # 格納バッチ index
    if "nooffset" in mode:
        j -= L["pad"]
    if "wrap" in mode and WRAP % (r * c) == 0:
        j = (j * r * c % WRAP) // (r * c)
    x = storage_batch(spec, name, j)
    if "stride" in mode and L["transposed"]:
        return x.t().contiguous().view(r, c)  # 格納 (c, r) を (r, c) として読む
    return x


HYPS = {"stride": {"stride"}, "wrap": {"wrap"}, "wrap+stride": {"wrap", "stride"}, "nooffset": {"nooffset"}}
GRAD_HYPS = {"stride": {"stride"}, "wrap": {"wrap"}, "wrap+stride": {"wrap", "stride"}}


def hyp_operand_T(spec, name, i, mode):
    """backward で使う転置 view(b^T または a^T。形 (c, r))を、カーネルが読んだと仮定する値で返す。
    stride: 転置 view の stride を無視して格納をそのまま (c, r) として読む。格納が連続 (r, c) なら読み違いになり、
    転置レイアウト(格納がすでに (c, r) の連続)なら正しい値と同じになる。wrap は hyp_operand と同じ。"""
    r, c = op_shapes(spec)[name]
    x = hyp_operand(spec, name, i, mode - {"stride"})  # 論理値 (r, c)
    if "stride" in mode and not layout_of(spec, name)["transposed"]:
        return x.contiguous().view(c, r)
    return x.t()


def test_grad_hypotheses(spec, grads, bads, tol):
    """勾配の簡易な仮説検定。autograd が grad_a = G @ b^T、grad_b = a^T @ G を転置 view のまま bmm に渡すと仮定する。"""
    res = {}
    for t, other in (("grad_a", "b"), ("grad_b", "a")):
        bad = bads[t]
        if not bad:
            continue
        sel = sorted(set(bad[:N_HYP] + bad[-N_HYP:]))
        hit, nbad = Counter(), 0
        for i in sel:
            o = grads[t][i:i + 1].cpu().double()[0]
            g = gen_logical(spec, "g", i, i + 1)[0].double()
            x = gen_logical(spec, other, i, i + 1)[0].double()
            ref = g @ x.t() if t == "grad_a" else x.t() @ g
            scale = float(ref.abs().max().clamp_min(1e-30))
            badm = (o - ref).abs() > tol * scale
            nbad += int(badm.sum())
            for h, mode in GRAD_HYPS.items():
                gh = hyp_operand(spec, "g", i, mode - {"stride"})
                xh = hyp_operand_T(spec, other, i, mode)
                hy = gh @ xh if t == "grad_a" else xh @ gh
                hit[h] += int((((o - hy).abs() <= tol * max(scale, float(hy.abs().max()))) & badm).sum())
        res[f"hyp_match_{t}"] = {h: round(hit[h] / nbad, 6) if nbad else None for h in GRAD_HYPS}
        res[f"hyp_n_bad_elem_{t}"] = nbad
    return res


def test_hypotheses(spec, out, bad, tol):
    """誤答バッチのうち最大 2*N_HYP 個で、各仮説の出力と実際の出力を比べる。
    返り値の hyp_match: 仮説 → 誤答要素のうち仮説の出力と許容誤差内で一致する割合。"""
    sel = sorted(set(bad[:N_HYP] + bad[-N_HYP:]))
    hit = Counter()
    nbad = 0
    for i in sel:
        o = out[i:i + 1].cpu().double()[0]
        ref = gen_logical(spec, "a", i, i + 1)[0].double() @ gen_logical(spec, "b", i, i + 1)[0].double()
        scale = float(ref.abs().max().clamp_min(1e-30))
        badm = (o - ref).abs() > tol * scale
        nbad += int(badm.sum())
        for h, mode in HYPS.items():
            hy = hyp_operand(spec, "a", i, mode) @ hyp_operand(spec, "b", i, mode)
            hs = float(hy.abs().max())
            hit[h] += int((((o - hy).abs() <= tol * max(scale, hs)) & badm).sum())
    return {"hyp_n_batches": len(sel), "hyp_n_bad_elem": nbad,
            "hyp_match": {h: round(hit[h] / nbad, 6) if nbad else None for h in HYPS}}


# ---------------------------------------------------------------------------- 実行

def run(spec, D: Device):
    tm = {"import": round(T_IMPORT, 3)}
    B, tol = spec["B"], spec["tol"]
    M, K, N = spec["M"], spec["K"], spec["N"]
    dt = DTYPES[spec["dtype"]]
    res = {"torch": torch.__version__, "python": platform.python_version(), "macos": platform.mac_ver()[0], **D.info()}

    bwd = spec.get("op", "bmm") == "bmm_bwd"
    names = ("a", "b", "g") if bwd else ("a", "b")
    # --- 入力: チャンクごとに論理値を CPU で生成 → 格納レイアウトにして device の格納テンソルへ copy_ ---
    t = time.perf_counter()
    store = {}
    for name in names:
        r, c = op_shapes(spec)[name]
        L = layout_of(spec, name)
        shp = (B + L["pad"], c, r) if L["transposed"] else (B + L["pad"], r, c)
        st = torch.empty(*shp, dtype=dt, device=D.dev)
        ch = max(1, CHUNK_ELEMS // (r * c))
        for j0 in range(0, B + L["pad"], ch):
            j1 = min(B + L["pad"], j0 + ch)
            st[j0:j1].copy_(gen_storage(spec, name, j0, j1))
        store[name] = st
        res[f"{name}_storage_shape"] = list(st.shape)
    D.sync()
    tm["gen_upload"] = time.perf_counter() - t
    a, b = view_of(store["a"], spec, "a"), view_of(store["b"], spec, "b")
    res["a_stride"], res["b_stride"] = list(a.stride()), list(b.stride())
    res["a_offset"], res["b_offset"] = a.storage_offset(), b.storage_offset()
    res["a_contig"], res["b_contig"] = a.is_contiguous(), b.is_contiguous()
    if bwd:  # view を leaf として勾配を取る(格納は requires_grad なしなので view が leaf になる)
        a.requires_grad_(True)
        b.requires_grad_(True)

    # --- 演算 ---
    t = time.perf_counter()
    try:
        out = torch.bmm(a, b)
        D.sync()
    except Exception as e:  # noqa: BLE001
        tm["bmm"] = time.perf_counter() - t
        res.update(cls="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}", stage="bmm")
        return res, tm
    tm["bmm"] = time.perf_counter() - t
    tensors = {"out": out}
    if bwd:
        t = time.perf_counter()
        try:
            ga, gb = torch.autograd.grad(out, (a, b), store["g"])
            D.sync()
        except Exception as e:  # noqa: BLE001
            tm["bwd"] = time.perf_counter() - t
            res.update(cls="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}", stage="bwd")
            return res, tm
        tm["bwd"] = time.perf_counter() - t
        out = out.detach()
        tensors = {"out": out, "grad_a": ga, "grad_b": gb}
        for k in ("grad_a", "grad_b"):
            res[f"{k}_shape"], res[f"{k}_stride"] = list(tensors[k].shape), list(tensors[k].stride())
            res[f"{k}_contig"] = tensors[k].is_contiguous()
    res.update(D.mem())
    if spec.get("test_corrupt"):  # テスト専用: 比較が壊れた出力を検出できることの確認。行にも残る
        with torch.no_grad():
            tensors[spec["test_corrupt"].get("tensor", "out")][int(spec["test_corrupt"]["batch"])].add_(1)
        res["test_corrupt"] = spec["test_corrupt"]
    res["out_shape"], res["out_stride"] = list(out.shape), list(out.stride())

    samp = sample_indices(spec, B)
    # --- 入力の読み戻し確認(サンプルしたバッチ。転送自体が壊れていないか) ---
    t = time.perf_counter()
    mism = {}
    for name in names:
        pad = layout_of(spec, name)["pad"]
        nbad = 0
        for i0, i1 in ranges(samp):
            got = store[name][i0 + pad:i1 + pad].cpu()
            want = to_storage(gen_logical(spec, name, i0, i1), spec, name)
            nbad += int((got != want).flatten(1).any(dim=1).sum())
        if spec["layout"] == "offset" and pad:  # pad の側も確認: 先頭・末尾と、格納の平坦 index が 2**31 / 2**32 をまたぐ付近
            r_, c_ = op_shapes(spec)[name]
            pj = set(range(min(3, pad))) | set(range(max(0, pad - 3), pad))
            for q in (2**31, 2**32):
                pj |= {j for k in range(1, pad * r_ * c_ // q + 1) for j in range(k * q // (r_ * c_) - 1, k * q // (r_ * c_) + 2)
                       if 0 <= j < pad}
            for j0, j1 in ranges(sorted(pj)):
                nbad += int((store[name][j0:j1].cpu() != gen_storage(spec, name, j0, j1)).flatten(1).any(dim=1).sum())
            res.setdefault("pad_batches_checked", {})[name] = len(pj)
        mism[name] = nbad
    res["input_mismatch_batches"] = mism
    tm["input_check"] = time.perf_counter() - t
    del a, b, store
    D.empty_cache()

    # --- 比較 ---
    all_stats = {k: Stats(B, spec, decode=(k == "out")) for k in tensors}
    stats = all_stats["out"]
    tm.update(download=0.0, regen=0.0, ref=0.0, metric=0.0)
    cb = max(1, CMP_ELEMS // max(M * N, M * K, K * N))
    if spec["compare"] == "full":
        blocks = [(i, min(B, i + cb)) for i in range(0, B, cb)]
    else:
        blocks = [(i0 + k, min(i1, i0 + k + cb)) for i0, i1 in ranges(samp) for k in range(0, i1 - i0, cb)]
    res["compare_n_blocks"] = len(blocks)
    try:
        for i0, i1 in blocks:
            t = time.perf_counter()
            got = {k: v[i0:i1].detach().cpu().double() for k, v in tensors.items()}
            t1 = time.perf_counter()
            ra = gen_logical(spec, "a", i0, i1).double()
            rb = gen_logical(spec, "b", i0, i1).double()
            rg = gen_logical(spec, "g", i0, i1).double() if bwd else None
            t2 = time.perf_counter()
            refs = {"out": torch.bmm(ra, rb)}
            if bwd:
                refs["grad_a"] = torch.bmm(rg, rb.transpose(1, 2))
                refs["grad_b"] = torch.bmm(ra.transpose(1, 2), rg)
            t3 = time.perf_counter()
            for k in tensors:
                all_stats[k].add(i0, got[k], refs[k])
            del got, refs
            t4 = time.perf_counter()
            tm["download"] += t1 - t
            tm["regen"] += t2 - t1
            tm["ref"] += t3 - t2
            tm["metric"] += t4 - t3
        rank = {"ok": 0, "truncated": 1, "wrong": 2}
        sums = {k: st.summary() for k, st in all_stats.items()}
        err_of = lambda k: math.inf if sums[k]["max_rel_err"] == "inf" else (sums[k]["max_rel_err"] or 0.0)  # noqa: E731
        worst = max(all_stats, key=lambda k: (rank[all_stats[k].cls()], err_of(k)))
        res.update(sums[worst])  # bmm では out そのもの(v1 と同じ)
        if bwd:
            res["worst_tensor"] = worst
            for k, sm in sums.items():
                res[f"cls_{k}"] = all_stats[k].cls()
                for key in ("max_rel_err", "n_batches_bad", "n_elem_bad", "frac_elem_bad", "frac_bad_elem_zero",
                            "bad_ranges", "n_bad_ranges", "first_bad_batch"):
                    res[f"{key}_{k}"] = sm[key]
        bad = stats.bad_batches()
        t = time.perf_counter()
        if bad:
            res.update(test_hypotheses(spec, out, bad, tol))
        if bwd:
            res.update(test_grad_hypotheses(spec, tensors, {k: all_stats[k].bad_batches() for k in tensors}, tol))
        if bad or bwd:
            tm["hyp"] = time.perf_counter() - t
    except Exception as e:  # noqa: BLE001
        res.update(cls="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}", stage="compare")
        return res, tm
    res["cls"] = "input_corrupt" if any(v > 0 for v in mism.values()) else all_stats[worst].cls()
    return res, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["mps", "cuda"], default=None)
    ap.add_argument("spec")
    args = ap.parse_args()
    spec = json.loads(args.spec)
    dev = args.device or spec.get("device") or ("cuda" if torch.cuda.is_available() else "mps")
    spec["device"] = dev
    t = time.perf_counter()
    try:
        D = Device(dev)
        res, tm = run(spec, D)
    except Exception as e:  # noqa: BLE001  (確保失敗など bmm 以外の段階)
        res, tm = {"cls": "error", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}",
                   "stage": "setup", "torch": torch.__version__, "device": dev}, {}
    tm["total"] = time.perf_counter() - t
    res["time"] = {k: round(v, 3) for k, v in tm.items()}
    res["max_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print("RESULT " + json.dumps(res, default=str), flush=True)
    os._exit(0)  # スレッドプールの後始末を待たない


if __name__ == "__main__":
    main()
