"""1 回の実行 = 1 つの (演算, device, shape (M,K,N), dtype, レイアウト, 入力の種類, B, seed, 比較方式) を実行して判定する。

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
- 入力の種類 kind:
  - random: [-1,1] 一様(非ゼロ)
  - idx_{a|b}_{batch|rc}: index 符号化。符号化する側の各要素を (field mod P) + 1 で埋める。field は batch = バッチ index、
    rc = バッチ内の論理平坦 index (r*C + c)。反対側は選択行列(one-hot)で、idx_b なら a[m,k] = [k == m mod K]
    (out[i,m,n] = b[i, m mod K, n])、idx_a なら b[k,n] = [k == n mod K](out[i,m,n] = a[i, m, n mod K])。
    積は厳密なので、誤った出力要素の値から「実際に読まれた (バッチ | 行, 列)」を逆算できる
- 誤答したバッチについて、読み違いの仮説(転置 view の stride 無視、2**32 要素での巻き戻り、storage_offset 無視)
  ごとに出力を CPU で再現し、誤答要素のうち仮説と一致する割合を記録する
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

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
# index 符号化の周期(素数)。fp32 は 2**24 未満で厳密に表せ、B・バッチ内の平坦 index の範囲で一意。
# fp16 は 2048 まで整数が厳密。2 の冪のずれ(例: 65536 バッチの巻き戻り)が delta ≡ 0 に隠れないよう素数にする
IDX_PERIOD = {"fp32": 1048573, "fp16": 2039}
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
    return {"a": (M, K), "b": (K, N)}


def layout_of(spec, name):
    lay = spec["layout"]
    return {"transposed": (lay == "aT" and name == "a") or (lay == "bT" and name == "b"),
            "pad": 1 if lay == "slice" else 0}


def seed_key(seed, opnd, i):
    """CPU の Generator (mt19937) は seed の下位 32 bit しか使わないので 32 bit に詰める。"""
    assert -1 <= i < 2**17 - 2 and 0 <= seed < 2**14
    return ((seed * 2 + opnd) << 17) + (i + 2)


def gen_logical(spec, name, i0, i1):
    """論理バッチ index [i0, i1)(-1 は slice の余分バッチ)の論理値 (n, r, c) を CPU で作る。レイアウトに依存しない。"""
    r, c = op_shapes(spec)[name]
    dt = DTYPES[spec["dtype"]]
    n = i1 - i0
    enc = parse_kind(spec["kind"])
    if enc is None:
        buf = torch.empty(n, r, c)
        opnd = 0 if name == "a" else 1

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
    またぐバッチ(およびタイル分割のタイル境界 floor(2**32/(M*N)) の倍数)の前後 ±2。"""
    M, K, N = spec["M"], spec["K"], spec["N"]
    s = DTYPES[spec["dtype"]].itemsize
    S = set(range(min(4, B))) | set(range(max(0, B - 8), B))
    S |= {round(x * (B - 1) / (N_EVEN - 1)) for x in range(N_EVEN)} if B > 1 else {0}
    marks = set()
    for per in (M * K, K * N, M * N):
        for q in (2**31, 2**32, 2**32 // s):
            k = 1
            while k * q < (B + 1) * per:
                marks.add((k * q) // per)
                k += 1
    tile = max(1, 2**32 // (M * N))
    marks |= {k * tile for k in range(1, B // tile + 1)}
    for m in marks:
        S |= {j for j in range(m - 2, m + 3) if 0 <= j < B}
    return sorted(S)


# ---------------------------------------------------------------------------- 誤差の集計

class Stats:
    def __init__(self, B, spec):
        self.B, self.tol, self.spec = B, spec["tol"], spec
        self.enc = parse_kind(spec["kind"])
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
    wrap: 格納の平坦 index が 2**32 要素で巻き戻る(バッチあたり要素数が 2**32 を割り切る場合のみ)
    nooffset: storage_offset を無視(slice で 1 バッチ前の格納を読む)"""
    r, c = op_shapes(spec)[name]
    L = layout_of(spec, name)
    j = i + L["pad"]  # 格納バッチ index
    if "nooffset" in mode:
        j -= L["pad"]
    if "wrap" in mode and WRAP % (r * c) == 0:
        j = (j * r * c % WRAP) // (r * c)
    li = j - L["pad"]
    x = gen_logical(spec, name, li, li + 1)[0].double()
    if "stride" in mode and L["transposed"]:
        return x.t().contiguous().view(r, c)  # 格納 (c, r) を (r, c) として読む
    return x


HYPS = {"stride": {"stride"}, "wrap": {"wrap"}, "wrap+stride": {"wrap", "stride"}, "nooffset": {"nooffset"}}


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

    # --- 入力: チャンクごとに論理値を CPU で生成 → 格納レイアウトにして device の格納テンソルへ copy_ ---
    t = time.perf_counter()
    store = {}
    for name in ("a", "b"):
        r, c = op_shapes(spec)[name]
        L = layout_of(spec, name)
        shp = (B + L["pad"], c, r) if L["transposed"] else (B + L["pad"], r, c)
        st = torch.empty(*shp, dtype=dt, device=D.dev)
        ch = max(1, CHUNK_ELEMS // (r * c))
        for j0 in range(0, B + L["pad"], ch):
            j1 = min(B + L["pad"], j0 + ch)
            st[j0:j1].copy_(to_storage(gen_logical(spec, name, j0 - L["pad"], j1 - L["pad"]), spec, name))
        store[name] = st
    D.sync()
    tm["gen_upload"] = time.perf_counter() - t
    a, b = view_of(store["a"], spec, "a"), view_of(store["b"], spec, "b")
    res["a_stride"], res["b_stride"] = list(a.stride()), list(b.stride())
    res["a_offset"], res["b_offset"] = a.storage_offset(), b.storage_offset()
    res["a_contig"], res["b_contig"] = a.is_contiguous(), b.is_contiguous()

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
    res.update(D.mem())
    res["out_shape"], res["out_stride"] = list(out.shape), list(out.stride())

    samp = sample_indices(spec, B)
    # --- 入力の読み戻し確認(サンプルしたバッチ。転送自体が壊れていないか) ---
    t = time.perf_counter()
    mism = {}
    for name in ("a", "b"):
        pad = layout_of(spec, name)["pad"]
        nbad = 0
        for i0, i1 in ranges(samp):
            got = store[name][i0 + pad:i1 + pad].cpu()
            want = to_storage(gen_logical(spec, name, i0, i1), spec, name)
            nbad += int((got != want).flatten(1).any(dim=1).sum())
        mism[name] = nbad
    res["input_mismatch_batches"] = mism
    tm["input_check"] = time.perf_counter() - t
    del a, b, store
    D.empty_cache()

    # --- 比較 ---
    stats = Stats(B, spec)
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
            o = out[i0:i1].cpu().double()
            t1 = time.perf_counter()
            ra = gen_logical(spec, "a", i0, i1).double()
            rb = gen_logical(spec, "b", i0, i1).double()
            t2 = time.perf_counter()
            r = torch.bmm(ra, rb)
            t3 = time.perf_counter()
            stats.add(i0, o, r)
            t4 = time.perf_counter()
            tm["download"] += t1 - t
            tm["regen"] += t2 - t1
            tm["ref"] += t3 - t2
            tm["metric"] += t4 - t3
        res.update(stats.summary())
        bad = stats.bad_batches()
        if bad:
            t = time.perf_counter()
            res.update(test_hypotheses(spec, out, bad, tol))
            tm["hyp"] = time.perf_counter() - t
    except Exception as e:  # noqa: BLE001
        res.update(cls="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}", stage="compare")
        return res, tm
    if any(v > 0 for v in mism.values()):
        cls = "input_corrupt"
    elif not stats.bad_batches():
        cls = "ok"
    elif stats.enc is None and stats.n_bad and stats.n_bad_zero / stats.n_bad >= 0.9:
        cls = "truncated"  # idx 入力は選択行列の読み違いでも 0 が出るので truncated にしない
    else:
        cls = "wrong"
    res["cls"] = cls
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
