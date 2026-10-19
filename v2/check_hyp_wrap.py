"""読み違い仮説(worker.hyp_operand / hyp_operand_T)と誤答位置の記録(worker.Stats の bad_batch_detail)の単体テスト。CPU だけ。

    uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python check_hyp_wrap.py

1. worker.WRAP を小さい値に差し替え、巻き戻りがバッチの途中・行の途中に来る小さい shape(2 の冪でないもの、*-250x* を含む)と
   全レイアウト(contig / bT / aT / slice / offset)・a / b の両方で、hyp_operand(mode) と hyp_operand_T(mode) を次の直接計算と比べる:
   bmm に渡す view(view_of(gen_storage(...)))の stride と storage_offset から、論理要素 (i, r, c) の格納の平坦 index f を求める。
   stride(stride 無視)なら view のバッチの先頭から行優先で数えた f、nooffset なら storage_offset = 0 とし、wrap なら f mod WRAP。
   格納全体を平らにした値の f 番目を読む。バッチ境界との位置関係を使わない式なので、一致すれば仮説は要素単位で平坦 index の巻き戻り
2. in-250x250x64 で WRAP = 2·62500 + 29796(要素 WRAP がバッチ 2 の位置 29796 = 行 119・列 46)とし、巻き戻った a から作った出力を
   Stats と test_hypotheses に通す。実機の B = 68720 / 68721 と要素 2**32(バッチ 68719 の位置 29796)の縮小版で、
   points_nonpow2.md の予測と同じ行・位置・delta になることを確かめる(バッチ 2, 3 が実機のバッチ 68719, 68720 に当たる)
3. Stats の先頭側・末尾側の記録(ブロックをまたぐ場合、誤答バッチが少なく重なる場合)
"""

from __future__ import annotations

import itertools

import torch

import worker as W

MODES = [set(), {"stride"}, {"wrap"}, {"wrap", "stride"}, {"nooffset"}]


def direct(spec, name, i, mode, T=False):
    """論理バッチ i のオペランド(T なら転置 view (c, r))を、格納の平坦 index から直接読む。"""
    r, c = W.op_shapes(spec)[name]
    L = W.layout_of(spec, name)
    st = W.gen_storage(spec, name, 0, spec["B"] + L["pad"])
    flat = st.contiguous().view(-1).double()
    v = W.view_of(st, spec, name)
    if T:
        v = v.transpose(1, 2)
    s0, s1, s2 = v.stride()
    R, C = v.shape[1:]
    off = 0 if "nooffset" in mode else v.storage_offset()
    rr, cc = torch.arange(R).view(R, 1), torch.arange(C).view(1, C)
    f = off + i * s0 + (rr * C + cc if "stride" in mode else rr * s1 + cc * s2)
    if "wrap" in mode:
        f = f % W.WRAP
    return flat[f]


def check_operands() -> int:
    shapes = [  # (M, K, N): 2 の冪でない小さい shape と、points の shape
        (5, 7, 3), (6, 4, 10), (250, 250, 64), (250, 64, 250), (8, 8, 8),
    ]
    n = 0
    for (M, K, N), lay in itertools.product(shapes, ["contig", "bT", "aT", "slice", "offset"]):
        big = M * K > 1000
        B = 3 if big else 7
        spec = {"M": M, "K": K, "N": N, "dtype": "fp32", "layout": lay, "kind": "random", "seed": 0, "B": B}
        if lay == "offset":
            spec["pad_batches"] = 2
        for name in ("a", "b"):
            r, c = W.op_shapes(spec)[name]
            per = r * c
            # 巻き戻りの位置: バッチの途中・行の途中に来るもの、バッチ境界ちょうど(割り切れる場合)、格納より大きい(巻き戻りなし)
            wraps = [per + per // 2 + 1, 2 * per - 1, 2 * per, 10**9]
            if big:
                wraps = [2 * 62500 + 29796, 62500 + 29796, 5 * 16000 + 12296, 10**9]
            for WR in wraps:
                W.WRAP = WR
                for i in range(B):
                    for mode in MODES:
                        for T in (False, True):
                            got = (W.hyp_operand_T if T else W.hyp_operand)(spec, name, i, mode)
                            want = direct(spec, name, i, mode, T)
                            if not torch.equal(got, want):
                                raise AssertionError(f"mismatch {spec} {name} i={i} mode={mode} T={T} WRAP={WR}")
                            n += 1
    W.WRAP = 2**32
    return n


def simulate_in_side() -> None:
    """in-250x250x64 contig、要素 WRAP がバッチ 2 の位置 29796 に来る縮小版(実機のバッチ 68719 に当たる)。"""
    P0 = 29796  # 実機: 2**32 - 68719·62500
    WR = 2 * 62500 + P0
    r0, c0 = divmod(P0, 250)
    assert (r0, c0) == (119, 46)
    tol = 1.5397693008385817e-05  # studio の fp32 の許容誤差(20260920T150512-studio)
    for kind, B in itertools.product(["random", "idx_a_rc", "idx_a_batch"], [3, 4]):
        spec = {"M": 250, "K": 250, "N": 64, "dtype": "fp32", "layout": "contig", "kind": kind, "seed": 0, "B": B,
                "tol": tol, "compare": "full"}
        W.WRAP = WR
        a_w = torch.stack([W.hyp_operand(spec, "a", i, {"wrap"}) for i in range(B)])
        b = W.gen_logical(spec, "b", 0, B).double()
        ref = torch.bmm(W.gen_logical(spec, "a", 0, B).double(), b)
        out = torch.bmm(a_w, b).float()  # 実機の出力の代わり(fp32)
        st = W.Stats(B, spec)
        st.add(0, out.double(), ref)
        sm = st.summary()
        det = {d["batch"]: d for d in sm["bad_batch_detail"]}
        # 期待: バッチ 2 は位置 29796 以降の a を巻き戻って読む。出力の行 m は a の行 m(位置 250m..250m+249)に依存
        idx = kind != "random"
        n2 = (250 - r0) * 64 - (c0 if idx else 0)  # idx: 出力 (m, n) = a[m, n](n < 64)なので行 119 は列 46 以降だけ
        exp = {2: {"n_bad": n2, "rows": [[r0, 250]], "pos_min": r0 * 64 + (c0 if idx else 0), "pos_max": 250 * 64 - 1}}
        if B == 4:  # バッチ 3 は全体が巻き戻る(実機のバッチ 68720)
            exp[3] = {"n_bad": 16000, "rows": [[0, 250]], "pos_min": 0, "pos_max": 15999}
        assert sorted(det) == sorted(exp), (kind, B, det)
        for bi, e in exp.items():
            for k, v in e.items():
                if k == "n_bad" and not idx:  # 乱数入力では、読み違えた要素がたまたま許容誤差内に入ることがある(実機でも 16383/16384)
                    assert v - max(3, v // 1000) <= det[bi][k] <= v, (kind, B, bi, k, det[bi][k], v)
                else:  # index 符号化は厳密(誤答要素の集合がそのまま決まる)
                    assert det[bi][k] == v, (kind, B, bi, k, det[bi][k], v)
        assert sm["first_bad_batch"] == 2 and sm["n_batches_bad"] == len(exp), sm
        if kind == "idx_a_rc":  # 位置のずれ: バッチ 2 は -29796、バッチ 3 は位置 < 29796 で +32704(= 62500 - 29796)、以降 -29796
            want = {-P0: n2 * (2 if B == 4 else 1)}
            if B == 4:
                want[62500 - P0] = r0 * 64 + c0
            assert dict(map(tuple, sm["idx_delta_top"])) == want, sm["idx_delta_top"]
        if kind == "idx_a_batch":  # バッチのずれ: バッチ 2 は -2(実機 -68719)、バッチ 3 は位置 < 29796 で -3(実機 -68720)、以降 -2
            want = {-2: n2 * (2 if B == 4 else 1)}
            if B == 4:
                want[-3] = r0 * 64 + c0
            assert dict(map(tuple, sm["idx_delta_top"])) == want, sm["idx_delta_top"]
        hy = W.test_hypotheses(spec, out, st.bad_batches(), tol)
        assert hy["hyp_match"]["wrap"] == 1.0 and hy["hyp_match"]["stride"] == 0.0, hy
        print(f"  {kind:11s} B={B}: detail={[(d['batch'], d['n_bad'], d['rows'], d['pos_min'], d['pos_max']) for d in sm['bad_batch_detail']]}"
              f" delta={sm.get('idx_delta_top', '-')} hyp={hy['hyp_match']}")
    W.WRAP = 2**32


def check_detail_bookkeeping() -> None:
    spec = {"M": 3, "K": 2, "N": 4, "dtype": "fp32", "layout": "contig", "kind": "random", "tol": 1e-6}
    for bads, blk in (({1, 5, 6, 7, 8, 12, 17, 19}, 3), ({2, 3}, 2), ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10)):
        B = 20
        st = W.Stats(B, spec)
        ref = torch.ones(B, 3, 4, dtype=torch.float64)
        out = ref.clone()
        for i in bads:
            out[i, 1, 2:] += 1  # 行 1 の列 2 以降
        for i0 in range(0, B, blk):
            st.add(i0, out[i0:i0 + blk], ref[i0:i0 + blk])
        got = [d["batch"] for d in st.summary()["bad_batch_detail"]]
        s = sorted(bads)
        assert got == sorted(set(s[:W.N_DETAIL] + s[-W.N_DETAIL:])), (bads, got)
        d0 = st.summary()["bad_batch_detail"][0]
        assert (d0["n_bad"], d0["rows"], d0["pos_min"], d0["pos_max"]) == (2, [[1, 2]], 6, 7), d0


if __name__ == "__main__":
    print(f"1. hyp_operand / hyp_operand_T と平坦 index からの直接計算: {check_operands()} 件一致")
    print("2. in-250x250x64 の縮小版(WRAP = 2·62500 + 29796)")
    simulate_in_side()
    check_detail_bookkeeping()
    print("3. Stats の先頭側・末尾側の記録: ok")
    print("ALL OK")
