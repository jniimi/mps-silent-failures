# Table 11 (tab:guard) の再集計: 論理要素数 vs 格納要素数

2026-10-26。IEEE Access の査読指摘(R1-1: 公開 guard の `_check` は `t.numel()`、つまり論理要素数を見るが、Table 11 の offset 行は view の背後の格納を数えている)への対応として、Table 11 の全行を raw の jsonl から再集計した。測定はしていない。スクリプトは `v2/analysis/guard_rule.py`(標準ライブラリのみ)。

```
uv run python v2/analysis/guard_rule.py            # 3 通りの数え方で表を出す
uv run python v2/analysis/guard_rule.py --diff     # 数え方で stopped が変わる実行を列挙
```

## 1. 使った raw と行の対応

すべて `v2/results/raw/studio/<run_id>.jsonl`(schema 2、host_tag studio)。版は manifest の `worker.torch`。v1/results/raw/studio にも 11 版の走査があるが(2.14.0 が 1584 行、計 5240 行)、表の 1596 / 5252 と一致するのは v2 側である。

| 表の行 | run_id | 行数 |
|---|---|---:|
| 2.4.1 | 20260920T192435-studio | 360 |
| 2.5.1 / 2.6.0 / 2.7.1 / 2.8.0 | 20260920T200227 / 203904 / 211533 / 214629-studio | 374 each |
| 2.9.1 | 20260920T221708-studio | 360 |
| 2.10.0 / 2.11.0 / 2.12.1 / 2.13.0 | 20260920T224546 / 231439 / 234329-studio, 20260921T001206-studio | 360 each |
| 2.14.0 (main sweep) | 20260920T150512-studio | 1596 |
| all versions | 上の 11 run の和 | 5252 |
| 2.14.0, (256, 256, 256) shape | 20260920T113543-studio + 20260920T135650-studio | 228 + 36 = 264 |
| 2.14.0, far points | 20260920T132549-studio | 124 |
| 2.14.0, offset series | 20260920T132133-studio | 104 |
| 2.14.0, backward pass | 20260920T120157-studio | 604 |

表に入らない studio の run: 20260920T084617 / 085410(試運転)、20260924T005506(2.14.0、30 行、別系列)。

## 2. 定義(表の数字が一致した定義)

- **runs**: jsonl の全行。`phase = calibrate` の行も含む(除くと 2.4.1 が 348 になり一致しない)。
- **silent**: `cls` が `wrong` または `truncated`。bmm_bwd は `cls_out` / `cls_grad_a` / `cls_grad_b` のどれかが wrong / truncated(集約の `cls` と一致する: ok 420、wrong 40 = out も勾配も wrong 20 + 勾配だけ wrong 20、error 144)。
- **correct**: `cls = ok`。`crash` / `error` はどちらにも入らない。
- **stopped**: その実行のテンソルのどれかが n ≥ 2^32 要素(右から 2 列目は n > 2^32)。
- テンソルの大きさ、3 通り:
  - **logical**(公開 guard の基準): bmm は a, b, out の `numel_*`。bmm_bwd はさらに上流勾配 G(`shape_out` と同形)、grad_a(`grad_a_shape`)、grad_b(`grad_b_shape`)。G, grad_a, grad_b は out, a, b と同じ要素数なので、全行で n = max(numel_a, numel_b, numel_out) に等しい。
  - **storage**: a, b, G は view の背後の格納全体(slice は B+1 バッチ = `prod(a_storage_shape)`、offset は pad + view = `a_offset + numel_a` = `storage_numel_a`、aT / bT / contig は論理と同じ)。out / grad_a / grad_b は新規確保なので論理要素数。
  - **paper**: offset layout だけ storage、それ以外は logical。

## 3. 再現: 現在の Table 11 を再現する数え方

**paper の数え方で 10 行すべて、全列が現在の表と一致した。** 行ごとに見ると:

| 表の行 | 再現する数え方 |
|---|---|
| 2.4.1 ... 2.14.0 (main sweep)、all versions | logical(storage で数えると 2.4.1 と all versions が変わる。4 節) |
| (256, 256, 256) shape、far points、backward pass | logical と storage のどちらでも同じ数字 |
| offset series(28 of 104) | storage のみ。a の格納(pad + view)が 2^32 以上の 28 行 = out-256x64x256 の pad_batches 262142〜262146(5 点 × fp32 / fp16)+ offset_fin 2 × 2 = 14、in-256x256x64 の pad_batches 65534〜65538 + offset_fin = 14。view の先頭(`a_offset`)が 2^32 以上という数え方だと 20 行なので、本文の "storage reaches 2^32" は格納の末尾(pad + view)の意味である |

2.5.1〜2.8.0 の 4 版、2.10.0〜2.13.0 の 4 版は、それぞれ版ごとの数字が完全に同じことも確認した(表の "(each)" のとおり)。

本文の派生した数字も一致: 481 of 5252、2.4.1 の 16 件(n > 2^32 で通る)、630 of 4095、拡張 + backward の silent 104 件(32 + 32 + 0 + 40)、勾配だけ wrong の 20 件、128 of 730(28 + 40 + 28 + 32 of 154 + 52 + 104 + 420)、そのうち offset 28 件。

## 4. 公開 guard の基準(logical)で再計算した表

| PyTorch | runs | silent runs | silent stopped n ≥ 2^32 | silent stopped n > 2^32 | correct runs stopped |
|---|---:|---:|---:|---:|---|
| 2.4.1 | 360 | 56 | 56 | 40 | 40 of 304 |
| 2.5.1–2.8.0 (each) | 374 | 42 | 42 | 42 | 48 of 282 |
| 2.9.1 | 360 | 32 | 32 | 32 | 40 of 274 |
| 2.10.0–2.13.0 (each) | 360 | 32 | 32 | 32 | 40 of 276 |
| 2.14.0 (main sweep) | 1596 | 97 | 97 | 97 | 198 of 1285 |
| all versions, main and reduced sweeps | 5252 | 481 | 481 | 465 | 630 of 4095 |
| 2.14.0, (256, 256, 256) shape | 264 | 32 | 32 | 32 | 28 of 154 |
| 2.14.0, far points | 124 | 32 | 32 | 32 | 40 of 52 |
| 2.14.0, offset series | 104 | 0 | 0 | 0 | **0 of 104**(現在 28 of 104) |
| 2.14.0, backward pass | 604 | 40 | 40 | 40 | 32 of 420 |

### 行ごとの差

- 変わるのは **offset 行の「correct runs stopped」だけ: 28 of 104 → 0 of 104**。offset 系列の論理要素数の最大は 2^28(a = 4096 × 256 × 256)で、guard はどの行も止めない。silent は 0 行なので見逃しも 0。
- 他の 9 行は全列が同じ。silent の「n ≥ 2^32」列は 3 通りのどの数え方でも全行同じ(481 of 481、拡張 + backward 104 of 104)。
- 本文で連動して変わる数字: "128 of 730 correct runs are stopped as well, 28 of them offset runs whose storage reaches 2^32 elements" → **100 of 730**(28 + 40 + 0 + 32)、offset の 28 件の句は不要になる。"the size of a run includes the upstream gradient, the gradients and the storage behind an offset view" のうち「the storage behind an offset view」は logical の基準では入らない。630 of 4095 と 481 of 5252 は offset 行を含まないので変わらない。

## 5. slice layout を格納(B+1 バッチ)で数えた場合

`--diff` で列挙(90 行 = offset 28 + slice 62)。slice の格納が論理と n ≥ 2^32 の判定で食い違うのは、**in-* shape の B = 65535 だけ**(a の論理 = 65535 × 2^16 < 2^32、格納 = 65536 × 2^16 = ちょうど 2^32)。62 行の内訳は error 58(どの列にも入らない)と **ok 4(2.4.1 の in-256x256x64 fp32 / fp16、probe と final)**。

- 変わる行: **2.4.1 の correct stopped 40 of 304 → 44 of 304**、**all versions 630 of 4095 → 634 of 4095**。
- 「n > 2^32」列: 2.4.1 の in-256x256x64 slice B = 65536 の silent 8 行(truncated 4 + wrong 4; a の論理 = ちょうど 2^32、格納 = 2^32 + 2^16)が > でも止まるようになり、**2.4.1 の 40 → 48、all versions の 465 → 473**。
- 2.14.0 main sweep(slice の差は error 12 行のみ)、(256,256,256)(error 2 行)、backward(error 2 行; G の格納は out と同形)、far(差なし)は変わらない。

## 6. 補足: 2.4.1 の「ちょうど 2^32 で零を返す 16 件」

2.4.1 の silent で max 論理要素数 = 2^32 ちょうどの行は 16(in-256x256x64、B = 65536、aT 8 + slice 8)。cls は truncated 8 + wrong 8(final の 4 + 4 が wrong)だが、16 行とも `frac_bad_elem_zero = 1.0`(誤った要素は全部 0)なので、本文の「view of exactly 2^32 elements returns zeros」という記述と整合する。

## 7. 再現できなかったもの・注意

- 再現できなかった数字はない。定義は 2 節のまま(数字合わせのための調整はしていない)。
- crash 行(2.5.1〜2.8.0 の 80 + 80 行)には `a_storage_shape` が無いので、storage の数え方では layout から復元している(slice: numel × (B+1) / B)。crash は silent にも correct にも入らないため、表の数字には影響しない。
- logical の基準は「記録された a, b, out(+ G, grad_a, grad_b)の論理要素数」であって、guard を実際に走らせたときに autograd の内部で見える中間テンソルは記録に無い。backward では guard は G や転置 view を見るが、これらの numel は out / a / b と同じなので、記録から判定できる範囲では一致する。
- v1/results/raw/studio の 11 走査(5240 行)は表に使われていない(行数が合わない)。
