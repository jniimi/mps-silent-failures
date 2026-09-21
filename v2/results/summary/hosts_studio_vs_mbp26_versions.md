# ホストをまたぐ比較

基準: **studio**。torch の版: 2.4.1, 2.8.0, 2.9.1, 2.14.0。`uv run python compare_hosts.py --source ../v1/results:studio --source results:mbp-macos26 --torch 2.4.1,2.8.0,2.9.1,2.14.0` で生成(`v2/` で実行)。

同じ点 = (shape, dtype, レイアウト, 入力の種類, B, seed, 比較の方法) が同じ実行。op = bmm、レイアウト contig / bT / aT / slice、phase ≠ point の行だけを使う。較正の行は「較正」の節だけで使う。分類が同じか違うかだけを書き、解釈はしない。

## 対象

| host-tag | torch | device | 機種 | チップ | RAM (GiB) | OS | GPU / driver / CUDA | Python | run(合併した数) | status | 行数 | うち較正 | 重複 | 比較に使う点 | 分類の内訳(全行) | revision | sha_scan / sha_worker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| studio | 2.4.1 | mps | Mac14,14 | Apple M2 Ultra | 192 | macOS 27.0 (26A428) | - | 3.12.12 | 20260919T152130-studio (1) | done | 360 | 12 | 0 | 348 | TRUNC 16, WRONG 40, ok 304 | 18dce3f | 7de38f9df9a3 / 5b2140fcc770 |
| studio | 2.8.0 | mps | Mac14,14 | Apple M2 Ultra | 192 | macOS 27.0 (26A428) | - | 3.12.12 | 20260919T174311-studio (1) | done | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | 059b073 | 7de38f9df9a3 / 5b2140fcc770 |
| studio | 2.9.1 | mps | Mac14,14 | Apple M2 Ultra | 192 | macOS 27.0 (26A428) | - | 3.12.12 | 20260919T181405-studio (1) | done | 360 | 12 | 0 | 348 | WRONG 32, err 54, ok 274 | 428ffe8 | 7de38f9df9a3 / 5b2140fcc770 |
| studio | 2.14.0 | mps | Mac14,14 | Apple M2 Ultra | 192 | macOS 27.0 (26A428) | - | 3.12.12 | 20260919T130741-studio (1) | done | 1584 | 12 | 0 | 1572 | WRONG 97, err 214, ok 1273 | d525e87 | 8ac27c7122fc / 5b2140fcc770 |
| mbp-macos26 | 2.4.1 | mps | Mac14,6 | Apple M2 Max | 96 | macOS 26.6.2 (25G83) | - | 3.12.11 | 20260920T145220-mbp-macos26 (1) | done | 360 | 12 | 0 | 348 | TRUNC 16, WRONG 40, ok 304 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | 2.8.0 | mps | Mac14,6 | Apple M2 Max | 96 | macOS 26.6.2 (25G83) | - | 3.12.11 | 20260920T152603-mbp-macos26 (1) | done | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | 2.9.1 | mps | Mac14,6 | Apple M2 Max | 96 | macOS 26.6.2 (25G83) | - | 3.12.11 | 20260920T155714-mbp-macos26 (1) | done | 360 | 12 | 0 | 348 | WRONG 32, err 54, ok 274 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | 2.14.0 | mps | Mac14,6 | Apple M2 Max | 96 | macOS 26.6.2 (25G83) | - | 3.12.11 | 20260920T110142-mbp-macos26 (1) | done | 1596 | 24 | 0 | 1572 | WRONG 97, err 214, ok 1285 | ed941f3 | abcd1fbb8854 / a264baa58030 |

行数 = 条件に合う行の全部(較正と重複を含む)。重複 = 同じ点が複数の run にあった数(後の run の行を使う。較正の行は数えない)。比較に使う点 = 較正を除き、重複をまとめた後の点の数。

使わなかった run:

- mbp-macos26: 20260920T162628-mbp-macos26: op=bmm_bwd

## 較正

較正の実行(contig、乱数入力、2^28 要素、seed 0/1/2、全要素比較)の最大相対誤差の最大と、manifest の許容誤差。較正する shape は版で違う(v1 は基本 shape だけ、v2 はその run の shape の全部)ので、機種の比較は下の shape ごとの表で見る。基準と値が違うセルは **太字**。

| torch | dtype | studio: 較正誤差の最大 / 許容誤差 | mbp-macos26: 較正誤差の最大 / 許容誤差 |
|---|---|---|---|
| 2.4.1 | fp16 | 4.211458e-04 / 0.00421 | 4.211458e-04 / 0.00421 |
| 2.4.1 | fp32 | 1.226142e-06 / 1.23e-05 | 1.226142e-06 / 1.23e-05 |
| 2.8.0 | fp16 | 4.211458e-04 / 0.00421 | 4.211458e-04 / 0.00421 |
| 2.8.0 | fp32 | 1.226142e-06 / 1.23e-05 | 1.226142e-06 / 1.23e-05 |
| 2.9.1 | fp16 | 4.211458e-04 / 0.00421 | 4.211458e-04 / 0.00421 |
| 2.9.1 | fp32 | 1.226142e-06 / 1.23e-05 | 1.226142e-06 / 1.23e-05 |
| 2.14.0 | fp16 | 4.211458e-04 / 0.00421 | 4.211458e-04 / 0.00421 |
| 2.14.0 | fp32 | 1.226142e-06 / 1.23e-05 | 1.539769e-06 / 1.54e-05 |

shape ごとの較正誤差の最大(3 seed の最大):

| torch | dtype | shape | studio | mbp-macos26 |
|---|---|---|---|---|
| 2.4.1 | fp16 | out-256x64x256 | 3.830090e-04 | 3.830090e-04 |
| 2.4.1 | fp16 | in-256x256x64 | 4.211458e-04 | 4.211458e-04 |
| 2.4.1 | fp32 | out-256x64x256 | 6.836674e-07 | 6.836674e-07 |
| 2.4.1 | fp32 | in-256x256x64 | 1.226142e-06 | 1.226142e-06 |
| 2.8.0 | fp16 | out-256x64x256 | 3.830090e-04 | 3.830090e-04 |
| 2.8.0 | fp16 | in-256x256x64 | 4.211458e-04 | 4.211458e-04 |
| 2.8.0 | fp32 | out-256x64x256 | 6.836674e-07 | 6.836674e-07 |
| 2.8.0 | fp32 | in-256x256x64 | 1.226142e-06 | 1.226142e-06 |
| 2.9.1 | fp16 | out-256x64x256 | 3.830090e-04 | 3.830090e-04 |
| 2.9.1 | fp16 | in-256x256x64 | 4.211458e-04 | 4.211458e-04 |
| 2.9.1 | fp32 | out-256x64x256 | 6.836674e-07 | 6.836674e-07 |
| 2.9.1 | fp32 | in-256x256x64 | 1.226142e-06 | 1.226142e-06 |
| 2.14.0 | fp16 | out-256x64x256 | 3.830090e-04 | 3.830090e-04 |
| 2.14.0 | fp16 | in-256x256x64 | 4.211458e-04 | 4.211458e-04 |
| 2.14.0 | fp32 | out-256x64x256 | 6.836674e-07 | 6.836674e-07 |
| 2.14.0 | fp32 | in-256x256x64 | 1.226142e-06 | 1.226142e-06 |
| 2.14.0 | fp32 | out-512x64x128 | n/a | 6.193911e-07 |
| 2.14.0 | fp32 | out-128x64x512 | n/a | 6.027965e-07 |
| 2.14.0 | fp32 | in-512x128x64 | n/a | 8.083443e-07 |
| 2.14.0 | fp32 | in-128x512x64 | n/a | 1.539769e-06 |

## 切り替わり(ホスト × 条件)

乱数入力 seed 0 の探索点と二分探索から、正常/異常が異なる隣接点(B の粒度)。書き方は `versions.md` と同じ: 「なし」は実行した点の範囲で切り替わりがない。括弧は誤答バッチで一致した読み違い仮説と一致率。`n/a` = その source にこの系列の行がない(メモリに入らない dtype を回していない、など。違いには数えない)。基準と違うセルは **太字**(仮説の括弧は比べない)。途中の source のセルには実行済みの点の数を `[n 点]` で付ける(基準の点の数は見出しの右の列)。

### torch 2.4.1

| shape | dtype | レイアウト | 基準の点の数 | studio | mbp-macos26 |
|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | aT | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | slice | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | 22 | 65536 ok→65537 WRONG | 65536 ok→65537 WRONG |
| out-256x64x256 | fp32 | bT | 22 | 65536 ok→65537 WRONG (stride 0.00) | 65536 ok→65537 WRONG (stride 0.00) |
| out-256x64x256 | fp32 | aT | 22 | 65536 ok→65537 WRONG (stride 0.00) | 65536 ok→65537 WRONG (stride 0.00) |
| out-256x64x256 | fp32 | slice | 22 | 65536 ok→65537 WRONG (nooffset 0.00) | 65536 ok→65537 WRONG (nooffset 0.00) |
| in-256x256x64 | fp16 | contig | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | 15 | 65535 ok→65536 TRUNC (stride 0.01) | 65535 ok→65536 TRUNC (stride 0.01) |
| in-256x256x64 | fp16 | slice | 15 | 65535 ok→65536 TRUNC (nooffset 0.01) | 65535 ok→65536 TRUNC (nooffset 0.01) |
| in-256x256x64 | fp32 | contig | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | 22 | 65535 ok→65536 TRUNC (stride 0.00) | 65535 ok→65536 TRUNC (stride 0.00) |
| in-256x256x64 | fp32 | slice | 22 | 65535 ok→65536 TRUNC (nooffset 0.00) | 65535 ok→65536 TRUNC (nooffset 0.00) |

### torch 2.8.0

| shape | dtype | レイアウト | 基準の点の数 | studio | mbp-macos26 |
|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | 15 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | aT | 15 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | slice | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | 22 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | bT | 22 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | aT | 22 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | slice | 22 | なし(ok) | なし(ok) |
| in-256x256x64 | fp16 | contig | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | 15 | 32767 ok→32768 CRASH | 32767 ok→32768 CRASH |
| in-256x256x64 | fp16 | slice | 15 | 32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00) | 32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | contig | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | 22 | 32767 ok→32768 CRASH | 32767 ok→32768 CRASH |
| in-256x256x64 | fp32 | slice | 22 | 32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00) | 32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00) |

### torch 2.9.1

| shape | dtype | レイアウト | 基準の点の数 | studio | mbp-macos26 |
|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | 15 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | aT | 15 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | slice | 15 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | 22 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | bT | 22 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | aT | 22 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | slice | 22 | なし(ok) | なし(ok) |
| in-256x256x64 | fp16 | contig | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 15 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | 15 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp16 | slice | 15 | 32766 ok→32767 err | 32766 ok→32767 err |
| in-256x256x64 | fp32 | contig | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 22 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | 22 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp32 | slice | 22 | 32766 ok→32767 err | 32766 ok→32767 err |

### torch 2.14.0

| shape | dtype | レイアウト | 基準の点の数 | studio | mbp-macos26 |
|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | 25 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | 25 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | aT | 25 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | slice | 25 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | 37 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | bT | 37 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | aT | 37 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | slice | 37 | なし(ok) | なし(ok) |
| in-256x256x64 | fp16 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 25 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | 25 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp16 | slice | 25 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp32 | contig | 37 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 37 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | 37 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp32 | slice | 37 | 32767 ok→32768 err | 32767 ok→32768 err |
| out-512x64x128 | fp32 | contig | 25 | なし(ok) | なし(ok) |
| out-512x64x128 | fp32 | bT | 25 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-512x64x128 | fp32 | aT | 25 | 65535 ok→65536 err | 65535 ok→65536 err |
| out-512x64x128 | fp32 | slice | 25 | 65535 ok→65536 err<br>65536 err→65537 ok | 65535 ok→65536 err<br>65536 err→65537 ok |
| out-128x64x512 | fp32 | contig | 25 | なし(ok) | なし(ok) |
| out-128x64x512 | fp32 | bT | 25 | 65535 ok→65536 err | 65535 ok→65536 err |
| out-128x64x512 | fp32 | aT | 25 | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-128x64x512 | fp32 | slice | 25 | 65535 ok→65536 err<br>65536 err→65537 ok | 65535 ok→65536 err<br>65536 err→65537 ok |
| in-512x128x64 | fp32 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-512x128x64 | fp32 | bT | 25 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-512x128x64 | fp32 | aT | 25 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-512x128x64 | fp32 | slice | 25 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-128x512x64 | fp32 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-128x512x64 | fp32 | bT | 25 | 65535 ok→65536 err | 65535 ok→65536 err |
| in-128x512x64 | fp32 | aT | 25 | 32767 ok→32768 err | 32767 ok→32768 err |
| in-128x512x64 | fp32 | slice | 25 | 32767 ok→32768 err | 32767 ok→32768 err |

## 点ごとの比較(基準 studio と同じ点で)

index 符号化の入力も含む。「分類が違う点」は、系列(shape, dtype, レイアウト, 入力, 比較)ごとに、共通の点の B の並びで連続するものを 1 行にまとめた(間に分類が同じ共通の点が入ると行を分ける。「a〜b(n 点)」は a から b までの共通の点 n 個で、間の B の全部ではない)。seed だけが違う行は 1 行にまとめた。「片方にしかない点」の内訳は phase・入力(random / index)・比較の方法ごとの数。

### mbp-macos26 vs studio(torch 2.4.1)

- 共通の点: 348(基準 348 点、mbp-macos26 348 点)
- 分類が同じ: 348、違う: 0
- 共通の点の基準側の分類: TRUNC 16, WRONG 40, ok 292、mbp-macos26 側: TRUNC 16, WRONG 40, ok 292
- 基準にしかない点: 0
- mbp-macos26 にしかない点: 0

### mbp-macos26 vs studio(torch 2.8.0)

- 共通の点: 362(基準 362 点、mbp-macos26 362 点)
- 分類が同じ: 362、違う: 0
- 共通の点の基準側の分類: CRASH 40, WRONG 42, err 10, ok 270、mbp-macos26 側: CRASH 40, WRONG 42, err 10, ok 270
- 基準にしかない点: 0
- mbp-macos26 にしかない点: 0

### mbp-macos26 vs studio(torch 2.9.1)

- 共通の点: 348(基準 348 点、mbp-macos26 348 点)
- 分類が同じ: 348、違う: 0
- 共通の点の基準側の分類: WRONG 32, err 54, ok 262、mbp-macos26 側: WRONG 32, err 54, ok 262
- 基準にしかない点: 0
- mbp-macos26 にしかない点: 0

### mbp-macos26 vs studio(torch 2.14.0)

- 共通の点: 1572(基準 1572 点、mbp-macos26 1572 点)
- 分類が同じ: 1572、違う: 0
- 共通の点の基準側の分類: WRONG 97, err 214, ok 1261、mbp-macos26 側: WRONG 97, err 214, ok 1261
- 基準にしかない点: 0
- mbp-macos26 にしかない点: 0

## まとめ

- **mbp-macos26**(torch 2.4.1): 共通の 348 点の全部で基準と分類が同じ。切り替わりの表で基準と違う系列: 0
- **mbp-macos26**(torch 2.8.0): 共通の 362 点の全部で基準と分類が同じ。切り替わりの表で基準と違う系列: 0
- **mbp-macos26**(torch 2.9.1): 共通の 348 点の全部で基準と分類が同じ。切り替わりの表で基準と違う系列: 0
- **mbp-macos26**(torch 2.14.0): 共通の 1572 点の全部で基準と分類が同じ。切り替わりの表で基準と違う系列: 0

