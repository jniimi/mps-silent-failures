# ホストをまたぐ比較

基準: **studio**。torch の版: 2.14.0。`uv run python compare_hosts.py --source results:studio --source results:a100 --torch 2.14.0` で生成(`v2/` で実行)。

同じ点 = (shape, dtype, レイアウト, 入力の種類, B, seed, 比較の方法) が同じ実行。op = bmm、レイアウト contig / bT / aT / slice、phase ≠ point の行だけを使う。較正の行は「較正」の節だけで使う。分類が同じか違うかだけを書き、解釈はしない。

## 対象

| host-tag | torch | device | 機種 | チップ | RAM (GiB) | OS | GPU / driver / CUDA | Python | run(合併した数) | status | 行数 | うち較正 | 重複 | 比較に使う点 | 分類の内訳(全行) | revision | sha_scan / sha_worker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| studio | 2.14.0 | mps | Mac14,14 | Apple M2 Ultra | 192 | macOS 27.0 (26A428) | - | 3.12.12 | 20260920T113543-studio … 20260920T150512-studio (2) | done 2 | 1824 | 30 | 0 | 1794 | WRONG 105, err 292, ok 1427 | 2e2e28a / 4423ad9 | abcd1fbb8854 / a264baa58030 |
| a100 | 2.14.0+cu130 | cuda | Google Compute Engine | Intel(R) Xeon(R) CPU @ 2.20GHz | 167 | Linux 6.6.122+ (Ubuntu 24.04.1 LTS) | NVIDIA A100-SXM4-80GB / driver 580.82.07 / CUDA 13.0 | 3.12.3 | 20260921T054329-a100 … 20260921T093433-a100 (18) | done 18 | 1494 | 54 | 0 | 1440 | ok 1494 | 8a89a4a | abcd1fbb8854 / a264baa58030 |

行数 = 条件に合う行の全部(較正と重複を含む)。重複 = 同じ点が複数の run にあった数(後の run の行を使う。較正の行は数えない)。比較に使う点 = 較正を除き、重複をまとめた後の点の数。

使わなかった run:

- studio: 20260920T084617-studio: 走査の行がない(--points の run か、較正だけ)
- studio: 20260920T085410-studio: op=bmm_bwd
- studio: 20260920T120157-studio: op=bmm_bwd
- studio: 20260920T132133-studio: 走査の行がない(--points の run か、較正だけ)
- studio: 20260920T132549-studio: 走査の行がない(--points の run か、較正だけ)
- studio: 20260920T135650-studio: 走査の行がない(--points の run か、較正だけ)

## 較正

較正の実行(contig、乱数入力、2^28 要素、seed 0/1/2、全要素比較)の最大相対誤差の最大と、manifest の許容誤差。較正する shape は版で違う(v1 は基本 shape だけ、v2 はその run の shape の全部)ので、機種の比較は下の shape ごとの表で見る。基準と値が違うセルは **太字**。

| torch | dtype | studio: 較正誤差の最大 / 許容誤差 | a100: 較正誤差の最大 / 許容誤差 |
|---|---|---|---|
| 2.14.0 | fp16 | 4.211458e-04 / 0.00387〜0.00421(2 通り) | 4.211458e-04 / 0.00383〜0.00421(2 通り) |
| 2.14.0 | fp32 | 1.539769e-06 / 1.15e-05〜1.54e-05(2 通り) | 1.539769e-06 / 6.03e-06〜1.54e-05(6 通り) |

shape ごとの較正誤差の最大(3 seed の最大):

| torch | dtype | shape | studio | a100 |
|---|---|---|---|---|
| 2.14.0 | fp16 | out-256x64x256 | 3.830090e-04 | 3.830090e-04 |
| 2.14.0 | fp16 | in-256x256x64 | 4.211458e-04 | 4.211458e-04 |
| 2.14.0 | fp16 | all-256x256x256 | 3.872996e-04 | n/a |
| 2.14.0 | fp32 | out-256x64x256 | 6.836674e-07 | 6.836674e-07 |
| 2.14.0 | fp32 | in-256x256x64 | 1.226142e-06 | 1.226142e-06 |
| 2.14.0 | fp32 | out-512x64x128 | 6.193911e-07 | 6.193911e-07 |
| 2.14.0 | fp32 | out-128x64x512 | 6.027965e-07 | 6.027965e-07 |
| 2.14.0 | fp32 | in-512x128x64 | 8.083443e-07 | 8.083443e-07 |
| 2.14.0 | fp32 | in-128x512x64 | 1.539769e-06 | 1.539769e-06 |
| 2.14.0 | fp32 | all-256x256x256 | 1.150926e-06 | n/a |

## 切り替わり(ホスト × 条件)

乱数入力 seed 0 の探索点と二分探索から、正常/異常が異なる隣接点(B の粒度)。書き方は `versions.md` と同じ: 「なし」は実行した点の範囲で切り替わりがない。括弧は誤答バッチで一致した読み違い仮説と一致率。`n/a` = その source にこの系列の行がない(メモリに入らない dtype を回していない、など。違いには数えない)。基準と違うセルは **太字**(仮説の括弧は比べない)。途中の source のセルには実行済みの点の数を `[n 点]` で付ける(基準の点の数は見出しの右の列)。

### torch 2.14.0

| shape | dtype | レイアウト | 基準の点の数 | studio | a100 |
|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | 25 | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | 25 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-256x64x256 | fp16 | aT | 25 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-256x64x256 | fp16 | slice | 25 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | 37 | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | bT | 37 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-256x64x256 | fp32 | aT | 37 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-256x64x256 | fp32 | slice | 37 | なし(ok) | なし(ok) |
| in-256x256x64 | fp16 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-256x256x64 | fp16 | bT | 25 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-256x256x64 | fp16 | aT | 25 | 32767 ok→32768 err | **なし(ok)** |
| in-256x256x64 | fp16 | slice | 25 | 32767 ok→32768 err | **なし(ok)** |
| in-256x256x64 | fp32 | contig | 37 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-256x256x64 | fp32 | bT | 37 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-256x256x64 | fp32 | aT | 37 | 32767 ok→32768 err | **なし(ok)** |
| in-256x256x64 | fp32 | slice | 37 | 32767 ok→32768 err | **なし(ok)** |
| out-512x64x128 | fp32 | contig | 25 | なし(ok) | なし(ok) |
| out-512x64x128 | fp32 | bT | 25 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-512x64x128 | fp32 | aT | 25 | 65535 ok→65536 err | **なし(ok)** |
| out-512x64x128 | fp32 | slice | 25 | 65535 ok→65536 err<br>65536 err→65537 ok | **なし(ok)** |
| out-128x64x512 | fp32 | contig | 25 | なし(ok) | なし(ok) |
| out-128x64x512 | fp32 | bT | 25 | 65535 ok→65536 err | **なし(ok)** |
| out-128x64x512 | fp32 | aT | 25 | 65536 ok→65537 WRONG (stride 1.00) | **なし(ok)** |
| out-128x64x512 | fp32 | slice | 25 | 65535 ok→65536 err<br>65536 err→65537 ok | **なし(ok)** |
| in-512x128x64 | fp32 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-512x128x64 | fp32 | bT | 25 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-512x128x64 | fp32 | aT | 25 | 32767 ok→32768 err | **なし(ok)** |
| in-512x128x64 | fp32 | slice | 25 | 32767 ok→32768 err | **なし(ok)** |
| in-128x512x64 | fp32 | contig | 25 | 65536 ok→65537 WRONG (wrap 1.00) | **なし(ok)** |
| in-128x512x64 | fp32 | bT | 25 | 65535 ok→65536 err | **なし(ok)** |
| in-128x512x64 | fp32 | aT | 25 | 32767 ok→32768 err | **なし(ok)** |
| in-128x512x64 | fp32 | slice | 25 | 32767 ok→32768 err | **なし(ok)** |
| all-256x256x256 | fp16 | contig | 17 | なし(ok) | n/a |
| all-256x256x256 | fp16 | bT | 17 | 32767 ok→32768 err | n/a |
| all-256x256x256 | fp16 | aT | 17 | 32767 ok→32768 err | n/a |
| all-256x256x256 | fp16 | slice | 17 | 32767 ok→32768 err<br>65536 err→65537 ok | n/a |
| all-256x256x256 | fp32 | contig | 25 | なし(ok) | n/a |
| all-256x256x256 | fp32 | bT | 25 | 32767 ok→32768 err | n/a |
| all-256x256x256 | fp32 | aT | 25 | 32767 ok→32768 err | n/a |
| all-256x256x256 | fp32 | slice | 25 | 32767 ok→32768 err<br>65536 err→65537 ok | n/a |

## 点ごとの比較(基準 studio と同じ点で)

index 符号化の入力も含む。「分類が違う点」は、系列(shape, dtype, レイアウト, 入力, 比較)ごとに、共通の点の B の並びで連続するものを 1 行にまとめた(間に分類が同じ共通の点が入ると行を分ける。「a〜b(n 点)」は a から b までの共通の点 n 個で、間の B の全部ではない)。seed だけが違う行は 1 行にまとめた。「片方にしかない点」の内訳は phase・入力(random / index)・比較の方法ごとの数。

### a100 vs studio(torch 2.14.0)

- 共通の点: 1314(基準 1794 点、a100 1440 点)
- 分類が同じ: 1111、違う: 203(基準 → source: err → ok 159, WRONG → ok 44)
- 共通の点の基準側の分類: WRONG 44, err 159, ok 1111、a100 側: ok 1314
- 基準にしかない点: 480(final index full 96, final random full 216, probe random full 8, probe random sample 160)
- a100 にしかない点: 126(final index full 48, final random full 78)

| shape | dtype | レイアウト | 入力 | 比較 | seed | B | 基準 → source | 仮説・壊れたバッチの区間 / エラー |
|---|---|---|---|---|---|---|---|---|
| out-256x64x256 | fp16 | bT | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | bT | idx_a_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | bT | idx_b_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | aT | idx_a_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | aT | idx_b_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp16 | aT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 257) |
| out-256x64x256 | fp32 | bT | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | bT | idx_a_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | bT | idx_b_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | aT | idx_a_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | aT | idx_b_rc | sample | 0 | 65537 | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-256x64x256 | fp32 | aT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| in-256x256x64 | fp16 | contig | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) |
| in-256x256x64 | fp16 | contig | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-256x256x64 | fp16 | bT | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) |
| in-256x256x64 | fp16 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-256x256x64 | fp16 | aT | idx_a_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | aT | idx_a_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | aT | idx_b_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | aT | idx_b_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | aT | random | sample | 0 | 32768〜65538(15 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | slice | idx_a_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | slice | idx_a_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | slice | idx_b_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | slice | idx_b_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp16 | slice | random | sample | 0 | 32768〜65538(15 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | contig | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) |
| in-256x256x64 | fp32 | contig | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-256x256x64 | fp32 | bT | idx_a_batch | sample | 0 | 65537 | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) |
| in-256x256x64 | fp32 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-256x256x64 | fp32 | aT | idx_a_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | aT | idx_a_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | aT | idx_b_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | aT | idx_b_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | aT | random | sample | 0 | 32768〜65538(15 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | slice | idx_a_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | slice | idx_a_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | slice | idx_b_batch | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | slice | idx_b_rc | sample | 0 | 32769〜65537(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-256x256x64 | fp32 | slice | random | sample | 0 | 32768〜65538(15 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| out-512x64x128 | fp32 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-512x64x128 | fp32 | aT | random | sample | 0 | 65536 | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| out-512x64x128 | fp32 | aT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-512x64x128 | fp32 | slice | random | sample | 0 | 65536 | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| out-128x64x512 | fp32 | bT | random | sample | 0 | 65536 | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| out-128x64x512 | fp32 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-128x64x512 | fp32 | aT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: stride 1.00; 壊れたバッチ [0,4),[257,258)…(計 259) |
| out-128x64x512 | fp32 | slice | random | sample | 0 | 65536 | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-512x128x64 | fp32 | contig | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-512x128x64 | fp32 | bT | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-512x128x64 | fp32 | aT | random | sample | 0 | 32768〜65538(11 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-512x128x64 | fp32 | slice | random | sample | 0 | 32768〜65538(11 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-128x512x64 | fp32 | contig | random | sample | 0 | 65537〜65538(2 点) | WRONG → ok | 基準: wrap 1.00; 壊れたバッチ [65536,65537) … B=65538: [65536,65538) |
| in-128x512x64 | fp32 | bT | random | sample | 0 | 65536〜65538(3 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-128x512x64 | fp32 | aT | random | sample | 0 | 32768〜65538(11 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |
| in-128x512x64 | fp32 | slice | random | sample | 0 | 32768〜65538(11 点) | err → ok | 基準: `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX` |

## まとめ

- **a100**(torch 2.14.0): 共通の 1314 点のうち 203 点で基準と分類が違う(基準 → source: err → ok 159, WRONG → ok 44)。切り替わりの表で基準と違う系列: 26

