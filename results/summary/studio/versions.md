# torch の版ごとの走査: studio

基本 shape(out-256x64x256 = 出力が大きい系列、in-256x256x64 = 入力 a が大きい系列)、fp32 / fp16、4 レイアウト、乱数入力。各版は `run_versions.sh` の縮小設定(境界 B0-2〜B0+2 + 境界の間 2 点のグリッド + 二分探索、最終点は seed 0 の全要素比較、index 符号化は切り替わりの異常側 1 点で rc 系のみ)。2.14.0 は全体走査の run(グリッド 7 点、seed 0/1/2)から基本 shape だけを使う。`uv run python versions.py --host-tag studio` で再生成。

## run

| torch | run_id | Python | 実行数 | 経過(分) | 許容誤差 fp32 / fp16 | 分類の内訳 |
|---|---|---|---|---|---|---|
| 2.14.0 | 20260919T130741-studio | 3.12.12 | 1088 | 114.6 | 1.2e-05 / 0.0042 | WRONG 68, err 136, ok 884 |

## 切り替わり(版 × 条件)

正常/異常が異なる隣接点(B の粒度)。「なし」は探索範囲(B=4096〜65538)で切り替わりがない。括弧は誤答バッチで一致した読み違い仮説(stride = 転置 view の stride 無視、wrap = 2^32 要素での巻き戻り)と一致率。2.14.0 と違うセルは **太字**。

| shape | dtype | レイアウト | 2.14.0 |
|---|---|---|---|
| out-256x64x256 | fp16 | contig | なし(ok) |
| out-256x64x256 | fp16 | bT | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | aT | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | slice | なし(ok) |
| out-256x64x256 | fp32 | contig | なし(ok) |
| out-256x64x256 | fp32 | bT | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | aT | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | slice | なし(ok) |
| in-256x256x64 | fp16 | contig | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | 32767 ok→32768 err |
| in-256x256x64 | fp16 | slice | 32767 ok→32768 err |
| in-256x256x64 | fp32 | contig | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | 32767 ok→32768 err |
| in-256x256x64 | fp32 | slice | 32767 ok→32768 err |

## 2.14.0 と分類が違う点(同じ B で比較)

- なし

## 理論境界の前後(B0-2, B0-1, B0, B0+1, B0+2 の分類)

**out-256x64x256 fp32**

| レイアウト | torch | 2^32 byte (B0=16384) | 2^31 elem (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|---|
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| slice | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |

**out-256x64x256 fp16**

| レイアウト | torch | 2^31 elem = 2^32 byte (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| slice | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok |

**in-256x256x64 fp32**

| レイアウト | torch | 2^32 byte (B0=16384) | 2^31 elem (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|---|
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.14.0 | ok ok ok ok ok | ok ok err err err | err err err err err |

**in-256x256x64 fp16**

| レイアウト | torch | 2^31 elem = 2^32 byte (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok err err err | err err err err err |
| slice | 2.14.0 | ok ok err err err | err err err err err |

## 最終点(全要素比較 seed 0)と index 符号化(rc)

切り替わりの下・上の全要素比較の分類 / 最大相対誤差 / 閾値超え要素の割合、index 符号化(rc、切り替わりの異常側)の上位 delta。

| torch | shape | dtype | レイアウト | B | 入力 | 分類 | 最大誤差 | 閾値超え割合 | 壊れたバッチの区間 | 仮説 | 上位の delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.14.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_batch | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 867: 20480, 3: 4224 |
| 2.14.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.14.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.14.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.14.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.14.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.14.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.14.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_batch | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 196611: 16384, 3: 128 |
| 2.14.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.14.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.14.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.14.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.14.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.14.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.14.0 | in-256x256x64 | fp16 | contig | 65537 | idx_a_batch | WRONG | 1 | 1.5e-05 | [65536,65537) | wrap 1.00 | -288: 16384 |
| 2.14.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.14.0 | in-256x256x64 | fp16 | bT | 65537 | idx_a_batch | WRONG | 1 | 1.5e-05 | [65536,65537) | wrap 1.00 | -288: 16384 |
| 2.14.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.14.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | slice | 32768 | idx_a_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | slice | 32768 | idx_b_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp16 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | contig | 65537 | idx_a_batch | WRONG | 1 | 1.5e-05 | [65536,65537) | wrap 1.00 | -65536: 16384 |
| 2.14.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.14.0 | in-256x256x64 | fp32 | bT | 65537 | idx_a_batch | WRONG | 1 | 1.5e-05 | [65536,65537) | wrap 1.00 | -65536: 16384 |
| 2.14.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.14.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | slice | 32768 | idx_a_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | slice | 32768 | idx_b_batch | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.14.0 | in-256x256x64 | fp32 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |

(異常だった最終点だけを載せる)

## エラー・クラッシュ(版ごと)

- **2.14.0**: error ×136 `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX`
