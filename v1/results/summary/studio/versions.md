# torch の版ごとの走査: studio

基本 shape(out-256x64x256 = 出力が大きい系列、in-256x256x64 = 入力 a が大きい系列)、fp32 / fp16、4 レイアウト、乱数入力。各版は `run_versions.sh` の縮小設定(境界 B0-2〜B0+2 + 境界の間 2 点のグリッド + 二分探索、最終点は seed 0 の全要素比較、index 符号化は切り替わりの異常側 1 点で rc 系のみ)。2.14.0 は全体走査の run(グリッド 7 点、seed 0/1/2)から基本 shape だけを使う。`uv run python versions.py --host-tag studio` で再生成。

## run

| torch | run_id | Python | 実行数 | 経過(分) | 許容誤差 fp32 / fp16 | 分類の内訳 |
|---|---|---|---|---|---|---|
| 2.4.1 | 20260919T152130-studio | 3.12.12 | 360 | 37.2 | 1.2e-05 / 0.0042 | TRUNC 16, WRONG 40, ok 304 |
| 2.5.1 | 20260919T155842-studio | 3.12.12 | 374 | 37.6 | 1.2e-05 / 0.0042 | CRASH 40, WRONG 42, err 10, ok 282 |
| 2.6.0 | 20260919T163619-studio | 3.12.12 | 374 | 36.4 | 1.2e-05 / 0.0042 | CRASH 40, WRONG 42, err 10, ok 282 |
| 2.7.1 | 20260919T171250-studio | 3.12.12 | 374 | 30.3 | 1.2e-05 / 0.0042 | CRASH 40, WRONG 42, err 10, ok 282 |
| 2.8.0 | 20260919T174311-studio | 3.12.12 | 374 | 30.8 | 1.2e-05 / 0.0042 | CRASH 40, WRONG 42, err 10, ok 282 |
| 2.9.1 | 20260919T181405-studio | 3.12.12 | 360 | 28.7 | 1.2e-05 / 0.0042 | WRONG 32, err 54, ok 274 |
| 2.10.0 | 20260919T184246-studio | 3.12.12 | 360 | 28.9 | 1.2e-05 / 0.0042 | WRONG 32, err 52, ok 276 |
| 2.11.0 | 20260919T191145-studio | 3.12.12 | 360 | 29.8 | 1.2e-05 / 0.0042 | WRONG 32, err 52, ok 276 |
| 2.12.1 | 20260919T194139-studio | 3.12.12 | 360 | 28.9 | 1.2e-05 / 0.0042 | WRONG 32, err 52, ok 276 |
| 2.13.0 | 20260919T201033-studio | 3.12.12 | 360 | 28.2 | 1.2e-05 / 0.0042 | WRONG 32, err 52, ok 276 |
| 2.14.0 | 20260919T130741-studio | 3.12.12 | 1088 | 114.6 | 1.2e-05 / 0.0042 | WRONG 68, err 136, ok 884 |

## 切り替わり(版 × 条件)

正常/異常が異なる隣接点(B の粒度)。「なし」は探索範囲(B=4096〜65538)で切り替わりがない。括弧は誤答バッチで一致した読み違い仮説(stride = 転置 view の stride 無視、wrap = 2^32 要素での巻き戻り)と一致率。2.14.0 と違うセルは **太字**。

| shape | dtype | レイアウト | 2.4.1 | 2.5.1 | 2.6.0 | 2.7.1 | 2.8.0 | 2.9.1 | 2.10.0 | 2.11.0 | 2.12.1 | 2.13.0 | 2.14.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| out-256x64x256 | fp16 | contig | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) |
| out-256x64x256 | fp16 | bT | **なし(ok)** | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | aT | **なし(ok)** | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp16 | slice | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | contig | **65536 ok→65537 WRONG** | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) |
| out-256x64x256 | fp32 | bT | 65536 ok→65537 WRONG (stride 0.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | aT | 65536 ok→65537 WRONG (stride 0.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) | 65536 ok→65537 WRONG (stride 1.00) |
| out-256x64x256 | fp32 | slice | **65536 ok→65537 WRONG (nooffset 0.00)** | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) | なし(ok) |
| in-256x256x64 | fp16 | contig | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | bT | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp16 | aT | **65535 ok→65536 TRUNC (stride 0.01)** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp16 | slice | **65535 ok→65536 TRUNC (nooffset 0.01)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 err** | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp32 | contig | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | bT | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) | 65536 ok→65537 WRONG (wrap 1.00) |
| in-256x256x64 | fp32 | aT | **65535 ok→65536 TRUNC (stride 0.00)** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | **32767 ok→32768 CRASH** | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err |
| in-256x256x64 | fp32 | slice | **65535 ok→65536 TRUNC (nooffset 0.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 CRASH<br>65535 err→65536 ok<br>65536 ok→65537 WRONG (wrap 1.00)** | **32766 ok→32767 err** | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err | 32767 ok→32768 err |

## 2.14.0 と分類が違う点(同じ B で比較)

| torch | shape | dtype | レイアウト | B | 2.14.0 | この版 | 仮説 / エラー |
|---|---|---|---|---|---|---|---|
| 2.4.1 | out-256x64x256 | fp16 | bT | 65537 | wrong | ok |  |
| 2.4.1 | out-256x64x256 | fp16 | bT | 65538 | wrong | ok |  |
| 2.4.1 | out-256x64x256 | fp16 | aT | 65537 | wrong | ok |  |
| 2.4.1 | out-256x64x256 | fp16 | aT | 65538 | wrong | ok |  |
| 2.4.1 | out-256x64x256 | fp32 | contig | 65537 | ok | wrong |  |
| 2.4.1 | out-256x64x256 | fp32 | contig | 65538 | ok | wrong |  |
| 2.4.1 | out-256x64x256 | fp32 | slice | 65537 | ok | wrong | nooffset 0.00 |
| 2.4.1 | out-256x64x256 | fp32 | slice | 65538 | ok | wrong | nooffset 0.00 |
| 2.4.1 | in-256x256x64 | fp16 | aT | 32768 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 32769 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 32770 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65534 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65535 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65536 | error | truncated | stride 0.01 |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65537 | error | truncated | wrap 0.03 |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65538 | error | truncated | wrap 0.06 |
| 2.4.1 | in-256x256x64 | fp16 | slice | 32768 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 32769 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 32770 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65534 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65535 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65536 | error | truncated | nooffset 0.01 |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65537 | error | truncated | wrap 0.03 |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65538 | error | truncated | wrap 0.06 |
| 2.4.1 | in-256x256x64 | fp32 | aT | 32768 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 32769 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 32770 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65534 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65535 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65536 | error | truncated | stride 0.00 |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65537 | error | truncated | wrap 0.03 |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65538 | error | truncated | wrap 0.06 |
| 2.4.1 | in-256x256x64 | fp32 | slice | 32768 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 32769 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 32770 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65534 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65535 | error | ok |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65536 | error | truncated | nooffset 0.00 |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65537 | error | truncated | wrap 0.03 |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65538 | error | truncated | wrap 0.06 |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp16 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65536 | error | ok |  |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp32 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65536 | error | ok |  |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp16 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65536 | error | ok |  |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp32 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65536 | error | ok |  |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp16 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65536 | error | ok |  |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp32 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65536 | error | ok |  |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp16 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65536 | error | ok |  |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 65535 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 65537 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp32 | aT | 65538 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32767 | ok | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32768 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32769 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32770 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65534 | error | crash | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65536 | error | ok |  |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65537 | error | wrong | wrap 1.00 |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65538 | error | wrong | wrap 1.00 |
| 2.9.1 | in-256x256x64 | fp16 | slice | 32767 | ok | error | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | slice | 32767 | ok | error | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |

## 理論境界の前後(B0-2, B0-1, B0, B0+1, B0+2 の分類)

**out-256x64x256 fp32**

| レイアウト | torch | 2^32 byte (B0=16384) | 2^31 elem (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|---|
| contig | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| bT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| slice | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| slice | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok ok ok |

**out-256x64x256 fp16**

| レイアウト | torch | 2^31 elem = 2^32 byte (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|
| contig | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok |
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok |
| bT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok |
| bT | 2.5.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.6.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.7.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.8.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.9.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.10.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.11.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.12.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.13.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok |
| aT | 2.5.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.6.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.7.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.8.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.9.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.10.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.11.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.12.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.13.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| slice | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok |
| slice | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok |

**in-256x256x64 fp32**

| レイアウト | torch | 2^32 byte (B0=16384) | 2^31 elem (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|---|
| contig | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.5.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.6.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.7.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.8.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.9.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.10.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.11.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.12.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.13.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok TRUNC TRUNC TRUNC |
| aT | 2.5.1 | ok ok ok ok ok | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.6.0 | ok ok ok ok ok | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.7.1 | ok ok ok ok ok | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.8.0 | ok ok ok ok ok | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.9.1 | ok ok ok ok ok | ok ok err err err | err err err err err |
| aT | 2.10.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| aT | 2.11.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| aT | 2.12.1 | ok ok ok ok ok | ok ok err err err | err err err err err |
| aT | 2.13.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| aT | 2.14.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.4.1 | ok ok ok ok ok | ok ok ok ok ok | ok ok TRUNC TRUNC TRUNC |
| slice | 2.5.1 | ok ok ok ok ok | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.6.0 | ok ok ok ok ok | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.7.1 | ok ok ok ok ok | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.8.0 | ok ok ok ok ok | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.9.1 | ok ok ok ok ok | ok err err err err | err err err err err |
| slice | 2.10.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.11.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.12.1 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.13.0 | ok ok ok ok ok | ok ok err err err | err err err err err |
| slice | 2.14.0 | ok ok ok ok ok | ok ok err err err | err err err err err |

**in-256x256x64 fp16**

| レイアウト | torch | 2^31 elem = 2^32 byte (B0=32768) | 2^32 elem (B0=65536) |
|---|---|---|---|
| contig | 2.4.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.5.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.6.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.7.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.8.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.9.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.10.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.11.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.12.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.13.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| contig | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.4.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.5.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.6.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.7.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.8.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.9.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.10.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.11.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.12.1 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.13.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| bT | 2.14.0 | ok ok ok ok ok | ok ok ok WRONG WRONG |
| aT | 2.4.1 | ok ok ok ok ok | ok ok TRUNC TRUNC TRUNC |
| aT | 2.5.1 | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.6.0 | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.7.1 | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.8.0 | ok ok CRASH CRASH CRASH | CRASH CRASH err WRONG WRONG |
| aT | 2.9.1 | ok ok err err err | err err err err err |
| aT | 2.10.0 | ok ok err err err | err err err err err |
| aT | 2.11.0 | ok ok err err err | err err err err err |
| aT | 2.12.1 | ok ok err err err | err err err err err |
| aT | 2.13.0 | ok ok err err err | err err err err err |
| aT | 2.14.0 | ok ok err err err | err err err err err |
| slice | 2.4.1 | ok ok ok ok ok | ok ok TRUNC TRUNC TRUNC |
| slice | 2.5.1 | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.6.0 | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.7.1 | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.8.0 | ok CRASH CRASH CRASH CRASH | CRASH err ok WRONG WRONG |
| slice | 2.9.1 | ok err err err err | err err err err err |
| slice | 2.10.0 | ok ok err err err | err err err err err |
| slice | 2.11.0 | ok ok err err err | err err err err err |
| slice | 2.12.1 | ok ok err err err | err err err err err |
| slice | 2.13.0 | ok ok err err err | err err err err err |
| slice | 2.14.0 | ok ok err err err | err err err err err |

## 最終点(全要素比較 seed 0)と index 符号化(rc)

切り替わりの下・上の全要素比較の分類 / 最大相対誤差 / 閾値超え要素の割合、index 符号化(rc、切り替わりの異常側)の上位 delta。

| torch | shape | dtype | レイアウト | B | 入力 | 分類 | 最大誤差 | 閾値超え割合 | 壊れたバッチの区間 | 仮説 | 上位の delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.4.1 | out-256x64x256 | fp32 | contig | 65537 | idx_a_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | contig | 65537 | idx_b_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | contig | 65537 | random | WRONG | 1.3 | 3.1e-05 | [0,1),[65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 1 | 1.5e-05 | [65536,65537) | stride 0.75 |  |
| 2.4.1 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 1.3 | 3.1e-05 | [0,1),[65536,65537) | stride 0.00 |  |
| 2.4.1 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 1.3 | 3.1e-05 | [0,1),[65536,65537) | stride 0.00 |  |
| 2.4.1 | out-256x64x256 | fp32 | slice | 65537 | idx_a_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | slice | 65537 | idx_b_rc | WRONG | 1 | 1.5e-05 | [65536,65537) |  |  |
| 2.4.1 | out-256x64x256 | fp32 | slice | 65537 | random | WRONG | 1.3 | 3.1e-05 | [0,1),[65536,65537) | nooffset 0.00 |  |
| 2.4.1 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.4.1 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65536 | idx_a_rc | WRONG | 1 | 1 | [0,65536) | stride 0.00 |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65536 | idx_b_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp16 | aT | 65536 | random | TRUNC | 1 | 0.99 | [0,65536) | stride 0.01 |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65536 | idx_a_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65536 | idx_b_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp16 | slice | 65536 | random | TRUNC | 1 | 0.99 | [0,65536) | nooffset 0.01 |  |
| 2.4.1 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.4.1 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65536 | idx_a_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65536 | idx_b_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp32 | aT | 65536 | random | TRUNC | 1 | 1 | [0,65536) | stride 0.00 |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65536 | idx_a_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65536 | idx_b_rc | WRONG | 1 | 1 | [0,65536) |  |  |
| 2.4.1 | in-256x256x64 | fp32 | slice | 65536 | random | TRUNC | 1 | 1 | [0,65536) | nooffset 0.00 |  |
| 2.5.1 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.5.1 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.5.1 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.5.1 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.5.1 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.5.1 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.5.1 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.5.1 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.5.1 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.5.1 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.5.1 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.5.1 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.5.1 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.5.1 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp16 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.5.1 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.5.1 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch.  |
| 2.5.1 | in-256x256x64 | fp32 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.6.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.6.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.6.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.6.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.6.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.6.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.6.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.6.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.6.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.6.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.6.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.6.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp16 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch.  |
| 2.6.0 | in-256x256x64 | fp32 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.7.1 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.7.1 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.7.1 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.7.1 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.7.1 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.7.1 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.7.1 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.7.1 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.7.1 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.7.1 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.7.1 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.7.1 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp16 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch.  |
| 2.7.1 | in-256x256x64 | fp32 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.8.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.8.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.8.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.8.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.8.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.8.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.8.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.8.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.8.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.8.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.8.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.8.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp16 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | aT | 32768 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32767 | idx_a_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32767 | idx_b_rc | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 32767 | random | CRASH | - | - | - | - | 31: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX' |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65535 | idx_a_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65535 | idx_b_rc | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65535 | random | err | - | - | - | - | rs/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch.  |
| 2.8.0 | in-256x256x64 | fp32 | slice | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.9.1 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.9.1 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.9.1 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.9.1 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.9.1 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.9.1 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.9.1 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.9.1 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.9.1 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.9.1 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.9.1 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.9.1 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.9.1 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.9.1 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.9.1 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp16 | slice | 32767 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp16 | slice | 32767 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp16 | slice | 32767 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.9.1 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.9.1 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | slice | 32767 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | slice | 32767 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.9.1 | in-256x256x64 | fp32 | slice | 32767 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.10.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.10.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.10.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.10.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.10.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.10.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.10.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.10.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.10.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.10.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.10.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.10.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.10.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.10.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp16 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp16 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp16 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.10.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.10.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.10.0 | in-256x256x64 | fp32 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.11.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.11.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.11.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.11.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.11.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.11.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.11.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.11.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.11.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.11.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.11.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.11.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.11.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.11.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp16 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp16 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp16 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.11.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.11.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.11.0 | in-256x256x64 | fp32 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.12.1 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.12.1 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.12.1 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.12.1 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.12.1 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.12.1 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.12.1 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.12.1 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.12.1 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.12.1 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.12.1 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.12.1 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.12.1 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.12.1 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp16 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp16 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp16 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.12.1 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.12.1 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.12.1 | in-256x256x64 | fp32 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX |
| 2.13.0 | out-256x64x256 | fp16 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | -983: 4128, -791: 4128 |
| 2.13.0 | out-256x64x256 | fp16 | bT | 65537 | idx_b_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -892: 524312, -938: 262180 |
| 2.13.0 | out-256x64x256 | fp16 | bT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.13.0 | out-256x64x256 | fp16 | aT | 65537 | idx_a_rc | WRONG | 0.99 | 0.99 | [0,65537) | stride 1.00 | -96: 131108, 159: 131108 |
| 2.13.0 | out-256x64x256 | fp16 | aT | 65537 | idx_b_rc | WRONG | 1 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -1015: 132096 |
| 2.13.0 | out-256x64x256 | fp16 | aT | 65537 | random | WRONG | 2.1 | 0.99 | [0,65537) | stride 1.00 |  |
| 2.13.0 | out-256x64x256 | fp32 | bT | 65537 | idx_a_rc | WRONG | 3 | 1 | [0,65537) | stride 1.00 | 99: 65552, 483: 65552 |
| 2.13.0 | out-256x64x256 | fp32 | bT | 65537 | idx_b_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -5100: 6156, -5037: 6156 |
| 2.13.0 | out-256x64x256 | fp32 | bT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.13.0 | out-256x64x256 | fp32 | aT | 65537 | idx_a_rc | WRONG | 0.98 | 1 | [0,65537) | stride 1.00 | -2079: 12300, 2142: 12300 |
| 2.13.0 | out-256x64x256 | fp32 | aT | 65537 | idx_b_rc | WRONG | 0.75 | 0.98 | [0,65537) | stride 1.00 | -256: 197632, -3840: 132096 |
| 2.13.0 | out-256x64x256 | fp32 | aT | 65537 | random | WRONG | 2.1 | 1 | [0,65537) | stride 1.00 |  |
| 2.13.0 | in-256x256x64 | fp16 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.13.0 | in-256x256x64 | fp16 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.13.0 | in-256x256x64 | fp16 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp16 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp16 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp16 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp16 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp16 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | contig | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.13.0 | in-256x256x64 | fp32 | bT | 65537 | random | WRONG | 1.7 | 1.5e-05 | [65536,65537) | wrap 1.00 |  |
| 2.13.0 | in-256x256x64 | fp32 | aT | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | aT | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | aT | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | slice | 32768 | idx_a_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | slice | 32768 | idx_b_rc | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
| 2.13.0 | in-256x256x64 | fp32 | slice | 32768 | random | err | - | - | - | - | RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX |
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

- **2.4.1**: truncated ×16 ``
- **2.5.1**: crash ×40 `ryDirectory.nZ8QWc/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:831: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'`; error ×10 `RuntimeError: srcNDArray INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":480, please report a bug to PyTorch. `
- **2.6.0**: crash ×40 `ryDirectory.nZ8QWc/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:831: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'`; error ×10 `RuntimeError: srcNDArray INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":486, please report a bug to PyTorch. `
- **2.7.1**: crash ×40 `ryDirectory.nZ8QWc/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:831: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'`; error ×10 `RuntimeError: srcNDArray INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":516, please report a bug to PyTorch. `
- **2.8.0**: crash ×40 `ryDirectory.nZ8QWc/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:831: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'`; error ×10 `RuntimeError: srcNDArray INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":517, please report a bug to PyTorch. `
- **2.9.1**: error ×54 `RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX`
- **2.10.0**: error ×52 `RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX`
- **2.11.0**: error ×52 `RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX`
- **2.12.1**: error ×52 `RuntimeError: MPSGaph does not support tensor dims larger than INT_MAX`
- **2.13.0**: error ×52 `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX`
- **2.14.0**: error ×136 `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX`
