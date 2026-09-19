# CUDA control: A100 での全走査(torch 2.14.0)

Colab A100-SXM4-80GB(ドライバ 580.82.07、CUDA 13.0、torch 2.14.0+cu130、Python 3.12.3)で、Studio の MPS の全走査(run 20260919T130741-studio)と同じ spec・生成器・誤差指標を `worker.py --device cuda` で回した。colab CLI のセッションが約 1 時間で操作できなくなるため、`colab_chunks.py` で (shape, dtype, レイアウト) のチャンクに分け、1 チャンク = 1 run とした。各 run の要約は同じディレクトリの `<run_id>.md`、生データは `results/raw/a100/`(git 管理外)。

| チャンク | run_id | 実行数 | 分類 |
|---|---|---|---|
| out-256x64x256_fp32_contig-bT | 20260919T121806-a100 | 147 | ok 147 |
| out-256x64x256_fp32_aT-slice | 20260919T121819-a100 | 147 | ok 147 |
| out-256x64x256_fp16_contig-bT | 20260919T125339-a100 | 107 | ok 107 |
| out-256x64x256_fp16_aT-slice | 20260919T125359-a100 | 107 | ok 107 |
| in-256x256x64_fp32_contig-bT | 20260919T132422-a100 | 147 | ok 147 |
| in-256x256x64_fp32_aT-slice | 20260919T132436-a100 | 147 | ok 147 |
| in-256x256x64_fp16_contig-bT | 20260919T140058-a100 | 107 | ok 107 |
| in-256x256x64_fp16_aT-slice | 20260919T140336-a100 | 107 | ok 107 |
| out-512x64x128_fp32_contig-bT | 20260919T142913-a100 | 56 | ok 56 |
| out-512x64x128_fp32_aT-slice | 20260919T143348-a100 | 56 | ok 56 |
| out-128x64x512_fp32_contig-bT | 20260919T144739-a100 | 56 | ok 56 |
| out-128x64x512_fp32_aT-slice | 20260919T145245-a100 | 56 | ok 56 |
| in-512x128x64_fp32_contig-bT | 20260919T150630-a100 | 56 | ok 56 |
| in-512x128x64_fp32_aT-slice | 20260919T151112-a100 | 56 | ok 56 |
| in-128x512x64_fp32_contig | 20260919T165109-a100 | 28 | ok 28 |
| in-128x512x64_fp32_bT | 20260919T165113-a100 | 28 | ok 28 |
| in-128x512x64_fp32_slice | 20260919T170150-a100 | 28 | ok 28 |
| in-128x512x64_fp32_aT | 20260919T170119-a100 | 28 | ok 28 |

合計 1464 実行: {'ok': 1464}。誤答・例外・crash・truncated は 0 件。

- `in-128x512x64_fp32_contig-bT` は 1 チャンクではセッションの寿命(約 1 時間)を超えたため、レイアウトごとの 4 チャンク(`in-128x512x64_fp32_{contig,bT,aT,slice}`)で回し直した。無効になった run は無い(完了したものだけを記録)
- 失敗した run `20260919T060440-a100`(環境取得の不具合で全件 skipped_memory)と、途中でセッションを失った run `20260919T110858-a100`(290 実行、全件 ok、コード 790483c 以前)は集計に含めない
- 各 run の manifest の git_rev は 790483c または 699d21d(両者の間の差分は DESIGN.md と arxiv-v1 のみで、scan.py / worker.py の sha256 は同一)
