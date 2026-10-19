# MPS の nll_loss / cross_entropy: 正解クラスの要素の平坦 index が 2^31 以上の行で損失が黙って 0 になる

MacBook Pro (Mac14,6, M2 Max, 96 GB)、macOS 27.0 (26A428)、Python 3.12.11、fp32。2026-09-24 実行。
実行は `uv run --no-project --python 3.12 --with torch==<版> --with numpy python firstpass/<script> ...`(v2/ から)。

## probe_nll_loss.py(cross_entropy / nll_loss / log_softmax / gather の切り分け、torch 2.14.0)

入力は CPU で生成して MPS に送り、先頭・中央・末尾の塊を読み戻して一致を確認。基準は CPU の小さな塊を MPS で計算した行ごとの損失(先頭と末尾の 2^14 行は CPU float64 と照合済み)。

`2**23 256`(2^31 要素ちょうど、randn。この実行のあと、生成を速めるためにスクリプトの入力を一様 [-4, 4) に変えた): すべて基準と一致(sum / mean / none、nll_loss(log_softmax)、gather の和)。

`2**24 256`(2^32 要素ちょうど、一様 [-4, 4)):

```
cross_entropy sum      / ref = nan
cross_entropy mean * R / ref = nan
cross_entropy none: sum/ref = nan, wrong rows = 8388567, first 8388608, last 16777215, zero among wrong = 8380834
log_softmax max abs diff vs CPU (3 chunks) = 9.54e-07
nll_loss(log_softmax) sum  / ref = nan
nll_loss none: sum/ref = nan, wrong rows = 8388567
gather then .sum() on MPS / ref = 1.000000
```

(行 8388608 以降の 8388608 行のうち 41 行が誤りに数えられていない。差の比較は NaN を誤りに数えないので、この 41 行は NaN の行と考えられる。sum が NaN になることと整合する。)

## probe_nll_loss_edge.py(境界が行か要素かの切り分け)

`2**23 + 2**20 250`(torch 2.14.0): 要素 2^31 は行 8589934 の列 148。

```
R=9437184 C=250 numel=2359296000 (2^31 x 1.0986); wrong rows = 847249
first wrong row 8589935; row start element 2147483750; 2^31 / C = 8589934.592
target offset >= 2^31  vs wrong:  both 847249, wrong only 0, offset only 0
  row 8589933: start 2147483250, target 119, offset 2147483369 (< 2^31), wrong=False, got=8.538, ref=8.538
  row 8589934: start 2147483500, target 43, offset 2147483543 (< 2^31), wrong=False, got=9.183, ref=9.183
  row 8589935: start 2147483750, target 158, offset 2147483908 (>= 2^31), wrong=True, got=-0, ref=9.573
  row 8589936: start 2147484000, target 48, offset 2147484048 (>= 2^31), wrong=True, got=-0, ref=5.976
wrong values: zero 736986, nan 0, inf 0, other 110263
```

`2**23 + 2**16 256` を 2.13.0、2.9.1、2.4.1 で: 3 版とも同じ結果(wrong rows = 65536、すべて目標の平坦 index ≥ 2^31 の行と一致、値はすべて 0)。

## issue_repro_nll_loss.py(Issue の再現コード、torch 2.14.0、約 9 秒)

```
wrong rows: 65536 | rows whose target lies at flat index >= 2**31: 65536 | same rows: True | wrong values that are 0: 65536
sum on MPS: 62565556.0 | CPU: 63054878.21647382
-log_softmax(x).gather(...).sum() on MPS: 63054876.0
```

## CUDA の対照(A100)

同じスクリプト(デバイス名だけ "cuda" に置換)を Colab の A100-SXM4-80GB、PyTorch 2.14.0+cu130 で実行。4 回ともすべての行が正しい。詳細は a100/README.md。
