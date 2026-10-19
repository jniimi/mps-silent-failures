# CUDA での対照実行(A100、torch 2.14.0): nll_loss / cross_entropy は正解クラスの平坦 index が 2^31 以上の行でも正しい

Google Colab の A100 ランタイム(NVIDIA A100-SXM4-80GB、80 GB、ドライバ 580.82.07、nvidia-smi の CUDA Version 13.0)、high-RAM(ホストのメモリ 167 GiB)、CPU Intel Xeon @ 2.20GHz(12 論理 CPU)、Ubuntu 24.04.1。Python 3.12.3、torch 2.14.0+cu130(`torch.version.cuda` = 13.0、cuDNN 9.24.0)、numpy 2.5.2、fp32。TF32 などの設定は既定のまま。2026-09-24 実行。環境の詳細(nvidia-smi、lscpu、インストールしたパッケージの一覧)は `env.txt`。

## 実行方法

スクリプトはリポジトリの `v2/firstpass/` のもの(commit 1554967 の時点、変更なし。sha256 は各 .log の冒頭に記録)を VM に送り、デバイスの文字列と出力のラベルだけを置き換えた複製を実行した。

```
sed -e 's/"mps"/"cuda"/g' -e 's/on MPS/on CUDA/g' orig/firstpass/<script> > v2/firstpass/<script>
```

差分は `.to("mps")` → `.to("cuda")` と、出力の `on MPS` → `on CUDA` だけ(各 .log の冒頭に `diff -u` を記録)。複製に残る「MPS」は probe_nll_loss.py の 1、7、15 行のコメントのみ。MPS 固有の呼び出しはない。

環境の構築:

```
uv venv /content/venv --python 3.12
uv pip install --python /content/venv/bin/python --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.14.0 numpy
```

実行(`v2/` に相当する作業ディレクトリから、4 件を 1 件ずつ順に):

```
PYTHONUNBUFFERED=1 python firstpass/issue_repro_nll_loss.py
PYTHONUNBUFFERED=1 python firstpass/probe_nll_loss_edge.py '2**23 + 2**20' 250
PYTHONUNBUFFERED=1 python firstpass/probe_nll_loss.py '2**23' 256
PYTHONUNBUFFERED=1 python firstpass/probe_nll_loss.py '2**24' 256
```

`PYTHONUNBUFFERED=1` は stdout と stderr を 1 つのログに出力順のまま記録するためで、計算には関係しない。

| 実行 | ログ | 終了コード | 実時間 | GPU メモリ使用量の最大(nvidia-smi、0.5 秒ごと) |
|---|---|---|---|---|
| issue_repro_nll_loss.py | `issue_repro_nll_loss.log` | 0 | 24.5 s | 17118 MiB |
| probe_nll_loss_edge.py `'2**23 + 2**20' 250` | `probe_nll_loss_edge_2p23+2p20_C250.log` | 0 | 29.1 s | 18614 MiB |
| probe_nll_loss.py `'2**23' 256` | `probe_nll_loss_2p23_C256.log` | 0 | 31.4 s | 19022 MiB |
| probe_nll_loss.py `'2**24' 256` | `probe_nll_loss_2p24_C256.log` | 0 | 51.6 s | 35470 MiB |

## issue_repro_nll_loss.py(Issue の再現コード)

```
wrong rows: 0 | rows whose target lies at flat index >= 2**31: 65536 | same rows: False | wrong values that are 0: 0
sum on CUDA: 63054880.0 | CPU: 63054878.216116905
-log_softmax(x).gather(...).sum() on CUDA: 63054880.0
```

CPU と合わない行は 0。`same rows: False` は誤りの行が 1 つもないことによる(正解クラスの平坦 index が 2^31 以上の 65536 行も正しい)。`reduction="sum"` は CPU の float64 の和との差が 1.8 で、6.3e7 付近の fp32 の間隔(4)の半分未満。MPS では 62565556.0 だった(65536 行が 0 になった分だけ小さい)。

CPU の和は MPS の記録(63054878.21647382)と小数第 4 位以下で異なる。CPU の fp32 の行ごとの損失の計算が x86 と Apple Silicon で異なるためと考えられるが、確認はしていない。判定には影響しない。

## probe_nll_loss_edge.py(基準は CPU float64)

`'2**23 + 2**20' 250`:

```
R=9437184 C=250 numel=2359296000 (2^31 x 1.0986); wrong rows = 0
```

誤りの行がないので、スクリプトの `if bad.any():` 以降の出力はない。MPS では同じ入力で 847249 行が誤り(値の大半が 0)だった。

## probe_nll_loss.py(cross_entropy / nll_loss / log_softmax / gather の切り分け)

入力は CPU で生成(一様 [-4, 4))して CUDA に送り、先頭・中央・末尾の塊を読み戻して一致を確認。基準は CPU の小さな塊(2^20 行)ごとに同じデバイス(ここでは CUDA)で計算した行ごとの損失で、先頭と末尾の 2^14 行は CPU float64 と照合済み(assert が通っている)。

`'2**23' 256`(2^31 要素ちょうど):

```
[  15.6s] R=8388608 (2^23) C=256 numel=2147483648 = 2^32 x 0.5000
[  20.4s] transfer verified
[  22.8s] reference sum = 62571332.3
[  22.8s] cross_entropy sum      / ref = 1.000000
[  22.9s] cross_entropy mean * R / ref = 1.000000
[  22.9s] cross_entropy none: sum/ref = 1.000000, wrong rows = 0
[  29.1s] log_softmax max abs diff vs CPU (3 chunks) = 9.54e-07
[  29.1s] nll_loss(log_softmax) sum  / ref = 1.000000
[  29.1s] nll_loss none: sum/ref = 1.000000, wrong rows = 0
[  29.2s] gather then .sum() on CUDA / ref = 1.000000
```

(nll_loss.md の MPS の `2**23 256` は randn と記載されており、入力の分布が現在のスクリプトと異なる。)

`'2**24' 256`(2^32 要素ちょうど):

```
[  31.2s] R=16777216 (2^24) C=256 numel=4294967296 = 2^32 x 1.0000
[  37.9s] transfer verified
[  42.6s] reference sum = 125142164.6
[  42.7s] cross_entropy sum      / ref = 1.000000
[  42.7s] cross_entropy mean * R / ref = 1.000000
[  42.9s] cross_entropy none: sum/ref = 1.000000, wrong rows = 0
[  49.2s] log_softmax max abs diff vs CPU (3 chunks) = 9.54e-07
[  49.2s] nll_loss(log_softmax) sum  / ref = 1.000000
[  49.3s] nll_loss none: sum/ref = 1.000000, wrong rows = 0
[  49.4s] gather then .sum() on CUDA / ref = 1.000000
```

MPS では同じ入力で sum / mean / none がすべて nan、誤りの行が 8388567 だった。

## まとめ

CUDA(A100、torch 2.14.0+cu130)では 4 件とも誤りの行は 0 で、和・平均も基準と一致した。MPS で見られた「正解クラスの要素の平坦 index が 2^31 以上の行で損失が 0 になる」現象は、同じコード(デバイスの文字列のみ変更)を CUDA で実行したときには起きない。
