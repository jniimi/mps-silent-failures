# bmm_bwd: Studio (M2 Ultra, macOS 27.0) と MBP (M2 Max, macOS 26.6.2) の比較

torch 2.14.0、`--op bmm_bwd --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16 --budget-hours 24`、同じ v2 のコード(scan.py abcd1fbb8854、worker.py a264baa58030)。`compare_hosts.py` は bmm_bwd を対象外にしているので、行を直接突き合わせた(キー = op, shape, dtype, レイアウト, 入力, B, seed, 比較の方法。較正を除く)。

| | Studio | MBP |
|---|---|---|
| run | 20260920T120157-studio | 20260920T162628-mbp-macos26 |
| 行(較正を含む) | 604 | 604 |
| 分類(較正を除く 592 点) | ok 408 / error 144 / wrong 40 | ok 408 / error 144 / wrong 40 |

共通の 592 点(片方にしかない点は 0)で、次の項目が全部同じ: 分類、tensor ごとの分類(`cls_out`、`cls_grad_a`、`cls_grad_b`)、最大誤差、grad_a の最大誤差、誤答のバッチの範囲(全体・grad_a・grad_b)、仮説の一致率、誤った要素数、許容誤差、例外のメッセージ。

MBP のデータは W&B の `final` の Artifact から復元した(`v2/results/raw/mbp-macos26/`、git 管理外)。
