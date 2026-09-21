# v2/analysis

2026-09-20 の Studio での軸の走査(`../results/summary/studio/v2_axes_studio.md`)で使った解析スクリプトと指定点のファイル。サブエージェントが scratch に作ったものを、失われないように無改変でここへ移した。指定点のファイルは、その後 MBP(macOS 26.6.2 / 27.0)と A100 の同じ系列でもそのまま使った(`compare_rows.py` は指定点のリストの sha256 で系列を対応づける)。

| ファイル | 内容 |
|---|---|
| `rules.py` | 規則の照合。`uv run python analysis/rules.py A\|B\|C <jsonl …>`(`v2/` で実行)。乱数入力の全行(較正を除く)について、規則の読み方 A / B / C の予測と記録された分類を突き合わせ、一致数と例外を出す。`--idx` で index 符号化の行。offset レイアウトと bmm_bwd の run は対象外 |
| `rules_bwd.py` | bmm_bwd の走査の全行を、forward と同じ規則(規則 B)を勾配の 2 つの積(grad_a = G b^T、grad_b = a^T G)に当てて照合する。out / grad_a / grad_b の分類、誤答の範囲、例外の stage。`uv run --no-project python analysis/rules_bwd.py results/raw/studio/<run_id>.jsonl`(原稿の 592 / 592) |
| `analyze.py` | run ごとの解析(系列ごとの区間、誤答の範囲、仮説の一致率、例外の stage、異常な分類)。bmm_bwd と offset にも対応 |
| `mkfar.py` | 遠い点(2 × 2**32 の前後)の指定点ファイルを作る |
| `far_points_full.json` / `far_points_sample.json` | 遠い点 112 点(全要素比較 / サンプル比較)。`scan.py --points` に渡す |
| `all256_supp_points.json` | all-256 の補足 30 点(aT / bT の 65537・65538 と連続の 65537、fp32 / fp16、seed 0〜2、全要素比較) |
| `chunks/` | `colab_chunks.py --series far,supp` が作る、Colab の 1 セッションに収まるように分けた指定点のファイル。遠い点を shape × dtype の 4 つ(`far_<shape>_<dtype>.json`、各 28 点、計 112)、all-256 の補足を dtype の 2 つ(`supp_all-256x256x256_<dtype>.json`、各 15 点、計 30)。中身は上の 2 つのファイルの点と同じで、A100 の run(遠い点 4 run、補足 2 run)の `--points` に渡した |
| `queue.sh` / `queue2.sh` | 実際に流したキュー(記録用。中のパスは当時の絶対パスのまま) |
| `rules_all.txt` | 照合の出力(v1 の全走査 + all-256 + 遠い点 + 補足、計 1376 行で B が全行一致) |

注意: `rules.py` の「A」は v1 の論文の表そのものではない。v1 の表(Table tab:rules)は規則 1 の下に「output > 2**32, otherwise → correct」の行を持つので、**v1 の表 = 規則 B**。A は、その行を落として規則 2・3 に進む読み方。

規則 B: 出力 > 2**32 要素なら「転置があれば全バッチ誤答、なければ ok」で確定。規則 2(view が 2**31 以上で例外)と規則 3(連続が 2**32 超で 2**32 より後ろだけ誤答)は、出力 ≤ 2**32 のときだけ適用する。
