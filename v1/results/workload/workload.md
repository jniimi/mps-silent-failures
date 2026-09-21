# 実ワークロードでの silent error: RoBERTa 感情分類(eager attention、大バッチ 1 回の forward)

`demo_workload.py` の出力。数値は `workload.json`、logits は `logits.npz`、層 0 の中間は `layer0.npz`、図は `fig_rel_by_position.png`(縦軸はテキストごとの rel、横軸はバッチ内の位置。rel = 0(ビット一致)は 1e-9 に、赤の点線は仮説 A の境界のテキスト)。

## 環境と条件

- macOS 27.0 (26A428)、Mac14,14 (Apple M2 Ultra, 192 GiB)
- Python 3.12.12、torch 2.14.0、transformers 5.17.0、datasets 5.0.1
- モデル: `cardiffnlp/twitter-roberta-base-sentiment-latest` (revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`)、tokenizer は同じリポジトリ (revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`)、12 heads、fp32
- データ: `cardiffnlp/tweet_eval` / `sentiment` / `test` (12284 件、revision `b3a375baf0f409c77e6bc7aa35102b7b3534f8be`) の先頭 2048 件(データセットの順)。ラベル件数 [631, 1024, 393] ({'0': 'negative', '1': 'neutral', '2': 'positive'})
- トークン化: padding=max_length, truncation=True, max_length=512(切り詰め 0 件、実トークン数の平均 25.6)。全条件で同一の input_ids / attention_mask
- torch.inference_mode(), model.eval()、`torch.manual_seed(0)`。eager と sdpa の違い以外は同一。コード git `66ce8db45fbd`

## 基準

MPS・eager・256 件ずつの分割実行(scores は 256×12×512×512 = 805,306,368 要素)を基準とする(サンプル間の相互作用がないので f(concat(X_i)) = concat(f(X_i)))。所要 16.6 秒。

独立確認: CPU fp32(eager、256 件ずつ)で 1000 件を計算(0-99, N/2-50..N/2+49, N-100..N-1, 1344-1384, rest = smallest top1-top2 margin in reference)。分割実行と CPU の差は rel 最大 6e-05、中央値 1e-06、予測が変わった件数 0(選抜内の margin の最大 4.53)。CPU 51 秒、16 threads。

指標: テキストごとに rel = max_i |Δlogit_i| / max_i |logit_ref,i|。rel > 0.001 を「壊れた」とする。

## 結果(logits)

| B | scores − 2**32 | FFN 中間 / 2**32 | 条件 | 所要 [s] | 壊れたテキスト | 壊れた位置(バッチ内) | 予測が変わった件数 (割合) | 最大 rel | 正常なテキストの最大 rel | 正解率 (基準) | CPU 選抜での最大 rel / 予測変化 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1365 | -1,048,576 | 0.500 | mps_eager_full | 11.5 | 0 (0.00%) | — | 0 (0.00%) | 0 | 0 | 0.7077 (0.7077) | 6e-05 / 0 (n=659) |
| 1366 | +2,097,152 | 0.500 | mps_eager_full | 11.4 | 1 (0.07%) | 1365 | 1 (0.07%) | 0.781 | 0 | 0.7072 (0.7079) | 0.78 / 1 (n=660) |
| 1366 | +2,097,152 | 0.500 | mps_sdpa_full | 8.8 | 0 (0.00%) | — | 0 (0.00%) | 1.94e-05 | 1.9e-05 | 0.7079 (0.7079) | 6.6e-05 / 0 (n=660) |
| 2048 | +2,147,483,648 | 0.750 | mps_eager_full | 17.3 | 683 (33.35%) | 1365–2047 | 386 (18.85%) | 5.16 | 0 | 0.6387 (0.7129) | 5.2 / 178 (n=1000) |
| 2048 | +2,147,483,648 | 0.750 | mps_sdpa_full | 13.1 | 0 (0.00%) | — | 0 (0.00%) | 1.94e-05 | 1.9e-05 | 0.7129 (0.7129) | 6.6e-05 / 0 (n=1000) |
| 2048 | +2,147,483,648 | 0.750 | mps_eager_full_shift1024 | 16.9 | 683 (33.35%) | 1365–2047 | 380 (18.55%) | 8.49 | 0 | 0.6538 (0.7129) | — |

循環シフト (B=2048, shift=1024): 壊れたテキスト(元の index)の区間 [[341, 1023]]、バッチ内の位置の区間 [[1365, 2047]]。シフトなしと壊れた位置が一致: True、壊れたテキストの重なり 0 件。

## 層 0 の attention の中間(一括 vs 分割、テキスト×ヘッドごと)

S = (q @ kᵀ)·scale、P = softmax(S + mask)、O = P @ v、O_isolated = P_ref @ v(2 つ目の積だけ)。

| B | 段 | 壊れたテキスト | 区間 | 最初の (text, head) | 境界のテキストの壊れた head | 最大 rel | 正常の最大 rel |
|---|---|---|---|---|---|---|---|
| 1365 | S | 0 | [] | None | None | 0 | 0 |
| 1365 | P | 0 | [] | None | None | 0 | 0 |
| 1365 | O | 0 | [] | None | None | 0 | 0 |
| 1365 | O_isolated | 0 | [] | None | None | 0 | 0 |
| 1366 | S | 0 | [] | None | [] | 0 | 0 |
| 1366 | P | 0 | [] | None | [] | 0 | 0 |
| 1366 | O | 1 | [[1365, 1365]] | [1365, 4] | [4, 5, 6, 7, 8, 9, 10, 11] | 1.36 | 0 |
| 1366 | O_isolated | 1 | [[1365, 1365]] | [1365, 4] | [4, 5, 6, 7, 8, 9, 10, 11] | 1.36 | 0 |
| 2048 | S | 0 | [] | None | [] | 0 | 0 |
| 2048 | P | 0 | [] | None | [] | 0 | 0 |
| 2048 | O | 683 | [[1365, 2047]] | [1365, 4] | [4, 5, 6, 7, 8, 9, 10, 11] | 6.74 | 0 |
| 2048 | O_isolated | 683 | [[1365, 2047]] | [1365, 4] | [4, 5, 6, 7, 8, 9, 10, 11] | 6.74 | 0 |

層 0 の診断の所要 21 秒。

## L=256 での abort(別実行の記録)

L=256 では FFN の中間 (B*256, 3072) が scores と同じ B=5462 で 2**32 を超える。B=5461 の一括 forward は基準と一致(rel 0)したが、B=5462 の一括 forward は `MPSNDArrayMatrixMultiplication encodeToCommandBuffer: destination [3072, 1398272] is too large for kernel` の assertion で abort した(exit 134、silent ではない)。ログは `abort_L256_B5462.log`。


全体の所要時間 173 秒。
