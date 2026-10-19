# 学習 1 ステップの一括実行 vs 分割累積: RoBERTa 感情分類(forward + backward)

`demo_train_step.py` の出力。数値は `train_step.json`、テンソルごとの勾配誤差は `train_step_tensors.csv`、logits とテキストごとの loss は `train_step_logits.npz`。

## 環境と条件

- macOS 27.0 (26A428)、Mac14,14 (Apple M2 Ultra, 192 GiB)、device mps
- Python 3.12.12、torch 2.14.0、transformers 5.17.0、datasets 5.0.1
- モデル: `cardiffnlp/twitter-roberta-base-sentiment-latest` (revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`)、12 層、12 heads、FFN 3072、fp32
- データ: `cardiffnlp/tweet_eval` / `sentiment` / `test`(12284 件、revision `b3a375baf0f409c77e6bc7aa35102b7b3534f8be`)の先頭 2048 件(データセットの順)。padding=max_length, truncation=True, max_length=512(切り詰め 0 件)。実トークン数の平均 25.6、ラベル件数 [631, 1024, 393]
- 学習ステップ: loss = cross_entropy(logits, dataset labels, reduction='sum'); all params requires_grad; no optimizer step; model.train() with every nn.Dropout p=0 and config dropouts 0
- gradient checkpointing: transformers gradient_checkpointing_enable(use_reentrant=False)。anomaly: forward+backward under set_detect_anomaly(True, check_nan=False)。seed 0。コード git `d735d4044133`
- 許容誤差: grad wrong if global rel L2 > tau = max(10 * noise, 1e-06); tensor wrong if rel L2 > max(10 * noise_t, 1e-05), except attention key biases (gradient exactly 0 in exact arithmetic, softmax shift invariance): wrong if abs L2 error > tau * ||g_ref|| (global); noise = ref c128 vs ref c256。forward は rel > 0.001 のテキストを「壊れた」とする
- PYTORCH_MPS_HIGH_WATERMARK_RATIO = None

## 予測(bmm の規則から)

- assumptions: predictions from the bmm rules found on PyTorch 2.14.0; they assume PyTorch's bmm backward formulas grad_self = grad @ mat2^T, grad_mat2 = self^T @ grad (transposed views) and that matmul's batch folding copies Q/K/V into contiguous 3-D tensors
- eager_below_2p31: B*H*L^2 < 2^31 (B <= 682): forward and gradients correct
- eager_2p31_to_2p32: 2^31 <= B*H*L^2 <= 2^32 (683 <= B <= 1365): forward correct; backward raises (rule 2 on P^T, a view with >= 2^31 elements, in the gradient of V for softmax(S)V)
- eager_above_2p32: B*H*L^2 > 2^32 (B >= 1366): forward wrong for texts from 1365 (head 4) on (rule 3 on P); backward raises (rule 2); grad_P = dO V^T would fall under rule 1 if reached
- eager_ffn: FFN intermediate (B*L, 3072) >= 2^31 from B = 1366: the FFN's 2-D matmuls (not covered by the bmm rules) have views >= 2^31 in backward and may raise or abort first; autograd reaches layer 11's FFN before its attention, so the anomaly trace is needed to tell which op failed
- sdpa: unknown (inference: dispatched to aten::_scaled_dot_product_attention_math_for_mps, forward correct). On torch 2.12.1 (MBP), with grad enabled and inputs requiring grad, MPS SDPA decomposed into aten.bmm x2 + aten._safe_softmax per layer instead; see refs.*.dispatch for the version actually run

## 基準

same impl and device, chunks <= 256 (every B a chunk boundary), grads accumulated in float64 on CPU, snapshot at each B。

| 基準 | 分割数 | 所要 [s] | loss の合計 | 正解率 | train=eval (ビット一致) | checkpointing 有無の勾配差 (rel L2) | forward で dispatch された演算(linear 以外) |
|---|---|---|---|---|---|---|---|
| eager_c256 | 11 | 138.9 | 1282.21 | 0.7129 | True | 4.5e-09 | bmm×24, _softmax×12 |
| eager_c128 | 20 | 132.6 | 1282.21 | 0.7129 | True | 4.2e-09 | bmm×24, _softmax×12 |
| sdpa_c256 | 11 | 191.6 | 1282.21 | 0.7129 | True | 4.1e-09 | bmm×24, _safe_softmax×12 |
| sdpa_c128 | 20 | 184.7 | 1282.21 | 0.7129 | True | 4.5e-09 | bmm×24, _safe_softmax×12 |

基準どうしの logits の差: eager_c128_vs_c256: 最大 rel 0、loss の合計の相対差 0、sdpa_c128_vs_c256: 最大 rel 0、loss の合計の相対差 0、sdpa_vs_eager_c256: 最大 rel 3.3e-05。

CPU 確認(eager、先頭 256 件を 1 回、55 秒、16 threads): eager_c256 vs CPU: logits 最大 rel 8.5e-06、勾配 rel L2 6.7e-06、1 − cos 2.2e-11、eager_c128 vs CPU: logits 最大 rel 8.5e-06、勾配 rel L2 6.8e-06、1 − cos 2.3e-11。

## 結果(B 件を 1 回で forward + backward)

| impl | B | scores / 2^32 | FFN / 2^31 | 予測 | status | 失敗の場所 (backward node @ forward の演算) | 壊れたテキスト (区間) | loss 相対差 | 勾配 rel L2 | 1 − cos | τ | 誤りテンソル | 最悪グループ (rel L2) | fwd / bwd [s] | driver メモリ fwd / bwd [GB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 256 (対照) | 0.1875 | 0.1875 | fwd correct, grads correct | ok | — | 0 | 0 | 3.31e-08 | 6.7e-16 | 1.5e-05 | 0/201 | embeddings (2.9e-07) | 2.3 / 15.0 | 20.6 / 28.6 |
| eager | 682 | 0.4995 | 0.4995 | fwd correct, grads correct | ok | — | 0 | 0 | 1.8e-06 | 1.6e-12 | 7.4e-06 | 0/201 | layer0 (2.1e-06) | 5.9 / 41.2 | 47.4 / 70.0 |
| eager | 683 | 0.5002 | 0.5002 | fwd correct, bwd raises (rule 2) | raised_backward | BmmBackward0 @ `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)` | 0 | 0 | — | — | — | — | — | 6.0 / — | 48.8 / 63.1 |
| eager | 1365 | 0.9998 | 0.9998 | fwd correct, bwd raises (rule 2) | raised_backward | BmmBackward0 @ `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)` | 0 | 0 | — | — | — | — | — | 11.7 / — | 93.6 / 121.5 |
| eager | 1366 | 1.0005 | 1.0005 | fwd wrong from text 1365 (rule 3), bwd raises (rule 2); FFN bwd may raise/abort first | raised_backward | BmmBackward0 @ `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)` | 1 (1365) | 0.0013 | — | — | — | — | — | 12.1 / — | 96.5 / 122.9 |
| eager | 2048 | 1.5000 | 1.5000 | fwd wrong from text 1365 (rule 3), bwd raises (rule 2); FFN bwd may raise/abort first | raised_backward | BmmBackward0 @ `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)` | 683 (1365–2047) | 0.3 | — | — | — | — | — | 18.9 / — | 139.7 / 178.4 |
| sdpa | 682 | 0.4995 | 0.4995 | unknown | ok | — | 0 | 0 | 1.81e-06 | 1.6e-12 | 7.5e-06 | 0/201 | layer0 (2.1e-06) | 6.3 / 50.3 | 51.3 / 72.8 |
| sdpa | 683 | 0.5002 | 0.5002 | unknown | raised_backward | BmmBackward0 @ `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(` | 0 | 0 | — | — | — | — | — | 7.3 / — | 52.9 / 66.1 |
| sdpa | 1365 | 0.9998 | 0.9998 | unknown | raised_backward | BmmBackward0 @ `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(` | 0 | 0 | — | — | — | — | — | 15.7 / — | 101.5 / 127.2 |
| sdpa | 1366 | 1.0005 | 1.0005 | unknown | raised_forward | forward: NotImplementedError | — | — | — | — | — | — | — | — / — | 62.4 / — |
| sdpa | 2048 | 1.5000 | 1.5000 | unknown | raised_forward | forward: NotImplementedError | — | — | — | — | — | — | — | — / — | 91.3 / — |

eager B=256 の一括 vs CPU: logits 最大 rel 8.5e-06、予測の変化 0、勾配 rel L2 6.7e-06、1 − cos 2.2e-11。

## 例外・異常終了の詳細

### eager B=683: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`、モデル内の場所: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)
```

### eager B=1365: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`、モデル内の場所: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)
```

### eager B=1366: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`、モデル内の場所: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)
```

### eager B=2048: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`、モデル内の場所: `modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
modeling_roberta.py:180 in eager_attention_forward: attn_output = torch.matmul(attn_weights, value)
```

### sdpa B=683: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(`、モデル内の場所: `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(
```

### sdpa B=1365: raised_backward

- 例外: `RuntimeError`: MPSGraph does not support tensor dims larger than INT_MAX
- backward node: `BmmBackward0`、forward の演算: `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(`、モデル内の場所: `sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(`
- forward の traceback(torch 以外のフレーム、末尾 8 個):

```
output_capturing.py:286 in wrapper: outputs = func(self, *args, **kwargs)
modeling_roberta.py:626 in forward: encoder_outputs = self.encoder(
modeling_roberta.py:509 in forward: hidden_states = layer_module(
modeling_layers.py:108 in __call__: return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
modeling_roberta.py:430 in forward: self_attention_output, _ = self.attention(
modeling_roberta.py:361 in forward: attention_output, attn_weights = self.self(
modeling_roberta.py:239 in forward: attn_output, attn_weights = attention_interface(
sdpa_attention.py:158 in sdpa_attention_forward: attn_output = torch.nn.functional.scaled_dot_product_attention(
```

### sdpa B=1366: raised_forward

- 例外: `NotImplementedError`: MPS all_reduction: tensors requiring 64-bit indexing are not supported (numel=4297064448)

### sdpa B=2048: raised_forward

- 例外: `NotImplementedError`: MPS all_reduction: tensors requiring 64-bit indexing are not supported (numel=6442450944)


全体の所要時間 1000 秒。work/ の大きさ 12.48 GB。
