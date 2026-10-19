# 出力側の構造の積 + loss head: 学習 1 ステップの一括実行 vs 分割

`probe_bwd_heads.py` の出力。数値は `bwd_heads.json`。

## 環境と条件

- macOS 27.0 (26A428)、Mac14,14 (Apple M2 Ultra, 192 GiB)、device mps
- Python 3.12.12、torch 2.14.0、numpy 2.5.3、コード git `00a01f26fa51`
- 形状: {'bmm': {'M': 256, 'K': 64, 'N': 256}, 'relation': {'R': 64, 'L': 512, 'd': 16}}、size {'bmm': [65536, 65537], 'relation': [256, 257]}、基準の chunk {'bmm': (8192, 4096), 'relation': (32, 16)}、fp32、seed 0
- 乱数: leaves, t and Z are generated per unit (bmm batch / relation sequence) on CPU from a Generator seeded with (seed, kind, unit index), so they do not depend on the size or the chunking; T = softmax(Z, -1) per unit. The full run copies the whole CPU tensors to the device once and verifies them by reading back in blocks (inputs_verified); the references generate each chunk the same way
- 基準: same device and head, units in chunks of `chunk` (second reference: chunk2 = chunk // 2, noise floor); loss summed in float64; leaf grads placed into full-size CPU float32 tensors (each element comes from exactly one chunk); on MPS the first 2 units are also computed on CPU (ref.vs_cpu)
- 指紋: per unit (sum x^2, sum w*x) in float64 on CPU (w fixed +-1) of Y (the head input) and of G (grad at the product output, via a tensor hook), in the full run and both references; per-unit rel = max(|d sum x^2| / (2 sum x_ref^2), |d sum w*x| / ||x_ref||)
- 許容誤差: unit wrong if rel > max(10 * noise_unit, 1e-05) (grads: rel L2 per unit; Y, G: fingerprint rel); global grad rel L2: tau = max(10 * noise, 1e-06); loss rel diff: max(10 * noise, 1e-05) (one wrong unit of 65537 moves the loss by only ~1.5e-5 x its own relative error; the Y check is the per-unit forward test); noise = second reference vs first
- anomaly: forward+backward under set_detect_anomaly(True, check_nan=False)。PYTORCH_MPS_HIGH_WATERMARK_RATIO = None

## 予測(bmm の規則から)

- 1. output > 2^32 elements and an operand is a transposed view -> silently wrong in every batch; output > 2^32 otherwise -> correct
- 2. an operand that is a view has >= 2^31 elements -> raises (INT_MAX)
- 3. a contiguous operand has > 2^32 elements -> silently wrong only in batches beyond element 2^32
- assumptions: bmm backward: grad_a = G @ b^T, grad_b = a^T @ G (transposed views); matmul folds (n, R) into one batch dim and copies the non-mergeable q and q^T into contiguous 3-D tensors (see dispatch)
- control: Y exactly 2^32 elements (bmm B=65536, relation n=256): forward and gradients correct; except bmm + sum, whose backward G = expand(ones) is a stride-0 view with 2^32 >= 2^31 elements -> rule 2 (raise in BmmBackward0), if rule 2 counts the logical numel of a stride-0 view
- above: Y just above 2^32 (bmm B=65537, relation n=257): forward product correct (rule 1, contiguous operands); the gradient products take G (contiguous, > 2^32) -> rule 3 -> leaf gradients wrong only in the units beyond element 2^32 of G (bmm: batch 65536; relation: element 2^32 is folded batch 16384 = sequence 256, head 0 -> sequence 256); the loss head may raise first if it reduces over > 2^32 elements
- reduction_check: MPS reductions over tensors requiring 64-bit indexing raise NotImplementedError (observed: full sum at numel 2^32 + 2^20; dim-wise all inside _safe_softmax at 2^32 + 2^21; not at numel between 2^31 and 2^32). Exactly 2^32 is untested (raises if the check is numel > UINT32_MAX)
- head ce: log_softmax (softmax family; softmax on > 2^32 elements was correct in inference) and nll_loss (reduces over rows = numel/C < 2^32): no reduction over > 2^32 elements in forward; backward (nll_loss_backward, _log_softmax_backward_data) unknown
- head kl_rows: dim-wise sum over > 2^32 elements (the dim-wise all in _safe_softmax raised at numel 2^32 + 2^21)
- head kl_sum: full sum over > 2^32 elements inside kl_div (MPS sum_reduction raised for a full sum at numel 2^32 + 2^20)
- head lse: logsumexp: dim-wise amax and sum over > 2^32 elements
- head amax: dim-wise amax over > 2^32 elements; its backward counts ties with a dim-wise sum
- head sum: full sum over > 2^32 elements; if passed, the backward G is expand(ones), a stride-0 view with > 2^31 elements: bmm -> rule 2 (raise); relation: the /sqrt(d) backward materializes G -> rule 3

## 結果(一括の forward + backward)

単位は bmm ではバッチ、relation では系列。Y / G の列は指紋で誤りと判定した単位の数(区間)。勾配は葉ごと。

| 積 | size | Y / 2^32 | head | 予測 | status | 失敗の場所 | loss 相対差 (noise) | Y 誤り | G 誤り | 勾配 rel L2 | 勾配の誤り単位 (区間) | fwd / bwd [s] | メモリ 推定 / 実測ピーク [GB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bmm | B=65536 (≤ 2^32) | 1 | ce | ok: fwd and grads correct | ok | — | 0.5 (2e-08) **誤** | 0 | 0 | a: 0<br>b: 0 | a: 0<br>b: 0 | 4.0 / 3.3 | 68.9 / 79.9 |
| bmm | B=65536 (≤ 2^32) | 1 | kl_rows | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).l`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 103.1 / 78.5 |
| bmm | B=65536 (≤ 2^32) | 1 | kl_sum | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_ta`; NotImplementedError | — (8e-09) | 0 | — | — | — | — / — | 103.1 / 94.6 |
| bmm | B=65536 (≤ 2^32) | 1 | lse | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 85.9 / 26.8 |
| bmm | B=65536 (≤ 2^32) | 1 | amax | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()`; NotImplementedError | — (5e-09) | 0 | — | — | — | — / — | 68.7 / 26.8 |
| bmm | B=65536 (≤ 2^32) | 1 | sum | bwd raises (rule 2 on the stride-0 G) (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 34.4 / 26.8 |
| relation | n=256 (≤ 2^32) | 1 | ce | ok: fwd and grads correct | ok | — | 0.5 (1e-08) **誤** | 0 | 0 | x: 0 | x: 0 | 4.5 / 3.5 | 52.7 / 72.2 |
| relation | n=256 (≤ 2^32) | 1 | kl_rows | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).l`; NotImplementedError | — (1e-08) | 0 | — | — | — | — / — | 87.0 / 71.5 |
| relation | n=256 (≤ 2^32) | 1 | kl_sum | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_ta`; NotImplementedError | — (5e-08) | 0 | — | — | — | — / — | 87.0 / 87.6 |
| relation | n=256 (≤ 2^32) | 1 | lse | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()`; NotImplementedError | — (4e-09) | 0 | — | — | — | — / — | 69.8 / 37.0 |
| relation | n=256 (≤ 2^32) | 1 | amax | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()`; NotImplementedError | — (7e-09) | 0 | — | — | — | — / — | 52.6 / 36.0 |
| relation | n=256 (≤ 2^32) | 1 | sum | ok: fwd and grads correct (exactly 2^32: head fwd raises first if the reduction check is numel >= 2^32) | raised_forward | fwd (head): `probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()`; NotImplementedError | — (1e-08) | 0 | — | — | — | — / — | 35.4 / 37.0 |
| bmm | B=65537 | 1.000015 | ce | fwd + loss correct; grads wrong in batch 65536 only (rule 3 on G) | ok | — | 0.5 (2e-08) **誤** | 0 | 0 | a: 0.00545<br>b: 0.00545 | a: 1 (65536)<br>b: 1 (65536) | 4.3 / 3.7 | 68.9 / 81.8 |
| bmm | B=65537 | 1.000015 | kl_rows | head fwd may raise; else grads wrong in batch 65536 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).l`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 103.1 / 80.9 |
| bmm | B=65537 | 1.000015 | kl_sum | head fwd raises; else grads wrong in batch 65536 only (rule 3 on G) | raised_forward | fwd (head): `functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_ta`; NotImplementedError | — (8e-09) | 0 | — | — | — | — / — | 103.1 / 97.6 |
| bmm | B=65537 | 1.000015 | lse | head fwd may raise; else grads wrong in batch 65536 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 85.9 / 27.6 |
| bmm | B=65537 | 1.000015 | amax | head fwd may raise; else grads wrong in batch 65536 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()`; NotImplementedError | — (5e-09) | 0 | — | — | — | — / — | 68.7 / 27.6 |
| bmm | B=65537 | 1.000015 | sum | head fwd raises; else bwd raises (rule 2 on the stride-0 G) | raised_forward | fwd (head): `probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()`; NotImplementedError | — (0) | 0 | — | — | — | — / — | 34.4 / 27.6 |
| relation | n=257 | 1.003906 | ce | fwd + loss correct; grads wrong in sequence 256 only (rule 3 on G) | ok | — | 0.5 (1e-08) **誤** | 0 | 0 | x: 0.0814 | x: 1 (256) | 4.6 / 3.5 | 52.9 / 73.9 |
| relation | n=257 | 1.003906 | kl_rows | head fwd may raise; else grads wrong in sequence 256 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).l`; NotImplementedError | — (1e-08) | 0 | — | — | — | — / — | 87.3 / 73.7 |
| relation | n=257 | 1.003906 | kl_sum | head fwd raises; else grads wrong in sequence 256 only (rule 3 on G) | raised_forward | fwd (head): `functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_ta`; NotImplementedError | — (5e-08) | 0 | — | — | — | — / — | 87.3 / 90.3 |
| relation | n=257 | 1.003906 | lse | head fwd may raise; else grads wrong in sequence 256 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()`; NotImplementedError | — (4e-09) | 0 | — | — | — | — / — | 70.1 / 38.2 |
| relation | n=257 | 1.003906 | amax | head fwd may raise; else grads wrong in sequence 256 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()`; NotImplementedError | — (7e-09) | 0 | — | — | — | — / — | 52.8 / 38.2 |
| relation | n=257 | 1.003906 | sum | head fwd raises; else grads wrong in sequence 256 only (rule 3 on G) | raised_forward | fwd (head): `probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()`; NotImplementedError | — (1e-08) | 0 | — | — | — | — / — | 35.6 / 38.2 |

## 基準と入力の確認

| 条件 | 入力の確認 | 基準 | 分割数 | 所要 [s] | loss (基準) | 勾配の単位の noise 最大 | Y / G の noise 最大 | 基準 vs CPU(先頭単位の勾配 rel L2、Y / G) |
|---|---|---|---|---|---|---|---|---|
| bmm_B65536_ce | ok | ok (c8192, c4096) | 8, 16 | 26.1, 27.2 | 101369581 | a: 0, b: 0 | 0 / 0 | a: 7.4e-08, b: 6.9e-08; 0 / 2.2e-08 |
| bmm_B65536_kl_rows | ok | ok (c8192, c4096) | 8, 16 | 74.2, 75.2 | 16603294.1 | — | 0 / — | a: 2.1e-07, b: 1.9e-07; 0 / 2e-07 |
| bmm_B65536_kl_sum | ok | ok (c8192, c4096) | 8, 16 | 74.5, 75.4 | 16603294.1 | — | 0 / — | a: 2.1e-07, b: 1.9e-07; 0 / 2e-07 |
| bmm_B65536_lse | ok | ok (c8192, c4096) | 8, 16 | 27.4, 28.3 | 101373235 | — | 0 / — | a: 1.6e-07, b: 1.3e-07; 0 / 1.1e-07 |
| bmm_B65536_amax | ok | ok (c8192, c4096) | 8, 16 | 26.9, 28.1 | 47251568.5 | — | 0 / — | a: 0, b: 0; 0 / 0 |
| bmm_B65536_sum | ok | ok (c8192, c4096) | 8, 16 | 26.8, 27.5 | 2039985.05 | — | 0 / — | a: 0, b: 0; 0 / 0 |
| relation_n256_ce | ok | ok (c32, c16) | 8, 16 | 6.7, 6.6 | 57494800.5 | x: 0 | 0 / 0 | x: 9.7e-08; 0 / 2.9e-08 |
| relation_n256_kl_rows | ok | ok (c32, c16) | 8, 16 | 47.7, 47.2 | 9331244 | — | 0 / — | x: 1.7e-07; 0 / 1.6e-07 |
| relation_n256_kl_sum | ok | ok (c32, c16) | 8, 16 | 47.8, 47.3 | 9331243.75 | — | 0 / — | x: 1.7e-07; 0 / 1.6e-07 |
| relation_n256_lse | ok | ok (c32, c16) | 8, 16 | 6.9, 6.8 | 57563032 | — | 0 / — | x: 1.6e-07; 0 / 3e-07 |
| relation_n256_amax | ok | ok (c32, c16) | 8, 16 | 6.5, 6.5 | 33855907 | — | 0 / — | x: 0; 0 / 0 |
| relation_n256_sum | ok | ok (c32, c16) | 8, 16 | 6.3, 6.3 | 33559578.5 | — | 0 / — | x: 0; 0 / 0 |
| bmm_B65537_ce | ok | ok (c8192, c4096) | 9, 17 | 27.8, 28.6 | 101371120 | a: 0, b: 0 | 0 / 0 | a: 7.4e-08, b: 6.9e-08; 0 / 2.2e-08 |
| bmm_B65537_kl_rows | ok | ok (c8192, c4096) | 9, 17 | 75.1, 76.4 | 16603540.7 | — | 0 / — | a: 2.1e-07, b: 1.9e-07; 0 / 2e-07 |
| bmm_B65537_kl_sum | ok | ok (c8192, c4096) | 9, 17 | 75.1, 76.2 | 16603540.7 | — | 0 / — | a: 2.1e-07, b: 1.9e-07; 0 / 2e-07 |
| bmm_B65537_lse | ok | ok (c8192, c4096) | 9, 17 | 27.7, 28.7 | 101374777 | — | 0 / — | a: 1.6e-07, b: 1.3e-07; 0 / 1.1e-07 |
| bmm_B65537_amax | ok | ok (c8192, c4096) | 9, 17 | 27.3, 28.3 | 47252282.3 | — | 0 / — | a: 0, b: 0; 0 / 0 |
| bmm_B65537_sum | ok | ok (c8192, c4096) | 9, 17 | 27.1, 28.0 | 2039770.6 | — | 0 / — | a: 0, b: 0; 0 / 0 |
| relation_n257_ce | ok | ok (c32, c16) | 9, 17 | 6.8, 6.6 | 57719755.3 | x: 0 | 0 / 0 | x: 9.7e-08; 0 / 2.9e-08 |
| relation_n257_kl_rows | ok | ok (c32, c16) | 9, 17 | 48.0, 47.6 | 9367838.55 | — | 0 / — | x: 1.7e-07; 0 / 1.6e-07 |
| relation_n257_kl_sum | ok | ok (c32, c16) | 9, 17 | 48.1, 47.8 | 9367838.3 | — | 0 / — | x: 1.7e-07; 0 / 1.6e-07 |
| relation_n257_lse | ok | ok (c32, c16) | 9, 17 | 7.0, 6.8 | 57788047.2 | — | 0 / — | x: 1.6e-07; 0 / 3e-07 |
| relation_n257_amax | ok | ok (c32, c16) | 9, 17 | 6.6, 6.6 | 33988489.9 | — | 0 / — | x: 0; 0 / 0 |
| relation_n257_sum | ok | ok (c32, c16) | 9, 17 | 6.5, 6.4 | 33698397.5 | — | 0 / — | x: 0; 0 / 0 |

## dispatch された aten 演算(2 単位の入力)

行列積の引数は shape / stride / 連続(c)か非連続(nc)か。backward の bmm の 1 つ目の引数が G のとき、その連続性と stride が規則 2 と 3 のどちらに当たるかを決める。

- bmm の積(forward): bmm×1

| 積 | head | head の forward | backward | 行列積の呼び出し |
|---|---|---|---|---|
| bmm | ce | view×1, _log_softmax×1, detach×1, nll_loss_forward×1 | ones_like×1, nll_loss_backward×1, detach×3, _log_softmax_backward_data×1, view×1, transpose.int×2, bmm×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(65536, 256, 1)/c)<br>backward: bmm((2, 256, 256)/(65536, 256, 1)/c, (2, 256, 64)/(16384, 1, 256)/nc) |
| bmm | kl_rows | _log_softmax×1, detach×1, clamp_min×1, log×1, sub.Tensor×1, mul.Tensor×1, sum.dim_IntList×1, sum×1 | ones_like×1, expand×2, unsqueeze×1, mul.Tensor×1, neg×1, detach×3, _log_softmax_backward_data×1, transpose.int×2, bmm×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(65536, 256, 1)/c)<br>backward: bmm((2, 256, 256)/(65536, 256, 1)/c, (2, 256, 64)/(16384, 1, 256)/nc) |
| bmm | kl_sum | _log_softmax×1, detach×1, xlogy.Tensor×1, mul.Tensor×1, sub.Tensor×1, sum×1 | ones_like×1, expand×1, neg×1, mul.Tensor×1, detach×3, _log_softmax_backward_data×1, transpose.int×2, bmm×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(65536, 256, 1)/c)<br>backward: bmm((2, 256, 256)/(65536, 256, 1)/c, (2, 256, 64)/(16384, 1, 256)/nc) |
| bmm | lse | logsumexp×1, detach×1, sum×1 | ones_like×1, expand×1, detach×3, unsqueeze×2, sub.Tensor×1, exp×1, mul.Tensor×1, transpose.int×2, bmm×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(65536, 256, 1)/c)<br>backward: bmm((2, 256, 256)/(65536, 256, 1)/c, (2, 256, 64)/(16384, 1, 256)/nc) |
| bmm | amax | amax×1, detach×1, sum×1 | ones_like×1, expand×1, detach×3, view×2, eq.Tensor×1, sum.dim_IntList×1, div.Tensor×1, mul.Tensor×1, transpose.int×2, bmm×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(65536, 256, 1)/c)<br>backward: bmm((2, 256, 256)/(65536, 256, 1)/c, (2, 256, 64)/(16384, 1, 256)/nc) |
| bmm | sum | sum×1 | ones_like×1, expand×1, transpose.int×2, bmm×2, detach×2 | product: bmm((2, 256, 64)/(16384, 64, 1)/c, (2, 64, 256)/(16384, 256, 1)/c)<br>backward: bmm((2, 64, 256)/(16384, 1, 64)/nc, (2, 256, 256)/(0, 0, 0)/nc)<br>backward: bmm((2, 256, 256)/(0, 0, 0)/nc, (2, 256, 64)/(16384, 1, 256)/nc) |
- relation の積(forward): view×1, transpose.int×2, expand×2, clone×2, _unsafe_view×3, bmm×1, div.Tensor×1

| 積 | head | head の forward | backward | 行列積の呼び出し |
|---|---|---|---|---|
| relation | ce | view×1, _log_softmax×1, detach×1, nll_loss_forward×1 | ones_like×1, nll_loss_backward×1, detach×2, _log_softmax_backward_data×1, view×4, div.Tensor×1, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |
| relation | kl_rows | _log_softmax×1, detach×1, clamp_min×1, log×1, sub.Tensor×1, mul.Tensor×1, sum.dim_IntList×1, sum×1 | ones_like×1, expand×2, unsqueeze×1, mul.Tensor×1, neg×1, detach×2, _log_softmax_backward_data×1, div.Tensor×1, view×3, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |
| relation | kl_sum | _log_softmax×1, detach×1, xlogy.Tensor×1, mul.Tensor×1, sub.Tensor×1, sum×1 | ones_like×1, expand×1, neg×1, mul.Tensor×1, detach×2, _log_softmax_backward_data×1, div.Tensor×1, view×3, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |
| relation | lse | logsumexp×1, detach×1, sum×1 | ones_like×1, expand×1, detach×2, unsqueeze×2, sub.Tensor×1, exp×1, mul.Tensor×1, div.Tensor×1, view×3, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |
| relation | amax | amax×1, detach×1, sum×1 | ones_like×1, expand×1, detach×2, view×5, eq.Tensor×1, sum.dim_IntList×1, div.Tensor×2, mul.Tensor×1, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |
| relation | sum | sum×1 | ones_like×1, expand×1, div.Tensor×1, view×3, transpose.int×4, bmm×2, add.Tensor×1, clone×1, _unsafe_view×1, detach×1 | product: bmm((128, 512, 16)/(8192, 16, 1)/c, (128, 16, 512)/(8192, 512, 1)/c)<br>backward: bmm((128, 16, 512)/(8192, 1, 16)/nc, (128, 512, 512)/(262144, 512, 1)/c)<br>backward: bmm((128, 512, 512)/(262144, 512, 1)/c, (128, 512, 16)/(8192, 1, 512)/nc) |

## 例外・異常終了の詳細

### bmm_B65536_kl_rows: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).log() - lp)).sum(-1).sum()
```

### bmm_B65536_kl_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:277 in h_kl_sum: return F.kl_div(F.log_softmax(ys.pop(), -1), T, reduction="sum")
functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)
```

### bmm_B65536_lse: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()
```

### bmm_B65536_amax: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()
```

### bmm_B65536_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()
```

### relation_n256_kl_rows: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).log() - lp)).sum(-1).sum()
```

### relation_n256_kl_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:277 in h_kl_sum: return F.kl_div(F.log_softmax(ys.pop(), -1), T, reduction="sum")
functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)
```

### relation_n256_lse: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()
```

### relation_n256_amax: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()
```

### relation_n256_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4294967296)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()
```

### bmm_B65537_kl_rows: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4295032832)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).log() - lp)).sum(-1).sum()
```

### bmm_B65537_kl_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4295032832)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:277 in h_kl_sum: return F.kl_div(F.log_softmax(ys.pop(), -1), T, reduction="sum")
functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)
```

### bmm_B65537_lse: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4295032832)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()
```

### bmm_B65537_amax: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4295032832)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()
```

### bmm_B65537_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4295032832)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()
```

### relation_n257_kl_rows: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4311744512)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:273 in h_kl_rows: return (T * (T.clamp_min(torch.finfo(T.dtype).tiny).log() - lp)).sum(-1).sum()
```

### relation_n257_kl_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4311744512)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:277 in h_kl_sum: return F.kl_div(F.log_softmax(ys.pop(), -1), T, reduction="sum")
functional.py:3470 in kl_div: reduced = torch.kl_div(input, target, reduction_enum, log_target=log_target)
```

### relation_n257_lse: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4311744512)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:281 in h_lse: return torch.logsumexp(ys.pop(), -1).sum()
```

### relation_n257_amax: raised_forward

- 例外(head): `NotImplementedError`: MPS max_reduction: tensors requiring 64-bit indexing are not supported (numel=4311744512)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:285 in h_amax: return ys.pop().amax(-1).sum()
```

### relation_n257_sum: raised_forward

- 例外(head): `NotImplementedError`: MPS sum_reduction: tensors requiring 64-bit indexing are not supported (numel=4311744512)
- traceback(末尾):

```
probe_bwd_heads.py:579 in job_case: box["loss"] = forward(sp, head, lv, tgt, w, st, dbg)
probe_bwd_heads.py:459 in forward: loss = HEAD_FN[head](ys, tgt)
probe_bwd_heads.py:289 in h_sum: return ys.pop().sum()
```


全体の所要時間 2311 秒。
