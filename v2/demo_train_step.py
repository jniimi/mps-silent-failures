"""実際の ML ワークロードの学習 1 ステップ(forward + backward)で MPS が誤答・例外を出すかを見る(RoBERTa 感情分類、B 件一括 vs 分割累積)。

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --no-project --python 3.12 --with torch==2.14.0 \
        --with transformers==5.17.0 --with datasets==5.0.1 python demo_train_step.py

    (demo_workload.py と同じく版を固定した使い捨て環境で動かす。動作確認は --tiny(乱数初期化の小さなモデルと合成入力。
    datasets もハブも使わない)で、例: --tiny --device mps --batch-sizes 6 7 12 --max-length 16 --chunk 4 --chunk2 2 --cpu-check-n 4)

目的: demo_workload.py(推論のみ)の事例を学習の 1 ステップへ広げる。題材・データ・トークン化は demo_workload.py と同じ
(MODEL を DATASET / CONFIG / SPLIT の先頭 N=max(B) 件(データセットの順)に適用、padding="max_length", truncation=True,
max_length=L、fp32)。B 件を 1 回で forward + backward した勾配を、同じテキストを小分けにして累積した勾配と比べる。

学習ステップ: loss = cross-entropy(データセットのラベル、reduction="sum")。sum なので一括の勾配は chunk ごとの勾配の和に
数学的に等しい。全パラメータが requires_grad、optimizer の step はしない。dropout は切る(eval と同じ意味)。

dropout と gradient checkpointing(方式 (a)): transformers の gradient checkpointing は module.training が True のときだけ働く
(5.17.0 の modeling_layers.py:81 `if self.gradient_checkpointing and self.training:`。RobertaLayer はこの
GradientCheckpointingLayer の子、modeling_roberta.py:401)。そこで config の hidden_dropout_prob / attention_probs_dropout_prob /
classifier_dropout を 0 にし、念のため全 nn.Dropout の p も 0 にして model.train() とし、
gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) を使う。attention に渡る dropout は
`0.0 if not self.training else self.dropout.p`(modeling_roberta.py:245)なので train でも 0。eager は
F.dropout(p=0, training=True)(:178)で恒等、SDPA は dropout_p=0 のまま scaled_dot_product_attention に渡る
(integrations/sdpa_attention.py:163)。tiny モデル(MBP、torch 2.12.1 / transformers 5.12.1)で確認した: train(dropout 0) と
eval の logits はビット一致(CPU / MPS、eager / sdpa)。checkpointing の有無で、CPU は logits も勾配もビット一致、MPS は logits が
ビット一致で勾配は embedding の 2 テンソルだけ 1e-7 程度の差(MPS の embedding の backward は checkpointing なしでも 2 回の実行で
ビット一致しない)。同じ確認を各 ref ジョブの冒頭で 2 件について行い refs.*.self_check に記録する(Studio の版と本物のモデルでも
確かめるため)。--no-checkpointing で無効化する。

予測(下の規則からの予測。PyTorch の bmm の backward が grad_self = grad @ mat2ᵀ, grad_mat2 = selfᵀ @ grad(ᵀ は転置の view)
であり、matmul がバッチを畳むときに Q/K/V を連続なテンソルにコピーすることを仮定する):
  規則(PyTorch 2.14.0、この順に適用) 1. 出力 > 2^32 要素でオペランドが転置の view → 全バッチで誤答、それ以外の出力 > 2^32 は正しい。
  2. view のオペランドが 2^31 要素以上 → 例外(INT_MAX)。3. 連続なオペランドが 2^32 要素超 → 要素 2^32 を越えるバッチだけ誤答。
  S = Q·Kᵀ、P = softmax(S + mask)、O = P·V。scores の要素数は B·H·L²(L=512, H=12 で B·3·2^20)。
  eager, B·H·L² < 2^31(B <= 682): forward も勾配も正しい
  eager, 2^31 <= B·H·L² <= 2^32(683 <= B <= 1365): forward は正しい。backward は例外(V の勾配 Pᵀ·dO の Pᵀ が 2^31 要素以上の
      view、規則 2)
  eager, B·H·L² > 2^32(B >= 1366): forward は境界のテキスト以降が誤る(P に規則 3、推論の事例と同じ)。backward は例外(規則 2)。
      P の勾配 dO·Vᵀ は、到達すれば規則 1 に当たる。FFN の中間 (B·L, 3072) は B >= 1366 で 2^31 以上になり、FFN の 2 次元の
      matmul(bmm の規則の対象外)は backward で 2^31 以上の view を持つので、そこで先に例外か abort になるかもしれない。
      autograd は層 11 の FFN を attention より先に通るので、どの演算で失敗したかは anomaly のトレースで見分ける。
      B=683 と 1365 は attention だけを切り出す条件
  sdpa: 不明(推論では aten::_scaled_dot_product_attention_math_for_mps に dispatch され、forward は正しかった)。ただし MBP の
      torch 2.12.1 では、grad が有効で入力が requires_grad のとき MPS の SDPA は _math_for_mps ではなく合成の math 実装
      (層ごとに aten.bmm ×2 と aten._safe_softmax)に分解された。各 ref ジョブで小さな入力について dispatch された行列積・softmax・
      SDPA・dropout 系の aten 演算を数え、refs.*.dispatch に記録する
  tiny では同じ式を tiny の config(H, L, FFN)で計算する(要素数が小さいので全条件が「正しい」の予測になる)。

基準(metamorphic relation: dropout 0 の Transformer はサンプル間の相互作用がないので、loss の和の勾配は chunk ごとの勾配の和):
同じ impl・同じ device で先頭 max(B) 件を --chunk(256)件以下ずつ forward + backward し、勾配を CPU の float64 で累積する。
要求された全ての B(と --cpu-check-n)が chunk の切れ目になるよう、区間 [0,B1), [B1,B2), … をそれぞれ chunk 以下に分ける。
各 B で累積した勾配を float32 で保存し((impl, chunk, B) ごとに 1 ファイル)、テキストごとの logits と loss も保存する。
基準の chunk が例外を出したら全体を止める(基準が正しいことが前提)。--chunk2(128)で同じことを行い、2 つの基準の差を丸め誤差の
目安(noise floor)とする。
CPU 確認(--cpu-check-n、既定 256、0 で省略): CPU・eager で先頭 n 件を 1 回で forward + backward し、2 つの基準と比べる。
対照として device・eager・B=n の一括の条件を自動で加え、CPU とも比べる。

条件(full): --impls × --batch-sizes の各組で先頭 B 件を 1 回の forward + backward。status(ok / raised_forward /
raised_backward / crashed / timeout)、例外の型とメッセージ、anomaly のトレース、forward と backward の所要時間、
device のメモリ(torch.mps.current_allocated_memory / driver_allocated_memory を数か所で)、logits、テキストごとの loss、
loss の合計を記録する。forward と backward を torch.autograd.set_detect_anomaly(True, check_nan=False) の中で行う
(forward の traceback は forward の時点で anomaly mode が有効なときだけ記録されるので forward も含める。check_nan=False なので
NaN は例外にならない)。backward の例外で PyTorch が出す警告 "Error detected in <Node>. Traceback of forward call that caused
the error: ..." を warnings.catch_warnings(record=True) で受け取り(backward を別スレッドで回す device でも、警告は呼び出し側の
スレッドで再発行される)、backward の node 名、torch 以外の最後のフレーム(forward の演算の場所)、切り詰めた本文を保存する
(--no-anomaly で無効)。

指標(各 full 条件。基準 = 同じ impl の --chunk の累積):
  forward: テキストごとの rel = max_i |Δlogit_i| / max_i |logit_ref,i|(demo_workload.text_rel)、rel > BAD_REL の件数と区間、
      予測の変化、非有限、loss の合計の相対差
  勾配(backward が完了したとき): 全パラメータを連結した相対 L2 誤差 ‖g−g_ref‖/‖g_ref‖ と cosine、テンソルごとの相対 L2、
      グループ(embeddings、layer 0…、classifier)ごとの相対 L2 と最大、非有限の数
  許容誤差: 同じ (impl, B) の 2 つの基準の差(chunk2 vs chunk)を noise とし、全体の相対 L2 > τ = max(10·noise, 1e-6) なら勾配を
      「誤り」とする。テンソルごとの相対 L2 > max(10·noise_t, 1e-5) の数も数える。ただし attention の key の bias は厳密な演算では
      勾配が 0(softmax が各 query の定数シフト q·b_k に不変)で、計算値も基準も丸め誤差だけなので相対誤差は意味を持たない。
      これらは絶対誤差 ‖g_t−g_ref,t‖ > τ·‖g_ref‖(全体の許容誤差の絶対値)で判定する。生の値も併記する
  予測: 上の仮説から(eager)。sdpa は unknown

構成: 親プロセスはトークン化(1 回)と out/work/inputs.pt への保存、子プロセスの起動と集計だけを行い、MPS に触れない。
重いジョブ(ref、cpu、full)は、このスクリプト自身を隠しオプション --child で起動した別の子プロセスで順に動かす。例外・Metal の
assertion による abort・メモリ不足が他の条件を巻き込まず、各条件は空の device メモリから始まる。子が結果を書かずに 0 以外で
終了したら "crashed"(終了コードと stderr の末尾)、--timeout-min(180 分)を超えたら "timeout" とする。

出力(--out、既定 results/train_step): train_step.json(環境、モデル・データの revision、全引数、仮説、B ごとの要素数、基準、
CPU 確認、結果)、train_step.md(要約表)、train_step_tensors.csv(テンソルごとの勾配誤差)、train_step_logits.npz(基準・CPU・
各 full 条件の logits とテキストごとの loss、ラベル)。途中のファイルは out/work/ に置き、
--keep-work がなければ最後に消す。

メモリ・ディスクの目安(RoBERTa-base、L=512、fp32): eager の attention 確率は B=2048 で層ごとに 2048·12·512²·4 B ≈ 25.8 GB。
checkpointing なしでは 12 層分を backward まで保持するので載らない。checkpointing ありでは層の入力(B=2048 で 1 層 3.2 GB、12 層で
約 39 GB)と、backward で再計算する 1 層分の S・P・dP など(数十 GB)が主になる。PYTORCH_MPS_HIGH_WATERMARK_RATIO で MPS の
確保の上限を変えられるが、このスクリプトは設定しない(設定されていれば値を info に記録する)。work/ の勾配のスナップショットは
1 つ約 0.5 GB(1.25 億パラメータ × 4 B)で、既定では 2 impl × 2 chunk × 6 境界 = 24 個(約 12 GB)。
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from demo_workload import BAD_REL, CONFIG, DATASET, MODEL, ROOT, SPLIT, compare, empty, hub_rev, sh, sync, text_rel

P31, P32 = 2**31, 2**32
TINY = dict(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, num_labels=3)
NO_DROPOUT = dict(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, classifier_dropout=0.0)
NOISE_MULT, TAU_MIN, TAU_T_MIN = 10.0, 1e-6, 1e-5
PROBE_OPS = re.compile(r"mm|linear|scaled_dot_product|softmax|dropout|bernoulli")
# 厳密な演算では勾配が 0 のテンソル: attention の key の bias(softmax は各 query の定数シフト q·b_k に不変)。計算値は丸め誤差だけ
EXACT_ZERO = re.compile(r"attention\.self\.key\.bias$")
ANOMALY = re.compile(r"Error detected in |Previous calculation was induced by ")
FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)\n[ \t]*([^\n]*)')


# ---------------------------------------------------------------- 共通


def env_info() -> dict:
    """demo_workload.env_info と同じ項目。datasets の無い環境(--tiny の動作確認)でも動く。"""
    import transformers
    try:
        import datasets
        dv = datasets.__version__
    except ImportError:
        dv = None
    return {
        "macos": sh("sw_vers", "-productVersion"), "macos_build": sh("sw_vers", "-buildVersion"),
        "hw_model": sh("sysctl", "-n", "hw.model"), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "mem_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__, "datasets": dv,
        "git_rev": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
    }


def tiny_config(L: int):
    from transformers import RobertaConfig
    return RobertaConfig(**TINY, max_position_embeddings=L + 2, **NO_DROPOUT)


def counts(B: int, H: int, L: int, Fd: int) -> dict:
    s, f = B * H * L * L, B * L * Fd
    return {"scores_elems": s, "scores_over_2p31": s / P31, "scores_over_2p32": s / P32,
            "ffn_elems": f, "ffn_over_2p31": f / P31, "ffn_over_2p32": f / P32}


def hypotheses(H: int, L: int, Fd: int, nl: int) -> dict:
    b31, b32 = -(-P31 // (H * L * L)), P32 // (H * L * L) + 1  # scores >= 2**31 / > 2**32 になる最小の B
    f31 = -(-P31 // (L * Fd))
    t0, h0 = divmod(P32 // (L * L), H)
    return {
        "assumptions": "predictions from the bmm rules found on PyTorch 2.14.0; they assume PyTorch's bmm backward formulas "
                       "grad_self = grad @ mat2^T, grad_mat2 = self^T @ grad (transposed views) and that matmul's batch "
                       "folding copies Q/K/V into contiguous 3-D tensors",
        "eager_below_2p31": f"B*H*L^2 < 2^31 (B <= {b31 - 1}): forward and gradients correct",
        "eager_2p31_to_2p32": f"2^31 <= B*H*L^2 <= 2^32 ({b31} <= B <= {b32 - 1}): forward correct; backward raises "
                              "(rule 2 on P^T, a view with >= 2^31 elements, in the gradient of V for softmax(S)V)",
        "eager_above_2p32": f"B*H*L^2 > 2^32 (B >= {b32}): forward wrong for texts from {t0} (head {h0}) on (rule 3 on P); "
                            "backward raises (rule 2); grad_P = dO V^T would fall under rule 1 if reached",
        "eager_ffn": f"FFN intermediate (B*L, {Fd}) >= 2^31 from B = {f31}: the FFN's 2-D matmuls (not covered by the bmm "
                     f"rules) have views >= 2^31 in backward and may raise or abort first; autograd reaches layer {nl - 1}'s "
                     "FFN before its attention, so the anomaly trace is needed to tell which op failed",
        "sdpa": "unknown (inference: dispatched to aten::_scaled_dot_product_attention_math_for_mps, forward correct). "
                "On torch 2.12.1 (MBP), with grad enabled and inputs requiring grad, MPS SDPA decomposed into aten.bmm x2 + "
                "aten._safe_softmax per layer instead; see refs.*.dispatch for the version actually run",
        "b_scores_ge_2p31": b31, "b_scores_gt_2p32": b32, "b_ffn_ge_2p31": f31, "boundary_text": t0, "boundary_head": h0,
    }


def predict(impl: str, B: int, H: int, L: int, Fd: int) -> dict:
    if impl != "eager":
        return {"forward": "unknown", "status": "unknown", "text": "unknown"}
    s, t0 = B * H * L * L, P32 // (L * L) // H
    if s < P31:
        p = {"forward": "correct", "status": "ok", "text": "fwd correct, grads correct"}
    elif s <= P32:
        p = {"forward": "correct", "status": "raised_backward", "text": "fwd correct, bwd raises (rule 2)"}
    else:
        p = {"forward": "wrong", "status": "raised_backward",
             "text": f"fwd wrong from text {t0} (rule 3), bwd raises (rule 2)"}
    if B * L * Fd >= P31:
        p["text"] += "; FFN bwd may raise/abort first"
    return p


def snap_path(work, impl: str, chunk: int, B: int) -> Path:
    return Path(work) / f"ref_{impl}_c{chunk}_B{B}.pt"


def outs_path(work, impl: str, chunk: int) -> Path:
    return Path(work) / f"ref_{impl}_c{chunk}_outputs.pt"


def group_of(name: str) -> str:
    m = re.search(r"encoder\.layer\.(\d+)\.", name)
    if m:
        return f"layer{int(m.group(1))}"
    return "embeddings" if ".embeddings." in name else ("classifier" if name.startswith("classifier.") else "other")


def rel_of(d2: float, r2: float) -> float:
    if not math.isfinite(d2):
        return math.inf
    if r2 == 0:
        return 0.0 if d2 == 0 else math.inf
    return math.sqrt(d2 / r2)


def gdiff(g: dict, r: dict) -> dict:
    """g と基準 r の差 ‖g−r‖/‖r‖(float64)をテンソルごと・グループごと・全体で。全体の cosine と非有限の数も。非有限は rel = inf。"""
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    per = {}
    for n, rt in r.items():
        a, b = g[n].double(), rt.double()
        v = [float((a - b).square().sum()), float(b.square().sum()), float(a.square().sum()), float((a * b).sum()),
             int((~torch.isfinite(a)).sum())]
        per[n] = {"numel": a.numel(), "nonfinite": v[4], "ref_norm": math.sqrt(v[1]), "rel_l2": rel_of(v[0], v[1]),
                  "abs_l2": math.sqrt(v[0]) if math.isfinite(v[0]) else math.inf}
        for k in ("all", group_of(n)):
            acc[k] = [x + y for x, y in zip(acc[k], v)]
    d2, r2, g2, ab, nf = acc.pop("all")
    ok = math.isfinite(ab) and math.isfinite(g2) and g2 > 0 and r2 > 0
    return {"global": {"rel_l2": rel_of(d2, r2), "cos": ab / math.sqrt(g2 * r2) if ok else math.nan, "nonfinite": nf,
                       "norm": math.sqrt(g2) if math.isfinite(g2) else math.inf, "ref_norm": math.sqrt(r2)},
            "groups": {k: {"rel_l2": rel_of(v[0], v[1]), "nonfinite": v[4]} for k, v in acc.items()}, "per": per}


def judge(d: dict, nz: dict) -> tuple[dict, list[dict]]:
    """noise(2 つの基準の差)から許容誤差を決めて勾配を判定する。テンソルごとの行(CSV 用)も返す。
    EXACT_ZERO のテンソルは基準自体が丸め誤差なので相対誤差では判定せず、‖g_t−r_t‖ > τ·‖r‖(全体の許容誤差の絶対値)で判定する。"""
    tau = max(NOISE_MULT * nz["global"]["rel_l2"], TAU_MIN)
    rows = []
    for n, t in d["per"].items():
        if EXACT_ZERO.search(n):
            rule, val, thr = "abs (exact-zero grad)", t["abs_l2"], tau * d["global"]["ref_norm"]
        else:
            rule, val, thr = "rel", t["rel_l2"], max(NOISE_MULT * nz["per"][n]["rel_l2"], TAU_T_MIN)
        rows.append({"tensor": n, "group": group_of(n), "numel": t["numel"], "rel_l2": t["rel_l2"],
                     "noise_rel_l2": nz["per"][n]["rel_l2"], "abs_l2": t["abs_l2"], "rule": rule, "threshold": thr,
                     "wrong": not val <= thr, "nonfinite": t["nonfinite"], "ref_norm": t["ref_norm"]})
    groups = {}
    for k, v in d["groups"].items():
        rs = [x for x in rows if x["group"] == k]
        groups[k] = {"rel_l2": v["rel_l2"], "noise_rel_l2": nz["groups"][k]["rel_l2"], "nonfinite": v["nonfinite"],
                     "max_tensor_rel_l2": max((x["rel_l2"] for x in rs if x["rule"] == "rel"), default=None),
                     "wrong_tensors": sum(x["wrong"] for x in rs)}
    g = d["global"]
    wrong = [x["tensor"] for x in rows if x["wrong"]]
    return {"rel_l2": g["rel_l2"], "cos": g["cos"], "nonfinite": g["nonfinite"], "norm": g["norm"],
            "ref_norm": g["ref_norm"], "noise_rel_l2": nz["global"]["rel_l2"], "tau": tau,
            "class": "correct" if g["rel_l2"] <= tau else "wrong", "n_tensors": len(rows),
            "n_wrong_tensors": len(wrong), "wrong_tensors": wrong[:30],
            "worst_group": max(groups, key=lambda k: groups[k]["rel_l2"]), "groups": groups}, rows


def vs_other(lg: torch.Tensor, g: dict | None, lg_o: torch.Tensor, g_o: dict) -> dict:
    """別の計算(CPU など)との差。g_o を基準にした勾配の相対 L2 と cosine、logits のテキストごとの rel。"""
    rel = text_rel(lg.numpy(), lg_o.numpy())
    d = gdiff(g, g_o)["global"] if g is not None else None
    return {"n": len(rel), "logits_max_rel": float(rel.max()), "logits_median_rel": float(np.median(rel)),
            "pred_changed": int((lg.argmax(1) != lg_o.argmax(1)).sum()),
            "grad_rel_l2": d["rel_l2"] if d else None, "grad_cos": d["cos"] if d else None}


# ---------------------------------------------------------------- 子プロセス: モデルと学習ステップ


class DebugRaise(torch.autograd.Function):
    """試験用(--debug-raise-backward): forward は恒等、backward で例外を出す。"""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("debug: raised inside backward (--debug-raise-backward)")


class OpCount(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.c = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if PROBE_OPS.search(str(func)):
            self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))


def mem(dev: str) -> dict | None:
    if dev != "mps":
        return None
    return {"current": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}


def set_ckpt(model, on: bool) -> None:
    if on:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    assert all(bool(x.gradient_checkpointing) == on for x in model.roberta.encoder.layer)


def build(job: dict, impl: str, dev: str):
    """dropout 0・train mode・(既定で)gradient checkpointing・全パラメータ requires_grad のモデル。"""
    from transformers import AutoModelForSequenceClassification
    if job["tiny"]:
        m = AutoModelForSequenceClassification.from_config(tiny_config(job["L"]), attn_implementation=impl)
        m.load_state_dict(torch.load(Path(job["work"]) / "tiny_weights.pt", weights_only=True))
    else:
        m = AutoModelForSequenceClassification.from_pretrained(MODEL, attn_implementation=impl, **NO_DROPOUT)
    drops = [x for x in m.modules() if isinstance(x, torch.nn.Dropout)]
    for x in drops:
        x.p = 0.0
    m = m.float().train().to(dev)
    set_ckpt(m, job["checkpointing"])
    for p in m.parameters():
        p.requires_grad_(True)
    assert m.config._attn_implementation == impl
    return m, {"attn_implementation": m.config._attn_implementation, "dropout_modules_set_to_0": len(drops),
               "config_dropouts": {k: getattr(m.config, k) for k in NO_DROPOUT}, "training": m.training,
               "checkpointed_layers": sum(bool(x.gradient_checkpointing) for x in m.roberta.encoder.layer),
               "n_params": sum(p.numel() for p in m.parameters())}


def loss_of(lg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(lg, y, reduction="sum")


def grads_cpu(model) -> dict[str, torch.Tensor]:
    return {n: (p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().cpu()
            for n, p in model.named_parameters()}


def step(model, ids, mask, y, dev: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """1 回の forward + backward(loss は sum)。logits、テキストごとの loss、勾配を CPU の float32 で返す。"""
    model.zero_grad(set_to_none=True)
    y = y.to(dev)
    lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
    loss_of(lg, y).backward()
    per = F.cross_entropy(lg.detach(), y, reduction="none")
    return lg.detach().float().cpu(), per.float().cpu(), grads_cpu(model)


def self_check(model, ids, mask, y, dev: str, ckpt: bool) -> dict:
    """数件で (1) train(dropout 0) と eval の logits、(2) checkpointing の有無で logits と勾配が一致するかを確かめる。"""
    with torch.no_grad():
        model.eval()
        le = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float().cpu()
        model.train()
        lt = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float().cpu()
    la, _, ga = step(model, ids, mask, y, dev)
    set_ckpt(model, not ckpt)
    lb, _, gb = step(model, ids, mask, y, dev)
    set_ckpt(model, ckpt)
    model.zero_grad(set_to_none=True)
    return {"n": int(ids.shape[0]), "train_eq_eval_bitwise": bool(torch.equal(le, lt)),
            "train_vs_eval_max_abs": float((le - lt).abs().max()),
            "ckpt_toggle_logits_bitwise": bool(torch.equal(la, lb)),
            "ckpt_toggle_grad_rel_l2": gdiff(gb, ga)["global"]["rel_l2"],
            "ckpt_toggle_grad_differing_tensors": [n for n in ga if not torch.equal(ga[n], gb[n])][:20]}


def dispatch_probe(model, ids, mask, y, dev: str) -> dict:
    """数件の forward / backward で dispatch された行列積・softmax・SDPA・dropout 系の aten 演算を数える。"""
    try:
        rec = OpCount()
        model.zero_grad(set_to_none=True)
        with rec:
            lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            fwd = dict(rec.c)
            rec.c.clear()
            loss_of(lg, y.to(dev)).backward()
        model.zero_grad(set_to_none=True)
        return {"forward": fwd, "backward": dict(rec.c)}
    except Exception as e:  # 記録は補助
        return {"error": repr(e)[:500]}


def load_inputs(job: dict, n: int):
    d = torch.load(Path(job["work"]) / "inputs.pt", weights_only=True)
    return d["ids"][:n], d["mask"][:n], d["labels"][:n]


def exc_info(e: BaseException) -> dict:
    s = str(e)
    return {"exception_type": type(e).__name__, "exception_msg": s[:4000], "oom": "out of memory" in s.lower()}


def trim_msg(m: str, keep: int = 30) -> str:
    ls = m.splitlines()
    return "\n".join(ls if len(ls) <= keep + 1 else ls[:1] + ["  ..."] + ls[-keep:])


def parse_anomaly(msgs: list[str]) -> dict | None:
    """anomaly mode の警告から、失敗した backward の node と forward の演算の場所(torch 以外の最後のフレーム)を取り出す。"""
    if not msgs:
        return None
    m = re.search(r"Error detected in (\S+?)\.\s", msgs[0])
    fr = [(f, f"{Path(f).name}:{ln} in {fn.strip()}: {code.strip()}") for f, ln, fn, code in FRAME.findall(msgs[0])
          if f"{os.sep}torch{os.sep}" not in f]
    model_fr = [s for f, s in fr if f"{os.sep}transformers{os.sep}" in f]
    return {"backward_node": m.group(1) if m else None, "forward_location": fr[-1][1] if fr else None,
            "model_location": model_fr[-1] if model_fr else None, "forward_frames": [s for _, s in fr[-8:]],
            "n_messages": len(msgs), "text": "\n\n".join(trim_msg(x) for x in msgs)[-6000:]}


# ---------------------------------------------------------------- 子プロセス: ジョブ


def job_ref(job: dict) -> dict:
    """分割実行の基準: 先頭 max(B) 件を chunk 以下ずつ forward + backward、勾配を CPU の float64 で累積し、各 B で保存。"""
    impl, c, dev, work = job["impl"], job["chunk_size"], job["device"], job["work"]
    tag = f"[ref {impl} c{c}]"
    ids, mask, y = load_inputs(job, job["bounds"][-1])
    model, minfo = build(job, impl, dev)
    chk = self_check(model, ids[:2], mask[:2], y[:2], dev, job["checkpointing"])
    probe = dispatch_probe(model, ids[:2], mask[:2], y[:2], dev)
    print(f"{tag} self-check {chk}\n{tag} dispatch {probe}", flush=True)
    acc = {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in model.named_parameters()}
    lgs, losses, pieces, s, t0 = [], [], 0, 0, time.perf_counter()
    for B in job["bounds"]:
        while s < B:
            e = min(s + c, B)
            lg, per, g = step(model, ids[s:e], mask[s:e], y[s:e], dev)
            for n, t in g.items():
                acc[n] += t.double()
            lgs.append(lg)
            losses.append(per)
            s, pieces = e, pieces + 1
        torch.save({n: t.float() for n, t in acc.items()}, snap_path(work, impl, c, B))
        print(f"{tag} B={B}: {pieces} pieces, {time.perf_counter() - t0:.1f}s", flush=True)
    lg, per = torch.cat(lgs), torch.cat(losses)
    torch.save({"logits": lg, "loss": per}, outs_path(work, impl, c))
    return {"status": "ok", "impl": impl, "chunk": c, "bounds": job["bounds"], "pieces": pieces,
            "wall_s": time.perf_counter() - t0, "loss_sum": float(per.double().sum()),
            "acc": float((lg.argmax(1) == y).float().mean()), "nonfinite_logits": int((~torch.isfinite(lg)).sum()),
            "model": minfo, "self_check": chk, "dispatch": probe, "mem_end": mem(dev),
            "recommended_max_memory": torch.mps.recommended_max_memory() if dev == "mps" else None}


def job_cpu(job: dict) -> dict:
    """CPU・eager で先頭 n 件を 1 回で forward + backward し、2 つの基準と比べる(結果は対照の full ジョブも使う)。"""
    n, work = job["n"], job["work"]
    ids, mask, y = load_inputs(job, n)
    model, minfo = build(job, "eager", "cpu")
    t0 = time.perf_counter()
    lg, per, g = step(model, ids, mask, y, "cpu")
    dt = time.perf_counter() - t0
    torch.save({"logits": lg, "loss": per, "grads": g}, Path(work) / "cpu.pt")
    res = {"status": "ok", "n": n, "wall_s": dt, "threads": torch.get_num_threads(), "model": minfo,
           "loss_sum": float(per.double().sum()), "refs_vs_cpu": {}}
    for c in (job["chunk"], job["chunk2"]):
        o = torch.load(outs_path(work, "eager", c), weights_only=True)
        res["refs_vs_cpu"][f"eager_c{c}"] = vs_other(o["logits"][:n], torch.load(snap_path(work, "eager", c, n),
                                                                                  weights_only=True), lg, g)
    print(f"[cpu n={n}] {dt:.1f}s refs vs cpu: {res['refs_vs_cpu']}", flush=True)
    return res


def job_full(job: dict) -> dict:
    """B 件を 1 回で forward + backward。例外・anomaly のトレース・時間・メモリを記録し、基準と比べる(生の勾配は保存しない)。"""
    impl, B, dev, work = job["impl"], job["B"], job["device"], job["work"]
    tag = f"[full {impl} B={B}]"
    ids, mask, y = load_inputs(job, B)
    model, minfo = build(job, impl, dev)
    r = {"status": None, "model": minfo, "mem": {"after_load": mem(dev)},
         "recommended_max_memory": torch.mps.recommended_max_memory() if dev == "mps" else None}
    if B in job["debug_raise_backward"]:
        model.roberta.encoder.layer[0].intermediate.register_forward_hook(lambda mod, inp, out: DebugRaise.apply(out))
        r["debug"] = "raise_backward (test only)"
    print(f"{tag} start", flush=True)
    box, lg, t0 = {}, None, time.perf_counter()
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        with torch.autograd.set_detect_anomaly(job["anomaly"], check_nan=False):
            try:
                sync(dev)
                t0 = time.perf_counter()
                yd = y.to(dev)
                box["out"] = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
                box["loss"] = loss_of(box["out"], yd)
                per = F.cross_entropy(box["out"].detach(), yd, reduction="none").float().cpu()
                lg = box["out"].detach().float().cpu()
                r["loss_sum_device"] = float(box["loss"].detach())
                sync(dev)
                r["forward_s"] = time.perf_counter() - t0
            except Exception as e:
                r.update(status="raised_forward", forward_s_until_error=time.perf_counter() - t0, **exc_info(e))
            r["mem"]["after_forward"] = mem(dev)
            print(f"{tag} forward {r['status'] or 'ok'} {r.get('forward_s', r.get('forward_s_until_error', 0)):.1f}s",
                  flush=True)
            if lg is not None:
                if B in job["debug_crash"]:
                    print(f"{tag} debug: os.abort() before backward (--debug-crash, test only)", file=sys.stderr, flush=True)
                    os.abort()
                try:
                    t0 = time.perf_counter()
                    box["loss"].backward()
                    sync(dev)
                    r.update(status="ok", backward_s=time.perf_counter() - t0)
                except Exception as e:
                    r.update(status="raised_backward", backward_s_until_error=time.perf_counter() - t0, **exc_info(e))
                r["mem"]["after_backward"] = mem(dev)
    msgs = [str(w.message) for w in W]
    an = [m for m in msgs if ANOMALY.search(m)]
    r["anomaly"] = parse_anomaly(an)
    r["other_warnings"] = [m[:300] for m in msgs if not ANOMALY.search(m)][:10]
    if r["status"] != "ok":
        a = r["anomaly"] or {}
        print(f"{tag} {r['status']}: {r.get('exception_type')}: {r.get('exception_msg', '')[:300]}\n"
              f"{tag} backward node {a.get('backward_node')} / forward {a.get('forward_location')}", flush=True)
    g = grads_cpu(model) if r["status"] == "ok" else None
    del model, box
    gc.collect()
    empty(dev)
    r["mem"]["after_free"] = mem(dev)

    ref = torch.load(outs_path(work, impl, job["chunk"]), weights_only=True)
    if lg is not None:
        torch.save({"logits": lg, "loss": per}, Path(work) / f"full_{impl}_B{B}_outputs.pt")
        f = compare(lg.numpy(), ref["logits"][:B].numpy(), y.numpy())
        ls, lr = float(per.double().sum()), float(ref["loss"][:B].double().sum())
        f.update(loss_sum=ls, loss_sum_ref=lr, loss_rel_diff=abs(ls - lr) / abs(lr) if lr else math.inf,
                 nonfinite_logits=int((~torch.isfinite(lg)).sum()))
        f["class"] = "correct" if f["bad_texts"] == 0 and f["nonfinite_texts"] == 0 else "wrong"
        r["forward"] = f
    if g is not None:
        r1 = torch.load(snap_path(work, impl, job["chunk"], B), weights_only=True)
        r2 = torch.load(snap_path(work, impl, job["chunk2"], B), weights_only=True)
        r["grad"], r["tensors"] = judge(gdiff(g, r1), gdiff(r2, r1))
    if job["cpu_ref"] and lg is not None:
        cp = torch.load(Path(work) / "cpu.pt", weights_only=True)
        r["vs_cpu"] = vs_other(lg, g, cp["logits"], cp["grads"])
    fw, gr = r.get("forward") or {}, r.get("grad") or {}
    print(f"{tag} {r['status']}: fwd bad={fw.get('bad_texts')} runs={fw.get('bad_runs_text_index', [])[:4]} "
          f"loss_rel={fw.get('loss_rel_diff')} grad rel_l2={gr.get('rel_l2')} tau={gr.get('tau')} "
          f"class={gr.get('class')} wrong_tensors={gr.get('n_wrong_tensors')}", flush=True)
    return r


def child(path: Path) -> None:
    job = json.loads(path.read_text())
    torch.manual_seed(job["seed"])
    res = {"ref": job_ref, "cpu": job_cpu, "full": job_full}[job["kind"]](job)
    Path(job["result"]).write_text(json.dumps(res, ensure_ascii=False))


# ---------------------------------------------------------------- 親プロセス


def run_child(work: Path, job: dict, timeout_min: float) -> dict:
    """子プロセスで 1 つのジョブを動かす。結果を書かずに終われば crashed(終了コードと stderr の末尾)。"""
    jd = work / "jobs"
    jd.mkdir(exist_ok=True)
    jf, rf = jd / f"{job['name']}.json", jd / f"{job['name']}.result.json"
    rf.unlink(missing_ok=True)
    jf.write_text(json.dumps({**job, "result": str(rf)}))
    print(f"== {job['name']}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(jf)],
                         stderr=subprocess.PIPE, text=True, errors="replace")
    tail: collections.deque[str] = collections.deque(maxlen=80)

    def pump():
        for line in p.stderr:
            sys.stderr.write(line)
            tail.append(line)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    status = None
    try:
        rc = p.wait(timeout=timeout_min * 60)
    except subprocess.TimeoutExpired:
        p.kill()
        rc, status = p.wait(), "timeout"
    th.join(10)
    res = json.loads(rf.read_text()) if status is None and rf.exists() else {"status": status or "crashed"}
    res.update(child_exit_code=rc, child_wall_s=time.perf_counter() - t0)
    if rc != 0 or res["status"] in ("crashed", "timeout"):
        res["stderr_tail"] = "".join(tail)[-4000:]
    return res


def prepare(a, N: int, work: Path):
    """入力を 1 回だけ作って work/inputs.pt に保存する(MPS には触れない)。tiny では乱数初期化の重みも保存する。"""
    L = a.max_length
    if a.tiny:
        from transformers import AutoModelForSequenceClassification
        cfg = tiny_config(L)
        torch.manual_seed(a.seed)
        torch.save(AutoModelForSequenceClassification.from_config(cfg).state_dict(), work / "tiny_weights.pt")
        g = torch.Generator().manual_seed(a.seed)
        lens = torch.randint(2, L + 1, (N,), generator=g)
        ids = torch.randint(3, cfg.vocab_size, (N, L), generator=g)
        mask = (torch.arange(L)[None] < lens[:, None]).long()
        ids[:, 0] = cfg.bos_token_id
        ids[torch.arange(N), lens - 1] = cfg.eos_token_id
        ids[mask == 0] = cfg.pad_token_id
        labels = torch.randint(0, cfg.num_labels, (N,), generator=g)
        meta = {"model": "tiny RobertaForSequenceClassification (random init)", "model_rev": None,
                "tiny_config": {**TINY, "max_position_embeddings": L + 2}, "id2label": cfg.id2label,
                "data": f"合成(乱数の token id、実長 2..{L} の乱数で padding、乱数のラベル)、{N} 件、max_length={L}"}
    else:
        import datasets
        from transformers import AutoConfig, AutoTokenizer
        ds = datasets.load_dataset(DATASET, CONFIG, split=SPLIT)
        n_split = len(ds)
        ds = ds.select(range(N))
        labels = torch.as_tensor(np.array(ds["label"]))
        texts = list(ds["text"])
        tok = AutoTokenizer.from_pretrained(MODEL)
        enc = tok(texts, padding="max_length", truncation=True, max_length=L, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        n_trunc = int(sum(len(x) > L for x in tok(texts)["input_ids"]))
        cfg = AutoConfig.from_pretrained(MODEL)
        meta = {"model": MODEL, "model_rev": getattr(cfg, "_commit_hash", None) or hub_rev("models", MODEL),
                "tokenizer": MODEL, "tokenizer_rev": hub_rev("models", MODEL), "dataset": DATASET,
                "dataset_config": CONFIG, "split": SPLIT, "split_size": n_split,
                "dataset_rev": hub_rev("datasets", DATASET), "text_order": "dataset order, first N",
                "tokenization": "padding=max_length, truncation=True", "n_truncated": n_trunc, "id2label": cfg.id2label,
                "data": f"`{DATASET}` / `{CONFIG}` / `{SPLIT}`({n_split} 件、revision `{hub_rev('datasets', DATASET)}`)"
                        f"の先頭 {N} 件(データセットの順)。padding=max_length, truncation=True, max_length={L}"
                        f"(切り詰め {n_trunc} 件)"}
    torch.save({"ids": ids, "mask": mask, "labels": labels}, work / "inputs.pt")
    meta.update(mean_real_tokens=float(mask.sum(1).float().mean()),
                label_counts=np.bincount(labels.numpy(), minlength=cfg.num_labels).tolist())
    return cfg, meta


def ref_noise(work: Path, impls: list[str], c1: int, c2: int) -> dict:
    """基準どうしの logits の差(chunk2 vs chunk、impl 間)。親は CPU のテンソルだけを読む。"""
    o = {(i, c): torch.load(outs_path(work, i, c), weights_only=True) for i in impls for c in (c1, c2)}
    res = {}
    for i in impls:
        a, b = o[(i, c1)], o[(i, c2)]
        la, lb = float(a["loss"].double().sum()), float(b["loss"].double().sum())
        res[f"{i}_c{c2}_vs_c{c1}"] = {"logits_max_rel": float(text_rel(b["logits"].numpy(), a["logits"].numpy()).max()),
                                      "loss_sum_rel_diff": abs(lb - la) / abs(la) if la else math.inf}
    if len(impls) > 1:
        a, b = o[(impls[0], c1)], o[(impls[1], c1)]
        res[f"{impls[1]}_vs_{impls[0]}_c{c1}"] = {
            "logits_max_rel": float(text_rel(b["logits"].numpy(), a["logits"].numpy()).max())}
    return res


def save(out: Path, info: dict, results: list[dict]) -> None:
    work = out / "work"
    arrs = {"labels": torch.load(work / "inputs.pt", weights_only=True)["labels"].numpy()}
    for f in sorted(work.glob("*_outputs.pt")) + sorted(work.glob("cpu.pt")):
        d = torch.load(f, weights_only=True)
        k = f.stem.removesuffix("_outputs")
        arrs[f"{k}_logits"], arrs[f"{k}_loss"] = d["logits"].numpy(), d["loss"].numpy()
    np.savez_compressed(out / "train_step_logits.npz", **arrs)
    rows = [{"impl": r["impl"], "B": r["B"], "control": r["control"], **t} for r in results for t in r.pop("tensors", [])]
    if rows:
        with open(out / "train_step_tensors.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    info["results"] = results
    (out / "train_step.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    (out / "train_step.md").write_text(render_md(info))


def parent(a) -> None:
    out = Path(a.out).resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    assert a.chunk > 0 and a.chunk2 > 0 and a.chunk != a.chunk2, "--chunk と --chunk2 は別の正の値にする"
    Bs = sorted(set(a.batch_sizes))
    n_cpu = a.cpu_check_n if "eager" in a.impls else 0
    bounds = sorted(set(Bs) | ({n_cpu} if n_cpu else set()))
    N, L = bounds[-1], a.max_length
    cfg, meta = prepare(a, N, work)
    H, Fd, nl = cfg.num_attention_heads, cfg.intermediate_size, cfg.num_hidden_layers
    info = {
        "env": env_info(), "device": a.device, "tiny": a.tiny, "args": {k: v for k, v in vars(a).items() if k != "child"},
        **meta, "n_texts": N, "max_length": L, "num_heads": H, "num_layers": nl, "ffn_size": Fd, "dtype": "fp32",
        "seed": a.seed, "chunk": a.chunk, "chunk2": a.chunk2, "bounds": bounds, "bad_rel_threshold": BAD_REL,
        "train_step": "loss = cross_entropy(logits, dataset labels, reduction='sum'); all params requires_grad; "
                      "no optimizer step; model.train() with every nn.Dropout p=0 and config dropouts 0",
        "checkpointing": None if a.no_checkpointing else "transformers gradient_checkpointing_enable(use_reentrant=False)",
        "anomaly": None if a.no_anomaly else "forward+backward under set_detect_anomaly(True, check_nan=False)",
        "tolerance": f"grad wrong if global rel L2 > tau = max({NOISE_MULT:g} * noise, {TAU_MIN:g}); tensor wrong if rel L2 > "
                     f"max({NOISE_MULT:g} * noise_t, {TAU_T_MIN:g}), except attention key biases (gradient exactly 0 in "
                     "exact arithmetic, softmax shift invariance): wrong if abs L2 error > tau * ||g_ref|| (global); "
                     f"noise = ref c{a.chunk2} vs ref c{a.chunk}",
        "reference": f"same impl and device, chunks <= {a.chunk} (every B a chunk boundary), grads accumulated in float64 "
                     "on CPU, snapshot at each B",
        "hypotheses": hypotheses(H, L, Fd, nl),
        "element_counts": {str(B): counts(B, H, L, Fd) for B in bounds},
        "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
    }
    print(json.dumps({k: v for k, v in info.items() if k not in ("env", "element_counts")}, ensure_ascii=False),
          flush=True)
    base = {"tiny": a.tiny, "L": L, "work": str(work), "device": a.device, "checkpointing": not a.no_checkpointing,
            "seed": a.seed, "chunk": a.chunk, "chunk2": a.chunk2}

    info["refs"], results = {}, []
    for impl in a.impls:
        for c in (a.chunk, a.chunk2):
            name = f"ref_{impl}_c{c}"
            r = run_child(work, {**base, "kind": "ref", "name": name, "impl": impl, "chunk_size": c, "bounds": bounds},
                          a.timeout_min)
            info["refs"][f"{impl}_c{c}"] = r
            if r["status"] != "ok":
                save(out, info, results)
                raise SystemExit(f"{name} failed ({r['status']}, exit {r.get('child_exit_code')}); the reference must "
                                 f"be valid. work files kept in {work}\n{r.get('stderr_tail', '')}")
    info["ref_noise"] = ref_noise(work, a.impls, a.chunk, a.chunk2)
    print(f"ref noise: {info['ref_noise']}", flush=True)

    info["cpu_check"] = None
    if n_cpu:
        info["cpu_check"] = run_child(work, {**base, "kind": "cpu", "name": f"cpu_n{n_cpu}", "n": n_cpu}, a.timeout_min)
    has_cpu = bool(info["cpu_check"] and info["cpu_check"]["status"] == "ok")

    for impl in a.impls:
        for B in (bounds if impl == "eager" and n_cpu else Bs):
            job = {**base, "kind": "full", "name": f"full_{impl}_B{B}", "impl": impl, "B": B,
                   "anomaly": not a.no_anomaly, "cpu_ref": has_cpu and impl == "eager" and B == n_cpu,
                   "debug_raise_backward": a.debug_raise_backward, "debug_crash": a.debug_crash}
            r = run_child(work, job, a.timeout_min)
            r = {"impl": impl, "B": B, "control": B not in Bs, **counts(B, H, L, Fd),
                 "predicted": predict(impl, B, H, L, Fd), **r}
            results.append(r)
            print(f"== full {impl} B={B}: {r['status']} (predicted: {r['predicted']['text']})", flush=True)

    info["total_wall_s"] = time.perf_counter() - t_all
    info["work_bytes"] = sum(f.stat().st_size for f in work.rglob("*") if f.is_file())
    info["work_kept"] = str(work) if a.keep_work else None
    save(out, info, results)
    if not a.keep_work:
        shutil.rmtree(work)
    print(f"done {info['total_wall_s']:.0f}s -> {out}", flush=True)


# ---------------------------------------------------------------- 要約(markdown)


def f3(x, spec: str = ".3g") -> str:
    return "—" if x is None else format(x, spec)


def short_ops(d: dict | None) -> str:
    if not d:
        return "—"
    if "error" in d:
        return d["error"][:80]
    keep = {k: v for k, v in d.get("forward", {}).items() if not re.search(r"linear|addmm|aten\.mm\.", k)}
    return ", ".join(f"{k.replace('aten.', '').replace('.default', '')}×{v}" for k, v in keep.items()) or "—"


def where(r: dict) -> str:
    if r["status"] in ("crashed", "timeout"):
        last = [x for x in (r.get("stderr_tail") or "").splitlines() if x.strip()][-1:] or [""]
        return f"exit {r.get('child_exit_code')}: `{last[0][:120]}`"
    a = r.get("anomaly")
    if r["status"] == "raised_backward":
        return f"{a['backward_node']} @ `{a['forward_location']}`" if a else "— (anomaly なし)"
    if r["status"] == "raised_forward":
        return f"forward: {r.get('exception_type')}"
    return "—"


def render_md(I: dict) -> str:
    e = I["env"]
    L = ["# 学習 1 ステップの一括実行 vs 分割累積: RoBERTa 感情分類(forward + backward)", "",
         "`demo_train_step.py` の出力。数値は `train_step.json`、テンソルごとの勾配誤差は `train_step_tensors.csv`、"
         "logits とテキストごとの loss は `train_step_logits.npz`。", "",
         "## 環境と条件", "",
         f"- macOS {e['macos']} ({e['macos_build']})、{e['hw_model']} ({e['chip']}, {e['mem_bytes'] / 2**30:.0f} GiB)、"
         f"device {I['device']}",
         f"- Python {e['python']}、torch {e['torch']}、transformers {e['transformers']}、datasets {e['datasets']}",
         f"- モデル: `{I['model']}` (revision `{I['model_rev']}`)、{I['num_layers']} 層、{I['num_heads']} heads、"
         f"FFN {I['ffn_size']}、fp32" + (f"、tiny config {I['tiny_config']}" if I["tiny"] else ""),
         f"- データ: {I['data']}。実トークン数の平均 {I['mean_real_tokens']:.1f}、ラベル件数 {I['label_counts']}",
         f"- 学習ステップ: {I['train_step']}",
         f"- gradient checkpointing: {I['checkpointing']}。anomaly: {I['anomaly']}。seed {I['seed']}。"
         f"コード git `{(e['git_rev'] or '')[:12]}`",
         f"- 許容誤差: {I['tolerance']}。forward は rel > {I['bad_rel_threshold']:g} のテキストを「壊れた」とする",
         f"- PYTORCH_MPS_HIGH_WATERMARK_RATIO = {I['mps_high_watermark_ratio']}", "",
         "## 予測(bmm の規則から)", ""]
    L += [f"- {k}: {v}" for k, v in I["hypotheses"].items() if isinstance(v, str)]
    L += ["", "## 基準", "", f"{I['reference']}。", "",
          "| 基準 | 分割数 | 所要 [s] | loss の合計 | 正解率 | train=eval (ビット一致) | checkpointing 有無の勾配差 (rel L2) | "
          "forward で dispatch された演算(linear 以外) |", "|---|---|---|---|---|---|---|---|"]
    for k, r in I["refs"].items():
        if r["status"] != "ok":
            L.append(f"| {k} | {r['status']} | | | | | | |")
            continue
        sc = r["self_check"]
        L.append(f"| {k} | {r['pieces']} | {r['wall_s']:.1f} | {r['loss_sum']:.6g} | {r['acc']:.4f} | "
                 f"{sc['train_eq_eval_bitwise']} | {sc['ckpt_toggle_grad_rel_l2']:.2g} | {short_ops(r['dispatch'])} |")
    if I.get("ref_noise"):
        L += ["", "基準どうしの logits の差: " + "、".join(
            f"{k}: 最大 rel {v['logits_max_rel']:.2g}" + (f"、loss の合計の相対差 {v['loss_sum_rel_diff']:.2g}"
                                                        if "loss_sum_rel_diff" in v else "")
            for k, v in I["ref_noise"].items()) + "。"]
    c = I.get("cpu_check")
    if c:
        if c["status"] == "ok":
            L += ["", f"CPU 確認(eager、先頭 {c['n']} 件を 1 回、{c['wall_s']:.0f} 秒、{c['threads']} threads): " + "、".join(
                f"{k} vs CPU: logits 最大 rel {v['logits_max_rel']:.2g}、勾配 rel L2 {v['grad_rel_l2']:.2g}、"
                f"1 − cos {1 - v['grad_cos']:.2g}" for k, v in c["refs_vs_cpu"].items()) + "。"]
        else:
            L += ["", f"CPU 確認: {c['status']} (exit {c.get('child_exit_code')})。"]
    L += ["", "## 結果(B 件を 1 回で forward + backward)", "",
          "| impl | B | scores / 2^32 | FFN / 2^31 | 予測 | status | 失敗の場所 (backward node @ forward の演算) | "
          "壊れたテキスト (区間) | loss 相対差 | 勾配 rel L2 | 1 − cos | τ | 誤りテンソル | 最悪グループ (rel L2) | "
          "fwd / bwd [s] | driver メモリ fwd / bwd [GB] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in I["results"]:
        fw, g = r.get("forward"), r.get("grad")
        fws = "—"
        if fw:
            runs = ", ".join(f"{x}–{y}" if x != y else f"{x}" for x, y in fw["bad_runs_text_index"][:3])
            fws = f"{fw['bad_texts']}" + (f" ({runs})" if runs else "")
        gs = (f"{g['rel_l2']:.3g} | {1 - g['cos']:.2g} | {g['tau']:.2g} | {g['n_wrong_tensors']}/{g['n_tensors']} | "
              f"{g['worst_group']} ({g['groups'][g['worst_group']]['rel_l2']:.2g})") if g else "— | — | — | — | —"
        mm = r.get("mem") or {}
        drv = [(mm.get(k) or {}).get("driver") for k in ("after_forward", "after_backward")]
        mem_s = " / ".join(f3(None if x is None else x / 1e9, ".1f") for x in drv)
        L.append(f"| {r['impl']} | {r['B']}{' (対照)' if r['control'] else ''} | {r['scores_over_2p32']:.4f} | "
                 f"{r['ffn_over_2p31']:.4f} | {r['predicted']['text']} | {r['status']} | {where(r)} | {fws} | "
                 f"{f3(fw and fw['loss_rel_diff'], '.2g')} | {gs} | "
                 f"{f3(r.get('forward_s'), '.1f')} / {f3(r.get('backward_s'), '.1f')} | {mem_s} |")
    ctrl = [r for r in I["results"] if r.get("vs_cpu")]
    for r in ctrl:
        v = r["vs_cpu"]
        L += ["", f"{r['impl']} B={r['B']} の一括 vs CPU: logits 最大 rel {v['logits_max_rel']:.2g}、予測の変化 "
                  f"{v['pred_changed']}、勾配 rel L2 {f3(v['grad_rel_l2'], '.2g')}、"
                  f"1 − cos {f3(None if v['grad_cos'] is None else 1 - v['grad_cos'], '.2g')}。"]
    bad = [r for r in I["results"] if r["status"] != "ok"]
    if bad:
        L += ["", "## 例外・異常終了の詳細", ""]
        for r in bad:
            L += [f"### {r['impl']} B={r['B']}: {r['status']}", ""]
            if r.get("exception_type"):
                L += [f"- 例外: `{r['exception_type']}`: " + r["exception_msg"][:600].replace("\n", " ").replace("|", "\\|")]
            a = r.get("anomaly")
            if a:
                L += [f"- backward node: `{a['backward_node']}`、forward の演算: `{a['forward_location']}`、"
                      f"モデル内の場所: `{a['model_location']}`", "- forward の traceback(torch 以外のフレーム、末尾 8 個):", "",
                      "```", *a["forward_frames"], "```"]
            if r.get("stderr_tail"):
                L += ["- stderr の末尾:", "", "```", *r["stderr_tail"].splitlines()[-15:], "```"]
            L.append("")
    if "total_wall_s" in I:
        L += ["", f"全体の所要時間 {I['total_wall_s']:.0f} 秒。work/ の大きさ {I['work_bytes'] / 1e9:.2f} GB。", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="学習 1 ステップ(forward + backward)の一括 vs 分割累積(MPS の事例)")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[682, 683, 1365, 1366, 2048])
    ap.add_argument("--impls", nargs="+", default=["eager", "sdpa"], choices=["eager", "sdpa"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256, help="基準の分割の大きさ")
    ap.add_argument("--chunk2", type=int, default=128, help="2 つ目の基準(noise floor 用)")
    ap.add_argument("--cpu-check-n", type=int, default=256, help="CPU 確認の件数(0 で省略)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", help="試験用に cpu も可")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--timeout-min", type=float, default=180, help="子プロセスごとの制限時間 [分]")
    ap.add_argument("--keep-work", action="store_true", help="out/work/ を消さない")
    ap.add_argument("--out", default=str(ROOT / "results" / "train_step"))
    ap.add_argument("--tiny", action="store_true", help="動作確認用: 乱数初期化の小さなモデルと合成入力(datasets・ハブを使わない)")
    ap.add_argument("--debug-raise-backward", type=int, nargs="*", default=[], metavar="B",
                    help="試験専用: この B の full 条件で backward の中で例外を出す(status / anomaly の記録の確認用)")
    ap.add_argument("--debug-crash", type=int, nargs="*", default=[], metavar="B",
                    help="試験専用: この B の full 条件の子プロセスを backward の前に os.abort() で落とす(crashed の確認用)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(Path(a.child))
    else:
        parent(a)


if __name__ == "__main__":
    main()
