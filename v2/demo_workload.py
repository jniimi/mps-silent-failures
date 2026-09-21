"""実際の ML ワークロードで MPS がエラーなしに誤答する例(RoBERTa 感情分類、eager attention、大バッチ 1 回の forward)。

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --no-project --python 3.12 --with torch==2.14.0 \
        --with transformers==5.17.0 --with datasets==5.0.1 --with matplotlib python demo_workload.py

    (worker.py と同じく torch はプロジェクトの依存に入れず、版を固定した使い捨て環境で動かす)

題材: cardiffnlp/twitter-roberta-base-sentiment-latest(12 heads、3 クラス)を cardiffnlp/tweet_eval (sentiment, test;
同じ 3 クラスのラベル体系) の先頭 N=max(B) 件(データセットの順)に適用する。全テキストを padding="max_length",
truncation=True, max_length=L でトークン化し、同一の input_ids / attention_mask を全条件で使う。fp32。

L の選び方: eager attention の scores (B, 12, L, L) と FFN の中間 (B*L, 3072) の要素数の比は L/256。
L=256 では両者が同じ B(5462)で 2**32 を超え、FFN の Linear(2 次元の matmul)が
"MPSNDArrayMatrixMultiplication ... is too large for kernel" の assertion で abort する(実測、silent ではない)。
そこで既定は L=512: scores は B >= 1366 で 2**32 を超え(B=1365 は 2**32 - 1,048,576)、FFN は B <= 2730 で 2**32 未満。

基準(metamorphic relation): eval 時の Transformer はサンプル間の相互作用がないので
f(concat(X1..Xk)) = concat(f(X1)..f(Xk)) が成り立つはず。MPS・eager・256 件ずつの分割実行を主な基準とし、
CPU fp32 との照合(約 1000 件の選抜)でこの関係の妥当性を独立に確認する。

条件(各 B について先頭 B 件):
  mps_eager_full   MPS、eager、B 件を 1 回の forward(壊れるはずの条件)
  mps_sdpa_full    MPS、fused SDPA、B 件を 1 回の forward(対照)
  mps_eager_full_shift4096   最大の B で同じテキスト集合を --shift だけ循環シフトして 1 回の forward
                            (比較はテキストごとに元の順へ戻して行い、位置は「シフト後のバッチ内の位置」でも記録する)
指標(基準とテキストごと): rel = max_i |Δlogit_i| / max_i |logit_ref,i|、予測ラベルの変化、正解率(補助)。
rel > BAD_REL のテキストを「壊れた」とし、その位置(連続区間)を記録する。

1 層目の attention の中間(層 0 の入力を分割実行で取り出し、eager_attention_forward と同じ演算を MPS 上で
B 件一括と 256 件ずつで計算して比較): S = (q @ kᵀ)·scale、P = softmax(S + mask)、O = P @ v(P は一括の値)、
O_isolated = P_ref @ v(P を分割実行の値にそろえ、2 つ目の積だけを切り出す)。テキスト×ヘッドごとの rel を記録する。

出力: results/workload/workload.json、workload.md、logits.npz、layer0.npz、fig_rel_by_position.png
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET, CONFIG, SPLIT = "cardiffnlp/tweet_eval", "sentiment", "test"
BAD_REL = 1e-3
ROOT = Path(__file__).resolve().parent


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def env_info() -> dict:
    import datasets
    import transformers
    return {
        "macos": sh("sw_vers", "-productVersion"), "macos_build": sh("sw_vers", "-buildVersion"),
        "hw_model": sh("sysctl", "-n", "hw.model"), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "mem_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__,
        "git_rev": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
    }


def hub_rev(kind: str, name: str) -> str:
    from huggingface_hub.constants import HF_HUB_CACHE
    p = Path(HF_HUB_CACHE) / f"{kind}--{name.replace('/', '--')}" / "refs" / "main"
    return p.read_text().strip() if p.exists() else ""


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def empty(dev: str) -> None:
    if dev == "mps":
        torch.mps.empty_cache()


def forward(model, ids, mask, dev: str, chunk: int | None) -> tuple[np.ndarray, float]:
    """logits (n, 3) を fp32 で返す。chunk=None なら全件を 1 回の forward。"""
    n = ids.shape[0]
    step = n if chunk is None else chunk
    out = []
    sync(dev)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, step):
            lg = model(input_ids=ids[s:s + step].to(dev), attention_mask=mask[s:s + step].to(dev)).logits
            out.append(lg.float().cpu())
    sync(dev)
    dt = time.perf_counter() - t0
    empty(dev)
    return torch.cat(out).numpy(), dt


def runs_of(idx: np.ndarray) -> list[list[int]]:
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) != 1)[0]
    return [[int(a), int(b)] for a, b in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]])]


def text_rel(lg: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.abs(lg.astype(np.float64) - ref.astype(np.float64)).max(1)
    rel = d / np.abs(ref.astype(np.float64)).max(1)
    rel[~np.isfinite(lg).all(1)] = np.inf
    return rel


def margin_of(lg: np.ndarray) -> np.ndarray:
    s = np.sort(lg, 1)
    return s[:, -1] - s[:, -2]


def compare(lg: np.ndarray, ref: np.ndarray, labels: np.ndarray, pos: np.ndarray | None = None) -> dict:
    """pos: 各テキストのバッチ内の位置(循環シフト用)。None なら index = 位置。"""
    rel = text_rel(lg, ref)
    bad = np.where(rel > BAD_REL)[0]
    flip = np.where(lg.argmax(1) != ref.argmax(1))[0]
    ok = np.setdiff1d(np.arange(len(ref)), bad)
    pos = np.arange(len(ref)) if pos is None else pos
    fin = rel[np.isfinite(rel)]
    return {
        "n": len(ref), "max_rel": float(rel.max()), "max_rel_finite": float(fin.max()) if fin.size else None,
        "median_rel": float(np.median(rel)), "max_rel_ok_texts": float(rel[ok].max()) if ok.size else None,
        "median_rel_bad_texts": float(np.median(rel[bad])) if bad.size else None,
        "nonfinite_texts": int((~np.isfinite(lg).all(1)).sum()),
        "bad_texts": int(bad.size), "bad_frac": bad.size / len(ref),
        "bad_runs_text_index": runs_of(bad), "bad_runs_batch_pos": runs_of(np.sort(pos[bad])),
        "pred_changed": int(flip.size), "pred_changed_frac": flip.size / len(ref),
        "pred_changed_runs_batch_pos": runs_of(np.sort(pos[flip]))[:50],
        "pred_changed_frac_among_bad": float(np.isin(bad, flip).mean()) if bad.size else None,
        "acc": float((lg.argmax(1) == labels).mean()), "acc_ref": float((ref.argmax(1) == labels).mean()),
    }


class Stop(Exception):
    pass


def layer0_inputs(model, ids, mask, dev: str, chunk: int):
    """層 0 の self-attention の入力 (hidden_states, attention_mask) を分割実行で取り出す(以降の層は計算しない)。"""
    attn = model.roberta.encoder.layer[0].attention.self
    box = {}

    def hook(mod, args, kw):
        box["h"] = args[0] if args else kw["hidden_states"]
        box["m"] = kw["attention_mask"] if "attention_mask" in kw else args[1]
        raise Stop

    hd = attn.register_forward_pre_hook(hook, with_kwargs=True)
    hs, ms = [], []
    try:
        with torch.inference_mode():
            for s in range(0, ids.shape[0], chunk):
                try:
                    model(input_ids=ids[s:s + chunk].to(dev), attention_mask=mask[s:s + chunk].to(dev))
                except Stop:
                    pass
                hs.append(box["h"].clone())
                ms.append(box["m"].clone())
    finally:
        hd.remove()
    return torch.cat(hs), torch.cat(ms)


def layer0_diag(model, h: torch.Tensor, am: torch.Tensor, chunk: int) -> dict[str, np.ndarray]:
    """eager_attention_forward と同じ演算を一括と分割で行い、テキスト×ヘッドごとの rel を返す。"""
    attn = model.roberta.encoder.layer[0].attention.self
    B, L, _ = h.shape
    hs = (B, L, -1, attn.attention_head_size)
    with torch.inference_mode():
        q = attn.query(h).view(*hs).transpose(1, 2)
        k = attn.key(h).view(*hs).transpose(1, 2)
        v = attn.value(h).view(*hs).transpose(1, 2)

        def S_of(sl):
            return torch.matmul(q[sl], k[sl].transpose(2, 3)) * attn.scaling

        def P_of(S, sl):
            return torch.nn.functional.softmax(S + am[sl], dim=-1)

        def rel(a, r):
            return ((a - r).abs().amax(dim=(-1, -2)) / r.abs().amax(dim=(-1, -2))).float().cpu().numpy()

        chunks = [slice(s, min(s + chunk, B)) for s in range(0, B, chunk)]
        out = {k_: np.zeros((B, q.shape[1])) for k_ in ("S", "P", "O", "O_isolated")}
        S = S_of(slice(None))
        for sl in chunks:
            out["S"][sl] = rel(S[sl], S_of(sl))
        P = P_of(S, slice(None))
        del S
        for sl in chunks:
            out["P"][sl] = rel(P[sl], P_of(S_of(sl), sl))
        O = torch.matmul(P, v)
        del P
        empty(h.device.type)
        # P の chunk は個別のテンソルとして持つ(2**31 要素を超える一括テンソルの slice view を matmul に渡すと
        # "MPSGraph does not support tensor dims larger than INT_MAX" の例外になるため)
        Prefs = [P_of(S_of(sl), sl) for sl in chunks]
        Pref = torch.cat(Prefs)
        O2 = torch.matmul(Pref, v)
        del Pref
        for sl, Pc in zip(chunks, Prefs):
            Or = torch.matmul(Pc, v[sl])
            out["O"][sl] = rel(O[sl], Or)
            out["O_isolated"][sl] = rel(O2[sl], Or)
        del O, O2, Prefs, q, k, v
    sync(h.device.type)
    empty(h.device.type)
    return out


def cpu_subset(N: int, ref_mps: np.ndarray, n_target: int, t0: int) -> np.ndarray:
    fixed = set(range(0, 100)) | set(range(N // 2 - 50, N // 2 + 50)) | set(range(N - 100, N)) | set(range(t0 - 21, t0 + 20))
    fixed = {i for i in fixed if 0 <= i < N}
    sel = list(fixed)
    for i in np.argsort(margin_of(ref_mps), kind="stable"):
        if len(sel) >= n_target:
            break
        if int(i) not in fixed:
            sel.append(int(i))
    return np.array(sorted(i for i in sel if i < N))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1365, 1366, 2048])
    ap.add_argument("--sdpa-batch-sizes", type=int, nargs="+", default=[1366, 2048])
    ap.add_argument("--shift", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--cpu-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", help="試験用(cpu で小さい B の動作確認)")
    ap.add_argument("--no-layer0", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "workload"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    t_all = time.perf_counter()

    import datasets
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    N = max(a.batch_sizes)
    ds = datasets.load_dataset(DATASET, CONFIG, split=SPLIT)
    n_split = len(ds)
    ds = ds.select(range(N))
    labels = np.array(ds["label"])
    texts = list(ds["text"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(texts, padding="max_length", truncation=True, max_length=a.max_length, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    n_trunc = int(sum(len(x) > a.max_length for x in tok(texts)["input_ids"]))

    models = {impl: AutoModelForSequenceClassification.from_pretrained(MODEL, attn_implementation=impl).eval()
              for impl in ("eager", "sdpa")}
    cfg = models["eager"].config
    H, L = cfg.num_attention_heads, a.max_length
    t0, h0 = divmod(2**32 // (L * L), H)  # 仮説 A: 平坦な (text, head) の slice index が 2**32/(L*L) に達する位置
    info = {
        "env": env_info(), "device": a.device,
        "model": MODEL, "model_rev": getattr(cfg, "_commit_hash", None) or hub_rev("models", MODEL),
        "tokenizer": MODEL, "tokenizer_rev": hub_rev("models", MODEL),
        "dataset": DATASET, "dataset_config": CONFIG, "split": SPLIT, "split_size": n_split,
        "dataset_rev": hub_rev("datasets", DATASET),
        "n_texts": N, "text_order": "dataset order, first N", "max_length": L,
        "tokenization": "padding=max_length, truncation=True", "n_truncated": n_trunc,
        "mean_real_tokens": float(mask.sum(1).float().mean()), "num_heads": H, "dtype": "fp32",
        "chunk": a.chunk, "seed": a.seed, "bad_rel_threshold": BAD_REL, "grad": "torch.inference_mode(), model.eval()",
        "label_counts": np.bincount(labels, minlength=3).tolist(), "id2label": cfg.id2label,
        "boundary_text": t0, "boundary_head": h0, "ffn_size": cfg.intermediate_size,
        "hypotheses": {
            "A_flat_index_wrap": f"2**32/(L*L) = {2**32 // (L * L)} = {t0}*{H} + {h0}: text {t0} head>={h0} onward; "
                                 f"only texts >= {t0} broken",
            "B_transposed_view_all": "q@k^T with k^T as transposed view: all texts broken once output > 2**32",
        },
    }
    print(json.dumps({k: v for k, v in info.items() if k != "env"}, ensure_ascii=False), flush=True)
    for m in models.values():
        m.to(a.device)
    saved = {"labels": labels}

    # 基準: MPS・eager・256 件ずつ(N 件。各 B では先頭 B 件。chunk の切れ目は B によらず同じ)
    torch.manual_seed(a.seed)
    ref, dt = forward(models["eager"], ids, mask, a.device, a.chunk)
    saved["mps_eager_c256"] = ref
    info["reference"] = {"name": "mps_eager_c256", "n": N, "wall_s": dt,
                         "acc": float((ref.argmax(1) == labels).mean())}
    print(f"ref mps_eager_c256: {dt:.1f}s acc={info['reference']['acc']:.4f}", flush=True)

    # CPU fp32 による独立確認(選抜した約 1000 件)
    sel = cpu_subset(N, ref, a.cpu_n, t0)
    cpu_m = AutoModelForSequenceClassification.from_pretrained(MODEL, attn_implementation="eager").eval()
    lg_cpu, dt = forward(cpu_m, ids[sel], mask[sel], "cpu", a.chunk)
    del cpu_m
    saved["cpu_sel_index"], saved["cpu_sel"] = sel, lg_cpu
    rc = text_rel(ref[sel], lg_cpu)
    info["cpu_check"] = {
        "n": int(sel.size), "wall_s": dt, "threads": torch.get_num_threads(),
        "selection": f"0-99, N/2-50..N/2+49, N-100..N-1, {t0 - 21}-{t0 + 19}, rest = smallest top1-top2 margin in reference",
        "margin_max_in_selection": float(margin_of(ref[sel]).max()),
        "c256_vs_cpu_max_rel": float(rc.max()), "c256_vs_cpu_median_rel": float(np.median(rc)),
        "c256_vs_cpu_pred_changed": int((ref[sel].argmax(1) != lg_cpu.argmax(1)).sum()),
    }
    print(f"cpu check: {info['cpu_check']}", flush=True)

    results = []
    for B in a.batch_sizes:
        r = {"B": B, "scores_elems": B * H * L * L, "ffn_elems": B * L * cfg.intermediate_size,
             "ffn_over_2p32": B * L * cfg.intermediate_size / 2**32, "scores_minus_2p32": B * H * L * L - 2**32,
             "scores_over_2p32": B * H * L * L / 2**32, "conds": {}}
        conds = [("mps_eager_full", "eager")] + ([("mps_sdpa_full", "sdpa")] if B in a.sdpa_batch_sizes else [])
        for name, impl in conds:
            torch.manual_seed(a.seed)
            lg, dt = forward(models[impl], ids[:B], mask[:B], a.device, None)
            c = compare(lg, ref[:B], labels[:B])
            m_sel = sel < B
            c["vs_cpu_on_selection"] = {"n": int(m_sel.sum()),
                                        "max_rel": float(text_rel(lg[sel[m_sel]], lg_cpu[m_sel]).max()),
                                        "pred_changed": int((lg[sel[m_sel]].argmax(1) != lg_cpu[m_sel].argmax(1)).sum())}
            c.update({"impl": impl, "wall_s": dt})
            r["conds"][name] = c
            saved[f"B{B}_{name}"] = lg
            print(f"B={B} {name}: {dt:.1f}s bad={c['bad_texts']} runs={c['bad_runs_text_index'][:4]} "
                  f"max_rel={c['max_rel']:.3g} pred_changed={c['pred_changed']} acc={c['acc']:.4f}/{c['acc_ref']:.4f}",
                  flush=True)
        if B == N and a.shift:
            perm = np.roll(np.arange(B), -a.shift)  # バッチ内の位置 p にテキスト perm[p]
            torch.manual_seed(a.seed)
            lg_s, dt = forward(models["eager"], ids[perm], mask[perm], a.device, None)
            lg = np.empty_like(lg_s)
            lg[perm] = lg_s
            pos = np.empty(B, dtype=int)
            pos[perm] = np.arange(B)
            c = compare(lg, ref[:B], labels[:B], pos=pos)
            c.update({"impl": "eager", "wall_s": dt, "shift": a.shift})
            full = r["conds"]["mps_eager_full"]
            b0 = set(map(int, np.where(text_rel(saved[f"B{B}_mps_eager_full"], ref[:B]) > BAD_REL)[0]))
            b1 = set(map(int, np.where(text_rel(lg, ref[:B]) > BAD_REL)[0]))
            c["overlap_bad_texts_with_unshifted"] = len(b0 & b1)
            c["same_bad_batch_positions_as_unshifted"] = c["bad_runs_batch_pos"] == full["bad_runs_batch_pos"]
            r["conds"][f"mps_eager_full_shift{a.shift}"] = c
            saved[f"B{B}_mps_eager_full_shift{a.shift}"] = lg
            print(f"B={B} shift{a.shift}: bad_pos={c['bad_runs_batch_pos'][:4]} bad_text={c['bad_runs_text_index'][:4]} "
                  f"overlap={c['overlap_bad_texts_with_unshifted']}", flush=True)
        results.append(r)
    info["results"] = results
    np.savez_compressed(out / "logits.npz", **saved)

    if not a.no_layer0:
        t_l0 = time.perf_counter()
        hN, mN = layer0_inputs(models["eager"], ids, mask, a.device, a.chunk)
        l0 = {}
        info["layer0"] = {"inputs": "layer-0 self-attention inputs from chunked run", "mask_shape": list(mN.shape[1:]),
                          "per_B": {}}
        for B in a.batch_sizes:
            d = layer0_diag(models["eager"], hN[:B], mN[:B], a.chunk)
            s = {}
            for st, x in d.items():
                l0[f"B{B}_{st}"] = x
                pt = x.max(1)
                bad = np.where(~(pt <= BAD_REL))[0]
                bh = np.argwhere(~(x <= BAD_REL))
                s[st] = {"max_rel": float(np.nanmax(x)), "nonfinite": int((~np.isfinite(x)).sum()),
                         "bad_texts": int(bad.size), "bad_runs": runs_of(bad),
                         "first_bad_text_head": bh[0].tolist() if bh.size else None,
                         "bad_heads_of_boundary_text": np.where(~(x[t0] <= BAD_REL))[0].tolist() if B > t0 else None,
                         "max_rel_ok": float(pt[pt <= BAD_REL].max()) if (pt <= BAD_REL).any() else None}
            info["layer0"]["per_B"][str(B)] = s
            print(f"layer0 B={B}: " + "; ".join(f"{k}: bad={v['bad_texts']} runs={v['bad_runs'][:3]} "
                                                f"first={v['first_bad_text_head']}" for k, v in s.items()), flush=True)
        del hN, mN
        np.savez_compressed(out / "layer0.npz", **l0)
        info["layer0"]["wall_s"] = time.perf_counter() - t_l0
    else:
        info["layer0"] = None

    info["total_wall_s"] = time.perf_counter() - t_all
    (out / "workload.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    try:
        plot(info, saved, out)
    except Exception as e:  # 図は補助
        print(f"plot failed: {e!r}", flush=True)
    (out / "workload.md").write_text(render_md(info))
    print(f"done {info['total_wall_s']:.0f}s -> {out}", flush=True)


def plot(I: dict, saved: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = saved["mps_eager_c256"]
    rows = []
    for r in I["results"]:
        B = r["B"]
        for k, c in r["conds"].items():
            rel = text_rel(saved[f"B{B}_{k}"], ref[:B])
            if "shift" in k:
                rel = rel[np.roll(np.arange(B), -c["shift"])]  # x = バッチ内の位置
                t = f"B={B}, eager, one forward, texts cyclically shifted by {c['shift']} (x = batch position)"
            else:
                t = f"B={B}, {c['impl']}, one forward"
            rows.append((t, rel))
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 1.8 * len(rows)), sharex=True)
    for ax, (t, rel) in zip(np.atleast_1d(axs), rows):
        y = np.where(np.isfinite(rel), np.maximum(rel, 1e-9), 1e3)
        ax.scatter(np.arange(len(y)), y, s=1, rasterized=True)
        ax.set_yscale("log")
        ax.axhline(BAD_REL, ls="--", lw=0.7, c="gray")
        ax.axvline(I["boundary_text"], ls=":", lw=0.7, c="red")
        ax.set_title(t, fontsize=9)
        ax.set_ylabel("rel. err.")
    np.atleast_1d(axs)[-1].set_xlabel("position in batch")
    fig.tight_layout()
    fig.savefig(out / "fig_rel_by_position.png", dpi=150)


def render_md(I: dict) -> str:
    e = I["env"]
    L = [
        "# 実ワークロードでの silent error: RoBERTa 感情分類(eager attention、大バッチ 1 回の forward)",
        "",
        "`demo_workload.py` の出力。数値は `workload.json`、logits は `logits.npz`、層 0 の中間は `layer0.npz`、"
        "図は `fig_rel_by_position.png`(縦軸はテキストごとの rel、横軸はバッチ内の位置。rel = 0(ビット一致)は 1e-9 に、"
        "赤の点線は仮説 A の境界のテキスト)。",
        "",
        "## 環境と条件",
        "",
        f"- macOS {e['macos']} ({e['macos_build']})、{e['hw_model']} ({e['chip']}, {e['mem_bytes'] / 2**30:.0f} GiB)",
        f"- Python {e['python']}、torch {e['torch']}、transformers {e['transformers']}、datasets {e['datasets']}",
        f"- モデル: `{I['model']}` (revision `{I['model_rev']}`)、tokenizer は同じリポジトリ (revision `{I['tokenizer_rev']}`)、"
        f"{I['num_heads']} heads、fp32",
        f"- データ: `{I['dataset']}` / `{I['dataset_config']}` / `{I['split']}` ({I['split_size']} 件、revision "
        f"`{I['dataset_rev']}`) の先頭 {I['n_texts']} 件(データセットの順)。ラベル件数 {I['label_counts']} ({I['id2label']})",
        f"- トークン化: {I['tokenization']}, max_length={I['max_length']}(切り詰め {I['n_truncated']} 件、"
        f"実トークン数の平均 {I['mean_real_tokens']:.1f})。全条件で同一の input_ids / attention_mask",
        f"- {I['grad']}、`torch.manual_seed({I['seed']})`。eager と sdpa の違い以外は同一。コード git `{e['git_rev'][:12]}`",
        "",
        "## 基準",
        "",
        f"MPS・eager・{I['chunk']} 件ずつの分割実行(scores は {I['chunk']}×{I['num_heads']}×{I['max_length']}×{I['max_length']} = "
        f"{I['chunk'] * I['num_heads'] * I['max_length'] ** 2:,} 要素)を基準とする"
        f"(サンプル間の相互作用がないので f(concat(X_i)) = concat(f(X_i)))。所要 {I['reference']['wall_s']:.1f} 秒。",
        "",
    ]
    c = I["cpu_check"]
    L += [f"独立確認: CPU fp32(eager、{I['chunk']} 件ずつ)で {c['n']} 件を計算({c['selection']})。"
          f"分割実行と CPU の差は rel 最大 {c['c256_vs_cpu_max_rel']:.2g}、中央値 {c['c256_vs_cpu_median_rel']:.2g}、"
          f"予測が変わった件数 {c['c256_vs_cpu_pred_changed']}(選抜内の margin の最大 {c['margin_max_in_selection']:.3g})。"
          f"CPU {c['wall_s']:.0f} 秒、{c['threads']} threads。", "",
          f"指標: テキストごとに rel = max_i |Δlogit_i| / max_i |logit_ref,i|。rel > {I['bad_rel_threshold']:g} を「壊れた」とする。", "",
          "## 結果(logits)", "",
          "| B | scores − 2**32 | FFN 中間 / 2**32 | 条件 | 所要 [s] | 壊れたテキスト | 壊れた位置(バッチ内) | 予測が変わった件数 (割合) | 最大 rel | 正常なテキストの最大 rel | 正解率 (基準) | CPU 選抜での最大 rel / 予測変化 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in I["results"]:
        for name, c in r["conds"].items():
            runs = ", ".join(f"{a}–{b}" if a != b else f"{a}" for a, b in c["bad_runs_batch_pos"][:4])
            if len(c["bad_runs_batch_pos"]) > 4:
                runs += f" 他 {len(c['bad_runs_batch_pos']) - 4} 区間"
            okr = f"{c['max_rel_ok_texts']:.2g}" if c["max_rel_ok_texts"] is not None else "—"
            vc = c.get("vs_cpu_on_selection")
            vcs = f"{vc['max_rel']:.2g} / {vc['pred_changed']} (n={vc['n']})" if vc else "—"
            L.append(f"| {r['B']} | {r['scores_minus_2p32']:+,} | {r['ffn_over_2p32']:.3f} | {name} | {c['wall_s']:.1f} | "
                     f"{c['bad_texts']} ({100 * c['bad_frac']:.2f}%) | {runs or '—'} | "
                     f"{c['pred_changed']} ({100 * c['pred_changed_frac']:.2f}%) | {c['max_rel']:.3g} | {okr} | "
                     f"{c['acc']:.4f} ({c['acc_ref']:.4f}) | {vcs} |")
    for r in I["results"]:
        for name, c in r["conds"].items():
            if "shift" in name:
                L += ["", f"循環シフト (B={r['B']}, shift={c['shift']}): 壊れたテキスト(元の index)の区間 "
                      f"{c['bad_runs_text_index'][:4]}、バッチ内の位置の区間 {c['bad_runs_batch_pos'][:4]}。"
                      f"シフトなしと壊れた位置が一致: {c['same_bad_batch_positions_as_unshifted']}、"
                      f"壊れたテキストの重なり {c['overlap_bad_texts_with_unshifted']} 件。"]
    if I.get("layer0"):
        L += ["", "## 層 0 の attention の中間(一括 vs 分割、テキスト×ヘッドごと)", "",
              "S = (q @ kᵀ)·scale、P = softmax(S + mask)、O = P @ v、O_isolated = P_ref @ v(2 つ目の積だけ)。", "",
              "| B | 段 | 壊れたテキスト | 区間 | 最初の (text, head) | 境界のテキストの壊れた head | 最大 rel | 正常の最大 rel |",
              "|---|---|---|---|---|---|---|---|"]
        for B, s in I["layer0"]["per_B"].items():
            for st, v in s.items():
                ok = f"{v['max_rel_ok']:.2g}" if v["max_rel_ok"] is not None else "—"
                L.append(f"| {B} | {st} | {v['bad_texts']} | {v['bad_runs'][:3]} | {v['first_bad_text_head']} | "
                         f"{v['bad_heads_of_boundary_text']} | {v['max_rel']:.3g} | {ok} |")
        L += ["", f"層 0 の診断の所要 {I['layer0']['wall_s']:.0f} 秒。"]
    L += ["", "## L=256 での abort(別実行の記録)", "",
          "L=256 では FFN の中間 (B*256, 3072) が scores と同じ B=5462 で 2**32 を超える。B=5461 の一括 forward は基準と一致"
          "(rel 0)したが、B=5462 の一括 forward は `MPSNDArrayMatrixMultiplication encodeToCommandBuffer: destination "
          "[3072, 1398272] is too large for kernel` の assertion で abort した(exit 134、silent ではない)。"
          "ログは `abort_L256_B5462.log`。", ""]
    L += ["", f"全体の所要時間 {I['total_wall_s']:.0f} 秒。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
