"""demo_workload.py が保存した logits から、case study の集計を作り直す(MPS での再実行は不要)。

    uv run --no-project --python 3.12 --with numpy python workload_stats.py [--results ../v1/results] [--out DIR]

対象は B=2048 の一括実行(eager)と、その循環シフト版。基準は 256 件ずつの分割実行。
  - 境界位置(2**32 要素に達する最初のテキスト)の前後での相対誤差の分離
  - 境界以降の位置での精度: 基準 vs 一括実行、対応のある比較(McNemar の正確検定、両側)
  - 境界以降の位置での予測クラスの内訳
出力: <results>/workload/stats.json、stats.md(--results の既定は v2/results。--out で出力先のディレクトリを変える。
v1 の結果を読むときは、v1 のファイルを上書きしないよう --out を付けること)
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "workload"
BAD_REL = 1e-3


def mcnemar_exact(b: int, c: int) -> float:
    """不一致対 (b, c) の両側正確検定。Binomial(b + c, 1/2) で min(b, c) 以下になる確率の 2 倍。"""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"), help="results ディレクトリ(v1 を読むなら ../v1/results)")
    ap.add_argument("--out", default=None, help="出力先のディレクトリ(既定は <results>/workload)")
    args = ap.parse_args()
    OUT = Path(args.results) / "workload"  # noqa: N806
    dst = Path(args.out) if args.out else OUT
    dst.mkdir(parents=True, exist_ok=True)
    info = json.loads((OUT / "workload.json").read_text())
    names = [info["id2label"][str(i)] for i in range(3)]
    z = np.load(OUT / "logits.npz")
    ref, y = z["mps_eager_c256"], z["labels"]
    N, H, L = len(ref), info["num_heads"], info["max_length"]
    first = 2**32 // (H * L * L)  # この位置のテキストから scores が 2**32 要素をまたぐ
    res = {"first_affected_position": first, "bad_rel": BAD_REL, "labels": names, "runs": {}}
    for key, shift in [("B2048_mps_eager_full", 0), ("B2048_mps_eager_full_shift1024", 1024)]:
        lg = z[key]
        rel = np.abs(lg - ref).max(1) / np.abs(ref).max(1)
        pos = (np.arange(N) + shift) % N  # テキスト i のバッチ内の位置
        lo, hi = pos < first, pos >= first
        ok_ref, ok_full = ref.argmax(1) == y, lg.argmax(1) == y
        b = int((ok_ref & ~ok_full & hi).sum())
        c = int((~ok_ref & ok_full & hi).sum())
        res["runs"][key] = {
            "shift": shift, "n_below": int(lo.sum()), "n_above": int(hi.sum()),
            "below_nonzero": int((rel[lo] > 0).sum()), "below_max_rel": float(rel[lo].max()),
            "above_corrupted": int((rel[hi] > BAD_REL).sum()), "above_min_rel": float(rel[hi].min()),
            "above_median_rel": float(np.median(rel[hi])),
            "above_acc_ref": float(ok_ref[hi].mean()), "above_acc_full": float(ok_full[hi].mean()),
            "mcnemar_ref_only_correct": b, "mcnemar_full_only_correct": c, "mcnemar_exact_p": mcnemar_exact(b, c),
            "above_pred_full": np.bincount(lg[hi].argmax(1), minlength=3).tolist(),
            "above_pred_ref": np.bincount(ref[hi].argmax(1), minlength=3).tolist(),
            "above_label": np.bincount(y[hi], minlength=3).tolist(),
            # クラスごとの精度(recall): 全テキストと、境界以降の位置だけ
            "recall_all_ref": [float(ok_ref[y == k].mean()) for k in range(3)],
            "recall_all_full": [float(ok_full[y == k].mean()) for k in range(3)],
            "recall_above_ref": [float(ok_ref[hi & (y == k)].mean()) for k in range(3)],
            "recall_above_full": [float(ok_full[hi & (y == k)].mean()) for k in range(3)],
        }
    (dst / "stats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    md = ["# case study の集計(workload_stats.py)", "",
          f"境界位置 = {first}(scores が 2**32 要素をまたぐ最初のテキスト)。corrupted = 相対 logit 誤差 > {BAD_REL:g}。基準は分割実行。", ""]
    for key, r in res["runs"].items():
        md += [f"## {key}", "",
               f"- 境界より前 {r['n_below']} 件: 誤差が 0 でないもの {r['below_nonzero']} 件(最大 {r['below_max_rel']:.3g})",
               f"- 境界以降 {r['n_above']} 件: corrupted {r['above_corrupted']} 件、誤差の最小 {r['above_min_rel']:.3g}、中央値 {r['above_median_rel']:.3g}",
               f"- 境界以降の精度: 基準 {r['above_acc_ref']:.3f} → 一括 {r['above_acc_full']:.3f}。基準だけ正解 {r['mcnemar_ref_only_correct']} 件、"
               f"一括だけ正解 {r['mcnemar_full_only_correct']} 件、McNemar の正確検定(両側)p = {r['mcnemar_exact_p']:.2g}", "",
               "| 境界以降の内訳 | " + " | ".join(names) + " |", "|---|" + "---|" * 3,
               "| 一括実行の予測 | " + " | ".join(map(str, r["above_pred_full"])) + " |",
               "| 基準の予測 | " + " | ".join(map(str, r["above_pred_ref"])) + " |",
               "| 正解ラベル | " + " | ".join(map(str, r["above_label"])) + " |", "",
               "| クラスごとの精度 (recall) | " + " | ".join(names) + " |", "|---|" + "---|" * 3]
        md += [f"| {lab} | " + " | ".join(f"{v:.3f}" for v in r[k]) + " |" for lab, k in [
            ("全テキスト、基準", "recall_all_ref"), ("全テキスト、一括実行", "recall_all_full"),
            ("境界以降、基準", "recall_above_ref"), ("境界以降、一括実行", "recall_above_full")]] + [""]
    (dst / "stats.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
