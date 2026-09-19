"""index 符号化入力で「規則は誤答を予測するが実際は正しかった」実行が、入力の不変性で説明できるかを CPU だけで確かめる。

    uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python check_idx_invariance.py [--run RUN_ID]

規則が誤答を予測する実行(出力 > 2**32 で転置あり → stride 無視、連続な入力 > 2**32 → 2**32 での巻き戻り)について、
その読み違いのもとでの積を worker.hyp_operand で作り、正しい積と一致するか(= 入力が読み違いに対して不変か)を調べる。
不変なら正しい出力が期待され、不変でなければ誤答が期待される。これを走査で記録された分類と突き合わせる。
調べるバッチ: stride は 0 と B-1、wrap は 2**32 要素より後ろの全バッチ(巻き戻りの影響を受けるのはそこだけ)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import worker as W

ROOT = Path(__file__).resolve().parent


def predicted_mode(r):
    lay, na, nb, no = r["layout"], r["numel_a"], r["numel_b"], r["numel_out"]
    if no > 2**32:
        return {"stride"} if lay in ("aT", "bT") else None
    view = {"contig": (0, 0), "bT": (0, 1), "aT": (1, 0), "slice": (1, 1)}[lay]
    if (view[0] and na >= 2**31) or (view[1] and nb >= 2**31):
        return None  # エラーになる条件
    return {"wrap"} if max(na, nb) > 2**32 else None


def invariant(spec, mode) -> bool:
    B = spec["B"]
    per = max(spec["M"] * spec["K"], spec["K"] * spec["N"])
    batches = [0, B - 1] if mode == {"stride"} else list(range(2**32 // per, B))
    for i in batches:
        ref = W.gen_logical(spec, "a", i, i + 1)[0].double() @ W.gen_logical(spec, "b", i, i + 1)[0].double()
        hy = W.hyp_operand(spec, "a", i, mode) @ W.hyp_operand(spec, "b", i, mode)
        if not bool((ref == hy).all()):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260919T130741-studio")
    args = ap.parse_args()
    host = args.run.rsplit("-", 1)[1]
    rows = [json.loads(l) for l in open(ROOT / "results" / "raw" / host / f"{args.run}.jsonl")]
    rows = [r for r in rows if r["kind"] != "random" and r["cls"] in ("ok", "wrong")]
    cache, tally, detail = {}, Counter(), Counter()
    for r in rows:
        mode = predicted_mode(r)
        if mode is None:
            continue
        key = (r["shape_id"], r["dtype"], r["layout"], r["kind"], r["B"])
        if key not in cache:
            cache[key] = invariant(r, mode)
        expect = "ok" if cache[key] else "wrong"
        tally[(expect, r["cls"])] += 1
        detail[(r["cross"], r["layout"], r["kind"], "invariant" if cache[key] else "changes", r["cls"])] += 1
    print("(不変性からの期待, 記録された分類): 件数")
    for k, v in sorted(tally.items()):
        print(" ", k, v)
    print("内訳 (cross, layout, kind, 入力は読み違いで, 記録): 件数")
    for k, v in sorted(detail.items()):
        print(" ", k, v)
    bad = sum(v for (e, c), v in tally.items() if e != c)
    print("食い違い:", bad)


if __name__ == "__main__":
    main()
