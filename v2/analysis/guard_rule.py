"""guard_rule.py: Table 11 (tab:guard) の停止規則を raw の jsonl から再集計する(標準ライブラリのみ)。

停止規則: その実行のどれかのテンソルが 2**32 要素以上なら止める。テンソルの「大きさ」の数え方を 3 通りで集計する。

  logical  公開 guard (guard/mps_guard.py の _check, t.numel()) が見る論理要素数。
           bmm: a, b, out。bmm_bwd: a, b, out, 上流勾配 G (= out の形), grad_a (= a の形), grad_b (= b の形)。
  storage  a, b, G は view の背後の格納全体 (slice: B+1 バッチ、offset: pad + view = a_offset + numel_a)、
           out / grad_a / grad_b は新規確保なので論理要素数。
  paper    論文の表の数え方: offset layout だけ storage、それ以外は logical。

行の定義: runs = jsonl の全行 (calibrate を含む)、silent = cls が wrong / truncated (bmm_bwd は cls_out /
cls_grad_a / cls_grad_b のどれかが wrong / truncated)、correct = cls が ok。crash / error はどちらでもない。

使い方 (repo 直下で):
  uv run python v2/analysis/guard_rule.py                       # 既定: v2/results/raw/studio
  uv run python v2/analysis/guard_rule.py --raw <dir> --diff    # 数え方で stopped が変わる実行を列挙
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LIMIT = 2**32
SILENT = {"wrong", "truncated"}

# 表の行 → run_id。v2/results/summary/studio/v2_axes_studio.md と各 run の manifest (worker.torch) による。
SWEEPS = [
    ("2.4.1", "20260920T192435-studio"), ("2.5.1", "20260920T200227-studio"),
    ("2.6.0", "20260920T203904-studio"), ("2.7.1", "20260920T211533-studio"),
    ("2.8.0", "20260920T214629-studio"), ("2.9.1", "20260920T221708-studio"),
    ("2.10.0", "20260920T224546-studio"), ("2.11.0", "20260920T231439-studio"),
    ("2.12.1", "20260920T234329-studio"), ("2.13.0", "20260921T001206-studio"),
    ("2.14.0 (main sweep)", "20260920T150512-studio"),
]
ROWS = [(name, [rid]) for name, rid in SWEEPS] + [
    ("all versions, main and reduced sweeps", [rid for _, rid in SWEEPS]),
    ("2.14.0, (256, 256, 256) shape", ["20260920T113543-studio", "20260920T135650-studio"]),
    ("2.14.0, far points", ["20260920T132549-studio"]),
    ("2.14.0, offset series", ["20260920T132133-studio"]),
    ("2.14.0, backward pass", ["20260920T120157-studio"]),
]


def load(raw: Path, rid: str) -> list[dict]:
    with open(raw / f"{rid}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_silent(r: dict) -> bool:
    cls = [r.get("cls")]
    if r.get("op") == "bmm_bwd":
        cls += [r.get("cls_out"), r.get("cls_grad_a"), r.get("cls_grad_b")]
    return any(c in SILENT for c in cls)


def storage_numel(r: dict, which: str) -> int:
    """a / b の格納全体の要素数。crash 行には storage shape が無いので layout から復元する。"""
    shape = r.get(f"{which}_storage_shape")
    if shape is not None:
        return math.prod(shape)
    n = r[f"numel_{which}"]
    if r.get("layout") == "offset":
        return r[f"storage_numel_{which}"]
    if r.get("layout") == "slice":  # 格納は B+1 バッチ
        return n // r["B"] * (r["B"] + 1)
    return n


def tensor_sizes(r: dict, convention: str) -> dict[str, int]:
    """停止規則が見るテンソルごとの要素数。"""
    logical = {"a": r["numel_a"], "b": r["numel_b"], "out": r["numel_out"]}
    if r.get("op") == "bmm_bwd":
        logical["G"] = math.prod(r.get("shape_out"))  # 上流勾配は out と同じ形
        logical["grad_a"] = math.prod(r.get("grad_a_shape") or r["shape_a"])
        logical["grad_b"] = math.prod(r.get("grad_b_shape") or r["shape_b"])
    if convention == "logical" or (convention == "paper" and r.get("layout") != "offset"):
        return logical
    sizes = dict(logical)
    sizes["a"] = storage_numel(r, "a")
    sizes["b"] = storage_numel(r, "b")
    if r.get("g_storage_shape") is not None:
        sizes["G"] = math.prod(r["g_storage_shape"])
    return sizes


def size(r: dict, convention: str) -> int:
    return max(tensor_sizes(r, convention).values())


def tally(rows: list[dict], convention: str) -> dict[str, int]:
    n = [size(r, convention) for r in rows]
    silent = [is_silent(r) for r in rows]
    correct = [r.get("cls") == "ok" for r in rows]
    return {
        "runs": len(rows),
        "silent": sum(silent),
        "silent_ge": sum(s and x >= LIMIT for s, x in zip(silent, n)),
        "silent_gt": sum(s and x > LIMIT for s, x in zip(silent, n)),
        "correct_stopped": sum(c and x >= LIMIT for c, x in zip(correct, n)),
        "correct": sum(correct),
    }


def print_table(raw: Path, convention: str) -> dict[str, dict[str, int]]:
    print(f"\n### convention = {convention}\n")
    print("| PyTorch / series | runs | silent | silent stopped n>=2^32 | silent stopped n>2^32 | correct stopped |")
    print("|---|---:|---:|---:|---:|---:|")
    out = {}
    for name, rids in ROWS:
        rows = [r for rid in rids for r in load(raw, rid)]
        t = out[name] = tally(rows, convention)
        print(f"| {name} | {t['runs']} | {t['silent']} | {t['silent_ge']} | {t['silent_gt']} | {t['correct_stopped']} of {t['correct']} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--raw", type=Path, default=root / "results" / "raw" / "studio", help="raw jsonl のディレクトリ")
    ap.add_argument("--conventions", default="logical,storage,paper", help="コンマ区切り: logical / storage / paper")
    ap.add_argument("--diff", action="store_true", help="logical と storage で stopped が変わる実行を列挙する")
    args = ap.parse_args()

    results = {c: print_table(args.raw, c) for c in args.conventions.split(",")}

    if len(results) > 1:
        base = "logical" if "logical" in results else next(iter(results))
        print(f"\n### rows that differ from convention = {base}\n")
        for other, tab in results.items():
            if other == base:
                continue
            for name in tab:
                a, b = results[base][name], tab[name]
                if a != b:
                    print(f"- {other} vs {base}, {name}: " + ", ".join(f"{k} {a[k]} -> {b[k]}" for k in a if a[k] != b[k]))

    if args.diff:
        print("\n### runs whose stopped status differs between logical and storage\n")
        print("| row | run_id | fingerprint | op | layout | shape | dtype | B | phase | cls | n_logical | n_storage |")
        print("|---|---|---|---|---|---|---|---:|---|---|---:|---:|")
        seen = set()
        for name, rids in ROWS:
            for rid in rids:
                for r in load(args.raw, rid):
                    nl, ns = size(r, "logical"), size(r, "storage")
                    if (nl >= LIMIT) != (ns >= LIMIT) and (rid, r["fingerprint"]) not in seen:
                        seen.add((rid, r["fingerprint"]))
                        print(f"| {name} | {rid} | {r['fingerprint']} | {r['op']} | {r['layout']} | {r['shape_id']} | {r['dtype']} | {r['B']} | {r['phase']} | {r['cls']} | {nl} | {ns} |")
        print(f"\n{len(seen)} distinct runs differ.")


if __name__ == "__main__":
    main()
