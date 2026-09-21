"""bmm_bwd の走査の全行を、forward と同じ規則で照合する。

    uv run --no-project python analysis/rules_bwd.py results/raw/studio/<run_id>.jsonl [...]

勾配は grad_a = G b^T、grad_b = a^T G(G は上流の勾配、出力と同じ形で連続)。この 2 つの積に forward の規則
(v1 の表 = 規則 B)をそのまま当てる。転置は view として入ると仮定する: 連続な operand の転置は転置 view、
転置 view として渡した operand の転置は元の連続な格納、slice の転置は転置 view。
照合するもの: out / grad_a / grad_b の分類(全要素比較では誤答の範囲まで: 全バッチ、または 2**32 要素より後ろのバッチだけ)と、
例外になった行の stage(forward の積が例外なら bmm、勾配の積のどちらかが例外なら bwd)。
"""
import json
import sys
from collections import Counter

L = 2**32
KINDS = {"contig": ("contig", "contig"), "aT": ("transposed", "contig"), "bT": ("contig", "transposed"), "slice": ("sliced", "sliced")}
T = {"contig": "transposed", "transposed": "contig", "sliced": "transposed"}


def rule(out_n, ops):
    if out_n > L:
        return "wrong_all" if any(k == "transposed" for _, k in ops) else "ok"
    if any(k != "contig" and n >= 2**31 for n, k in ops):
        return "error"
    if any(k == "contig" and n > L for n, k in ops):
        return "wrong_tail"
    return "ok"


def main():
    for path in sys.argv[1:]:
        rows = [r for r in map(json.loads, open(path)) if r["phase"] != "calibrate" and r.get("op") == "bmm_bwd"]
        res, bad = Counter(), []
        for r in rows:
            B, M, K, N = r["B"], r["M"], r["K"], r["N"]
            ka, kb = KINDS[r["layout"]]
            na, nb, no = B * M * K, B * K * N, B * M * N
            pred = {"out": rule(no, [(na, ka), (nb, kb)]),
                    "grad_a": rule(na, [(no, "contig"), (nb, T[kb])]),
                    "grad_b": rule(nb, [(na, T[ka]), (no, "contig")])}
            if r["cls"] == "error":
                stage = "bmm" if pred["out"] == "error" else ("bwd" if "error" in (pred["grad_a"], pred["grad_b"]) else None)
                ok = r.get("stage") == stage
                res[("stage", ok)] += 1
                if not ok:
                    bad.append((r["series_id"], B, "stage", r.get("stage"), pred))
                continue
            for name, p in pred.items():
                c = r.get(f"cls_{name}")
                rng = r.get(f"bad_ranges_{name}") if name != "out" else r.get("bad_ranges_out", r.get("bad_ranges"))
                if c == "wrong" and r["compare"] == "full" and rng is not None:
                    o = "wrong_all" if rng == [[0, B]] else ("wrong_tail" if rng == [[L // (M * N), B]] else "wrong_other")
                else:
                    o = c
                ok = (o == p) or (o == "wrong" and p.startswith("wrong"))
                res[(name, ok)] += 1
                if not ok:
                    bad.append((r["series_id"], B, name, "pred", p, "obs", o, rng))
        print(f"# {path}: rows={len(rows)}", {f"{k[0]}:{'agree' if k[1] else 'DISAGREE'}": v for k, v in sorted(res.items())})
        for b in bad[:20]:
            print("  ", b)


if __name__ == "__main__":
    main()
