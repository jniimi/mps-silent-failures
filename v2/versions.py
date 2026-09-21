"""torch の版ごとの走査結果を 1 つの表にまとめる(版 × 条件の分類表)。

    uv run python versions.py --host-tag studio [--runs RUN_ID,...] [--results ../v1/results] [--out PATH]

--runs を省くと results/raw/<host-tag>/ の manifest から、device ごと・torch の版ごとに最新の status=done の run を使う。
比較の基準は torch 2.14.0。基本 shape(out-256x64x256, in-256x256x64)だけを比べる。
出力: <results>/summary/<host-tag>/versions.md(--results の既定は v2/results。--out で出力先を変える。
v1 の結果を v2 から読むときは、v1 のファイルを上書きしないよう --out を付けること)。
--scale-log2 付きのテスト run(manifest の scale_log2 != 0)は比較に入れない。明示した --runs に含まれていれば中止する
(--allow-scaled で許可)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scan import LAYOUTS, ROOT, SHAPE, boundaries, is_ok, transitions

BASE = ["out-256x64x256", "in-256x256x64"]
REF = "2.14.0"


def vkey(v: str):
    return tuple(int(x) for x in v.split("+")[0].split("."))


def load(host: str, runs: list[str] | None, device: str, results: Path, allow_scaled: bool = False):
    d = results / "raw" / host
    mans = {}
    for p in sorted(d.glob("*.manifest.json")):
        m = json.loads(p.read_text())
        if m.get("device", "mps") != device or m.get("status") != "done":
            continue
        if runs and m["run_id"] not in runs:
            continue
        if m.get("op", "bmm") != "bmm" or m.get("mode", "sweep") != "sweep" or m.get("points"):
            continue  # 版の比較は bmm の走査だけ(--op bmm_bwd と --points の run は対象外)
        if m.get("scale_log2") and not allow_scaled:  # テスト用の縮小 run は版の比較に使わない
            if runs:
                raise SystemExit(f"{m['run_id']}: scale_log2={m['scale_log2']} のテスト run は比較できない(--allow-scaled で許可)")
            print(f"# {m['run_id']}: scale_log2={m['scale_log2']} のテスト run なので除外")
            continue
        v = m["torch_req"]
        if v not in mans or m["run_id"] > mans[v]["run_id"]:
            mans[v] = m
    data = {}
    for v, m in mans.items():
        rows = {}
        for line in (d / f"{m['run_id']}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("scale_log2") and not allow_scaled:
                raise SystemExit(f"{m['run_id']}: scale_log2 付きの行がある(--allow-scaled で許可)")
            rows[r["fingerprint"]] = r
        data[v] = (m, [r for r in rows.values() if r["shape_id"] in BASE])
    return dict(sorted(data.items(), key=lambda kv: vkey(kv[0])))


def short(c: str) -> str:
    return {"ok": "ok", "wrong": "WRONG", "error": "err", "crash": "CRASH", "timeout": "TO", "truncated": "TRUNC",
            "input_corrupt": "INPUT", "skipped_memory": "skip", "skipped_budget": "skip"}.get(c, c)


def best_hyp(r: dict) -> str:
    h = r.get("hyp_match") or {}
    h = {k: v for k, v in h.items() if v}
    if not h:
        return ""
    k = max(h, key=h.get)
    return f"{k} {h[k]:.2f}"


def series_rows(rows):
    """(shape, dtype, layout) → {B: row}(乱数入力、seed 0 の探索点と二分探索、小 shape 点)。"""
    ser = defaultdict(dict)
    for r in rows:
        if r["kind"] == "random" and r["phase"] in ("probe", "bisect"):
            ser[(r["shape_id"], r["dtype"], r["layout"])][r["B"]] = r
    return ser


def trans_str(known: dict) -> str:
    tr = transitions(known)
    pts = sorted(b for b in known if is_ok(known[b]) is not None)
    if not pts:
        return "判定不能"
    if not tr:
        cl = sorted({short(known[b]["cls"]) for b in pts})
        return f"なし({'/'.join(cl)})"
    out = []
    for lo, hi in tr:
        bad = known[hi] if is_ok(known[lo]) else known[lo]
        h = best_hyp(bad)
        out.append(f"{lo} {short(known[lo]['cls'])}→{hi} {short(known[hi]['cls'])}" + (f" ({h})" if h else ""))
    return "<br>".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tag", default="studio")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"), help="results ディレクトリ(v1 を読むなら ../v1/results)")
    ap.add_argument("--out", default=None, help="出力先(既定は <results>/summary/<host-tag>/versions.md)")
    ap.add_argument("--allow-scaled", action="store_true", help="--scale-log2 付きのテスト run も読む")
    args = ap.parse_args()
    results = Path(args.results).resolve()
    data = load(args.host_tag, args.runs.split(",") if args.runs else None, args.device, results, args.allow_scaled)
    vs = list(data)
    L = [f"# torch の版ごとの走査: {args.host_tag}", "",
         "基本 shape(out-256x64x256 = 出力が大きい系列、in-256x256x64 = 入力 a が大きい系列)、fp32 / fp16、4 レイアウト、乱数入力。"
         "各版は `run_versions.sh` の縮小設定(境界 B0-2〜B0+2 + 境界の間 2 点のグリッド + 二分探索、最終点は seed 0 の全要素比較、"
         "index 符号化は切り替わりの異常側 1 点で rc 系のみ)。2.14.0 は全体走査の run(グリッド 7 点、seed 0/1/2)から基本 shape だけを使う。"
         "`uv run python versions.py --host-tag " + args.host_tag + "` で再生成。", "",
         "## run", "", "| torch | run_id | Python | 実行数 | 経過(分) | 許容誤差 fp32 / fp16 | 分類の内訳 |", "|---|---|---|---|---|---|---|"]
    for v, (m, rows) in data.items():
        tol = m.get("tol", {})
        cnt = Counter(short(r["cls"]) for r in rows)
        L.append(f"| {v} | {m['run_id']} | {m.get('worker', {}).get('python')} | {len(rows)} | {m.get('elapsed_min')} | "
                 f"{tol.get('fp32', 0):.2g} / {tol.get('fp16', 0):.2g} | "
                 + ", ".join(f"{k} {n}" for k, n in sorted(cnt.items())) + " |")
    L.append("")

    sers = {v: series_rows(rows) for v, (m, rows) in data.items()}
    keys = sorted({k for s in sers.values() for k in s}, key=lambda k: (BASE.index(k[0]), k[1], LAYOUTS.index(k[2])))

    L += ["## 切り替わり(版 × 条件)", "",
          "正常/異常が異なる隣接点(B の粒度)。「なし」は探索範囲(B=4096〜65538)で切り替わりがない。括弧は誤答バッチで一致した読み違い仮説"
          "(stride = 転置 view の stride 無視、wrap = 2^32 要素での巻き戻り)と一致率。2.14.0 と違うセルは **太字**。", ""]
    L += ["| shape | dtype | レイアウト | " + " | ".join(vs) + " |", "|---|---|---|" + "---|" * len(vs)]
    diffs = []
    for k in keys:
        cells = []
        ref = trans_str(sers[REF][k]) if REF in sers and k in sers[REF] else None
        for v in vs:
            if k not in sers[v]:
                cells.append("-")
                continue
            t = trans_str(sers[v][k])
            same = ref is None or t.split(" (")[0] == ref.split(" (")[0] and \
                all(a.split(" (")[0] == b.split(" (")[0] for a, b in zip(t.split("<br>"), ref.split("<br>")))
            cells.append(t if same else f"**{t}**")
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")
    L.append("")

    # 共通の B で分類が 2.14.0 と違う点
    if REF in sers:
        for v in vs:
            if v == REF:
                continue
            for k in keys:
                a, b = sers[v].get(k, {}), sers[REF].get(k, {})
                for B in sorted(set(a) & set(b)):
                    if a[B]["cls"] != b[B]["cls"]:
                        e = a[B].get("error") or a[B].get("stderr_tail") or ""
                        diffs.append(f"| {v} | {k[0]} | {k[1]} | {k[2]} | {B} | {b[B]['cls']} | {a[B]['cls']} | "
                                     f"{best_hyp(a[B]) or e[-120:]} |")
    L += ["## 2.14.0 と分類が違う点(同じ B で比較)", ""]
    if diffs:
        L += ["| torch | shape | dtype | レイアウト | B | 2.14.0 | この版 | 仮説 / エラー |", "|---|---|---|---|---|---|---|---|"] + diffs
    else:
        L.append("- なし")
    L.append("")

    # 境界 ±2 の詳細(版 × 境界)
    L += ["## 理論境界の前後(B0-2, B0-1, B0, B0+1, B0+2 の分類)", ""]
    for sid in BASE:
        for dt in ("fp32", "fp16"):
            bs = boundaries(SHAPE[sid], dt)
            L += [f"**{sid} {dt}**", "", "| レイアウト | torch | " + " | ".join(f"{lab} (B0={b0})" for lab, b0 in bs) + " |",
                  "|---|---|" + "---|" * len(bs)]
            for lay in LAYOUTS:
                for v in vs:
                    kn = sers[v].get((sid, dt, lay), {})
                    if not kn:
                        continue
                    L.append(f"| {lay} | {v} | " + " | ".join(" ".join(short(kn[b0 + d]["cls"]) if b0 + d in kn else "-"
                                                                       for d in (-2, -1, 0, 1, 2)) for _, b0 in bs) + " |")
            L.append("")

    # 最終点の全要素比較と index 符号化
    L += ["## 最終点(全要素比較 seed 0)と index 符号化(rc)", "",
          "切り替わりの下・上の全要素比較の分類 / 最大相対誤差 / 閾値超え要素の割合、index 符号化(rc、切り替わりの異常側)の上位 delta。", "",
          "| torch | shape | dtype | レイアウト | B | 入力 | 分類 | 最大誤差 | 閾値超え割合 | 壊れたバッチの区間 | 仮説 | 上位の delta |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, (m, rows) in data.items():
        fin = [r for r in rows if r["phase"] == "final" and r["seed"] == 0 and r["cls"] != "ok"]
        for r in sorted(fin, key=lambda r: (BASE.index(r["shape_id"]), r["dtype"], LAYOUTS.index(r["layout"]), r["B"], r["kind"])):
            br = r.get("bad_ranges") or []
            brs = ",".join(f"[{a},{b})" for a, b in br[:2]) + ("…" if len(br) > 2 else "")
            top = ", ".join(f"{a}: {b}" for a, b in (r.get("idx_delta_top") or [])[:2])
            me = r.get("max_rel_err")
            L.append(f"| {v} | {r['shape_id']} | {r['dtype']} | {r['layout']} | {r['B']} | {r['kind']} | {short(r['cls'])} | "
                     f"{me:.2g} | {r.get('frac_elem_bad', 0):.2g} | {brs} | {best_hyp(r)} | {top} |"
                     if isinstance(me, float) else
                     f"| {v} | {r['shape_id']} | {r['dtype']} | {r['layout']} | {r['B']} | {r['kind']} | {short(r['cls'])} | - | - | - | - | "
                     f"{(r.get('error') or r.get('stderr_tail') or '')[-120:]} |")
    L += ["", "(異常だった最終点だけを載せる)", ""]

    # エラー・クラッシュ
    L += ["## エラー・クラッシュ(版ごと)", ""]
    for v, (m, rows) in data.items():
        c = Counter((r["cls"], (r.get("error") or r.get("stderr_tail") or "")[-200:]) for r in rows
                    if r["cls"] in ("error", "crash", "timeout", "input_corrupt", "truncated"))
        if c:
            L.append(f"- **{v}**: " + "; ".join(f"{cl} ×{n} `{msg}`" for (cl, msg), n in c.most_common()))
    out = Path(args.out) if args.out else results / "summary" / args.host_tag / "versions.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
