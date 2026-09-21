"""ホストをまたいで走査結果を行ごとに照合する(全系列・全項目。同じか違うかだけを書く)。

    uv run python compare_rows.py --source results:studio --source results:mbp-macos26 [--source ...] \\
        [--ref results:studio] [--all-pairs] [--out results/summary/hosts_rows.md]

--source ROOT:HOSTTAG は compare_hosts.py と同じ(ROOT/raw/HOSTTAG/ の <run_id>.jsonl と <run_id>.manifest.json を読む)。
既定は「基準(--ref。既定は最初の --source)vs 各 source」、--all-pairs なら source の全部の組。

compare_hosts.py は op = bmm の走査(レイアウト contig / bT / aT / slice)の分類だけを比べる。こちらは backward (op = bmm_bwd)、
all-256、--points の run(遠い点、all-256 の補足)、offset レイアウトも含めて、status = done・scale_log2 = 0 の run の全部を対象にする。

run は「系列」に分ける。系列の signature = (op, torch の公開版, mode, 較正以外の行の shape の集合, レイアウトの集合,
--points の run なら指定点のリストの sha256)。同じ source に同じ signature の run が複数あれば行を合併し、source の間で
signature が同じ系列どうしを照合する。片方にしかない系列は列挙するだけ(失敗にしない)。

同じ点 = (op, shape, dtype, レイアウト, 入力, B, seed, 比較, pad_batches) が同じ行。較正の行(phase = calibrate)は照合から除いて
数だけ数える。同じ点が 2 回あれば後の行(run_id の順、行の順)を残し、重複を数える。共通の点について、EXCLUDE の項目以外の
全項目が等しいかを見る(片方にしかない項目も違いに数える)。

終了コード: 対応した系列の中に、違う点・片方にしかない点・manifest の不一致(sha / tol / torch / points)があれば 1、なければ 0。
このスクリプトは読むだけで、GPU は使わない。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

from compare_hosts import cls_counts, join, os_str, parse_source, pub, read_jsonl, rev_str
from scan import ROOT
from versions import vkey

# 比較から外す項目(理由ごと)。ここに無い項目は全部比べる
EXCLUDE = {
    "識別(run ごとに違う)": ("run_id", "fingerprint", "host_tag", "ts"),
    "環境(ホストの記述そのもの)": ("os_release", "macos", "macos_build", "hw_model", "chip", "mem_bytes", "linux_distro", "python",
                       "git_rev", "git_dirty_code", "mem_limit_host_bytes", "mem_limit_dev_bytes", "mem_limit_note"),
    "時間とメモリの実測(実行ごとに揺れる)": ("time", "max_rss", "wall_sec", "dev_driver_alloc", "dev_current_alloc"),
    "crash の stderr(Apple の build root のパスが入る)": ("stderr_tail",),
}
EXCLUDED = {f for fs in EXCLUDE.values() for f in fs}
MAX_DIFF_POINTS = 50
MISSING = "(項目なし)"


def row_key(r: dict) -> tuple:
    """compare_hosts.row_key に op と pad_batches を足したもの(offset は同じ B で pad_batches が違う行がある)。"""
    return (r.get("op", "bmm"), r["shape_id"], r["dtype"], r["layout"], r["kind"], r["B"], r["seed"], r["compare"], r.get("pad_batches"))


def key_str(k: tuple) -> str:
    return f"{k[0]} / {k[1]} / {k[2]} / {k[3]} / {k[4]} / B={k[5]} / seed {k[6]} / {k[7]}" + (f" / pad_batches={k[8]}" if k[8] is not None else "")


def points_sha(points: list | None) -> str:
    """--points の指定点のリストの正規化 JSON(label を除き、キーと点を整列)の sha256。走査の run は ""。"""
    if not points:
        return ""
    norm = sorted(json.dumps({k: v for k, v in p.items() if k != "label"}, sort_keys=True) for p in points)
    return hashlib.sha256(json.dumps(norm).encode()).hexdigest()


def label_of(sig: tuple) -> str:
    op, v, mode, shapes, layouts, _ = sig
    if mode == "points":
        if all(s.startswith("all-") for s in shapes):
            name = "all-256 supplement"
        elif op == "bmm" and not any(s.startswith("all-") for s in shapes) and "offset" not in layouts:
            name = "far points"
        else:
            name = "other points"  # 試験用の小さい run など
    elif set(layouts) == {"offset"}:
        name = "offset"
    elif op == "bmm_bwd":
        name = "backward"
    elif set(shapes) == {"all-256x256x256"}:
        name = "all-256"
    else:
        name = "main sweep" if len(shapes) >= 3 else "reduced sweep"
    return f"{name} {v}"


LABEL_ORDER = ["main sweep", "reduced sweep", "backward", "all-256", "all-256 supplement", "far points", "offset", "other points"]


def sig_order(sig: tuple) -> tuple:
    return (LABEL_ORDER.index(label_of(sig).rsplit(" ", 1)[0]), vkey(sig[1]) if sig[1] else (), sig)


def sig_id(sig: tuple) -> str:
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:8]


class Series:
    """1 つの source の 1 つの系列(signature が同じ run を合併したもの)。"""

    def __init__(self):
        self.mans: list[dict] = []
        self.points: dict[tuple, dict] = {}  # 較正以外の行(点 → 行)
        self.n_rows = 0
        self.n_cal = 0
        self.n_dup = 0
        self.n_badlines = 0
        self.cls = Counter()


def load_source(root: Path, host: str) -> tuple[dict[tuple, Series], list[str]]:
    """signature → Series と、使わなかった run の一覧(理由つき)。"""
    d = root / "raw" / host
    out: dict[tuple, Series] = {}
    skipped = []
    for p in sorted(d.glob("*.manifest.json")):  # run_id は時刻で始まるので、名前の順 = 時間の順
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            skipped.append(f"{p.name}: manifest が読めない")
            continue
        rid = m.get("run_id") or p.name.removesuffix(".manifest.json")
        if m.get("status") != "done":
            skipped.append(f"{rid}: status={m.get('status')}")
            continue
        if m.get("scale_log2"):
            skipped.append(f"{rid}: scale_log2={m['scale_log2']} のテスト run")
            continue
        rows, bad = read_jsonl(p.with_name(f"{rid}.jsonl"))
        if any(r.get("scale_log2") for r in rows):
            skipped.append(f"{rid}: scale_log2 付きの行がある")
            continue
        body = [r for r in rows if r["phase"] != "calibrate"]
        if not body:
            skipped.append(f"{rid}: 較正以外の行がない")
            continue
        v = pub(m.get("torch_req") or (m.get("worker") or {}).get("torch") or next((r.get("torch") for r in rows if r.get("torch")), ""))
        mode = m.get("mode") or ("points" if m.get("points") else "sweep")  # 古い manifest には mode が無い
        sig = (m.get("op") or "bmm", v, mode, tuple(sorted({r["shape_id"] for r in body})), tuple(sorted({r["layout"] for r in body})),
               points_sha(m.get("points")) if mode == "points" else "")
        g = out.setdefault(sig, Series())
        g.mans.append(m)
        g.n_badlines += bad
        g.n_rows += len(rows)
        g.n_cal += len(rows) - len(body)
        for r in rows:
            g.cls[r["cls"]] += 1
        for r in body:
            k = row_key(r)
            g.n_dup += k in g.points
            g.points[k] = r
    return out, skipped


def man_facts(g: Series) -> dict[str, set]:
    """manifest の照合に使う項目(合併した run の値の集合)。"""
    code = [m.get("code") or {} for m in g.mans]
    return {"sha_scan": {c.get("sha_scan") for c in code}, "sha_worker": {c.get("sha_worker") for c in code},
            "tol": {json.dumps(m.get("tol"), sort_keys=True) for m in g.mans},
            "torch": {pub((m.get("worker") or {}).get("torch")) for m in g.mans},
            "points": {points_sha(m.get("points")) for m in g.mans}}


def compare(a: dict[tuple, dict], b: dict[tuple, dict]) -> tuple[dict[tuple, list[str]], Counter, Counter]:
    """共通の点の照合。返り値: (違う点 → 違った項目, 違った項目の数, 外した項目のうち値が違った点の数)。"""
    diff: dict[tuple, list[str]] = {}
    fields, excl = Counter(), Counter()
    for k in sorted(set(a) & set(b), key=str):
        ra, rb = a[k], b[k]
        bad = []
        for f in sorted(set(ra) | set(rb)):
            if ra.get(f, MISSING) != rb.get(f, MISSING):
                if f in EXCLUDED:
                    excl[f] += 1
                else:
                    bad.append(f)
        if bad:
            diff[k] = bad
            fields.update(bad)
    return diff, fields, excl


def val_str(v) -> str:
    s = v if v == MISSING else json.dumps(v, ensure_ascii=False)
    s = s.replace("|", "\\|").replace("\n", " ")
    return f"`{s[:200]}`" + ("…" if len(s) > 200 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="ROOT:HOSTTAG(ROOT/raw/HOSTTAG/ を読む)。何個でも")
    ap.add_argument("--ref", default=None, help="比較の基準 ROOT:HOSTTAG(既定は最初の --source)")
    ap.add_argument("--all-pairs", action="store_true", help="基準 vs 各 source ではなく、source の全部の組を照合する")
    ap.add_argument("--out", default=str(ROOT / "results" / "summary" / "hosts_rows.md"))
    args = ap.parse_args()

    srcs = list(dict.fromkeys(parse_source(s) for s in args.source))
    ref = parse_source(args.ref) if args.ref else srcs[0]
    if ref not in srcs:
        srcs.insert(0, ref)
    tags = [h for _, h in srcs]
    name = {s: (s[1] if tags.count(s[1]) == 1 else f"{s[1]} ({s[0]})") for s in srcs}  # 同じ host-tag が別の ROOT にあれば ROOT も書く
    data, skipped = {}, {}
    for s in srcs:
        data[s], skipped[s] = load_source(s[0], s[1])
        if not data[s]:
            print(f"# {name[s]}: 使える run がない({s[0] / 'raw' / s[1]})")
    pairs = list(itertools.combinations(srcs, 2)) if args.all_pairs else [(ref, s) for s in srcs if s != ref]

    L = ["# ホストをまたぐ行ごとの照合", "",
         f"基準: **{name[ref]}**。組: {', '.join(f'{name[a]} vs {name[b]}' for a, b in pairs) or '-'}。"
         "`" + "uv run python compare_rows.py " + " ".join(f"--source {s}" for s in args.source)
         + (f" --ref {args.ref}" if args.ref else "") + (" --all-pairs" if args.all_pairs else "") + "` で生成(`v2/` で実行)。", "",
         "同じ点 = (op, shape, dtype, レイアウト, 入力の種類, B, seed, 比較の方法, pad_batches) が同じ行(op が無い行は bmm、pad_batches が無い行は None)。"
         "status = done、scale_log2 = 0 の run の全部を使う。較正の行(phase = calibrate)は照合から除き、数だけ数える。phase は点の定義に入れない。"
         "同じ点が 2 回あれば後の行を残す。共通の点について、下の「外した項目」以外の全項目が等しいかを見る(片方にしかない項目も違いに数える)。"
         "同じか違うかだけを書き、解釈はしない。", "",
         "系列 = (op, torch の公開版, mode, 較正以外の行の shape の集合, レイアウトの集合, --points の run なら指定点のリストの sha256)が同じ run の集まり"
         "(指定点の sha256 は label を除き、整列した JSON から計算する)。系列の名前は表示用で、source の間の対応づけは signature(表の sig 列はその sha256 の先頭 8 桁)で行う。", "",
         "外した項目:", ""]
    L += [f"- {why}: " + ", ".join(f"`{f}`" for f in fs) for why, fs in EXCLUDE.items()] + [""]

    # 1. 対象
    L += ["## 対象", "",
          "| host-tag | 系列 | sig | run | status | 機種 / チップ / OS | torch | 行数 | うち較正 | 重複 | 比較に使う点 | 分類の内訳(全行) | revision | sha_scan / sha_worker |",
          "|" + "---|" * 14]
    for s in srcs:
        for sig in sorted(data[s], key=sig_order):
            g = data[s][sig]
            env = [m.get("env") or {} for m in g.mans]
            bad = f"(読めない行 {g.n_badlines})" if g.n_badlines else ""
            L.append(
                f"| {name[s]} | {label_of(sig)} | {sig_id(sig)} | {join((m.get('run_id') for m in g.mans), '<br>')} | {join(m.get('status') for m in g.mans)} | "
                f"{join(e.get('hw_model') for e in env)} / {join(e.get('chip') for e in env)} / {join(os_str(e) for e in env)} | "
                f"{join((m.get('worker') or {}).get('torch') or m.get('torch_req') for m in g.mans)} | {g.n_rows}{bad} | {g.n_cal} | {g.n_dup} | "
                f"{len(g.points)} | {cls_counts(g.cls)} | {join(rev_str(m) for m in g.mans)} | "
                f"{join(((m.get('code') or {}).get('sha_scan') or '-')[:12] for m in g.mans)} / "
                f"{join(((m.get('code') or {}).get('sha_worker') or '-')[:12] for m in g.mans)} |")
    L += ["", "行数 = run の行の全部(較正と重複を含む)。重複 = 同じ点が 2 回あった数(後の行を使う)。比較に使う点 = 較正を除き、重複をまとめた後の点の数。", ""]

    # 2. 組ごとの照合
    L += ["## 照合", "",
          "A = 組の左、B = 組の右。manifest = その系列の run の sha_scan / sha_worker / tol / torch(公開版)/ points が A と B で同じか"
          "(違う項目があれば名前を書く)。", ""]
    fail = False
    summary, details, excl_lines, unmatched = [], [], [], []
    for sa, sb in pairs:
        A, B = data[sa], data[sb]
        L += [f"### {name[sa]} vs {name[sb]}", "",
              "| 系列 | sig | 共通の点 | A にしかない点 | B にしかない点 | 全項目が同じ点 | 違う点 | 違った項目(点の数) | manifest |", "|" + "---|" * 9]
        tot = Counter()
        excl_tot = Counter()
        for sig in sorted(set(A) & set(B), key=sig_order):
            a, b = A[sig].points, B[sig].points
            diff, fields, excl = compare(a, b)
            excl_tot.update(excl)
            fa, fb = man_facts(A[sig]), man_facts(B[sig])
            mbad = [f for f in fa if fa[f] != fb[f]]
            n_common, only_a, only_b = len(set(a) & set(b)), sorted(set(a) - set(b), key=str), sorted(set(b) - set(a), key=str)
            tot.update(series=1, common=n_common, only_a=len(only_a), only_b=len(only_b), diff=len(diff), man=int(bool(mbad)))
            L.append(f"| {label_of(sig)} | {sig_id(sig)} | {n_common} | {len(only_a)} | {len(only_b)} | {n_common - len(diff)} | {len(diff)} | "
                     + (", ".join(f"`{f}` {n}" for f, n in sorted(fields.items())) or "-") + " | " + ("同じ" if not mbad else "**違う**: " + ", ".join(mbad)) + " |")
            if diff or only_a or only_b or mbad:
                details += [f"### {name[sa]} vs {name[sb]}: {label_of(sig)}", ""]
                for f in mbad:
                    details.append(f"- manifest の `{f}`: A {val_str(sorted(map(str, fa[f])))}、B {val_str(sorted(map(str, fb[f])))}")
                for who, ks in (("A", only_a), ("B", only_b)):
                    details += [f"- {who} にしかない点: {key_str(k)}" for k in ks[:MAX_DIFF_POINTS]]
                    details += [f"- {who} にしかない点: ほか {len(ks) - MAX_DIFF_POINTS} 点"] if len(ks) > MAX_DIFF_POINTS else []
                if diff:
                    details += ["", "| 点 | 項目 | A | B |", "|---|---|---|---|"]
                    for k in list(diff)[:MAX_DIFF_POINTS]:
                        details += [f"| {key_str(k)} | `{f}` | {val_str(a[k].get(f, MISSING))} | {val_str(b[k].get(f, MISSING))} |" for f in diff[k]]
                    details += ["", f"(違う点 {len(diff)} のうち最初の {MAX_DIFF_POINTS} 点)"] if len(diff) > MAX_DIFF_POINTS else []
                details.append("")
        L += [f"| **合計({tot['series']} 系列)** | | {tot['common']} | {tot['only_a']} | {tot['only_b']} | {tot['common'] - tot['diff']} | {tot['diff']} | | "
              + ("同じ" if not tot["man"] else f"**違う系列 {tot['man']}**") + " |", ""]
        bad = tot["diff"] or tot["only_a"] or tot["only_b"] or tot["man"]
        fail |= bool(bad)
        summary.append(f"- **{name[sa]} vs {name[sb]}**: 対応する系列 {tot['series']}、共通の点 {tot['common']}、全項目が同じ {tot['common'] - tot['diff']}、"
                       f"違う {tot['diff']}、片方にしかない点 {tot['only_a']} + {tot['only_b']}、manifest が違う系列 {tot['man']}")
        excl_lines += [f"- {name[sa]} vs {name[sb]}(共通の {tot['common']} 点): "
                       + (", ".join(f"`{f}` {excl_tot[f]}" for fs in EXCLUDE.values() for f in fs if excl_tot[f]) or "なし")]
        for who, X, Y in ((sa, A, B), (sb, B, A)):
            unmatched += [f"- {name[sa]} vs {name[sb]}: {name[who]} にしかない系列 {label_of(sig)}(sig {sig_id(sig)}、"
                          f"{join(m.get('run_id') for m in X[sig].mans)}、{len(X[sig].points)} 点、shape {', '.join(sig[3])}、レイアウト {', '.join(sig[4])})"
                          for sig in sorted(set(X) - set(Y), key=sig_order)]
    if not pairs:
        L += ["- 比べる組がない", ""]

    # 3. 違う点
    L += ["## 違う点・片方にしかない点", ""] + (details or ["- なし", ""])

    # 4. 外した項目の実際の違い
    L += ["## 外した項目のうち、実際に値が違った点の数", "",
          "共通の点のうち、外した項目の値が A と B で違った点の数(項目ごと。片方にしかない場合も数える)。ここに無い外した項目は、全部の共通の点で値が同じだった。", ""]
    L += (excl_lines or ["- 比べる組がない"]) + [""]

    # 5. 使わなかった run、対応のない系列
    L += ["## 使わなかった run、対応のない系列", ""]
    L += ([f"- 使わなかった run: {name[s]}: {x}" for s in srcs for x in skipped[s]] or ["- 使わなかった run: なし"])
    L += (unmatched or ["- 対応のない系列: なし"]) + [""]

    L += ["## まとめ", ""] + (summary or ["- 比べる組がない"]) + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")
    for x in summary + unmatched:
        print(x)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
