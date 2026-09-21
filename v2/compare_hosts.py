"""ホスト(機種・macOS の版・CUDA の対照)をまたいで走査結果を比べる(ホスト × 条件の表)。

    uv run python compare_hosts.py --source ../v1/results:studio --source results:mbp-macos26 [--source ...] \\
        [--ref ../v1/results:studio] [--torch 2.14.0[,2.9.1,...]] [--allow-running] [--out results/summary/hosts.md]

--source ROOT:HOSTTAG は ROOT/raw/HOSTTAG/ の <run_id>.jsonl と <run_id>.manifest.json を読む(何個でも)。
--ref は比較の基準(既定は最初の --source)。--torch は比べる torch の版(既定は基準にある版の全部)。
torch の版は公開部分で比べる(2.14.0+cu130 = 2.14.0)。出力の既定は v2/results/summary/hosts.md。

run の選び方(source ごと・torch の版ごと): manifest の status = done(--allow-running なら running / interrupted も。
報告に「途中」と書く)、scale_log2 = 0、op = bmm、走査の行を 1 つ以上含む run の全部。同じ版の run が複数あれば
(A100 のチャンクなど)行を合併する。同じ点(shape, dtype, レイアウト, 入力, B, seed, 比較)が 2 回あれば後の run の行を残し、
重複の数を数える。行は op = bmm(キーなしも bmm)、phase ≠ point、レイアウト contig / bT / aT / slice だけを使う。
較正の行は結果の表に入れない(較正の節だけで使う)。

このスクリプトは読むだけで、GPU は使わない。分類の解釈はしない(同じか違うかだけを書く)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scan import LAYOUTS, ROOT, SHAPE
from versions import best_hyp, short, trans_str, vkey

PARTIAL = ("running", "interrupted")
SHAPE_ORDER = list(SHAPE)
ABNORMAL_NO_DETAIL = ("error", "crash", "timeout", "truncated", "input_corrupt")


def pub(v: str | None) -> str:
    """torch の版の公開部分(2.14.0+cu130 → 2.14.0)。"""
    return (v or "").split("+")[0]


def row_key(r: dict) -> tuple:
    return (r["shape_id"], r["dtype"], r["layout"], r["kind"], r["B"], r["seed"], r["compare"])


def series_order(k: tuple) -> tuple:
    sid, dt, lay = k[0], k[1], k[2]
    return (SHAPE_ORDER.index(sid) if sid in SHAPE_ORDER else len(SHAPE_ORDER), sid, dt,
            LAYOUTS.index(lay) if lay in LAYOUTS else len(LAYOUTS))


def read_jsonl(p: Path) -> tuple[list[dict], int]:
    """JSONL を読む。途中までしか書かれていない行(走査中のコピーの最終行など)は捨てて数える。"""
    rows, bad = [], 0
    if not p.exists():
        return rows, bad
    for line in p.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(r, dict) and all(k in r for k in ("shape_id", "dtype", "layout", "kind", "B", "seed", "compare", "cls", "phase")):
            rows.append(r)
        else:
            bad += 1
    return rows, bad


def usable(r: dict) -> bool:
    return r.get("op", "bmm") == "bmm" and r["phase"] != "point" and r["layout"] in LAYOUTS and not r.get("scale_log2")


class Group:
    """1 つの source の 1 つの torch の版(合併した run の集まり)。"""

    def __init__(self):
        self.mans: list[dict] = []
        self.points: dict[tuple, dict] = {}  # 較正以外の行(点 → 行)
        self.cal: list[dict] = []
        self.n_rows = 0
        self.n_dup = 0
        self.n_badlines = 0
        self.cls = Counter()

    @property
    def partial(self) -> bool:
        return any(m.get("status") != "done" for m in self.mans)


def load_source(root: Path, host: str, allow_running: bool) -> tuple[dict[str, Group], list[str]]:
    """torch の版 → Group と、使わなかった run の一覧(理由つき)。"""
    d = root / "raw" / host
    groups: dict[str, Group] = defaultdict(Group)
    skipped = []
    for p in sorted(d.glob("*.manifest.json")):  # run_id は時刻で始まるので、名前の順 = 時間の順
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            skipped.append(f"{p.name}: manifest が読めない")
            continue
        rid = m.get("run_id") or p.name.removesuffix(".manifest.json")
        st = m.get("status")
        if st != "done" and not (allow_running and st in PARTIAL):
            skipped.append(f"{rid}: status={st}")
            continue
        if m.get("scale_log2"):
            skipped.append(f"{rid}: scale_log2={m['scale_log2']} のテスト run")
            continue
        if m.get("op", "bmm") != "bmm":
            skipped.append(f"{rid}: op={m.get('op')}")
            continue
        rows, bad = read_jsonl(p.with_name(f"{rid}.jsonl"))
        if any(r.get("scale_log2") for r in rows):
            skipped.append(f"{rid}: scale_log2 付きの行がある")
            continue
        rows = [r for r in rows if usable(r)]
        if not any(r["phase"] != "calibrate" for r in rows):
            skipped.append(f"{rid}: 走査の行がない(--points の run か、較正だけ)")
            continue
        v = pub(m.get("torch_req") or (m.get("worker") or {}).get("torch") or next((r.get("torch") for r in rows if r.get("torch")), ""))
        g = groups[v]
        g.mans.append(m)
        g.n_badlines += bad
        for r in rows:
            g.n_rows += 1
            g.cls[r["cls"]] += 1
            if r["phase"] == "calibrate":
                g.cal.append(r)
                continue
            k = row_key(r)
            g.n_dup += k in g.points
            g.points[k] = r
    return dict(groups), skipped


# ---------------------------------------------------------------------------- 書式

def uniq(xs) -> list:
    return list(dict.fromkeys(x for x in xs if x not in (None, "")))


def join(xs, sep=" / ") -> str:
    return sep.join(str(x) for x in uniq(xs)) or "-"


def os_str(env: dict) -> str:
    if env.get("os", "Darwin") == "Darwin":
        return f"macOS {env.get('macos')} ({env.get('macos_build')})"
    return f"{env.get('os')} {env.get('os_release')} ({env.get('linux_distro')})"


def gpu_str(m: dict) -> str:
    env, wk = m.get("env") or {}, m.get("worker") or {}
    name = env.get("gpu_name") or wk.get("gpu_name")
    return f"{name} / driver {env.get('gpu_driver', '-')} / CUDA {wk.get('cuda', '-')}" if name else ""


def rev_str(m: dict) -> str:
    c = m.get("code") or {}
    rev = (c.get("git_rev") or "-")[:7]
    return rev + (" (dirty)" if c.get("git_dirty_code") else "")


def fmt_tol(vals: list[float]) -> str:
    vals = sorted(set(vals))
    if not vals:
        return "-"
    return f"{vals[0]:.3g}" if len(vals) == 1 else f"{vals[0]:.3g}〜{vals[-1]:.3g}({len(vals)} 通り)"


def cls_counts(c: Counter) -> str:
    out = Counter()
    for k, n in c.items():
        out[short(k)] += n
    return ", ".join(f"{k} {n}" for k, n in sorted(out.items()))


def series_rows(points: dict[tuple, dict]) -> dict[tuple, dict[int, dict]]:
    """(shape, dtype, レイアウト) → {B: 行}。versions.py と同じく乱数入力の探索点と二分探索だけ。"""
    ser: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in points.values():
        if r["kind"] == "random" and r["phase"] in ("probe", "bisect"):
            ser[(r["shape_id"], r["dtype"], r["layout"])][r["B"]] = r
    return ser


def strip_hyp(t: str) -> list[str]:
    return [x.split(" (")[0] for x in t.split("<br>")]


def bad_ranges_str(r: dict) -> str:
    br = r.get("bad_ranges") or []
    n = r.get("n_bad_ranges") or len(br)
    return ",".join(f"[{a},{b})" for a, b in br[:2]) + (f"…(計 {n})" if n > 2 else "")


def hyp_name(r: dict) -> str:
    h = best_hyp(r)
    return h.rsplit(" ", 1)[0] if h else ""


def detail(rows: list[dict]) -> str:
    """分類が違う点の補足: wrong なら一致した仮説(一致率の範囲)と壊れたバッチの区間、それ以外の異常ならエラーの末尾。"""
    r0 = rows[0]
    if r0["cls"] == "wrong":
        rates = [max(v for v in (r.get("hyp_match") or {}).values() if v) for r in rows if hyp_name(r)]
        h = hyp_name(r0)
        hs = (f"{h} {min(rates):.2f}" + (f"〜{max(rates):.2f}" if max(rates) != min(rates) else "")) if h and rates else "仮説なし"
        br = bad_ranges_str(r0)
        if len(rows) > 1 and bad_ranges_str(rows[-1]) != br:
            br += f" … B={rows[-1]['B']}: {bad_ranges_str(rows[-1])}"
        return f"{hs}; 壊れたバッチ {br or '-'}"
    if r0["cls"] in ABNORMAL_NO_DETAIL:
        e = (r0.get("error") or r0.get("stderr_tail") or "").strip().splitlines()
        return f"`{e[-1][-100:]}`" if e else ""
    return ""


def diff_lines(ref: dict[tuple, dict], src: dict[tuple, dict]) -> tuple[list[str], int, Counter]:
    """共通の点で分類が違うものを、系列ごとに B の並びでまとめた表の行。返り値: (行, 違う点の数, (基準の分類, source の分類) の数)。"""
    common = set(ref) & set(src)
    by_ser: dict[tuple, list[int]] = defaultdict(list)  # (shape, dtype, layout, kind, seed, compare) → B
    for k in common:
        by_ser[(k[0], k[1], k[2], k[3], k[5], k[6])].append(k[4])
    pairs = Counter()
    merged: dict[tuple, list[int]] = defaultdict(list)  # seed 以外が同じ行は 1 行にまとめる
    for s, Bs in by_ser.items():
        run: list[tuple[dict, dict]] = []
        sig_prev = None

        def flush():
            if not run:
                return
            a0, b0 = run[0]
            Bt = str(a0["B"]) if len(run) == 1 else f"{a0['B']}〜{run[-1][0]['B']}({len(run)} 点)"
            dr, ds = detail([a for a, _ in run]), detail([b for _, b in run])
            det = " / ".join(x for x in ((f"基準: {dr}" if dr else ""), (f"source: {ds}" if ds else "")) if x)
            merged[(s[0], s[1], s[2], s[3], s[5], Bt, a0["cls"], b0["cls"], det, a0["B"])].append(s[4])
            run.clear()

        for B in sorted(Bs):
            k = (s[0], s[1], s[2], s[3], B, s[4], s[5])
            a, b = ref[k], src[k]
            sig = (a["cls"], b["cls"], hyp_name(a), hyp_name(b)) if a["cls"] != b["cls"] else None
            if sig != sig_prev:
                flush()
            if sig:
                run.append((a, b))
                pairs[(a["cls"], b["cls"])] += 1
            sig_prev = sig
        flush()
    lines = []
    for (sid, dt, lay, kind, cmp_, Bt, ca, cb, det, b_first), seeds in sorted(
            merged.items(), key=lambda kv: (series_order(kv[0][:3]), kv[0][3], kv[0][4], kv[0][9])):
        lines.append(f"| {sid} | {dt} | {lay} | {kind} | {cmp_} | {','.join(str(x) for x in sorted(seeds))} | {Bt} | "
                     f"{short(ca)} → {short(cb)} | {det} |")
    return lines, sum(pairs.values()), pairs


def only_str(keys: set, points: dict[tuple, dict]) -> str:
    if not keys:
        return "0"
    c = Counter((points[k]["phase"], "random" if k[3] == "random" else "index", k[6]) for k in keys)
    return f"{len(keys)}(" + ", ".join(f"{ph} {kd} {cm} {n}" for (ph, kd, cm), n in sorted(c.items())) + ")"


# ---------------------------------------------------------------------------- 本体

def parse_source(s: str) -> tuple[Path, str]:
    root, sep, host = s.rpartition(":")
    if not sep or not root or not host:
        raise SystemExit(f"--source / --ref は ROOT:HOSTTAG の形で書く: {s}")
    return Path(root).resolve(), host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="ROOT:HOSTTAG(ROOT/raw/HOSTTAG/ を読む)。何個でも")
    ap.add_argument("--ref", default=None, help="比較の基準 ROOT:HOSTTAG(既定は最初の --source)")
    ap.add_argument("--torch", default=None, help="比べる torch の版(カンマ区切り。既定は基準にある版の全部)")
    ap.add_argument("--allow-running", action="store_true", help="status = running / interrupted の run も読む(報告に「途中」と書く)")
    ap.add_argument("--out", default=str(ROOT / "results" / "summary" / "hosts.md"))
    args = ap.parse_args()

    srcs = list(dict.fromkeys(parse_source(s) for s in args.source))
    ref = parse_source(args.ref) if args.ref else srcs[0]
    if ref not in srcs:
        srcs.insert(0, ref)
    tags = [h for _, h in srcs]
    name = {s: (s[1] if tags.count(s[1]) == 1 else f"{s[1]} ({s[0]})") for s in srcs}  # 同じ host-tag が別の ROOT にあれば ROOT も書く
    data, skipped = {}, {}
    for s in srcs:
        data[s], skipped[s] = load_source(s[0], s[1], args.allow_running)
        if not data[s]:
            print(f"# {name[s]}: 使える run がない({s[0] / 'raw' / s[1]})")
    vs = [pub(v) for v in args.torch.split(",")] if args.torch else sorted(data[ref], key=vkey)
    if not vs:
        raise SystemExit(f"基準 {name[ref]} に使える run がない(--torch で版を指定するか、--allow-running を付ける)")
    others = [s for s in srcs if s != ref]

    L = ["# ホストをまたぐ比較", "",
         f"基準: **{name[ref]}**。torch の版: {', '.join(vs)}。"
         "`" + "uv run python compare_hosts.py " + " ".join(f"--source {s}" for s in args.source)
         + (f" --ref {args.ref}" if args.ref else "") + (f" --torch {args.torch}" if args.torch else "")
         + (" --allow-running" if args.allow_running else "") + "` で生成(`v2/` で実行)。", "",
         "同じ点 = (shape, dtype, レイアウト, 入力の種類, B, seed, 比較の方法) が同じ実行。op = bmm、レイアウト contig / bT / aT / slice、"
         "phase ≠ point の行だけを使う。較正の行は「較正」の節だけで使う。分類が同じか違うかだけを書き、解釈はしない。", ""]

    # 1. 対象
    L += ["## 対象", "",
          "| host-tag | torch | device | 機種 | チップ | RAM (GiB) | OS | GPU / driver / CUDA | Python | run(合併した数) | status | 行数 | うち較正 | 重複 | 比較に使う点 | "
          "分類の内訳(全行) | revision | sha_scan / sha_worker |",
          "|" + "---|" * 18]
    for s in srcs:
        for v in vs:
            g = data[s].get(v)
            if not g:
                L.append(f"| {name[s]} | {v} | " + "n/a | " * 16)
                continue
            ms = g.mans
            env = [m.get("env") or {} for m in ms]
            rids = [m.get("run_id") for m in ms]
            runs = rids[0] if len(rids) == 1 else f"{rids[0]} … {rids[-1]}"
            st = Counter(m.get("status") for m in ms)
            sts = ", ".join(f"{k} {n}" if len(ms) > 1 else str(k) for k, n in st.items()) + ("(**途中**)" if g.partial else "")
            bad = f"(読めない行 {g.n_badlines})" if g.n_badlines else ""
            L.append(
                f"| {name[s]} | {join((m.get('worker') or {}).get('torch') or m.get('torch_req') for m in ms)} | "
                f"{join(m.get('device', 'mps') for m in ms)} | {join(e.get('hw_model') for e in env)} | {join(e.get('chip') for e in env)} | "
                f"{join(round(e.get('mem_bytes', 0) / 2**30) for e in env)} | {join(os_str(e) for e in env)} | {join(gpu_str(m) for m in ms)} | "
                f"{join((m.get('worker') or {}).get('python') for m in ms)} | {runs} ({len(ms)}) | {sts} | {g.n_rows}{bad} | {len(g.cal)} | "
                f"{g.n_dup} | {len(g.points)} | {cls_counts(g.cls)} | {join(rev_str(m) for m in ms)} | "
                f"{join(((m.get('code') or {}).get('sha_scan') or '-')[:12] for m in ms)} / "
                f"{join(((m.get('code') or {}).get('sha_worker') or '-')[:12] for m in ms)} |")
    L += ["", "行数 = 条件に合う行の全部(較正と重複を含む)。重複 = 同じ点が複数の run にあった数(後の run の行を使う。較正の行は数えない)。"
          "比較に使う点 = 較正を除き、重複をまとめた後の点の数。", ""]
    if any(g.partial for s in srcs for g in data[s].values()):
        L += ["**途中** と書いた source は走査が終わっていない(`--allow-running`)。まだ実行されていない点は「片方にしかない点」に数えられ、"
              "切り替わりの表は実行済みの点だけから作られる。", ""]
    sk = [f"- {name[s]}: {x}" for s in srcs for x in skipped[s]]
    if sk:
        L += ["使わなかった run:", ""] + sk + [""]

    # 2. 較正
    L += ["## 較正", "",
          "較正の実行(contig、乱数入力、2^28 要素、seed 0/1/2、全要素比較)の最大相対誤差の最大と、manifest の許容誤差。"
          "較正する shape は版で違う(v1 は基本 shape だけ、v2 はその run の shape の全部)ので、機種の比較は下の shape ごとの表で見る。"
          "基準と値が違うセルは **太字**。", "",
          "| torch | dtype | " + " | ".join(f"{name[s]}: 較正誤差の最大 / 許容誤差" for s in srcs) + " |", "|---|---|" + "---|" * len(srcs)]
    cal = {}  # (source, torch) → {(dtype, shape): 最大誤差}
    for s in srcs:
        for v in vs:
            g = data[s].get(v)
            d: dict[tuple, float] = {}
            for r in (g.cal if g else []):
                if isinstance(r.get("max_rel_err"), float):
                    k = (r["dtype"], r["shape_id"])
                    d[k] = max(d.get(k, 0.0), r["max_rel_err"])
            cal[(s, v)] = d
    for v in vs:
        dts = sorted({k[0] for s in srcs for k in cal[(s, v)]} | {dt for s in srcs if data[s].get(v) for m in data[s][v].mans for dt in (m.get("tol") or {})})
        for dt in dts:
            cells = []
            for s in srcs:
                g = data[s].get(v)
                if not g:
                    cells.append("n/a")
                    continue
                errs = [e for (d_, _), e in cal[(s, v)].items() if d_ == dt]
                tols = [m["tol"][dt] for m in g.mans if dt in (m.get("tol") or {})]
                cells.append((f"{max(errs):.6e}" if errs else "-") + " / " + fmt_tol(tols))
            L.append(f"| {v} | {dt} | " + " | ".join(cells) + " |")
    L += ["", "shape ごとの較正誤差の最大(3 seed の最大):", "",
          "| torch | dtype | shape | " + " | ".join(name[s] for s in srcs) + " |", "|---|---|---|" + "---|" * len(srcs)]
    for v in vs:
        for k in sorted({k for s in srcs for k in cal[(s, v)]}, key=lambda k: (k[0], series_order((k[1], k[0], "contig")))):
            rv = cal[(ref, v)].get(k)
            cells = []
            for s in srcs:
                e = cal[(s, v)].get(k)
                t = "n/a" if e is None else f"{e:.6e}"
                cells.append(f"**{t}**" if e is not None and rv is not None and e != rv else t)
            L.append(f"| {v} | {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. 切り替わり
    L += ["## 切り替わり(ホスト × 条件)", "",
          "乱数入力 seed 0 の探索点と二分探索から、正常/異常が異なる隣接点(B の粒度)。書き方は `versions.md` と同じ: 「なし」は実行した点の範囲で"
          "切り替わりがない。括弧は誤答バッチで一致した読み違い仮説と一致率。`n/a` = その source にこの系列の行がない"
          "(メモリに入らない dtype を回していない、など。違いには数えない)。基準と違うセルは **太字**(仮説の括弧は比べない)。"
          "途中の source のセルには実行済みの点の数を `[n 点]` で付ける(基準の点の数は見出しの右の列)。", ""]
    sers = {(s, v): series_rows(data[s][v].points) if data[s].get(v) else {} for s in srcs for v in vs}
    trans_diff = Counter()
    for v in vs:
        cols = [s for s in srcs if data[s].get(v)]
        L += [f"### torch {v}", ""]
        if not cols:
            L += ["- この版の run はどの source にもない", ""]
            continue
        keys = sorted({k for s in cols for k in sers[(s, v)]}, key=series_order)
        L += ["| shape | dtype | レイアウト | 基準の点の数 | " + " | ".join(name[s] for s in srcs) + " |", "|---|---|---|---|" + "---|" * len(srcs)]
        for k in keys:
            rk = sers[(ref, v)].get(k)
            rt = trans_str(rk) if rk else None
            cells = []
            for s in srcs:
                kn = sers[(s, v)].get(k)
                if not kn:
                    cells.append("n/a")
                    continue
                t = trans_str(kn)
                same = rt is None or strip_hyp(t) == strip_hyp(rt)
                trans_diff[(s, v)] += not same
                t += f" [{len(kn)} 点]" if data[s][v].partial else ""
                cells.append(t if same else f"**{t}**")
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rk) if rk else '-'} | " + " | ".join(cells) + " |")
        L.append("")

    # 4. 点ごとの比較
    L += [f"## 点ごとの比較(基準 {name[ref]} と同じ点で)", "",
          "index 符号化の入力も含む。「分類が違う点」は、系列(shape, dtype, レイアウト, 入力, 比較)ごとに、共通の点の B の並びで連続するものを "
          "1 行にまとめた(間に分類が同じ共通の点が入ると行を分ける。「a〜b(n 点)」は a から b までの共通の点 n 個で、間の B の全部ではない)。"
          "seed だけが違う行は 1 行にまとめた。「片方にしかない点」の内訳は phase・入力(random / index)・比較の方法ごとの数。", ""]
    summary = []
    for s in others:
        for v in vs:
            gr, gs = data[ref].get(v), data[s].get(v)
            if not gr or not gs:
                if gs or gr:
                    summary.append(f"- **{name[s]}**(torch {v}): {'基準' if not gr else 'この source'} に run がないので比べられない")
                continue
            a, b = gr.points, gs.points
            common = set(a) & set(b)
            lines, n_diff, pairs = diff_lines(a, b)
            part = "(**途中**)" if gs.partial or gr.partial else ""
            L += [f"### {name[s]} vs {name[ref]}(torch {v}){part}", "",
                  f"- 共通の点: {len(common)}(基準 {len(a)} 点、{name[s]} {len(b)} 点)",
                  f"- 分類が同じ: {len(common) - n_diff}、違う: {n_diff}"
                  + (f"(基準 → source: " + ", ".join(f"{short(x)} → {short(y)} {n}" for (x, y), n in pairs.most_common()) + ")" if pairs else ""),
                  "- 共通の点の基準側の分類: " + (cls_counts(Counter(a[k]["cls"] for k in common)) or "-")
                  + f"、{name[s]} 側: " + (cls_counts(Counter(b[k]["cls"] for k in common)) or "-"),
                  f"- 基準にしかない点: {only_str(set(a) - set(b), a)}",
                  f"- {name[s]} にしかない点: {only_str(set(b) - set(a), b)}", ""]
            if lines:
                L += ["| shape | dtype | レイアウト | 入力 | 比較 | seed | B | 基準 → source | 仮説・壊れたバッチの区間 / エラー |", "|" + "---|" * 9] + lines + [""]
            cl = ", ".join(f"{short(x)} → {short(y)} {n}" for (x, y), n in pairs.most_common())
            summary.append(f"- **{name[s]}**(torch {v}){part}: 共通の {len(common)} 点"
                           + ("がない" if not common else "の全部で基準と分類が同じ" if not n_diff else f"のうち {n_diff} 点で基準と分類が違う(基準 → source: {cl})")
                           + f"。切り替わりの表で基準と違う系列: {trans_diff[(s, v)]}")
    if not others:
        L += ["- 基準のほかに source がない", ""]

    # 5. まとめ
    L += ["## まとめ", ""] + (summary or ["- 比べる source がない"]) + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")
    for x in summary:
        print(x)


if __name__ == "__main__":
    main()
