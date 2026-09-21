"""Per-run analysis: class counts, per-series ranges by class, switch points, extent of wrong rows, errors/stages."""
import json, sys, ast
from collections import Counter, defaultdict

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
print(f"# {path}  rows={len(rows)}")
print("cls:", dict(Counter(r["cls"] for r in rows)))
print("phase:", dict(Counter(r["phase"] for r in rows)))
print("phase x cls:", dict(Counter((r["phase"], r["cls"]) for r in rows)))
print("compare:", dict(Counter(r.get("compare") for r in rows)))
ts = sorted(r["ts"] for r in rows); print("ts:", ts[0], "->", ts[-1], " sum wall_sec=%.0f" % sum(r.get("wall_sec") or 0 for r in rows))
print("tol:", sorted({(r["dtype"], r.get("tol")) for r in rows}))
print("input_mismatch nonzero:", sum(1 for r in rows if any((r.get("input_mismatch_batches") or {}).values())))
print("errors:", dict(Counter((r.get("stage"), r.get("error")) for r in rows if r["cls"] == "error")))
odd = [r for r in rows if r["cls"] not in ("ok", "wrong", "error")]
for r in odd: print("ODD", r["cls"], r["series_id"], r["B"], r.get("pad_batches"), r.get("rc"), str(r.get("error"))[:200])

def L(v):
    if isinstance(v, str):
        try: return ast.literal_eval(v)
        except Exception: return v
    return v

def ranges(xs):
    xs = sorted(set(xs)); out = [];
    return xs

is_bwd = rows[0].get("op") == "bmm_bwd"
def sig(r):
    """outcome signature of a row"""
    if r["cls"] == "error": return f"error[{r.get('stage')}]"
    if is_bwd and r["cls"] in ("ok", "wrong"):
        return "/".join(f"{t}:{r.get('cls_'+t)}" for t in ("out", "grad_a", "grad_b"))
    return r["cls"]

ser = defaultdict(list)
for r in rows:
    if r["phase"] == "calibrate": continue
    ser[(r["shape_id"], r["dtype"], r["layout"], r["kind"])].append(r)

def xval(r): return r["pad_batches"] if r["layout"] == "offset" else r["B"]

for key in sorted(ser):
    rs = sorted(ser[key], key=lambda r: (xval(r), r["compare"], r["seed"]))
    print(f"\n## {key}  n={len(rs)}")
    # by x: set of signatures (consistency across seeds / compare)
    byx = defaultdict(set)
    for r in rs: byx[xval(r)].add(sig(r))
    incons = {x: s for x, s in byx.items() if len(s) > 1}
    if incons: print("  INCONSISTENT across seeds/compare:", incons)
    xs = sorted(byx)
    # runs of the same signature
    seg = []
    for x in xs:
        s = "|".join(sorted(byx[x]))
        if seg and seg[-1][0] == s: seg[-1][2] = x; seg[-1][3] += 1
        else: seg.append([s, x, x, 1])
    print("  segments:", "  ".join(f"{s}: {a}..{b} ({n} pts)" for s, a, b, n in seg))
    # wrong rows detail
    for r in rs:
        if r["cls"] != "wrong": continue
        tens = ("out", "grad_a", "grad_b") if is_bwd else ("",)
        for t in tens:
            sfx = "_" + t if t else ""
            if is_bwd and r.get("cls_" + t) != "wrong": continue
            br = L(r.get("bad_ranges" + sfx)); nb = r.get("n_batches_bad" + sfx)
            hm = r.get("hyp_match" + sfx) if t in ("", ) else r.get("hyp_match_" + t)
            if t == "out": hm = r.get("hyp_match")
            brs = str(br) if br is not None and len(str(br)) < 120 else (str(br)[:60] + f"...(n_ranges={r.get('n_bad_ranges'+sfx)})...{str(br)[-40:]}")
            print(f"   WRONG x={xval(r)} B={r['B']} seed={r['seed']} {r['compare']} ph={r['phase']} {t} err={r.get('max_rel_err'+sfx):.3g} bad={nb}/{r.get('n_batches_compared')} first_bad={r.get('first_bad_batch'+sfx)} frac_elem_bad={r.get('frac_elem_bad'+sfx)} zero={r.get('frac_bad_elem_zero'+sfx)} ranges={brs} hyp={hm}")
