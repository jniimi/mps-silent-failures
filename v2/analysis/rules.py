"""Check rule sets against every non-calibrate bmm row (standard layouts) of the given JSONL files.
usage: rules.py RULESET file.jsonl [file.jsonl ...]     RULESET = v1 | (others added later)
Prediction per row: 'ok' | 'error' | 'wrong_all' | ('wrong_tail', thr)  (batches >= thr wrong, others ok)
"""
import json, sys, ast
from collections import Counter, defaultdict

T31, T32 = 2**31, 2**32

def L(v):
    return ast.literal_eval(v) if isinstance(v, str) else v

def operands(r):
    lay = r["layout"]; M, K, N, B = r["M"], r["K"], r["N"], r["B"]
    a = dict(numel=B * M * K, per=M * K, transposed=lay == "aT", view=lay in ("aT", "slice"))
    b = dict(numel=B * K * N, per=K * N, transposed=lay == "bT", view=lay in ("bT", "slice"))
    return a, b, B * M * N

def _r2(a,b): return any(o["view"] and o["numel"] >= T31 for o in (a, b))
def _r3(a,b):
    big = [o for o in (a, b) if not o["view"] and o["numel"] > T32]
    return ("wrong_tail", min(-(-T32 // o["per"]) for o in big)) if big else None

def rule_A(r):
    """order 1,2,3; rule 1 fires only with a transposed operand, otherwise falls through to 2 and 3"""
    a, b, out = operands(r)
    if out > T32 and (a["transposed"] or b["transposed"]): return "wrong_all"
    if _r2(a, b): return "error"
    return _r3(a, b) or "ok"

def rule_B(r):
    """order 1,2,3; out > 2^32 is terminal: transposed -> wrong_all, otherwise ok (rules 2, 3 apply only when out <= 2^32)"""
    a, b, out = operands(r)
    if out > T32: return "wrong_all" if (a["transposed"] or b["transposed"]) else "ok"
    if _r2(a, b): return "error"
    return _r3(a, b) or "ok"

def rule_C(r):
    """like B for rule 2 (skipped when out > 2^32) but rule 3 still applies when out > 2^32 and nothing is transposed"""
    a, b, out = operands(r)
    if out > T32 and (a["transposed"] or b["transposed"]): return "wrong_all"
    if out <= T32 and _r2(a, b): return "error"
    return _r3(a, b) or "ok"

RULES = {"A": rule_A, "B": rule_B, "C": rule_C}

def observed(r, pred):
    """returns (label, exactness). label comparable with pred."""
    c = r["cls"]
    if c in ("ok", "error"): return c, "exact"
    if c != "wrong": return c, "exact"
    nb, nc = r["n_batches_bad"], r["n_batches_compared"]
    if nb == nc: return "wrong_all", "exact" if r["compare"] == "full" else "sampled"
    br = L(r["bad_ranges"]); lo = min(x[0] for x in br)
    if isinstance(pred, tuple):
        thr = pred[1]
        if r["compare"] == "full":
            if br == [[thr, r["B"]]]: return pred, "exact"
        else:
            # sampled: every bad batch >= thr, and every compared batch >= thr is bad (via histogram bins lying fully >= thr)
            edges, hb, hc = L(r["hist_edges"]), r["hist_bad"], r["hist_compared"]
            full_bins_ok = all(hb[i] == hc[i] for i in range(len(hb)) if edges[i] >= thr)
            low_bins_ok = all(hb[i] == 0 for i in range(len(hb)) if edges[i + 1] <= thr)
            if lo >= thr and full_bins_ok and low_bins_ok: return pred, "sampled"
    return ("wrong_partial", lo, nb, nc), "exact"

def main():
    rule = RULES[sys.argv[1]]
    for path in [x for x in sys.argv[2:] if x != '--idx']:
        rows = [json.loads(l) for l in open(path)]
        KINDS = ("random",) if "--idx" not in sys.argv else ("idx_a_batch","idx_a_rc","idx_b_batch","idx_b_rc")
        rows = [r for r in rows if r["kind"] in KINDS and r["phase"] != "calibrate" and r.get("op", "bmm") == "bmm" and r["layout"] != "offset"
                and r["cls"] not in ("skipped_memory", "skipped_budget")]
        agree = Counter(); dis = []
        for r in rows:
            p = rule(r); o, ex = observed(r, p)
            pk = p if isinstance(p, str) else p[0]
            if o == p: agree[(pk, ex)] += 1
            else: dis.append((r["shape_id"], r["dtype"], r["layout"], r["kind"], r["B"], r["seed"], r["compare"], r["phase"], "pred=" + str(p), "obs=" + str(o),
                              "hyp=" + str(r.get("hyp_match"))))
        print(f"# {sys.argv[1]} on {path}: rows={len(rows)} agree={sum(agree.values())} disagree={len(dis)}")
        print("  agree by predicted class:", dict(sorted(agree.items())))
        g = defaultdict(list)
        for d in dis: g[(d[0], d[1], d[2], d[8].split(',')[0], d[9].split(',')[0])].append(d)
        for k, v in sorted(g.items()):
            print("  DISAGREE", k, "n=", len(v), "B=", sorted({d[4] for d in v}), "kinds=", sorted({d[3] for d in v}))
            print("     e.g.", v[0])

main()
