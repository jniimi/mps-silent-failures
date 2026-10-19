# nll_loss の誤りが「要素 2^31 以降の行」か「目標の要素の平坦 index が 2^31 以上」かを C=250 で分ける
import sys, time, torch, torch.nn.functional as F
R, C = int(eval(sys.argv[1])), int(sys.argv[2])
torch.manual_seed(0)
xc = torch.rand(R, C).mul_(8).sub_(4); tc = torch.randint(0, C, (R,))
x, t = xc.to("mps"), tc.to("mps")
lsm = F.log_softmax(x, -1)
got = F.nll_loss(lsm, t, reduction="none").cpu().double()
ch = 2**20
ref = torch.cat([F.nll_loss(F.log_softmax(xc[i:i+ch].double(), -1), tc[i:i+ch], reduction="none") for i in range(0, R, ch)])
bad = (got - ref).abs() > 1e-4
off = torch.arange(R, dtype=torch.int64) * C + tc          # 目標の要素の平坦 index
print(f"R={R} C={C} numel={R*C} (2^31 x {R*C/2**31:.4f}); wrong rows = {int(bad.sum())}")
if bad.any():
    fw = int(bad.nonzero()[0])
    print(f"first wrong row {fw}; row start element {fw*C}; 2^31 / C = {2**31/C:.3f}")
    print(f"target offset >= 2^31  vs wrong:  both {int((bad & (off >= 2**31)).sum())}, "
          f"wrong only {int((bad & (off < 2**31)).sum())}, offset only {int((~bad & (off >= 2**31)).sum())}")
    r0 = 2**31 // C
    for r in range(r0 - 1, r0 + 3):
        print(f"  row {r}: start {r*C}, target {int(tc[r])}, offset {int(off[r])} ({'>=' if off[r] >= 2**31 else '<'} 2^31), "
              f"wrong={bool(bad[r])}, got={got[r].item():.4g}, ref={ref[r].item():.4g}")
    w = got[bad]
    print(f"wrong values: zero {int((w == 0).sum())}, nan {int(w.isnan().sum())}, inf {int(w.isinf().sum())}, other {int((w.isfinite() & (w != 0)).sum())}")
