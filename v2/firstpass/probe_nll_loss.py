# MPS の cross_entropy の損失値が大きな入力で基準と合わない件の切り分け(PyTorch 2.14.0)
import sys, time, torch, torch.nn.functional as F
R, C = int(eval(sys.argv[1])), int(sys.argv[2])
T0 = time.time()
def lap(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)
torch.manual_seed(0)
xc = torch.rand(R, C).mul_(8).sub_(4)       # CPU で生成(MPS の乱数は使わない)。一様 [-4, 4)
tc = torch.randint(0, C, (R,))
lap(f"R={R} (2^{R.bit_length()-1}) C={C} numel={R*C} = 2^32 x {R*C/2**32:.4f}")
x, t = xc.to("mps"), tc.to("mps")
ch = 2**20
for i in (0, (R // 2) // ch * ch, R - ch):   # 転送の確認(先頭・中央・末尾の塊)
    assert torch.equal(x[i:i+ch].cpu(), xc[i:i+ch]), f"transfer mismatch at row {i}"
lap("transfer verified")
# 基準: 行ごとの損失を小さな塊ごとに MPS で(CPU の塊を送る。大きなテンソルは使わない)
ref_rows = torch.cat([F.cross_entropy(xc[i:i+ch].to("mps"), tc[i:i+ch].to("mps"), reduction="none").cpu() for i in range(0, R, ch)]).double()
for i in (0, R - 2**14):                     # 基準の CPU float64 との照合(小さな塊)
    c = F.cross_entropy(xc[i:i+2**14].double(), tc[i:i+2**14], reduction="none")
    assert (c - ref_rows[i:i+2**14]).abs().max() < 1e-4, "chunk reference disagrees with CPU"
ref = ref_rows.sum().item()
del xc, tc
lap(f"reference sum = {ref:.1f}")
s = F.cross_entropy(x, t, reduction="sum").item();  lap(f"cross_entropy sum      / ref = {s/ref:.6f}")
m = F.cross_entropy(x, t, reduction="mean").item(); lap(f"cross_entropy mean * R / ref = {m*R/ref:.6f}")
n = F.cross_entropy(x, t, reduction="none").cpu().double()
bad = (n - ref_rows).abs() > 1e-4
lap(f"cross_entropy none: sum/ref = {n.sum().item()/ref:.6f}, wrong rows = {int(bad.sum())}"
    + (f", first {int(bad.nonzero()[0])}, last {int(bad.nonzero()[-1])}, zero among wrong = {int((n[bad] == 0).sum())}" if bad.any() else ""))
del n
lsm = F.log_softmax(x, -1)
d = max((lsm[i:i+ch].cpu() - F.log_softmax(x[i:i+ch].cpu(), -1)).abs().max().item() for i in (0, (R // 2) // ch * ch, R - ch))
lap(f"log_softmax max abs diff vs CPU (3 chunks) = {d:.2e}")
ns = F.nll_loss(lsm, t, reduction="sum").item();   lap(f"nll_loss(log_softmax) sum  / ref = {ns/ref:.6f}")
nn_ = F.nll_loss(lsm, t, reduction="none").cpu().double()
lap(f"nll_loss none: sum/ref = {nn_.sum().item()/ref:.6f}, wrong rows = {int(((nn_ - ref_rows).abs() > 1e-4).sum())}")
g = -lsm.gather(1, t[:, None]).squeeze(1)
lap(f"gather then .sum() on MPS / ref = {g.sum().item()/ref:.6f}")
