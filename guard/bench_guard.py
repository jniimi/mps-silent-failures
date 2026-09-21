"""mps_guard のオーバーヘッド計測: DBM 規模の小さい学習ループと、LLM 規模の大きい forward。"""
import sys, time, torch, mps_guard
M = torch.device("mps")

def small_loop(steps=2000):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(120, 512), torch.nn.Sigmoid(), torch.nn.Linear(512, 120)).to(M)
    opt = torch.optim.Adam(net.parameters(), 1e-3); x = torch.rand(1024, 120, device=M)
    for _ in range(steps):
        loss = ((net(x) - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()

def big_forward(reps=5):
    q = torch.randn(64, 32, 1024, 128, device=M, dtype=torch.float16)
    for _ in range(reps):
        w = (q @ q.transpose(-1, -2)).softmax(-1); o = w @ q
    torch.mps.synchronize()

for name, fn in [("small_loop (2000 steps)", small_loop), ("big_forward (attention 64x32x1024)", big_forward)]:
    fn()  # warmup
    t = time.perf_counter(); fn(); base = time.perf_counter() - t
    mps_guard.install(); fn(); t = time.perf_counter(); fn(); g = time.perf_counter() - t; mps_guard.uninstall()
    print(f"{name:36s} off {base:6.2f}s  on {g:6.2f}s  x{g/base:.2f}")
