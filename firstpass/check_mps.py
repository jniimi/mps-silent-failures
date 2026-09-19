"""MPS で以前(macOS 14.6.1 + torch 2.11)失敗していた演算が現 OS で通るかを確認する。

    uv run python check_mps.py            # 全テストを 1 つずつ別プロセスで実行
    uv run python check_mps.py conv1d_long_350k   # 単体実行

Metal 側のエラーはプロセスごと abort することがあるので、各テストは subprocess で隔離する。
正しさは CPU (fp32/fp64) の結果との最大相対誤差で判定する。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

MPS = torch.device("mps")


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def conv_case(conv_cls, in_shape, grad=True, **kw):
    torch.manual_seed(0)
    m_cpu = conv_cls(**kw)
    m_mps = conv_cls(**kw).to(MPS)
    m_mps.load_state_dict(m_cpu.state_dict())
    x = torch.randn(*in_shape)
    x_cpu = x.clone().requires_grad_(grad)
    x_mps = x.to(MPS).requires_grad_(grad)
    y_mps = m_mps(x_mps)
    y_cpu = m_cpu(x_cpu)
    out = {"out_shape": list(y_mps.shape), "fwd_rel_err": rel_err(y_mps, y_cpu)}
    if grad:
        g = torch.randn_like(y_cpu)
        y_cpu.backward(g)
        y_mps.backward(g.to(MPS))
        out["dx_rel_err"] = rel_err(x_mps.grad, x_cpu.grad)
        out["dw_rel_err"] = rel_err(m_mps.weight.grad, m_cpu.weight.grad)
    return out


# --- 以前実際に踏んだもの ---------------------------------------------------

def conv1d_long_350k():
    """vocoder 相当: 44.1kHz 波形 1 チャンク ≈ 35 万サンプル。"""
    return conv_case(torch.nn.Conv1d, (1, 32, 350_000), in_channels=32, out_channels=32, kernel_size=7, padding=3)


def conv1d_long_640k():
    """音声変換モデル相当: 16kHz × 40 s = 64 万サンプル。"""
    return conv_case(torch.nn.Conv1d, (1, 16, 640_000), in_channels=16, out_channels=16, kernel_size=9, padding=4)


def conv_transpose1d_long():
    """NSF/HiFi-GAN の upsample 相当: 出力長 40 万。"""
    return conv_case(torch.nn.ConvTranspose1d, (1, 64, 50_000), in_channels=64, out_channels=32,
                     kernel_size=16, stride=8, padding=4)


def conv1d_out_channels_70k():
    """文字通り out_channels > 65536。"""
    return conv_case(torch.nn.Conv1d, (2, 8, 32), in_channels=8, out_channels=70_000, kernel_size=3, padding=1)


def conv2d_wide():
    """Conv2d でも同じ 65536 チェックがかかっていた (W = 100k)。"""
    return conv_case(torch.nn.Conv2d, (1, 4, 2, 100_000), in_channels=4, out_channels=8, kernel_size=(1, 5), padding=(0, 2))


def bmm_over_2pow32():
    """DistilBERT の forward の attention score: (5712*12, 256, 64) @ (.., 64, 256) = 4.49e9 要素。"""
    B, L, D = 5712 * 12, 256, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, D, device=MPS)
    k = torch.randn(B, D, L, device=MPS)
    t = time.perf_counter()
    s = torch.bmm(q, k)
    torch.mps.synchronize()
    dt = time.perf_counter() - t
    # 先頭・中央・末尾のバッチを CPU で検算(タイル境界をまたぐ末尾が重要)
    errs = {}
    for i in (0, B // 2, B - 1):
        errs[i] = rel_err(s[i], q[i].cpu() @ k[i].cpu())
    return {"numel": s.numel(), "numel_over_2pow32": s.numel() / 2**32, "sec": round(dt, 2),
            "rel_err_by_batch": errs}


def bmm_over_2pow32_kT_view():
    """同じ規模で batch2 を transpose の view のまま渡す (attention の q @ k.transpose(-1,-2) と同じ形)。
    2**32 要素を超えてタイル分割経路に入ると stride が無視され、エラーなしで誤った値が返る。"""
    B, L, D = 65536 + 16, 256, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, D, device=MPS)
    k = torch.randn(B, L, D, device=MPS)
    s = torch.bmm(q, k.transpose(1, 2))
    torch.mps.synchronize()
    return {"numel_over_2pow32": s.numel() / 2**32,
            "rel_err_by_batch": {i: rel_err(s[i], q[i].cpu() @ k[i].cpu().T) for i in (0, B - 1)}}


def bmm_over_2pow32_q_view():
    """batch1 が非連続 (transpose を戻した view) の場合。"""
    B, L, D = 65536 + 16, 256, 64
    torch.manual_seed(0)
    q = torch.randn(B, D, L, device=MPS).transpose(1, 2)
    k = torch.randn(B, D, L, device=MPS)
    s = torch.bmm(q, k)
    torch.mps.synchronize()
    return {"rel_err_by_batch": {i: rel_err(s[i], q[i].cpu() @ k[i].cpu()) for i in (0, B - 1)}}


def attention_distilbert_full_val():
    """同じ shape を eager attention (matmul → softmax → matmul) で通す。"""
    B, H, L, D = 5712, 12, 256, 64
    torch.manual_seed(0)
    q = torch.randn(B, H, L, D, device=MPS)
    k = torch.randn(B, H, L, D, device=MPS)
    v = torch.randn(B, H, L, D, device=MPS)
    t = time.perf_counter()
    w = (q @ k.transpose(-1, -2) / D**0.5).softmax(-1)
    o = w @ v
    torch.mps.synchronize()
    dt = time.perf_counter() - t
    i = B - 1
    ref = ((q[i].cpu() @ k[i].cpu().transpose(-1, -2)) / D**0.5).softmax(-1) @ v[i].cpu()
    return {"score_numel_over_2pow32": w.numel() / 2**32, "sec": round(dt, 2), "last_batch_rel_err": rel_err(o[i], ref)}


def sdpa_fused_full_val():
    """同じ shape を fused SDPA (HF の既定 attn_implementation="sdpa" の経路) で通す。"""
    B, H, L, D = 5712, 12, 256, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, device=MPS) for _ in range(3))
    o = F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize()
    errs = {i: rel_err(o[i:i + 1], F.scaled_dot_product_attention(q[i:i + 1].cpu(), k[i:i + 1].cpu(), v[i:i + 1].cpu()))
            for i in (0, B - 1)}
    return {"rel_err_by_batch": errs}


def sdpa_dropout():
    """`scaled_dot_product_attention for MPS does not support dropout`。"""
    q = torch.randn(4, 8, 128, 64, device=MPS)
    o = F.scaled_dot_product_attention(q, q, q, dropout_p=0.1)
    torch.mps.synchronize()
    return {"out_shape": list(o.shape)}


def mps_tensor_cpu_generator():
    """device mismatch 系 (CPU generator で MPS テンソルを生成)。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(4, device=MPS, generator=g)
    return {"x": x.cpu().tolist()}


def matmul_transposed_lhs_after_big_alloc():
    """pytorch#193487: 大きな確保の後、LHS が transpose view の fp32 matmul が allocator 状態次第で 10-30% ずれる
    (報告は macOS 26 / torch 2.7-2.12)。issue の再現コードそのまま。"""
    import numpy as np
    rng = np.random.default_rng(0)
    n, p = 50_000, 25
    A = torch.tensor(rng.standard_normal((n, p)), dtype=torch.float32)
    w = torch.tensor(rng.random(n), dtype=torch.float32)
    Aw = A * w.unsqueeze(-1)
    ref = Aw.double().T @ A.double()
    scale = ref.abs().max().item()

    def err():
        return float(((Aw.to(MPS).T @ A.to(MPS)).cpu().double() - ref).abs().max()) / scale

    fresh = err()
    big = torch.tensor(rng.standard_normal((200_000, p)), dtype=torch.float32)
    wb = torch.tensor(rng.random(200_000), dtype=torch.float32)
    ((big * wb.unsqueeze(-1)).to(MPS).T @ big.to(MPS)).cpu()
    triggered = err()
    return {"fresh_rel_err": fresh, "triggered_rel_err": triggered}


# --- 入力側が 2**32 要素を超える演算 (出力は小さくても壊れるか) ---------------

N_BIG = 2**32 + 2**20  # 2**32 をわずかに超える要素数


def bmm_in_over_2pow32():
    """attention の w @ v: 入力 w が 2**32 要素超、出力は小さい。入力はすべて連続。"""
    B, L, D = 65536 + 16, 256, 64
    torch.manual_seed(0)
    w = torch.randn(B, L, L, device=MPS)
    v = torch.randn(B, L, D, device=MPS)
    o = torch.bmm(w, v)
    torch.mps.synchronize()
    return {"rel_err_by_batch": {i: rel_err(o[i], w[i].cpu() @ v[i].cpu()) for i in (0, B // 2, B - 1)}}


def linear_in_over_2pow32():
    """F.linear の入力 (N, 64) が 2**32 要素超 (長系列 × 大バッチの hidden state 相当)。"""
    n = N_BIG // 64 + 1
    torch.manual_seed(0)
    x = torch.randn(n, 64, device=MPS)
    lin = torch.nn.Linear(64, 32).to(MPS)
    y = lin(x)
    torch.mps.synchronize()
    idx = torch.tensor([0, n // 2, n - 1])
    ref = torch.nn.functional.linear(x[idx.to(MPS)].cpu(), lin.weight.cpu(), lin.bias.cpu())
    return {"rel_err": rel_err(y[idx.to(MPS)], ref)}


def elementwise_over_2pow32():
    """fp16 で 2**32 要素超の x * 2 + 1。末尾を確認。"""
    x = torch.ones(N_BIG, dtype=torch.float16, device=MPS)
    y = x * 2 + 1
    tail = y[-4:].cpu().float()
    return {"rel_err": rel_err(tail, torch.full((4,), 3.0))}


def sum_over_2pow32():
    x = torch.ones(N_BIG, dtype=torch.uint8, device=MPS)
    s = x.sum().item()
    return {"rel_err": abs(s - N_BIG) / N_BIG}


def transpose_contiguous_over_2pow32():
    """2**32 要素超のテンソルの transpose().contiguous() (mps_guard のような回避策自体が壊れないか)。"""
    B, L = 65536 + 16, 256
    torch.manual_seed(0)
    x = torch.randn(B, L, L, dtype=torch.float16, device=MPS)
    y = x.transpose(1, 2).contiguous()
    return {"rel_err_by_batch": {i: rel_err(y[i], x[i].cpu().T) for i in (0, B - 1)}}


def cat_over_2pow32():
    a = torch.zeros(2**31 + 2**19, dtype=torch.float16, device=MPS)
    b = torch.ones(2**31 + 2**19, dtype=torch.float16, device=MPS)
    c = torch.cat([a, b])
    return {"rel_err": rel_err(c[-4:], torch.ones(4))}


def index_over_2pow32():
    """2**32 を超える位置の読み出し (fancy index)。"""
    x = torch.arange(N_BIG, dtype=torch.int64, device=MPS) % 1000
    idx = torch.tensor([0, 2**32 + 5, N_BIG - 1], device=MPS)
    got = x[idx].cpu()
    ref = torch.tensor([0, (2**32 + 5) % 1000, (N_BIG - 1) % 1000])
    return {"rel_err": rel_err(got.double(), ref.double())}


# --- OS バージョンで分岐しているその他の経路 -------------------------------

def masked_fill_over_2pow32():
    """torch 側の無条件チェック(OS に関係なく残るはず)。uint8 で 4.3e9 要素 ≈ 4.3GB。"""
    x = torch.zeros(2**32 + 16, dtype=torch.uint8, device=MPS)
    m = torch.zeros_like(x, dtype=torch.bool)
    m[-1] = True
    x.masked_fill_(m, 7)
    return {"last": int(x[-1].item())}


def big_elementwise_over_2pow32_bytes():
    """単一 NDArray > 4GB の elementwise / reduction (以前 MPSNDArray の 2**32 bytes 制限の噂があった経路)。"""
    n = 1_500_000_000  # fp32 6GB
    x = torch.ones(n, device=MPS)
    y = (x * 2 + 1).sum()
    return {"sum": y.item(), "expected": 3.0 * n}


def gather_scatter_int32():
    """`torch.int32 is only supported on MacOS15+` (scatter_reduce 系)。"""
    src = torch.arange(12, dtype=torch.int32, device=MPS).reshape(3, 4)
    idx = torch.tensor([[0, 1, 0, 1]] * 3, device=MPS)
    out = torch.zeros(3, 2, dtype=torch.int32, device=MPS).scatter_reduce(1, idx, src, reduce="amax")
    ref = torch.zeros(3, 2, dtype=torch.int32).scatter_reduce(1, idx.cpu(), src.cpu(), reduce="amax")
    return {"equal": bool(torch.equal(out.cpu(), ref))}


def conv2d_channels_last():
    """channels_last の conv (macOS 15 未満は contiguous に落としていた)。"""
    torch.manual_seed(0)
    m = torch.nn.Conv2d(64, 128, 3, padding=1)
    x = torch.randn(8, 64, 128, 128)
    y_cpu = m(x)
    m2 = m.to(MPS, memory_format=torch.channels_last)
    y = m2(x.to(MPS).contiguous(memory_format=torch.channels_last))
    return {"rel_err": rel_err(y, y_cpu), "out_channels_last": y.is_contiguous(memory_format=torch.channels_last)}


def nonzero_native():
    x = torch.randn(64, 1024, device=MPS) > 0
    return {"equal": bool(torch.equal(x.nonzero().cpu(), x.cpu().nonzero()))}


def bf16_matmul():
    a = torch.randn(1024, 1024)
    y = (a.to(MPS, torch.bfloat16) @ a.to(MPS, torch.bfloat16)).float()
    return {"rel_err": rel_err(y, a @ a)}


TESTS = {f.__name__: f for f in [
    conv1d_long_350k, conv1d_long_640k, conv_transpose1d_long, conv1d_out_channels_70k, conv2d_wide,
    bmm_over_2pow32, bmm_over_2pow32_kT_view, bmm_over_2pow32_q_view, attention_distilbert_full_val, sdpa_fused_full_val, sdpa_dropout, mps_tensor_cpu_generator, matmul_transposed_lhs_after_big_alloc,
    bmm_in_over_2pow32, linear_in_over_2pow32, elementwise_over_2pow32, sum_over_2pow32,
    transpose_contiguous_over_2pow32, cat_over_2pow32, index_over_2pow32,
    masked_fill_over_2pow32, big_elementwise_over_2pow32_bytes, gather_scatter_int32,
    conv2d_channels_last, nonzero_native, bf16_matmul,
]}


def max_err(d) -> float:
    """結果 dict 中の *err* キーの最大値 (ネスト対応)。"""
    m = 0.0
    for k, v in d.items():
        if isinstance(v, dict):
            m = max(m, max_err(v) if "err" not in k else max(v.values()))
        elif "err" in k and isinstance(v, float):
            m = max(m, v)
    return m


ERR_TOL = 1e-2  # bf16 を含めても CPU 比でこれを超えたら誤答扱い


def run_one(name: str) -> None:
    import os
    if os.environ.get("MPS_GUARD") == "1":
        import mps_guard
        mps_guard.install()
    t = time.perf_counter()
    try:
        res = {"status": "ok", **TESTS[name]()}
        if max_err(res) > ERR_TOL:
            res["status"] = "WRONG"
    except Exception as e:  # noqa: BLE001
        res = {"status": "error", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    res["wall_sec"] = round(time.perf_counter() - t, 2)
    print("RESULT " + json.dumps(res, default=str))


def main() -> None:
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
        return
    print(f"# macOS {platform.mac_ver()[0]} / torch {torch.__version__} / python {platform.python_version()}")
    results = {}
    for name in TESTS:
        p = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[7:])
        else:
            tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            res = {"status": f"crash(rc={p.returncode})", "error": tail[:200]}
        results[name] = res
        print(f"{name:36s} {res['status']:10s} {json.dumps({k: v for k, v in res.items() if k != 'status'}, default=str)}",
              flush=True)
    import os
    guard = "_guard" if os.environ.get("MPS_GUARD") == "1" else ""
    out = f"results/macos{platform.mac_ver()[0]}_torch{torch.__version__.split('+')[0]}{guard}.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"macos": platform.mac_ver()[0], "torch": torch.__version__, "results": results}, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
