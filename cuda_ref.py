"""CUDA (A100) の参考値: MPS で壊れた条件と同じ shape を CUDA で回し、fp64 の正解と全要素比較する。

Colab で `colab exec -f cuda_ref.py` として流す想定(argv は使わず、条件は下の定数で与える)。
- TF32 は無効化する
- 正解は同じ GPU 上の fp64 をチャンクごとに計算して比較し、捨てる(fp64 の出力を全要素持たない)
- 念のため数バッチは CPU の fp64 とも照合する
- 結果は 1 条件 1 行の JSON を RESULT_PATH に追記し、最後に番兵を print する
"""

import json
import platform
import time

import torch

RESULT_PATH = "/content/cuda_ref.jsonl"
SEED = 0
CHUNK = 2048  # 比較時のバッチのチャンク
BATCHES = [65520, 65552, 68544]  # (·,256,·) で要素数 2**32 をまたぐ
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
LAYOUTS = ["contig", "b_T", "a_T", "sliced"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device("cuda")

ENV = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "python": platform.python_version(),
    "tf32": False,
}


def emit(rec):
    rec = {**ENV, **rec}
    with open(RESULT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k not in ENV}), flush=True)


def make(shape, dt, layout_T):
    """CPU で固定 seed から生成して GPU へ。layout_T なら転置した view を返す(中身は同じ行列)。"""
    g = torch.Generator().manual_seed(SEED)
    if layout_T:
        base = (torch.rand(shape[0], shape[2], shape[1], generator=g) * 2 - 1).to(dt).to(DEV)
        return base.transpose(1, 2)
    return (torch.rand(*shape, generator=g) * 2 - 1).to(dt).to(DEV)


def rel_err_full(out, a, b):
    """out と fp64 正解を、バッチのチャンクごとに比べる。"""
    B = out.shape[0]
    max_abs, max_ref, bad_batches = 0.0, 0.0, 0
    for i in range(0, B, CHUNK):
        ref = torch.bmm(a[i:i + CHUNK].double(), b[i:i + CHUNK].double())
        d = (out[i:i + CHUNK].double() - ref).abs()
        max_abs = max(max_abs, d.max().item())
        max_ref = max(max_ref, ref.abs().max().item())
        per = d.flatten(1).max(1).values / ref.abs().flatten(1).max(1).values.clamp_min(1e-12)
        bad_batches += int((per > 1e-2).sum().item())
        del ref, d
    return max_abs / max(max_ref, 1e-12), bad_batches


def cpu_spot(out, a, b, idx):
    errs = {}
    for i in idx:
        ref = a[i].cpu().double() @ b[i].cpu().double()
        errs[int(i)] = ((out[i].cpu().double() - ref).abs().max() / ref.abs().max()).item()
    return errs


def run_bmm(series, B, dname, layout):
    dt = DTYPES[dname]
    if series == "out":  # 出力 (B,256,256) が境界をまたぐ
        sa, sb = (B, 256, 64), (B, 64, 256)
    else:                # 入力 a (B,256,256) が境界をまたぐ
        sa, sb = (B, 256, 256), (B, 256, 64)
    rec = {"case": "bmm", "series": series, "B": B, "dtype": dname, "layout": layout,
           "a_shape": sa, "b_shape": sb}
    try:
        if layout == "sliced":
            a = make((sa[0] + 1, *sa[1:]), dt, False)[1:]
            b = make(sb, dt, False)
        else:
            a = make(sa, dt, layout == "a_T")
            b = make(sb, dt, layout == "b_T")
        rec["out_numel"] = sa[0] * sa[1] * sb[2]
        rec["max_in_numel"] = max(a.numel(), b.numel())
        t = time.perf_counter()
        out = torch.bmm(a, b)
        torch.cuda.synchronize()
        rec["sec_bmm"] = round(time.perf_counter() - t, 3)
        rec["rel_err_full_vs_gpu_fp64"], rec["bad_batches"] = rel_err_full(out, a, b)
        rec["rel_err_cpu_fp64_spot"] = cpu_spot(out, a, b, [0, B // 2, B - 1])
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    a = b = out = None  # noqa: F841  次の条件の前にメモリを返す
    torch.cuda.empty_cache()
    emit(rec)


def run_attention(kind):
    """(5712,12,256,64) の attention。eager は in-place softmax でメモリを抑える。末尾と先頭のバッチを CPU fp64 と比較。"""
    B, H, L, D = 5712, 12, 256, 64
    rec = {"case": f"attention_{kind}", "shape": [B, H, L, D], "dtype": "fp32"}
    try:
        g = torch.Generator().manual_seed(SEED)
        q, k, v = ((torch.rand(B, H, L, D, generator=g) * 2 - 1).to(DEV) for _ in range(3))
        if kind == "eager":
            s = q @ k.transpose(-1, -2)
            s.mul_(D ** -0.5)
            s.sub_(s.amax(-1, keepdim=True)).exp_()
            s.div_(s.sum(-1, keepdim=True))
            o = s @ v
            del s
        else:
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        errs = {}
        for i in (0, B // 2, B - 1):
            qi, ki, vi = q[i].cpu().double(), k[i].cpu().double(), v[i].cpu().double()
            ref = torch.softmax(qi @ ki.transpose(-1, -2) * D ** -0.5, -1) @ vi
            errs[i] = ((o[i].cpu().double() - ref).abs().max() / ref.abs().max()).item()
        rec["rel_err_cpu_fp64_spot"] = errs
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    torch.cuda.empty_cache()
    emit(rec)


def run_arange():
    N = 2**32 + 2**20
    rec = {"case": "arange_int64", "N": N}
    try:
        x = torch.arange(N, dtype=torch.int64, device=DEV)
        idx = [0, 2**20 - 1, 2**20, 2**31, 2**32, N - 1]
        rec["values"] = {i: x[i].item() for i in idx}
        rec["status"] = "ok" if all(rec["values"][i] == i for i in idx) else "wrong"
        del x
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    torch.cuda.empty_cache()
    emit(rec)


print(json.dumps(ENV), flush=True)
for series in ("out", "in"):
    for dname in DTYPES:
        for layout in LAYOUTS:
            for B in BATCHES:
                run_bmm(series, B, dname, layout)
run_attention("eager")
run_attention("sdpa")
run_arange()
print("__CUDA_REF_DONE__", flush=True)
