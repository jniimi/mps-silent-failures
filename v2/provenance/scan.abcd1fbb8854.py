"""MPS の bmm が 2**31 / 2**32 境界付近でどう壊れるかを走査するドライバ。

    uv run python scan.py --host-tag studio --dry-run         # 条件数・メモリ・所要時間の見積もりだけ
    uv run python scan.py --host-tag studio                   # 新しい run_id で走査
    uv run python scan.py --host-tag studio --resume <run_id> # 同じ run_id の続きから
    uv run python scan.py --host-tag studio --summary <run_id>  # その run の summary を作り直す
    uv run python scan.py --host-tag mac15 --wandb --backup-cmd 'rsync -a {jsonl} {manifest} me@host:mpsb/'  # 退避つき

v2 で足した軸(どれも既定の引数では無効で、既定の計画は v1 と同じ):
    uv run python scan.py --host-tag studio --layouts offset --shapes out-256x64x256,in-256x256x64   # 大きい storage_offset の系列
    uv run python scan.py --host-tag studio --shapes all-256x256x256 --dtypes fp16                   # a・b・出力が同時に 2**32 超
    uv run python scan.py --host-tag studio --far 3                                                  # 2·2**32、3·2**32 の前後
    uv run python scan.py --host-tag studio --dtypes bf16 --shapes out-256x64x256,in-256x256x64      # bfloat16
    uv run python scan.py --host-tag studio --points points.json                                     # 指定した点だけ
    uv run python scan.py --host-tag studio --op bmm_bwd --shapes out-256x64x256,in-256x256x64       # backward(下の --op)
  offset 系列: B は固定(--offset-B、既定 4096)で、a, b の格納の先頭に置く pad_batches O を振る(view = 格納[O:])。
    storage_offset(a, b の大きい方の要素数)が各境界に達する O0 の O0-2〜O0+2、O = 1、その間に --grid-offset 点。
    全要素比較、切り替わりは二分探索。基本 shape、fp32 / fp16 だけ。行に pad_batches、offset_numel_a/b、storage_numel_a/b
  --far K: 各系列に B0·k-1, B0·k, B0·k+1(k = 2..K、B0 = 2**32 要素の B)をサンプル比較で足す(phase = far)。
    基本 shape は B0·k+1 で index 符号化も。切り替わりの探索・最終点には使わない
  --op bmm_bwd: forward + backward を 1 実行で行い、out / grad_a / grad_b を同じ基準で比べる(詳細は worker.py)。分類は 3 つの
    うち最悪で、tensor ごとの結果は cls_out / cls_grad_a / cls_grad_b など。較正も同じ演算で行う。index 符号化は回さない。
    メモリ見積もりに G, grad_a, grad_b を含む。読み違い仮説の検定は out だけ v1 と同じで、勾配は簡易な検定(hyp_match_grad_a/b)
  --points FILE: 較正 + 指定点だけ(phase = point)。実測の追試や、予算の都合で走査できない条件の単発測定用

1 実行 = 1 つの (演算, device, shape (M,K,N), dtype, レイアウト, 入力の種類, B, seed, 比較方式) を worker.py の
別プロセスで実行する(`uv run --no-project --python 3.12 --with torch==X`)。クラッシュ・タイムアウトも 1 件として記録する。

記録(run ごとに分離):
  results/raw/<host-tag>/<run_id>.manifest.json  run manifest(コードの git revision と dirty、scan/worker のハッシュ、
                                                 macOS の版と build、機種、メモリ、Python・torch の版、実行コマンド)
  results/raw/<host-tag>/<run_id>.jsonl          1 実行 1 行(正本)。行と manifest に "schema": 2。v1 のキーは同じ意味のまま
  results/summary/<host-tag>/<run_id>.md         要約
結果の退避(どちらも既定はなし。JSONL に書いた後に行い、失敗しても走査は止めない):
  --wandb                走査 1 回 = W&B の run 1 つ(name = id = run_id、group = host-tag)。行ごとに分類の累計を log し、
                         --wandb-every 行ごとと終了時(中断・例外・SIGTERM を含む)に JSONL と manifest を Artifact
                         (type sweep-results、name = run_id、正常終了の版に alias final)として上げる。失敗は <run_id>.wandb.log
  --backup-cmd CMD       --backup-every 行ごとと終了時にシェルで実行({jsonl} {manifest} {dir} を置換)。失敗は <run_id>.backup.log
テスト専用: --scale-log2 S は理論境界と B の点を 2**S で割る(ドライバ・退避の経路を安く通すため)。行と manifest に scale_log2 が
付き、versions.py / check_idx_invariance.py は拒否する。S = 0(既定)の計画は v1 と同じ(較正の対象だけが違う。下の 1.)。
キャッシュ(再開)は同じ run_id の中だけで、環境・コード・spec・tol から作る fingerprint が一致した実行を再利用する。

手順:
  1. 較正: 境界をまたぐ量が 2**28 要素の B、連続、乱数入力、seed 0/1/2 の全要素比較で dtype ごとの正常時の最大相対誤差を
     測り、許容誤差 = TOL_FACTOR 倍。その run に含まれる shape 全部で測る(v1 は基本 shape だけ)。内訳は manifest の tol_calibration
  2. 探索(乱数入力、seed 0、サンプル比較): 各理論境界 B0 の B0-2〜B0+2 を無条件で測り、境界の間には粗いグリッドを置く。
     正常/異常が異なる隣接点の間を B の粒度まで二分探索する(非単調なら切り替わりが複数見つかる)
  3. index 符号化入力(基本 shape のみ): 小 shape 点と各境界の B0±1 をサンプル比較
  4. 最終点: 各系列の切り替わりの下・上を全要素比較(乱数入力は seed 0/1/2、基本 shape は index 符号化も)。
     切り替わりがない系列は最大の B。時間切れで省いた全要素比較は skipped_budget として残す
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "worker.py"
PROJECT = "mps-boundary"

# 系列: cross = 境界をまたぐ量(out = 出力、in = 入力 a)。どれもバッチあたり 65536 要素で、同じ要素数を別の (M,K,N) に分解する
SHAPES = [
    {"id": "out-256x64x256", "M": 256, "K": 64, "N": 256, "cross": "out", "base": True},
    {"id": "in-256x256x64", "M": 256, "K": 256, "N": 64, "cross": "in", "base": True},
    {"id": "out-512x64x128", "M": 512, "K": 64, "N": 128, "cross": "out", "base": False},
    {"id": "out-128x64x512", "M": 128, "K": 64, "N": 512, "cross": "out", "base": False},
    {"id": "in-512x128x64", "M": 512, "K": 128, "N": 64, "cross": "in", "base": False},
    {"id": "in-128x512x64", "M": 128, "K": 512, "N": 64, "cross": "in", "base": False},
]
# v2 の追加 shape(既定の --shapes には入れない。既定の計画を v1 と同じに保つため)。cross = all: a・b・出力が同時に B·2**16 要素。
# all_dtypes: 追加 shape だが --extra-dtypes に絞らず --dtypes の全部で回す
EXTRA_SHAPES = [
    {"id": "all-256x256x256", "M": 256, "K": 256, "N": 256, "cross": "all", "base": False, "all_dtypes": True},
]
SHAPE = {s["id"]: s for s in SHAPES + EXTRA_SHAPES}
ITEMSIZE = {"fp32": 4, "fp16": 2, "bf16": 2}
LAYOUTS = ["contig", "bT", "aT", "slice"]
OFFSET = "offset"  # v2: 大きい storage_offset の系列(B 固定で pad_batches を振る)。--layouts offset か --with-offset で有効
OFFSET_DTYPES = ["fp32", "fp16"]
IDX_KINDS = ["idx_a_batch", "idx_a_rc", "idx_b_batch", "idx_b_rc"]
CAL_ELEMS = 2**28  # 較正・小 shape 点: 境界をまたぐ量の要素数
CAL_SEEDS = [0, 1, 2]
FINAL_SEEDS = [0, 1, 2]
TOL_FACTOR = 10
TOL_FLOOR = {"fp32": 1e-6, "fp16": 1e-3, "bf16": 1e-2}  # 較正値が極端に小さい場合の下限
TOL_FALLBACK = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 1e-2}  # 較正が 1 件も取れなかった場合だけ使う
TIMEOUT = {"sample": 1200, "full": 3600}
MAX_TRANS = 4  # 1 系列で全要素比較する切り替わりの上限
SCHEMA = 2  # 行と manifest の版。v1 のキーは同じ意味のまま残し、v2 のキーを足す
OP = "bmm"  # --op。bmm_bwd は forward + backward(out, grad_a, grad_b を比べる)
SCALE_LOG2 = 0  # --scale-log2(テスト専用): 理論境界と B の点を 2**S で割る。shape は変えない


# ---------------------------------------------------------------------------- 環境と provenance

def sh(*cmd: str, cwd=None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_text(p: str) -> str:
    try:
        return Path(p).read_text()
    except Exception:  # noqa: BLE001
        return ""


def host_env(device: str = "mps") -> dict:
    """ホストの環境。mem_bytes はホストの RAM(Apple Silicon では GPU と共有)。
    macOS / Linux の両方で取る。device=cuda なら nvidia-smi で GPU 名・メモリ・ドライバ版も取る。"""
    d = {"os": platform.system(), "os_release": platform.release(), "arch": platform.machine(),
         "macos": "", "macos_build": "", "linux_distro": "", "hw_model": "", "chip": "", "mem_bytes": 0}
    if d["os"] == "Darwin":
        d.update(macos=platform.mac_ver()[0], macos_build=sh("sw_vers", "-buildVersion"),
                 hw_model=sh("sysctl", "-n", "hw.model"), chip=sh("sysctl", "-n", "machdep.cpu.brand_string"),
                 mem_bytes=int(sh("sysctl", "-n", "hw.memsize") or 0))
    elif d["os"] == "Linux":
        osr = dict(l.split("=", 1) for l in read_text("/etc/os-release").splitlines() if "=" in l)
        cpu = next((l.split(":", 1)[1].strip() for l in read_text("/proc/cpuinfo").splitlines()
                    if l.startswith("model name")), "")
        mem = next((int(l.split()[1]) * 1024 for l in read_text("/proc/meminfo").splitlines()
                    if l.startswith("MemTotal:")), 0)
        d.update(linux_distro=osr.get("PRETTY_NAME", "").strip('"'), chip=cpu, mem_bytes=mem,
                 hw_model=read_text("/sys/class/dmi/id/product_name").strip(),
                 n_cpu=len([l for l in read_text("/proc/cpuinfo").splitlines() if l.startswith("processor")]))
    if device == "cuda":
        q = sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits")
        if q:
            name, mem, drv = [x.strip() for x in q.splitlines()[0].split(",")]
            d.update(gpu_name=name, gpu_mem_bytes=int(float(mem)) * 2**20, gpu_driver=drv, gpu_mem_source="nvidia-smi")
    return d


def env_str(env: dict, wk: dict | None = None) -> str:
    wk = wk or {}
    if env.get("os", "Darwin") == "Darwin":
        s = f"macOS {env.get('macos')} ({env.get('macos_build')}) / {env.get('chip')} ({env.get('hw_model')})"
    else:
        s = f"{env.get('os')} {env.get('os_release')} ({env.get('linux_distro')}) / {env.get('chip')}"
    s += f" / RAM {env.get('mem_bytes', 0)/2**30:.0f} GiB"
    if env.get("gpu_name") or wk.get("gpu_name"):
        s += (f" / GPU {env.get('gpu_name') or wk.get('gpu_name')} "
              f"{(env.get('gpu_mem_bytes') or wk.get('gpu_mem_bytes') or 0)/2**30:.0f} GiB"
              f" driver {env.get('gpu_driver', '-')} CUDA {wk.get('cuda', '-')}")
    return s


def code_info() -> dict:
    """git の revision と dirty。.git が無い環境(tar で送った Colab など)では環境変数 MPSB_GIT_REV を使う。"""
    code_files = ["scan.py", "worker.py"]
    d = {"sha_scan": sha256(ROOT / "scan.py"), "sha_worker": sha256(WORKER)}
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, cwd=ROOT)
        in_git = p.returncode == 0 and len(p.stdout.strip()) == 40
    except Exception:  # noqa: BLE001
        in_git = False
    env_rev = os.environ.get("MPSB_GIT_REV")
    if in_git:
        d.update(git_rev=p.stdout.strip(), git_rev_source="git",
                 git_dirty_repo=bool(sh("git", "status", "--porcelain", cwd=ROOT)),
                 git_dirty_code=bool(sh("git", "status", "--porcelain", "--", *code_files, cwd=ROOT)))
    else:  # dirty は分からない(scan/worker のハッシュで同一性を確認する)
        d.update(git_rev=env_rev, git_rev_source="MPSB_GIT_REV" if env_rev else "none",
                 git_dirty_repo=None, git_dirty_code=None)
    return d


def worker_env(args) -> dict:
    """ワーカー側の Python / torch の版(ワーカーと同じ uv の使い捨て環境で調べる)。"""
    out = sh(*worker_cmd(args)[:-3], "-c",
             "import json,platform,sys,torch;d={'python':platform.python_version(),"
             "'torch':torch.__version__,'executable':sys.executable,'cuda':torch.version.cuda}\n"
             "if torch.cuda.is_available():\n p=torch.cuda.get_device_properties(0);"
             "d.update(gpu_name=p.name,gpu_mem_bytes=p.total_memory)\n"
             "print(json.dumps(d))")
    try:
        return json.loads(out.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {}


def worker_cmd(args) -> list[str]:
    return ["uv", "run", "-q", "--no-project", "--python", args.python, "--with", f"torch=={args.torch}",
            "--with", "numpy", "python", str(WORKER), "--device", args.device]


# ---------------------------------------------------------------------------- 条件

def per_batch(sh_: dict) -> int:
    """境界をまたぐ量のバッチあたり要素数(cross = in は a、out / all は出力)。"""
    return sh_["M"] * sh_["K"] if sh_["cross"] == "in" else sh_["M"] * sh_["N"]


def scaled(b: int) -> int:
    """--scale-log2 S(テスト専用)のとき B の点を 2**S で割る。S = 0 なら何もしない。"""
    return b >> SCALE_LOG2


def cal_B(sh_: dict) -> int:
    return scaled(CAL_ELEMS // per_batch(sh_))


def boundaries(sh_: dict, dtype: str, p: int | None = None) -> list[tuple[str, int]]:
    """(ラベル, 境界ちょうどの B0)。同じ B0 になる境界は 1 つにまとめる。p = バッチあたり要素数(既定は境界をまたぐ量)。"""
    p = p or per_batch(sh_)
    raw = [("2^31 elem", scaled(2**31 // p)), ("2^32 elem", scaled(2**32 // p)),
           ("2^32 byte", scaled(2**32 // (p * ITEMSIZE[dtype])))]
    merged: dict[int, list[str]] = defaultdict(list)
    for lab, b0 in raw:
        merged[b0].append(lab)
    return sorted(((" = ".join(v), k) for k, v in merged.items()), key=lambda x: x[1])


def probe_points(sh_: dict, dtype: str, grid: int) -> list[tuple[int, str]]:
    """小 shape 点 + 各境界の B0-2〜B0+2 + 隣り合う基準点の間に grid 点ずつの等間隔グリッド。"""
    pts = {cal_B(sh_): "small (2^28 elem)"}
    anchors = [cal_B(sh_)]
    for lab, b0 in boundaries(sh_, dtype):
        for d in (-2, -1, 0, 1, 2):
            pts[b0 + d] = f"{lab} {d:+d}" if d else lab
        anchors.append(b0)
    for lo, hi in zip(anchors, anchors[1:]):
        for k in range(1, grid + 1):
            b = lo + round((hi - lo) * k / (grid + 1))
            pts.setdefault(b, "grid")
    return sorted((b, lab) for b, lab in pts.items() if b >= 1)  # b < 1 は --scale-log2 が大きいときだけ起こる


def idx_points(sh_: dict, dtype: str) -> list[tuple[int, str]]:
    pts = {cal_B(sh_): "small (2^28 elem)"}
    for lab, b0 in boundaries(sh_, dtype):
        pts[b0 - 1], pts[b0 + 1] = f"{lab} -1", f"{lab} +1"
    return sorted((b, lab) for b, lab in pts.items() if b >= 1)


def offset_per_batch(sh_: dict) -> int:
    """offset 系列で storage_offset を数えるオペランド(a, b のうちバッチあたり要素数が大きい方。同じなら両方)の要素数。"""
    return max(sh_["M"] * sh_["K"], sh_["K"] * sh_["N"])


def offset_B(args) -> int:
    return max(4, scaled(args.offset_B))


def offset_points(sh_: dict, dtype: str, grid: int) -> list[tuple[int, str]]:
    """offset 系列の pad_batches O の点: O = 1(slice と同じ offset)、storage_offset が各境界に達する O0 の O0-2〜O0+2、
    その間に grid 点。O0-2, O0-1 は view が境界をまたぎ、O0 以上は view 全体が境界の先にある。"""
    pts = {1: "O=1"}
    anchors = [1]
    for lab, o0 in boundaries(sh_, dtype, offset_per_batch(sh_)):
        for d in (-2, -1, 0, 1, 2):
            pts.setdefault(o0 + d, f"offset {lab} {d:+d}" if d else f"offset {lab}")
        anchors.append(o0)
    for lo, hi in zip(anchors, anchors[1:]):
        for k in range(1, grid + 1):
            pts.setdefault(lo + round((hi - lo) * k / (grid + 1)), "grid")
    return sorted((o, lab) for o, lab in pts.items() if o >= 1)


def far_points(sh_: dict, far: int) -> list[tuple[int, str]]:
    """--far K: 2**32 要素の B0 の k 倍(k = 2..K)の前後 1 点。"""
    b0 = scaled(2**32 // per_batch(sh_))
    return [(b0 * k + d, f"{k}x2^32 elem {d:+d}" if d else f"{k}x2^32 elem") for k in range(2, far + 1) for d in (-1, 0, 1)]


def make_spec(shape_id, dtype, layout, kind, B, seed, compare, tol, device, pad_batches=None) -> dict:
    s = SHAPE[shape_id]
    spec = {"op": OP, "device": device, "shape_id": shape_id, "M": s["M"], "K": s["K"], "N": s["N"],
            "cross": s["cross"], "dtype": dtype, "layout": layout, "kind": kind, "B": int(B), "seed": seed,
            "compare": compare, "tol": tol}
    if layout == OFFSET:  # v1 のレイアウトの spec には足さない(キーを増やさない)
        spec["pad_batches"] = int(pad_batches)
    return spec


def series_id(s: dict) -> str:
    return f"{s.get('op', 'bmm')}/{s['shape_id']}/{s['dtype']}/{s['layout']}/{s['kind']}"


def shape_info(spec: dict) -> dict:
    M, K, N, B = spec["M"], spec["K"], spec["N"], spec["B"]
    s = ITEMSIZE[spec["dtype"]]
    na, nb, no = B * M * K, B * K * N, B * M * N
    cross = na if spec["cross"] == "in" else no
    d = {"shape_a": [B, M, K], "shape_b": [B, K, N], "shape_out": [B, M, N],
         "numel_a": na, "numel_b": nb, "numel_out": no, "bytes_a": na * s, "bytes_b": nb * s, "bytes_out": no * s,
         "crossing_numel": cross, "crossing_log2": round(math.log2(cross), 6)}
    if spec["layout"] == OFFSET:  # numel_a / numel_b は view の要素数のまま。格納と offset は別のキー
        O = spec["pad_batches"]
        d.update(offset_numel_a=O * M * K, offset_numel_b=O * K * N, storage_numel_a=(B + O) * M * K,
                 storage_numel_b=(B + O) * K * N,
                 offset_log2=round(math.log2(O * max(M * K, K * N)), 6) if O else None)
    return d


HOST_OVERHEAD = 3 * 2**30  # CPU 側: torch 本体 + 生成・転送のチャンク + fp64 正解の比較チャンク(1 チャンク 256MB × 数本)


def mem_est(spec: dict) -> dict:
    """メモリの見積もり(バイト)。dev = device 上の a, b(slice は 1 バッチ多い), out の 1.05 倍。
    実測(M2 Ultra, torch 2.14)では bT / aT でも連続化のコピーは作られず driver_allocated ≈ a+b+out。
    host = CPU 側。MPS は unified memory なので dev も host の RAM を使う(host = dev + HOST_OVERHEAD)。
    CUDA では host = HOST_OVERHEAD(入力・正解はチャンクごとに作って捨てるので B に比例しない)。"""
    M, K, N, B = spec["M"], spec["K"], spec["N"], spec["B"]
    s = ITEMSIZE[spec["dtype"]]
    pad = {"slice": 1, OFFSET: spec.get("pad_batches", 0)}.get(spec["layout"], 0)  # offset は大きい格納を含める
    elems = (B + pad) * (M * K + K * N) + B * M * N
    if spec.get("op") == "bmm_bwd":  # 上流の勾配 G(出力と同じ形)と grad_a, grad_b
        elems += B * M * N + B * (M * K + K * N)
    dev = int(elems * s * 1.05)
    host = dev + HOST_OVERHEAD if spec.get("device", "mps") == "mps" else HOST_OVERHEAD
    return {"dev": dev, "host": host}


class MemLimits:
    """MPS: host(= unified)の RAM × mem_frac。CUDA: GPU メモリ × gpu_frac と、ホスト RAM × mem_frac の両方。"""

    def __init__(self, env: dict, wenv: dict, args):
        self.device = args.device
        self.host = int(env.get("mem_bytes", 0) * args.mem_frac)
        self.dev = None
        self.note = ""
        if self.device == "cuda":
            gm = env.get("gpu_mem_bytes") or wenv.get("gpu_mem_bytes")
            if getattr(args, "gpu_mem_gib", None):
                gm, self.note = int(args.gpu_mem_gib * 2**30), "--gpu-mem-gib で指定"
            if not gm:
                gm, self.note = 40 * 2**30, "GPU メモリを取得できず 40 GiB と仮定"
            self.dev = int(gm * args.gpu_frac)

    def check(self, spec: dict) -> tuple[bool, dict]:
        est = mem_est(spec)
        over = est["host"] > self.host or (self.dev is not None and est["dev"] > self.dev)
        return over, est

    def as_dict(self) -> dict:
        return {"mem_limit_host_bytes": self.host, "mem_limit_dev_bytes": self.dev, "mem_limit_note": self.note}

    def __str__(self) -> str:
        s = f"host {self.host/2**30:.0f}GiB"
        return s + (f" / GPU {self.dev/2**30:.0f}GiB {self.note}" if self.dev is not None else "")


def time_est(spec: dict) -> float:
    """1 実行の所要時間の見積もり(秒)。Studio (M2 Ultra) での実測から当てはめた粗い式。"""
    M, K, N, B = spec["M"], spec["K"], spec["N"], spec["B"]
    ab, out = B * (M * K + K * N), B * M * N
    gen = (1.0e-9 if spec["kind"] == "random" else 0.2e-9) * ab
    t = 1.3 + gen + 0.3e-9 * spec.get("pad_batches", 0) * (M * K + K * N)  # pad は定数で埋めるだけ
    if spec["compare"] == "full":
        t += gen + 2 * B * M * K * N / 1.2e11 + 0.5e-9 * out
    return t * (3 if spec.get("op") == "bmm_bwd" else 1)  # G の生成と、正解の積が 3 つ


def fingerprint(spec: dict, env: dict, code: dict, wenv: dict) -> str:
    key = {"spec": spec, "env": {k: env.get(k) for k in ("os", "os_release", "macos", "macos_build", "hw_model",
                                                          "chip", "mem_bytes", "gpu_name", "gpu_driver")},
           "worker": {k: wenv.get(k) for k in ("python", "torch")},
           "code": {k: code.get(k) for k in ("git_rev", "git_dirty_code", "sha_scan", "sha_worker")}}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------- 実行と記録

SECRET_KEY_PAT = ("key", "token", "secret", "password", "passwd", "netrc")


def scrub(x, secrets: tuple[str, ...] = ()):
    """W&B の config に渡す前に、秘密らしいキー(key / token / secret / password)と、値が API キーそのものの文字列を落とす。
    manifest には環境変数を書いていないが、argv に紛れた場合の保険。"""
    if isinstance(x, dict):
        return {k: scrub(v, secrets) for k, v in x.items() if not any(w in str(k).lower() for w in SECRET_KEY_PAT)}
    if isinstance(x, (list, tuple)):
        return [scrub(v, secrets) for v in x]
    if isinstance(x, str) and any(sec and sec in x for sec in secrets):
        return "<redacted>"
    return x


class WandbSink:
    """結果の退避としての W&B(--wandb のときだけ)。走査 1 回 = run 1 つ(name = id = run_id、group = host_tag)。

    - 行ごと: 分類ごとの累計・行数・最後の行の series / B / cls を log する
    - every 行ごとと終了時: <run_id>.jsonl と <run_id>.manifest.json を Artifact(type sweep-results、name = run_id)として上げる。
      ファイルをそのまま入れるので、Artifact を落とせば JSONL をバイト単位で復元できる。正常終了の版に alias `final`
    - wandb の import はここで初めて行う。どの呼び出しも try/except で包み、失敗は <run_id>.wandb.log に書いて走査は続ける。
      init に失敗したら次の退避のタイミングで再試行する(最大 INIT_TRIES 回)。JSONL は必ずこの前に書いてある
    - API キーは環境変数 WANDB_API_KEY か ~/.netrc から wandb が読む。ここでは読まないし、どこにも書かない"""

    INIT_TRIES = 3
    INIT_TIMEOUT = 60
    FINISH_TIMEOUT = 600

    def __init__(self, R: "Runner", project: str, every: int, rows: list[dict]):
        self.R, self.project, self.every = R, project, max(1, every)
        self.wandb = self.run = self.url = self.error = None
        self.tries = 0
        self.counts = Counter(r["cls"] for r in rows)  # --resume のときは既存の行から数え直す
        self.n_rows = len(rows)
        self.n_art = 0
        self.start()

    def info(self) -> dict:
        return {"enabled": True, "project": self.project, "every": self.every, "url": self.url,
                "artifact": f"{self.R.run_id} (type sweep-results)", "init_error": self.error}

    def start(self) -> None:
        if self.run is not None or self.tries >= self.INIT_TRIES:
            return
        self.tries += 1
        try:
            import wandb  # --wandb のときだけ import する
            self.wandb = wandb
            man = json.loads(self.R.manifest_path.read_text())
            cfg = scrub({k: v for k, v in man.items() if k not in ("updated", "status", "wandb")},
                        (os.environ.get("WANDB_API_KEY", ""),))
            self.run = wandb.init(project=self.project, name=self.R.run_id, id=self.R.run_id, resume="allow",
                                  group=self.R.args.host_tag, job_type="sweep", config=cfg, dir=str(ROOT / "results"),
                                  settings=wandb.Settings(silent=True, console="off", save_code=False,
                                                          init_timeout=self.INIT_TIMEOUT))
            self.url, self.error = self.run.url, None
        except Exception as e:  # noqa: BLE001
            self.run = None
            self.error = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
            try:  # 失敗した init が service 側に run ID を残すと再試行が "run ID is in use" になるので片付ける
                self.wandb.teardown()
            except Exception:  # noqa: BLE001
                pass
            self.R.wb_note(f"init failed ({self.tries}/{self.INIT_TRIES}): {self.error}")

    def after_row(self, row: dict) -> None:
        self.counts[row["cls"]] += 1
        self.n_rows += 1
        if self.run is not None:
            try:
                self.run.log({"rows": self.n_rows, **{f"count/{k}": v for k, v in self.counts.items()},
                              "last/series": row["series_id"], "last/B": row["B"], "last/cls": row["cls"],
                              "last/phase": row["phase"], "last/wall_sec": row.get("wall_sec") or 0})
            except Exception as e:  # noqa: BLE001
                self.R.wb_note(f"log failed: {type(e).__name__}: {e}")
        if self.n_rows % self.every == 0:
            self.upload(["latest"])

    def upload(self, aliases: list[str]) -> bool:
        if self.run is None:
            self.start()
            if self.run is not None:
                self.R.write_manifest(None, {"wandb": self.info()})
        if self.run is None:
            return False
        try:
            art = self.wandb.Artifact(name=self.R.run_id, type="sweep-results",
                                      metadata={"n_rows": self.n_rows, "counts": dict(self.counts), "schema": SCHEMA})
            art.add_file(str(self.R.raw))  # この時点のファイルのコピーが staging に取られる
            art.add_file(str(self.R.manifest_path))
            self.run.log_artifact(art, aliases=aliases)
            self.n_art += 1
            return True
        except Exception as e:  # noqa: BLE001
            self.R.wb_note(f"artifact failed: {type(e).__name__}: {e}")
            return False

    def close(self, status: str) -> None:
        """最後の Artifact を上げて run を閉じる。finish は別スレッドで待ち、FINISH_TIMEOUT を超えたら諦める。"""
        ok = self.upload(["final"] if status == "done" else [status])
        if self.run is None:
            return
        done = threading.Event()

        def fin():
            try:
                self.run.finish(exit_code=0 if status == "done" else 1)
            except Exception as e:  # noqa: BLE001
                self.R.wb_note(f"finish failed: {type(e).__name__}: {e}")
            done.set()

        threading.Thread(target=fin, daemon=True).start()
        if not done.wait(self.FINISH_TIMEOUT):
            self.R.wb_note(f"finish: {self.FINISH_TIMEOUT}s で終わらず打ち切り")
        elif ok:
            self.R.wb_note(f"closed: status={status} rows={self.n_rows} artifacts={self.n_art}")


class Runner:
    def __init__(self, args, run_id: str):
        self.args = args
        self.run_id = run_id
        self.env = host_env(args.device)
        self.code = code_info()
        self.wenv = worker_env(args)
        self.limits = MemLimits(self.env, self.wenv, args)
        d = ROOT / "results" / "raw" / args.host_tag
        d.mkdir(parents=True, exist_ok=True)
        self.raw = d / f"{run_id}.jsonl"
        self.manifest_path = d / f"{run_id}.manifest.json"
        self.wblog = d / f"{run_id}.wandb.log"
        self.bklog = d / f"{run_id}.backup.log"
        self.cache: dict[str, dict] = {}
        old_rows = []
        if self.raw.exists():
            for line in self.raw.read_text().splitlines():
                r = json.loads(line)
                old_rows.append(r)
                if r["cls"] not in ("skipped_memory", "skipped_budget"):
                    self.cache[r["fingerprint"]] = r
        self.t_start = time.time()
        self.n_new = 0
        self.closed = False
        self.write_manifest("running", {"wandb": {"enabled": False}}, new_invocation=True)
        self.sink = None
        if args.wandb:
            self.sink = WandbSink(self, args.wandb_project, args.wandb_every, old_rows)
            self.write_manifest(None, {"wandb": self.sink.info()})

    def write_manifest(self, status: str | None, extra: dict | None = None, new_invocation: bool = False) -> None:
        """status = None なら前の status のまま、extra だけ書き足す。"""
        old = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        m = {**old, "schema": SCHEMA, "run_id": self.run_id, "host_tag": self.args.host_tag,
             "status": status or old.get("status"),
             "updated": datetime.datetime.now().isoformat(timespec="seconds"),
             "env": self.env, "code": self.code, "worker": self.wenv, "device": self.args.device,
             "torch_req": self.args.torch, "python_req": self.args.python, "scale_log2": SCALE_LOG2, "op": OP,
             "mode": "points" if getattr(self.args, "points_list", None) is not None else "sweep",
             "worker_cmd": " ".join(shlex.quote(x) for x in worker_cmd(self.args)) + " '<spec>'",
             **self.limits.as_dict(), **(extra or {})}
        m.setdefault("invocations", [])
        if new_invocation:
            m["invocations"].append({"started": m["updated"], "argv": sys.argv, "code": self.code})
        self.manifest_path.write_text(json.dumps(m, indent=1, default=str) + "\n")

    def elapsed_h(self) -> float:
        return (time.time() - self.t_start) / 3600

    def run(self, spec: dict, phase: str, probe: str | None = None, skip: str | None = None) -> dict:
        fp = fingerprint(spec, self.env, self.code, self.wenv)
        if fp in self.cache:
            return self.cache[fp]
        row = {"schema": SCHEMA, "run_id": self.run_id, "fingerprint": fp, "host_tag": self.args.host_tag,
               "ts": datetime.datetime.now().isoformat(timespec="seconds"), "phase": phase,
               "series_id": series_id(spec), **spec, **shape_info(spec), "probe": probe,
               "torch_req": self.args.torch, "python_req": self.args.python, **self.env,
               "git_rev": self.code["git_rev"], "git_dirty_code": self.code["git_dirty_code"],
               **self.limits.as_dict()}
        if SCALE_LOG2:
            row["scale_log2"] = SCALE_LOG2  # テスト用の縮小 run の印。versions.py などはこの行を拒否する
        over, est = self.limits.check(spec)
        row.update(mem_est_dev_bytes=est["dev"], mem_est_host_bytes=est["host"])
        if skip:
            row.update(cls=skip)
        elif over:
            row.update(cls="skipped_memory")
        else:
            t = time.time()
            try:
                p = subprocess.run(worker_cmd(self.args) + [json.dumps(spec)], capture_output=True, text=True,
                                   timeout=TIMEOUT[spec["compare"]], cwd=ROOT)
                line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
                row["rc"] = p.returncode
                if line:
                    row.update(json.loads(line[7:]))
                else:
                    tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-3:]
                    row.update(cls="crash", stderr_tail=" | ".join(tail)[-500:])
            except subprocess.TimeoutExpired:
                row.update(cls="timeout", rc=None)
            row["wall_sec"] = round(time.time() - t, 2)
        with self.raw.open("a") as f:  # W&B・退避コマンドより先に必ず書く
            f.write(json.dumps(row, default=str) + "\n")
        if row["cls"] not in ("skipped_memory", "skipped_budget"):
            self.cache[fp] = row
        self.n_new += 1
        if self.sink:
            self.sink.after_row(row)
        if self.args.backup_cmd and self.n_new % max(1, self.args.backup_every) == 0:
            self.backup()
        err = row.get("max_rel_err")
        hyp = row.get("hyp_match")
        hs = ""
        if hyp:
            best = max(hyp.items(), key=lambda kv: kv[1] or 0)
            hs = f" hyp={best[0]}:{best[1]}"
        o_str = f" O={spec['pad_batches']}" if "pad_batches" in spec else ""
        print(f"[{self.elapsed_h()*60:6.1f}m] {phase:9s} {series_id(spec):38s} B={spec['B']:6d}{o_str} s{spec['seed']} "
              f"{spec['compare']:6s} -> {row['cls']:14s} err={err if not isinstance(err, float) else f'{err:.3g}'} "
              f"bad={row.get('n_batches_bad')}/{row.get('n_batches_compared')}{hs} wall={row.get('wall_sec')}s",
              flush=True)
        return row

    def backup(self) -> None:
        """--backup-cmd をシェルで実行する({jsonl} {manifest} {dir} を絶対パスに置換、shlex.quote 済み)。
        失敗・タイムアウトは <run_id>.backup.log に書くだけで、走査は止めない。"""
        cmd = self.args.backup_cmd
        if not cmd:
            return
        for k, v in (("{jsonl}", self.raw), ("{manifest}", self.manifest_path), ("{dir}", self.raw.parent)):
            cmd = cmd.replace(k, shlex.quote(str(v.resolve())))
        t = time.time()
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.args.backup_timeout)
            msg = f"rc={p.returncode} {time.time()-t:.1f}s rows={self.n_new}"
            if p.returncode != 0:
                msg += f" FAILED stderr={(p.stderr or '').strip()[-300:]!r}"
        except subprocess.TimeoutExpired:
            msg = f"TIMEOUT after {self.args.backup_timeout}s rows={self.n_new}"
        except Exception as e:  # noqa: BLE001
            msg = f"FAILED {type(e).__name__}: {e}"
        with self.bklog.open("a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")

    def close(self, status: str, extra: dict | None = None) -> None:
        """終了処理(正常終了・中断・例外のどれでも 1 回だけ): manifest → 退避コマンド → W&B の最後の Artifact。"""
        if self.closed:
            return
        self.closed = True
        try:
            self.write_manifest(status, {"elapsed_min": round(self.elapsed_h() * 60, 1), "n_rows_new": self.n_new,
                                         **({"wandb": self.sink.info()} if self.sink else {}), **(extra or {})})
        except Exception as e:  # noqa: BLE001
            print(f"# manifest の更新に失敗: {e}", flush=True)
        self.backup()
        if self.sink:
            try:
                self.sink.close(status)
            except Exception as e:  # noqa: BLE001
                self.wb_note(f"close failed: {type(e).__name__}: {e}")

    def wb_note(self, msg: str) -> None:
        with self.wblog.open("a") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")


# ---------------------------------------------------------------------------- 走査

def is_ok(r: dict) -> bool | None:
    """二分探索用: ok → True、異常 → False、判定不能(メモリ超過・予算超過) → None。"""
    if r["cls"] in ("skipped_memory", "skipped_budget"):
        return None
    return r["cls"] == "ok"


def calibrate(R: Runner, pairs: list[tuple[dict, str]]) -> tuple[dict, dict]:
    """その run に含まれる (shape, dtype) 全部で較正する(v1 は基本 shape だけで、追加 shape だけの run は固定値 1e-4 に落ちていた)。
    dtype ごとに 許容誤差 = max(TOL_FACTOR × 較正誤差の最大, TOL_FLOOR)。返り値の 2 つ目は manifest に残す内訳。"""
    tol, info = {}, {}
    for dt in dict.fromkeys(d for _, d in pairs):
        errs: dict[str, list[float]] = {}
        for s in [x for x, d in pairs if d == dt]:
            for sd in CAL_SEEDS:
                r = R.run(make_spec(s["id"], dt, "contig", "random", cal_B(s), sd, "full", 1.0, R.args.device),
                          "calibrate")
                if isinstance(r.get("max_rel_err"), float):
                    errs.setdefault(s["id"], []).append(r["max_rel_err"])
        flat = [e for v in errs.values() for e in v]
        tol[dt] = max(TOL_FACTOR * max(flat), TOL_FLOOR[dt]) if flat else TOL_FALLBACK[dt]
        info[dt] = {"source": "calibrated" if flat else "fallback", "shapes": sorted(errs), "n_runs": len(flat),
                    "max_err": max(flat) if flat else None, "max_err_by_shape": {k: max(v) for k, v in errs.items()},
                    "factor": TOL_FACTOR, "floor": TOL_FLOOR[dt], "tol": tol[dt]}
    return tol, info


def cal_pairs(shapes, dtypes, args) -> list[tuple[dict, str]]:
    """較正する (shape, dtype)。dtype の順 → shape の順。offset だけの run でも基本 shape は含まれる。"""
    return [(s, dt) for dt in dtypes for s in shapes if dt in dtypes_for(s, dtypes, args)]


def transitions(known: dict[int, dict]) -> list[tuple[int, int]]:
    pts = sorted(b for b in known if is_ok(known[b]) is not None)
    return [(lo, hi) for lo, hi in zip(pts, pts[1:]) if is_ok(known[lo]) != is_ok(known[hi])]


def bisect(R: Runner, known: dict[int, dict], mk, phase: str) -> dict:
    """known(走査した量 → 行)の、正常/異常が異なる隣接点の間を粒度 1 まで二分探索する。mk(x) が spec を作る。"""
    while True:
        gaps = [(lo, hi) for lo, hi in transitions(known) if hi - lo > 1]
        if not gaps:
            break
        lo, hi = gaps[0]
        mid = (lo + hi) // 2
        known[mid] = R.run(mk(mid), phase)
        if is_ok(known[mid]) is None:
            break
    pts = sorted(b for b in known if is_ok(known[b]) is not None)
    return {"transitions": transitions(known), "max_B": pts[-1] if pts else None,
            "ok": {b: is_ok(known[b]) for b in pts}}


def scan_series(R: Runner, s, dt, lay, tol, grid) -> dict:
    """乱数入力 seed 0 のサンプル比較で探索点を回し、正常/異常が異なる隣接点の間を二分探索する。"""
    dev = R.args.device
    known: dict[int, dict] = {}
    for B, lab in probe_points(s, dt, grid):
        cmp_ = "full" if B == cal_B(s) else "sample"
        known[B] = R.run(make_spec(s["id"], dt, lay, "random", B, 0, cmp_, tol, dev), "probe", lab)
    return bisect(R, known, lambda mid: make_spec(s["id"], dt, lay, "random", mid, 0, "sample", tol, dev), "bisect")


def scan_offset_series(R: Runner, s, dt, tol, args, final_seeds, over) -> dict:
    """offset 系列: B は固定(小さい)で pad_batches O を振る。B が小さいので探索点から全要素比較にする。
    切り替わりの下・上(なければ最大の O)は seed 0 以外の final_seeds でも全要素比較する。max_B の欄は O の最大。"""
    dev, B = R.args.device, offset_B(args)
    mk = lambda O, sd=0: make_spec(s["id"], dt, OFFSET, "random", B, sd, "full", tol, dev, pad_batches=O)  # noqa: E731
    known = {O: R.run(mk(O), "offset", lab) for O, lab in offset_points(s, dt, args.grid_offset)}
    info = bisect(R, known, mk, "offset_bis")
    for O in final_points(info):
        for sd in [x for x in final_seeds if x != 0]:
            R.run(mk(O, sd), "offset_fin", skip="skipped_budget" if over() else None)
    return info


def final_points(info: dict) -> list[int]:
    if info["transitions"]:
        return sorted({b for t in info["transitions"][:MAX_TRANS] for b in t})
    return [info["max_B"]] if info["max_B"] else []


def abnormal_points(info: dict) -> list[int]:
    """各切り替わりの異常側の点(正常 → 異常なら上側、異常 → 正常なら下側)。"""
    return sorted({hi if info["ok"][lo] else lo for lo, hi in info["transitions"][:MAX_TRANS]})


def dtypes_for(s: dict, dtypes: list[str], args) -> list[str]:
    return dtypes if s["base"] or s.get("all_dtypes") else [d for d in dtypes if d in args.extra_dtypes.split(",")]


def load_points(path: str) -> list[dict]:
    """--points FILE: 指定した点だけを回す(JSON の配列)。各要素は {"shape": id, "dtype", "layout", "B"} が必須で、
    "kind"(既定 random)、"seed"(0)、"compare"(full)、"pad_batches"(layout = offset のとき必須)を足せる。"""
    pts = json.loads(Path(path).read_text())
    for p in pts:
        assert p["shape"] in SHAPE and p["dtype"] in ITEMSIZE and p["layout"] in LAYOUTS + [OFFSET], p
        assert p["layout"] != OFFSET or "pad_batches" in p, p
    return pts


def new_run_id(host_tag: str) -> str:
    return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{host_tag}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-tag", required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda"])
    ap.add_argument("--op", default="bmm", choices=["bmm", "bmm_bwd"],
                    help="bmm_bwd = forward + backward(out, grad_a, grad_b を比べ、cls は最悪のもの)。index 符号化は回さない")
    ap.add_argument("--torch", default="2.14.0")
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--dtypes", default="fp32,fp16", help="fp32 / fp16 / bf16(bf16 は既定に入れていない)")
    ap.add_argument("--layouts", default=",".join(LAYOUTS),
                    help=f"contig,bT,aT,slice と {OFFSET}({OFFSET} = 大きい storage_offset の系列。既定には入れていない)")
    ap.add_argument("--with-offset", action="store_true", help=f"--layouts に {OFFSET} を足すのと同じ")
    ap.add_argument("--offset-B", type=int, default=4096, help="offset 系列の B(固定。振るのは pad_batches)")
    ap.add_argument("--grid-offset", type=int, default=3, help="offset 系列で境界の間に置くグリッド点の数")
    ap.add_argument("--shapes", default=",".join(s["id"] for s in SHAPES),
                    help="既定は v1 の 6 つ。追加: " + ", ".join(s["id"] for s in EXTRA_SHAPES))
    ap.add_argument("--far", type=int, default=0, metavar="K",
                    help="各系列に 2^32 要素の B0 の k 倍(k = 2..K)の前後 1 点を足す(サンプル比較。基本 shape は k·B0+1 で index 符号化も)")
    ap.add_argument("--points", metavar="FILE", default=None,
                    help="走査の代わりに、JSON で指定した点だけを回す(較正はその点の shape / dtype で行う)。書式は load_points")
    ap.add_argument("--idx-kinds", default=",".join(IDX_KINDS), help="index 符号化入力(基本 shape のみ)。空で省略")
    ap.add_argument("--grid-base", type=int, default=7, help="基本 shape で境界の間に置くグリッド点の数")
    ap.add_argument("--grid-extra", type=int, default=3, help="追加 shape で境界の間に置くグリッド点の数")
    ap.add_argument("--extra-dtypes", default="fp32", help="追加 shape で回す dtype(dtype の効果は基本 shape で見る)")
    ap.add_argument("--mem-frac", type=float, default=0.6, help="メモリ見積もりの上限(ホスト RAM 比。MPS は unified)")
    ap.add_argument("--gpu-frac", type=float, default=0.9, help="CUDA: device 側の見積もりの上限(GPU メモリ比)")
    ap.add_argument("--gpu-mem-gib", type=float, default=None, help="CUDA: GPU メモリ量を指定(取得できない dry-run 用)")
    ap.add_argument("--budget-hours", type=float, default=2.2, help="超えたら全要素比較・index 符号化を skipped_budget にする")
    ap.add_argument("--final-seeds", default=",".join(map(str, FINAL_SEEDS)), help="最終点の全要素比較の seed")
    ap.add_argument("--final-idx-kinds", default=None, help="最終点の index 符号化(既定は --idx-kinds と同じ)")
    ap.add_argument("--final-idx-where", default="both", choices=["both", "abnormal"],
                    help="最終点の index 符号化を切り替わりの両側で回すか、異常側 1 点だけか")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", metavar="RUN_ID", help="この run_id の続きから(同じ run の中だけでキャッシュが効く)")
    ap.add_argument("--fresh", action="store_true", help="新しい run_id を切る(既定の動作。明示用)")
    ap.add_argument("--summary", metavar="RUN_ID", help="実行せずその run の summary を作り直す")
    ap.add_argument("--wandb", action="store_true",
                    help="結果の退避として W&B に送る(既定は送らない)。走査 1 回 = run 1 つ。キーは WANDB_API_KEY か ~/.netrc")
    ap.add_argument("--wandb-project", default=PROJECT)
    ap.add_argument("--wandb-every", type=int, default=50, help="この行数ごとに JSONL と manifest を Artifact として上げる")
    ap.add_argument("--backup-cmd", default=None,
                    help="退避コマンド(シェル)。{jsonl} {manifest} {dir} を絶対パスに置換して実行する。"
                         "例: 'rsync -a {jsonl} {manifest} me@host:mps-boundary/'。失敗しても走査は続ける")
    ap.add_argument("--backup-every", type=int, default=200, help="この行数ごと(と終了時)に --backup-cmd を実行")
    ap.add_argument("--backup-timeout", type=float, default=300, help="--backup-cmd 1 回の上限(秒)")
    ap.add_argument("--scale-log2", type=int, default=0, metavar="S",
                    help="テスト専用: 理論境界と B の点を 2**S で割る(shape はそのまま。S=10 なら 2^32 境界が B=64)。"
                         "行と manifest に scale_log2 が付き、versions.py などは拒否する")
    args = ap.parse_args()
    global SCALE_LOG2, OP
    SCALE_LOG2, OP = args.scale_log2, args.op
    dtypes, layouts = args.dtypes.split(","), [x for x in args.layouts.split(",") if x]
    if args.with_offset and OFFSET not in layouts:
        layouts.append(OFFSET)
    shapes = [SHAPE[x] for x in args.shapes.split(",")]
    args.points_list = load_points(args.points) if args.points else None
    idx_kinds = [k for k in args.idx_kinds.split(",") if k] if args.op == "bmm" else []  # 勾配は符号から逆算できない
    final_seeds = [int(x) for x in args.final_seeds.split(",") if x]
    final_idx_kinds = idx_kinds if args.final_idx_kinds is None else [k for k in args.final_idx_kinds.split(",") if k]
    final_idx_kinds = final_idx_kinds if args.op == "bmm" else []

    if args.dry_run:
        dry_run(args, dtypes, layouts, shapes, idx_kinds, final_seeds, final_idx_kinds)
        return
    if args.summary:
        write_summary(args.host_tag, args.summary)
        return

    run_id = args.resume or new_run_id(args.host_tag)
    R = Runner(args, run_id)
    print(f"# run_id={run_id} {R.env} worker={R.wenv} code={R.code['git_rev'][:10]}"
          f"{' (dirty)' if R.code['git_dirty_code'] else ''} limit={R.limits}"
          + (f" scale_log2={SCALE_LOG2}(テスト用)" if SCALE_LOG2 else "")
          + (f" wandb={R.sink.url or R.sink.error}" if R.sink else ""), flush=True)
    def on_term(signum, frame):  # クラウドの VM の停止は SIGTERM で来る。Ctrl-C と同じ終了処理に回す
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_term)
    try:  # 中断・例外でも manifest の status、退避コマンド、W&B の最後の Artifact までは行う
        sweep(R, args, dtypes, layouts, shapes, idx_kinds, final_seeds, final_idx_kinds)
    except KeyboardInterrupt:
        R.close("interrupted")
        raise
    except BaseException as e:
        R.close("failed", {"failure": f"{type(e).__name__}: {e}"})
        raise
    print(f"# 新規 {R.n_new} 実行, {R.elapsed_h()*60:.1f} 分" + ("。W&B の送信完了を待つ" if R.sink else ""), flush=True)
    R.close("done")
    write_summary(args.host_tag, run_id)


def run_points(R: Runner, args) -> None:
    """--points: 指定点だけ。較正は点に出てくる (shape, dtype) で行い、その許容誤差で各点を 1 回ずつ実行する。"""
    pts = args.points_list
    pairs = list(dict.fromkeys((p["shape"], p["dtype"]) for p in pts))
    tol, cal_info = calibrate(R, [(SHAPE[sid], dt) for sid, dt in sorted(pairs, key=lambda x: x[1])])
    R.write_manifest(None, {"tol": tol, "tol_calibration": cal_info, "points": pts})
    print(f"# tol = {tol}", flush=True)
    for p in pts:
        B = p["B"] if p["layout"] != OFFSET else p.get("B", offset_B(args))
        R.run(make_spec(p["shape"], p["dtype"], p["layout"], p.get("kind", "random"), B, p.get("seed", 0),
                        p.get("compare", "full"), tol[p["dtype"]], args.device, pad_batches=p.get("pad_batches")),
              "point", p.get("label"))


def sweep(R: Runner, args, dtypes, layouts, shapes, idx_kinds, final_seeds, final_idx_kinds) -> None:
    if args.points_list is not None:
        return run_points(R, args)
    with_offset = OFFSET in layouts
    layouts = [x for x in layouts if x != OFFSET]  # 以下の B を振る系列は標準のレイアウトだけ
    off_series = [(s, dt) for dt in dtypes if dt in OFFSET_DTYPES for s in shapes if s["base"]] if with_offset else []
    pairs = cal_pairs(shapes, dtypes, args) if layouts else []
    pairs += [x for x in off_series if x not in pairs]
    tol, cal_info = calibrate(R, pairs)
    R.write_manifest(None, {"tol": tol, "tol_calibration": cal_info})
    if R.limits.check(make_spec(shapes[0]["id"], dtypes[0], "contig", "random", cal_B(shapes[0]), 0, "full", 1.0,
                                args.device))[0]:
        print(f"# 警告: 較正の小 shape 点すらメモリ上限 ({R.limits}) を超える。環境の取得を確認すること", flush=True)
    print(f"# tol = {tol}", flush=True)
    grid = lambda s: args.grid_base if s["base"] else args.grid_extra  # noqa: E731
    over = lambda: R.elapsed_h() > args.budget_hours  # noqa: E731
    infos = {}
    for base in (True, False):  # 探索(乱数入力): 基本 shape → 追加 shape
        for dt in dtypes:
            for s in [x for x in shapes if x["base"] == base and dt in dtypes_for(x, dtypes, args)]:
                for lay in layouts:
                    infos[(s["id"], dt, lay)] = scan_series(R, s, dt, lay, tol[dt], grid(s))
    for s, dt in off_series:  # offset 系列(基本 shape、fp32 / fp16)
        scan_offset_series(R, s, dt, tol[dt], args, final_seeds, over)
    for kd in idx_kinds:  # index 符号化入力(基本 shape)
        for dt in dtypes:
            for s in [x for x in shapes if x["base"]]:
                for lay in layouts:
                    for B, lab in idx_points(s, dt):
                        R.run(make_spec(s["id"], dt, lay, kd, B, 0, "sample", tol[dt], args.device), "idx_probe", lab,
                              skip="skipped_budget" if over() else None)
    for dt in dtypes:  # 最終点の全要素比較: fp32 → fp16、基本 shape → 追加 shape
        for base in (True, False):
            for s in [x for x in shapes if x["base"] == base and dt in dtypes_for(x, dtypes, args)]:
                for lay in layouts:
                    for B in final_points(infos[(s["id"], dt, lay)]):
                        for sd in final_seeds:
                            R.run(make_spec(s["id"], dt, lay, "random", B, sd, "full", tol[dt], args.device), "final",
                                  skip="skipped_budget" if over() else None)
                    info = infos[(s["id"], dt, lay)]
                    idx_B = abnormal_points(info) if args.final_idx_where == "abnormal" else final_points(info)
                    for B in (idx_B if base else []):
                        for kd in final_idx_kinds:
                            R.run(make_spec(s["id"], dt, lay, kd, B, 0, "full", tol[dt], args.device), "final",
                                  skip="skipped_budget" if over() else None)
    for dt in dtypes:  # --far: 2^32 の倍数の先(サンプル比較)。切り替わりの探索・最終点には使わない。いちばん重いので最後
        for s in [x for x in shapes if dt in dtypes_for(x, dtypes, args)] if args.far >= 2 else []:
            for lay in layouts:
                for B, lab in far_points(s, args.far):
                    R.run(make_spec(s["id"], dt, lay, "random", B, 0, "sample", tol[dt], args.device), "far", lab,
                          skip="skipped_budget" if over() else None)
                    if s["base"] and lab.endswith("+1"):
                        for kd in idx_kinds:
                            R.run(make_spec(s["id"], dt, lay, kd, B, 0, "sample", tol[dt], args.device), "far", lab,
                                  skip="skipped_budget" if over() else None)


def dry_run(args, dtypes, layouts, shapes, idx_kinds, final_seeds, final_idx_kinds) -> None:
    """探索点・index 符号化・較正の実行数と時間の見積もり。二分探索と最終点は切り替わりの数による(前回の結果から見積もる)。"""
    env = host_env(args.device)
    lim = MemLimits(env, {}, args)
    print(f"# {env_str(env)}  device {args.device}  上限 {lim}")
    n = defaultdict(int)
    t = defaultdict(float)
    n_skip = 0
    if args.points_list is not None:
        for p in args.points_list:
            sp = make_spec(p["shape"], p["dtype"], p["layout"], p.get("kind", "random"), p.get("B", offset_B(args)),
                           p.get("seed", 0), p.get("compare", "full"), 0, args.device, pad_batches=p.get("pad_batches"))
            over, est = lim.check(sp)
            print(f"point {series_id(sp)} B={sp['B']} O={sp.get('pad_batches', '-')} {sp['compare']}: "
                  f"host {est['host']/2**30:.1f} GiB{' -> skipped_memory' if over else ''} 見積もり {time_est(sp):.0f}s")
        return
    with_offset = OFFSET in layouts
    layouts = [x for x in layouts if x != OFFSET]
    for s in shapes if layouts else []:
        g = args.grid_base if s["base"] else args.grid_extra
        for dt in dtypes_for(s, dtypes, args):
            pts = probe_points(s, dt, g)
            print(f"\n## {s['id']} {dt}: 境界 {boundaries(s, dt)} 探索点 {len(pts)}: {[b for b, _ in pts]}")
            for lay in layouts:
                for B, _ in pts:
                    sp = make_spec(s["id"], dt, lay, "random", B, 0, "full" if B == cal_B(s) else "sample", 0,
                                   args.device)
                    if lim.check(sp)[0]:
                        n_skip += 1
                        continue
                    n["probe"] += 1
                    t["probe"] += time_est(sp)
                if s["base"]:
                    for kd in idx_kinds:
                        for B, _ in idx_points(s, dt):
                            n["idx"] += 1
                            t["idx"] += time_est(make_spec(s["id"], dt, lay, kd, B, 0, "sample", 0, args.device))
                # 最終点: 切り替わり 1 つ(2 点)と仮定、B は最大の境界付近
                Bf = max(b for b, _ in pts)
                for j in range(2):
                    for sd in final_seeds:
                        n["final"] += 1
                        t["final"] += time_est(make_spec(s["id"], dt, lay, "random", Bf, sd, "full", 0, args.device))
                    if s["base"] and (j == 0 or args.final_idx_where == "both"):
                        for kd in final_idx_kinds:
                            n["final"] += 1
                            t["final"] += time_est(make_spec(s["id"], dt, lay, kd, Bf, 0, "full", 0, args.device))
                n["bisect"] += 3
                t["bisect"] += 3 * time_est(make_spec(s["id"], dt, lay, "random", Bf // 2, 0, "sample", 0, args.device))
                for B, lab in far_points(s, args.far):
                    for kd in ["random"] + (idx_kinds if s["base"] and lab.endswith("+1") else []):
                        sp = make_spec(s["id"], dt, lay, kd, B, 0, "sample", 0, args.device)
                        if lim.check(sp)[0]:
                            n_skip += 1
                            continue
                        n["far"] += 1
                        t["far"] += time_est(sp)
            if args.far >= 2:
                print(f"   far: {[b for b, _ in far_points(s, args.far)]}")
    off_series = [(s, dt) for dt in dtypes if dt in OFFSET_DTYPES for s in shapes if s["base"]] if with_offset else []
    for s, dt in off_series:
        pts = offset_points(s, dt, args.grid_offset)
        print(f"\n## {s['id']} {dt} {OFFSET}: B={offset_B(args)} 固定、storage_offset の境界 "
              f"{boundaries(s, dt, offset_per_batch(s))} pad_batches {len(pts)} 点: {[o for o, _ in pts]}")
        for O, _ in pts:
            sp = make_spec(s["id"], dt, OFFSET, "random", offset_B(args), 0, "full", 0, args.device, pad_batches=O)
            if lim.check(sp)[0]:
                n_skip += 1
                continue
            n["offset"] += 1
            t["offset"] += time_est(sp)
    pairs = cal_pairs(shapes, dtypes, args) if layouts else []
    n["calibrate"] = len(pairs + [x for x in off_series if x not in pairs]) * len(CAL_SEEDS)  # 含まれる shape 全部で較正
    t["calibrate"] = n["calibrate"] * 3.0
    print("\n| 段階 | 実行数 | 見積もり(分) |\n|---|---|---|")
    for k in ("calibrate", "probe", "bisect", "idx", "final", "offset", "far"):
        if k in ("offset", "far") and not n[k]:
            continue  # 既定の引数では v1 と同じ表にする
        print(f"| {k} | {n[k]} | {t[k]/60:.1f} |")
    print(f"| 計 | {sum(n.values())} | {sum(t.values())/60:.1f} |")
    print(f"\nskipped_memory {n_skip} 点。二分探索は系列あたり 3 実行、最終点は切り替わり 1 つと仮定した見積もり。")


# ---------------------------------------------------------------------------- 要約

def fmt_err(v) -> str:
    return f"{v:.2g}" if isinstance(v, float) else ("-" if v is None else str(v))


def cls_short(r: dict | None) -> str:
    if r is None:
        return "-"
    return {"ok": "ok", "wrong": "WRONG", "error": "err", "crash": "CRASH", "timeout": "timeout",
            "truncated": "TRUNC", "input_corrupt": "INPUT", "skipped_memory": "skip_mem",
            "skipped_budget": "skip_bud"}.get(r["cls"], r["cls"])


def hyp_str(r: dict) -> str:
    h = r.get("hyp_match")
    if not h:
        return "-"
    return " ".join(f"{k}={v:.2f}" for k, v in h.items() if v)  or "該当なし"


def write_summary(host_tag: str, run_id: str) -> None:
    d = ROOT / "results" / "raw" / host_tag
    rows_all = [json.loads(x) for x in (d / f"{run_id}.jsonl").read_text().splitlines()]
    man = json.loads((d / f"{run_id}.manifest.json").read_text())
    global SCALE_LOG2
    SCALE_LOG2 = int(man.get("scale_log2") or 0)  # 境界の位置を manifest の縮小率に合わせる
    last: dict[str, dict] = {}
    for r in rows_all:
        last[r["fingerprint"]] = r
    rows = list(last.values())
    env, code, wk = man["env"], man["code"], man.get("worker", {})
    L = [f"# 走査結果の要約: {run_id}", "",
         f"- 生成: {datetime.datetime.now().isoformat(timespec='minutes')}(`scan.py --host-tag {host_tag} --summary {run_id}` で再生成)",
         f"- 演算: {man.get('op', 'bmm')}" + ("(forward + backward。分類は out / grad_a / grad_b のうち最悪。下の各表の分類・最大誤差も同じ)"
                                            if man.get("op") == "bmm_bwd" else ""),
         f"- 環境: {env_str(env, wk)} / device {man.get('device')} / torch {wk.get('torch')} / Python {wk.get('python')}",
         f"- メモリ上限: host {(man.get('mem_limit_host_bytes') or man.get('mem_limit_bytes') or 0)/2**30:.0f} GiB"
         + (f" / device {man['mem_limit_dev_bytes']/2**30:.0f} GiB" if man.get("mem_limit_dev_bytes") else "")
         + (f"({man['mem_limit_note']})" if man.get("mem_limit_note") else ""),
         f"- コード: git {(code.get('git_rev') or '不明')[:12]}(取得元 {code.get('git_rev_source', 'git')})"
         f"{'(scan/worker に未 commit の変更あり)' if code.get('git_dirty_code') else ''}"
         f" / worker.py sha256 {code['sha_worker'][:12]} / scan.py sha256 {code['sha_scan'][:12]}",
         f"- 実行数 {len(rows)}、状態 {man.get('status')}、経過 {man.get('elapsed_min', '-')} 分"
         f"(各実行の wall の和 {sum(r.get('wall_sec') or 0 for r in rows)/60:.1f} 分)",
         *([f"- **注記: {man['failure']}**"] if man.get("failure") else []),
         *([f"- **テスト用の縮小 run(scale_log2 = {SCALE_LOG2}: 理論境界と B を 2^{SCALE_LOG2} で割っている)。"
            "ドライバの動作確認用で、測定結果としては使わないこと**"] if SCALE_LOG2 else []),
         f"- 生データ `results/raw/{host_tag}/{run_id}.jsonl`、manifest `…/{run_id}.manifest.json`、W&B "
         + (f"{(man.get('wandb') or {}).get('url')}(Artifact `{run_id}`、type sweep-results)"
            if (man.get("wandb") or {}).get("url") else "なし"),
         ""]

    cal = [r for r in rows if r["phase"] == "calibrate"]
    tol = man.get("tol", {})
    if cal:
        L += ["## 許容誤差の較正", "",
              f"境界をまたぐ量が 2^28 要素の B、連続、乱数入力([-1,1] 一様、非ゼロ)、全要素比較。誤差はバッチごとの "
              f"max|MPS − CPU fp64| / max|CPU fp64| の最大。許容誤差 = {TOL_FACTOR} × (dtype ごとの最大)、下限 {TOL_FLOOR}。", "",
              "| dtype | shape | seed 0 | seed 1 | seed 2 | 許容誤差 |", "|---|---|---|---|---|---|"]
        for dt in ITEMSIZE:
            for sid in sorted({r["shape_id"] for r in cal}):
                c = {r["seed"]: r for r in cal if r["dtype"] == dt and r["shape_id"] == sid}
                if c:
                    L.append(f"| {dt} | {sid} | " + " | ".join(fmt_err(c[s].get("max_rel_err")) if s in c else "-"
                                                               for s in CAL_SEEDS) + f" | {fmt_err(tol.get(dt))} |")
        L.append("")

    rnd = [r for r in rows if r["kind"] == "random" and r["phase"] in ("probe", "bisect", "final")]
    ser: dict[tuple, dict] = defaultdict(dict)
    full: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rnd:
        k = (r["shape_id"], r["dtype"], r["layout"])
        if r["compare"] == "sample" and r["seed"] == 0 or r["phase"] == "probe":
            ser[k][r["B"]] = r
        if r["phase"] == "final":
            full[k][r["B"]][r["seed"]] = r
    order = {s["id"]: i for i, s in enumerate(SHAPES + EXTRA_SHAPES)}
    keys = sorted(ser, key=lambda k: (order[k[0]], k[1], LAYOUTS.index(k[2])))

    # 境界ごとの B0-2..B0+2(レイアウト対比較)
    L += ["## 理論境界の前後(乱数入力 seed 0、サンプル比較)とレイアウト対比較", "",
          "同じ seed では全レイアウトで論理的に同一の行列積(入力の論理値が同じ)。各セルは B0-2, B0-1, B0, B0+1, B0+2 の分類。", ""]
    for sid in sorted({k[0] for k in keys}, key=order.get):
        for dt in ITEMSIZE:
            ks = [k for k in keys if k[0] == sid and k[1] == dt]
            if not ks:
                continue
            bs = boundaries(SHAPE[sid], dt)
            L += [f"**{sid} {dt}**", "", "| レイアウト | " + " | ".join(f"{lab} (B0={b0})" for lab, b0 in bs) + " |",
                  "|---|" + "---|" * len(bs)]
            for k in ks:
                cells = []
                for _, b0 in bs:
                    cells.append(" ".join(cls_short(ser[k].get(b0 + dd)) for dd in (-2, -1, 0, 1, 2)))
                L.append(f"| {k[2]} | " + " | ".join(cells) + " |")
            L.append("")

    # 切り替わり
    L += ["## 実際の境界(グリッド + 二分探索)と最終点の全要素比較", "",
          "切り替わり = 正常/異常が異なる隣接点(B の粒度)。複数あれば非単調。括弧は境界をまたぐ量の log2(要素数)。"
          "全要素比較は seed ごとに 分類 / 最大相対誤差 / 閾値超え要素の割合 / 壊れたバッチの区間。仮説は誤答要素のうち各読み違い仮説の"
          "出力と一致する割合(サンプル比較の最初の異常点)。", "",
          "| shape | dtype | レイアウト | 探索点数 | 切り替わり | 異常の分類 | 読み違い仮説 | 全要素比較(下) | 全要素比較(上) |",
          "|---|---|---|---|---|---|---|---|---|"]
    nonmono = []
    for k in keys:
        known = {b: r for b, r in ser[k].items()}
        tr = transitions(known)
        if len(tr) > 1:
            nonmono.append(k)

        def fcell(B):
            if B not in full[k]:
                return "-"
            out = []
            for sd, r in sorted(full[k][B].items()):
                br = r.get("bad_ranges") or []
                brs = ",".join(f"[{a},{b})" for a, b in br[:2]) + ("…" if len(br) > 2 else "")
                out.append(f"s{sd}: {cls_short(r)} {fmt_err(r.get('max_rel_err'))} {fmt_err(r.get('frac_elem_bad'))}"
                           + (f" {brs}" if brs else ""))
            return "<br>".join(out)

        pts = sorted(b for b in known if is_ok(known[b]) is not None)
        if not pts:
            cl = "/".join(sorted({r["cls"] for r in known.values()}))
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(known)} | 判定できた点なし(全 {len(known)} 点が {cl}) | - | - | - | - |")
            continue
        if not tr:
            cl = "/".join(sorted({known[b]["cls"] for b in pts}))
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(known)} | なし(B={pts[0]}〜{pts[-1]} で {cl}) | - | - | - | "
                     f"{fcell(pts[-1])} |")
            continue
        for lo, hi in tr:
            bad = known[hi] if known[lo]["cls"] == "ok" else known[lo]
            p = per_batch(SHAPE[k[0]])
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(known)} | {lo} ({math.log2(lo*p):.5f}) {cls_short(known[lo])} → "
                     f"{hi} ({math.log2(hi*p):.5f}) {cls_short(known[hi])} | {bad['cls']} | {hyp_str(bad)} | "
                     f"{fcell(lo)} | {fcell(hi)} |")
    L += ["", "非単調(切り替わりが 2 つ以上)の系列: " + (", ".join("/".join(k) for k in nonmono) or "なし"), ""]

    # サンプルと全要素の食い違い
    mism = []
    for k in keys:
        for B, byseed in full[k].items():
            s0 = byseed.get(0)
            smp = ser[k].get(B)
            if s0 and smp and smp["compare"] == "sample" and (s0["cls"] == "ok") != (smp["cls"] == "ok"):
                mism.append(f"- {'/'.join(k)} B={B}: サンプル {smp['cls']} / 全要素 {s0['cls']}")
            cl = {r["cls"] for r in byseed.values() if not r["cls"].startswith("skipped")}
            if len(cl) > 1:
                mism.append(f"- {'/'.join(k)} B={B}: seed によって分類が違う {sorted(cl)}")
    L += ["## サンプル比較と全要素比較・seed 間の食い違い", ""] + (mism or ["- なし"]) + [""]

    # index 符号化
    idx = [r for r in rows if r["kind"] != "random" and not r["cls"].startswith("skipped")]
    if idx:
        L += ["## index 符号化入力", "",
              "idx_{a|b}_{batch|rc}: 符号化する側の各要素 = (field mod P)+1、反対側は選択行列。batch はバッチ index、rc はバッチ内の"
              "論理平坦 index r·C+c。delta = 読まれた field − 期待 field(mod P を [-P/2, P/2) に写す。誤答要素が多いときはブロックごとに最大 65536 要素を等間隔に間引いて逆算するので、要素数は間引き後の数)。rc の例は "
              "(期待 (行,列)) → (読まれた (行,列))(fp32 のみ一意に逆算)。P は fp32 1048573、fp16 2039。異常が出た実行だけ載せる"
              "(error は下の表)。選択行列の側が転置 view で読み違えられると、選択行列の 1 が 0 個や 2 個以上の行・列ができ、出力は符号の"
              "整数倍や和になる。その場合の delta はずれた位置を意味しないので、読み違い仮説の一致率と合わせて読むこと。", "",
              "| shape | dtype | レイアウト | 入力 | B | 比較 | 分類 | 壊れたバッチ | 上位の delta: 要素数 | rc の例 | 解釈不能 | 読み違い仮説 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        n_ok = 0
        for r in sorted(idx, key=lambda r: (order[r["shape_id"]], r["dtype"], LAYOUTS.index(r["layout"]), r["kind"], r["B"])):
            if r["cls"] in ("ok", "error"):
                n_ok += r["cls"] == "ok"
                continue
            top = ", ".join(f"{a}: {b}" for a, b in (r.get("idx_delta_top") or [])[:3])
            ex = ", ".join(f"({a[0]},{a[1]})→({b[0]},{b[1]})" for a, b, _ in (r.get("idx_rc_examples") or [])[:3])
            L.append(f"| {r['shape_id']} | {r['dtype']} | {r['layout']} | {r['kind']} | {r['B']} | {r['compare']} | "
                     f"{cls_short(r)} | {r.get('n_batches_bad')}/{r.get('n_batches_compared')} | {top} | {ex or '-'} | "
                     f"{fmt_err(r.get('idx_frac_invalid'))} | {hyp_str(r)} |")
        L += ["", f"index 符号化入力で ok: {n_ok} / {len(idx)}", ""]

    # v2 の軸: offset 系列、2^32 の倍数の先、指定点(どれも 1 実行 1 行で載せる)
    def v2_table(title: str, note: str, rs: list[dict]) -> None:
        nonlocal L
        if not rs:
            return
        L += [f"## {title}", "", note, "",
              "| 演算 | shape | dtype | レイアウト | 入力 | B | pad_batches (log2 offset) | seed | 比較 | 探索点 | 分類 | 最大誤差 | 壊れたバッチ | "
              "壊れたバッチの区間 | 読み違い仮説 | tensor ごと(bmm_bwd) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            br = r.get("bad_ranges") or []
            brs = ",".join(f"[{a},{b})" for a, b in br[:3]) + ("…" if len(br) > 3 else "")
            o = f"{r['pad_batches']} ({r.get('offset_log2')})" if "pad_batches" in r else "-"
            msg = (r.get("error") or r.get("stderr_tail") or "")[-100:]
            per_t = []
            for k in ("out", "grad_a", "grad_b"):
                if f"cls_{k}" in r:
                    kb = r.get(f"bad_ranges_{k}") or []
                    kh = r.get(f"hyp_match_{k}") or {}
                    per_t.append(f"{k}: {r[f'cls_{k}']} {fmt_err(r.get(f'max_rel_err_{k}'))}"
                                 + (" " + ",".join(f"[{a},{b})" for a, b in kb[:2]) + ("…" if len(kb) > 2 else "") if kb else "")
                                 + (" (" + " ".join(f"{h}={v:.2f}" for h, v in kh.items() if v) + ")" if any(kh.values()) else ""))
            L.append(f"| {r.get('op', 'bmm')} | {r['shape_id']} | {r['dtype']} | {r['layout']} | {r['kind']} | {r['B']} | {o} | {r['seed']} | "
                     f"{r['compare']} | {r.get('probe') or '-'} | {cls_short(r)} | {fmt_err(r.get('max_rel_err'))} | "
                     f"{r.get('n_batches_bad', '-')}/{r.get('n_batches_compared', '-')} | {brs or msg or '-'} | {hyp_str(r)} | "
                     f"{'<br>'.join(per_t) or '-'} |")
        L.append("")

    skey = lambda r: (order[r["shape_id"]], r["dtype"], r["layout"], r["kind"], r.get("pad_batches", 0), r["B"], r["seed"])  # noqa: E731
    v2_table("offset 系列(大きい storage_offset にある小さい view)",
             "B は固定で、格納の先頭の pad_batches(a, b とも)を振る。括弧は log2(storage_offset の要素数。a, b の大きい方)。"
             "仮説 nooffset = 格納の先頭から読む(pad の定数)、wrap = 格納の平坦 index が 2^32 要素で巻き戻る。",
             sorted([r for r in rows if r["phase"].startswith("offset")], key=skey))
    v2_table("2^32 要素の倍数の先(--far)", "2^32 要素の B0 の k 倍の前後 1 点。サンプル比較。",
             sorted([r for r in rows if r["phase"] == "far"], key=skey))
    v2_table("指定点(--points)", "走査ではなく、指定した点を 1 回ずつ実行したもの。",
             [r for r in rows if r["phase"] == "point"])

    # 時間
    L += ["## 1 実行あたりの時間(秒、平均)", "",
          "wall はプロセス起動を含む。gen_upload = CPU で入力生成 + device へ転送、regen = 正解用の入力を CPU で再生成、"
          "ref = CPU fp64 の bmm、metric = 誤差計算、hyp = 読み違い仮説の検定。", "",
          "| 区分 | 件数 | wall | gen_upload | bmm | download | regen | ref (CPU fp64) | metric | hyp | device 確保 最大 (GiB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        if "time" not in r:
            continue
        groups[f"{'random' if r['kind'] == 'random' else 'idx'} / {r['compare']} / "
               f"{'B>16384' if r['B'] > 16384 else 'B≤16384'}"].append(r)
    for g, rs in sorted(groups.items()):
        def mean(k, rs=rs):
            v = [r["time"].get(k, 0) for r in rs]
            return f"{sum(v)/len(v):.1f}"
        wall = sum(r.get("wall_sec") or 0 for r in rs) / len(rs)
        drv = max(r.get("dev_driver_alloc") or 0 for r in rs) / 2**30
        L.append(f"| {g} | {len(rs)} | {wall:.1f} | {mean('gen_upload')} | {mean('bmm')} | {mean('download')} | "
                 f"{mean('regen')} | {mean('ref')} | {mean('metric')} | {mean('hyp')} | {drv:.1f} |")
    L.append("")

    other = defaultdict(list)
    for r in rows:
        if r["cls"] in ("crash", "timeout", "error", "input_corrupt", "skipped_memory", "skipped_budget"):
            msg = r.get("error") or r.get("stderr_tail") or ""
            other[(r["cls"], msg)].append(f"{r['shape_id']}/{r['dtype']}/{r['layout']}/{r['kind']} B={r['B']} s{r['seed']} {r['compare']}")
    L += ["## エラー・クラッシュ・タイムアウト・入力の転送異常・スキップ", ""]
    if not other:
        L.append("- なし")
    for (c, msg), items in sorted(other.items()):
        L.append(f"- **{c}** {msg}({len(items)} 件): " + "; ".join(items[:6]) + (" …" if len(items) > 6 else ""))
    out = ROOT / "results" / "summary" / host_tag / f"{run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
