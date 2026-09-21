"""Colab A100 で CUDA の走査を小分けに回すドライバ(手元で実行する)。

colab CLI(v0.6.0)のセッションは、立ててから約 1 時間で 401/404 になり手元から操作できなくなる。
走査の run は環境の fingerprint を持つので、別の VM へまたいで --resume することもできない。
そこで走査を (shape, dtype, layout の組) の小さいチャンクに分け、1 チャンク = 1 run として回す。
セッションは立ててから SESSION_MAX_MIN を超えたら閉じて新しく立て、途中で失われたチャンクはキューに戻す。

    uv run python colab_chunks.py --workers 2
    uv run python colab_chunks.py --workers 2 --wandb   # VM 側の scan.py に --wandb を付ける(走査 1 回 = W&B の run 1 つ)

    uv run --env-file ../.env python colab_chunks.py --workers 2 --wandb --series bwd,all256,offset   # main 以外の系列

--wandb は手元の環境変数 WANDB_API_KEY(.env)を netrc の形の一時ファイル(mode 600)にして `colab upload` で VM の /root/.netrc に
置く(colab CLI のセッションは Colab の secret を読めない。exec するコードにはキーを書かないので、CLI の履歴にも残らない)。
一時ファイルは送ったら消す。キーが無ければ VM 側の scan.py が <run_id>.wandb.log に失敗を書いて走査だけ続ける。

--series: main = 全走査(6 shape。in-128x512x64 は 1 チャンクが長いのでレイアウトごと)、bwd = bmm_bwd(基本 shape)、
all256 = all-256x256x256、offset = offset レイアウト(基本 shape)、
far = 遠い点(analysis/far_points_full.json を shape × dtype の 4 つに分けた --points の run)、supp = all-256 の補足
(analysis/all256_supp_points.json を dtype ごとの 2 run に分ける。1 run だと CLI のセッションの寿命を超える)。分けた指定点のファイルは analysis/chunks/ に作って tar に入れる。チャンクの id は main 以外は系列名で始まる。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "raw" / "a100"
STATE = OUT / "chunks_state.json"
TAG = ""  # --tag: 2 本のドライバを同時に流すときに、状態ファイル・セッション名・一時ファイルを分ける
SESSION_MAX_MIN = 30

BASE = ["out-256x64x256", "in-256x256x64"]
EXTRA = ["out-512x64x128", "out-128x64x512", "in-512x128x64", "in-128x512x64"]
LAYOUTS = [["contig", "bT"], ["aT", "slice"]]


def all_chunks(series=("main",)):
    ch = []
    if "main" in series:
        for shape, dt, lays in itertools.product(BASE, ["fp32", "fp16"], LAYOUTS):
            ch.append({"shape": shape, "dtype": dt, "layouts": lays})
        for shape, lays in itertools.product(EXTRA, LAYOUTS):
            # in-128x512x64 は 2 レイアウトだとセッションの寿命に収まらなかったので、レイアウトごとに分ける
            for ls in ([[l] for l in lays] if shape == "in-128x512x64" else [lays]):
                ch.append({"shape": shape, "dtype": "fp32", "layouts": ls})
    if "bwd" in series:
        for shape, dt, lays in itertools.product(BASE, ["fp32", "fp16"], LAYOUTS):
            ch.append({"shape": shape, "dtype": dt, "layouts": lays, "series": "bwd", "extra": "--op bmm_bwd "})
    if "all256" in series:
        for dt, lays in itertools.product(["fp32", "fp16"], LAYOUTS):
            ch.append({"shape": "all-256x256x256", "dtype": dt, "layouts": lays, "series": "all256"})
    if "offset" in series:
        for shape, dt in itertools.product(BASE, ["fp32", "fp16"]):
            ch.append({"shape": shape, "dtype": dt, "layouts": ["offset"], "series": "offset"})
    if "far" in series or "supp" in series:
        (ROOT / "analysis" / "chunks").mkdir(exist_ok=True)
    if "far" in series:
        pts = json.loads((ROOT / "analysis" / "far_points_full.json").read_text())
        for shape, dt in itertools.product(BASE, ["fp32", "fp16"]):
            f = f"analysis/chunks/far_{shape}_{dt}.json"
            (ROOT / f).write_text(json.dumps([p for p in pts if p["shape"] == shape and p["dtype"] == dt], indent=1))
            ch.append({"shape": shape, "dtype": dt, "layouts": ["points"], "series": "far", "points": f})
    if "supp" in series:  # 1 run だと CLI のセッションの寿命(約 1 時間)を超えるので、dtype ごとに分ける
        pts = json.loads((ROOT / "analysis" / "all256_supp_points.json").read_text())
        for dt in ["fp32", "fp16"]:
            f = f"analysis/chunks/supp_all-256x256x256_{dt}.json"
            (ROOT / f).write_text(json.dumps([p for p in pts if p["dtype"] == dt], indent=1))
            ch.append({"shape": "all-256x256x256", "dtype": dt, "layouts": ["points"], "series": "supp", "points": f})
    for c in ch:
        c["id"] = (c["series"] + "_" if c.get("series") else "") + f"{c['shape']}_{c['dtype']}_{'-'.join(c['layouts'])}"
    return ch


def push_netrc(ses, w):
    """WANDB_API_KEY を netrc の形で VM の /root/.netrc に置く。キーは exec するコードにも標準出力にも出さない。"""
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        return False
    fd, tmp = tempfile.mkstemp(prefix=f".netrc_w{w}_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"machine api.wandb.ai\n  login user\n  password {key}\n")
        sh(f"colab upload -s {ses} {tmp} /root/.netrc")
    finally:
        os.unlink(tmp)
    return True


def sh(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # ワーカーのスレッドを落とさない(落ちるとセッションが枠を占めたまま残る)
        return "__SH_TIMEOUT__"
    return r.stdout + r.stderr


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "tries": {}}

    def save(self):
        STATE.write_text(json.dumps(self.d, indent=1))

    def take(self, chunks):
        with self.lock:
            for c in chunks:
                if (c["id"] not in self.d["done"] and c["id"] not in self.d.setdefault("running", {})
                        and self.d["tries"].get(c["id"], 0) < 3):
                    self.d["running"][c["id"]] = time.time()
                    self.save()
                    return c
        return None

    def finish(self, cid, run_id):
        with self.lock:
            self.d["running"].pop(cid, None)
            self.d["done"][cid] = run_id
            self.save()

    def requeue(self, cid):
        with self.lock:
            self.d["running"].pop(cid, None)
            self.d["tries"][cid] = self.d["tries"].get(cid, 0) + 1
            self.save()


def log(w, msg):
    line = f"{time.strftime('%H:%M')} [w{w}] {msg}"
    print(line, flush=True)
    with open(OUT / "chunks.log", "a") as f:
        f.write(line + "\n")


def worker(w, st, chunks, rev, tgz, use_wandb=False):
    n = 0
    while True:
        ses = f"mpsb{TAG}w{w}x{n}"
        n += 1
        if "READY" not in sh(f"colab new -s {ses} --gpu A100 --high-mem"):
            log(w, f"{ses}: new failed, retry in 2 min")
            time.sleep(120)
            continue
        born = time.time()
        sh(f"colab upload -s {ses} {tgz} /content/src.tgz")
        if use_wandb and not push_netrc(ses, w):
            log(w, "WANDB_API_KEY が手元の環境に無い(--env-file ../.env を付ける)。W&B なしで続ける")
        boot = ROOT / f".boot_{TAG}w{w}.py"
        boot.write_text(
            "import subprocess\n"
            "subprocess.run('mkdir -p /content/mb && cd /content/mb && tar xzf /content/src.tgz && pip install -q uv; chmod 600 /root/.netrc 2>/dev/null', shell=True)\n"
            "print('__BOOT__')\n")
        if "__BOOT__" not in sh(f"colab exec -s {ses} -f {boot} --timeout 240", timeout=300):
            log(w, f"{ses}: boot failed")
            sh(f"colab stop -s {ses}")
            continue
        while (time.time() - born) / 60 < SESSION_MAX_MIN:
            c = st.take(chunks)
            if c is None:
                sh(f"colab stop -s {ses}")
                log(w, "no more chunks")
                return
            log(w, f"{ses}: start {c['id']}")
            run = ROOT / f".run_{TAG}w{w}.py"
            wb_with, wb_flag = ("--with wandb ", "--wandb ") if use_wandb else ("", "")
            what = (f"--points {c['points']} " if c.get("points") else
                    f"--shapes {c['shape']} --dtypes {c['dtype']} --layouts {','.join(c['layouts'])} ")
            run.write_text(
                "import subprocess, glob\n"
                f"cmd = 'cd /content/mb && MPSB_GIT_REV={rev} uv run --no-project --python 3.12 --with numpy {wb_with}python scan.py "
                f"--host-tag a100 --device cuda --torch 2.14.0 {wb_flag}{c.get('extra', '')}--budget-hours 1 "
                f"{what}> /content/chunk.log 2>&1'\n"
                "rc = subprocess.run(cmd, shell=True).returncode\n"
                "print('RC', rc)\n"
                "print('FILES', ' '.join(glob.glob('/content/mb/results/raw/a100/*')))\n"
                "print('__RUN__')\n")
            out = sh(f"colab exec -s {ses} -f {run} --timeout 3600", timeout=3700)
            if "__RUN__" not in out:
                log(w, f"{ses}: lost during {c['id']} -> requeue")
                st.requeue(c["id"])
                break
            files = next((l.split()[1:] for l in out.splitlines() if l.startswith("FILES")), [])
            run_id = None
            for f in files:
                name = Path(f).name
                if (OUT / name).exists():
                    continue
                sh(f"colab download -s {ses} {f} {OUT / name}")
                if name.endswith(".jsonl"):
                    run_id = name[:-6]
            sh(f"colab download -s {ses} /content/chunk.log {OUT / (c['id'] + '.log')}")
            if run_id is None:
                log(w, f"{ses}: {c['id']} produced no new jsonl -> requeue")
                st.requeue(c["id"])
                continue
            st.finish(c["id"], run_id)
            log(w, f"{ses}: done {c['id']} -> {run_id}")
        sh(f"colab stop -s {ses}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--series", default="main", help="main, bwd, all256, offset, far, supp(カンマ区切り)")
    ap.add_argument("--wandb", action="store_true", help="VM 側の scan.py に --wandb を付ける(既定はなし)")
    ap.add_argument("--tag", default="", help="状態ファイル chunks_state_<tag>.json、セッション名、一時ファイルを分ける(別のドライバと同時に流すとき)")
    args = ap.parse_args()
    global STATE, TAG
    if args.tag:
        TAG, STATE = args.tag, OUT / f"chunks_state_{args.tag}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    rev = sh(f"git -C {ROOT} rev-parse --short HEAD").strip()
    tgz = ROOT / f".src{TAG}.tgz"
    chunks = all_chunks(args.series.split(","))
    extra = " ".join(sorted({c["points"] for c in chunks if c.get("points")}))  # 分けた指定点のファイル(git の管理外)も送る
    sh(f"cd {ROOT} && (git ls-files -z -- . ':!results'; for f in {extra}; do printf '%s\\0' $f; done) | sort -zu | "
       f"tar --null -czf {tgz} -T -")  # 結果 (raw) は送らない
    st = State()
    st.d["running"] = {}
    st.save()
    log(0, f"rev {rev}, {len(chunks)} chunks, done {len(st.d['done'])}")
    ts = [threading.Thread(target=worker, args=(w, st, chunks, rev, tgz, args.wandb)) for w in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(0, f"all done: {len(st.d['done'])}/{len(chunks)}")


if __name__ == "__main__":
    main()
