"""Colab A100 で CUDA の走査を小分けに回すドライバ(手元で実行する)。

colab CLI(v0.6.0)のセッションは、立ててから約 1 時間で 401/404 になり手元から操作できなくなる。
走査の run は環境の fingerprint を持つので、別の VM へまたいで --resume することもできない。
そこで走査を (shape, dtype, layout の組) の小さいチャンクに分け、1 チャンク = 1 run として回す。
セッションは立ててから SESSION_MAX_MIN を超えたら閉じて新しく立て、途中で失われたチャンクはキューに戻す。

    uv run python colab_chunks.py --workers 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "raw" / "a100"
STATE = OUT / "chunks_state.json"
SESSION_MAX_MIN = 30

BASE = ["out-256x64x256", "in-256x256x64"]
EXTRA = ["out-512x64x128", "out-128x64x512", "in-512x128x64", "in-128x512x64"]
LAYOUTS = [["contig", "bT"], ["aT", "slice"]]


def all_chunks():
    ch = []
    for shape, dt, lays in itertools.product(BASE, ["fp32", "fp16"], LAYOUTS):
        ch.append({"shape": shape, "dtype": dt, "layouts": lays})
    for shape, lays in itertools.product(EXTRA, LAYOUTS):
        ch.append({"shape": shape, "dtype": "fp32", "layouts": lays})
    for c in ch:
        c["id"] = f"{c['shape']}_{c['dtype']}_{'-'.join(c['layouts'])}"
    return ch


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
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


def worker(w, st, chunks, rev, tgz):
    n = 0
    while True:
        ses = f"mpsbw{w}x{n}"
        n += 1
        if "READY" not in sh(f"colab new -s {ses} --gpu A100 --high-mem"):
            log(w, f"{ses}: new failed, retry in 2 min")
            time.sleep(120)
            continue
        born = time.time()
        sh(f"colab upload -s {ses} {tgz} /content/src.tgz")
        boot = ROOT / f".boot_w{w}.py"
        boot.write_text(
            "import subprocess\n"
            "subprocess.run('mkdir -p /content/mb && cd /content/mb && tar xzf /content/src.tgz && pip install -q uv', shell=True)\n"
            "print('__BOOT__')\n")
        if "__BOOT__" not in sh(f"colab exec -s {ses} -f {boot} --timeout 600"):
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
            run = ROOT / f".run_w{w}.py"
            run.write_text(
                "import subprocess, glob\n"
                f"cmd = 'cd /content/mb && MPSB_GIT_REV={rev} uv run --no-project --python 3.12 --with numpy python scan.py "
                f"--host-tag a100 --device cuda --torch 2.14.0 --no-wandb --budget-hours 1 "
                f"--shapes {c['shape']} --dtypes {c['dtype']} --layouts {','.join(c['layouts'])} "
                f"> /content/chunk.log 2>&1'\n"
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
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rev = sh(f"git -C {ROOT} rev-parse --short HEAD").strip()
    tgz = ROOT / ".src.tgz"
    sh(f"cd {ROOT} && git ls-files -z | tar --null -czf {tgz} -T -")
    st = State()
    st.d["running"] = {}
    st.save()
    chunks = all_chunks()
    log(0, f"rev {rev}, {len(chunks)} chunks, done {len(st.d['done'])}")
    ts = [threading.Thread(target=worker, args=(w, st, chunks, rev, tgz)) for w in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(0, f"all done: {len(st.d['done'])}/{len(chunks)}")


if __name__ == "__main__":
    main()
