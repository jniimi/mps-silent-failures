"""demo_workload.py が保存した結果から、テキストごとの rel の図(fig_rel_by_position)だけを描き直す(MPS での再実行は不要)。

    uv run --no-project --python 3.12 --with torch==2.14.0 --with matplotlib==3.11.2 python plot_workload.py \
        --results ../v1/results --out ../ieee-access/figs --formats pdf

描画は demo_workload.py の plot() をそのまま使う(demo_workload を import するので torch が要るが、モデルの実行や
MPS での計算はしない)。matplotlib 3.11.2 は v1/results/workload/fig_rel_by_position.png を描いた版。
入力: <results>/workload/workload.json、logits.npz。出力: <out>/fig_rel_by_position.<ext>(--out の既定は <results>/workload。
v1 の結果を読むときは、v1 のファイルを上書きしないよう --out を付けること)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_workload import plot

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"), help="results ディレクトリ(v1 を読むなら ../v1/results)")
    ap.add_argument("--out", default=None, help="出力先のディレクトリ(既定は <results>/workload)")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], help="保存する形式(拡張子)")
    args = ap.parse_args()
    src = Path(args.results) / "workload"
    dst = Path(args.out) if args.out else src
    dst.mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "workload.json").read_text())
    with np.load(src / "logits.npz") as z:
        saved = {k: z[k] for k in z.files}
    plot(info, saved, dst, formats=tuple(args.formats))
    print("\n".join(str(dst / f"fig_rel_by_position.{ext}") for ext in args.formats))


if __name__ == "__main__":
    main()
