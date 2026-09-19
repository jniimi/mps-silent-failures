"""走査結果から境界マップの図を作る(基本 shape、乱数入力、1 つの run)。

    uv run --no-project --python 3.12 --with matplotlib python boundary_map.py [--run RUN_ID] [--dtype fp32]

行 = レイアウト、列 = 走査した B(昇順、等間隔ではない)、セル = 分類(ok / error / wrong)。
上段は出力が境界をまたぐ系列 (out-256x64x256)、下段は入力 a がまたぐ系列 (in-256x256x64)。
同じ (系列, B) に複数の実行(seed、比較方式)があれば、全部が同じ分類であることを確かめる。
出力: arxiv-v1/figs/fig_boundary_map.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
LAYOUTS = [("contig", "contiguous"), ("bT", "$b$ transposed"), ("aT", "$a$ transposed"), ("slice", "sliced (offset)")]
PANELS = [("out-256x64x256", "output crosses: $(B,256,64)\\times(B,64,256)$, output has $2^{16}B$ elements"),
          ("in-256x256x64", "input crosses: $(B,256,256)\\times(B,256,64)$, $a$ has $2^{16}B$ elements")]
COLOR = {"ok": "#e9e9e6", "error": "#0072B2", "wrong": "#D55E00"}
LETTER = {"ok": "", "error": "E", "wrong": "W"}
MARKS = {16384: "$2^{32}$ bytes", 32768: "$2^{31}$ el.", 65536: "$2^{32}$ el."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260919T130741-studio")
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()
    host = args.run.rsplit("-", 1)[1]
    rows = [json.loads(l) for l in open(ROOT / "results" / "raw" / host / f"{args.run}.jsonl")]
    cls = defaultdict(set)
    for r in rows:
        if r["kind"] == "random" and r["dtype"] == args.dtype and r["phase"] != "calibrate":
            cls[(r["shape_id"], r["layout"], r["B"])].add(r["cls"])
    mixed = {k: v for k, v in cls.items() if len(v) > 1}
    assert not mixed, f"分類が実行間で食い違う: {mixed}"

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 2.7), constrained_layout=True)
    for ax, (shape, title) in zip(axes, PANELS):
        Bs = sorted({b for (s, _, b) in cls if s == shape})
        for i, (lay, _) in enumerate(LAYOUTS):
            for j, b in enumerate(Bs):
                c = next(iter(cls[(shape, lay, b)]))
                ax.add_patch(plt.Rectangle((j + 0.06, i + 0.08), 0.88, 0.84, color=COLOR[c], lw=0))
                if LETTER[c]:
                    ax.text(j + 0.5, i + 0.5, LETTER[c], ha="center", va="center", color="white", fontsize=5.5, fontweight="bold")
        ax.set_xlim(0, len(Bs)); ax.set_ylim(len(LAYOUTS), 0)
        ax.set_yticks([i + 0.5 for i in range(len(LAYOUTS))], [n for _, n in LAYOUTS])
        ticks = [(Bs.index(b) + 0.5, f"{b}\n({lab})") for b, lab in MARKS.items() if b in Bs]
        ticks = [(0.5, str(Bs[0]))] + ticks
        ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.tick_params(length=2, pad=1.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=7, pad=2)
    axes[1].set_xlabel("batch size $B$ (every value tested, in increasing order; not to scale)", labelpad=2)
    fig.legend(handles=[Patch(color=COLOR[k], label=l) for k, l in [("ok", "correct"), ("error", "E: raises an error"), ("wrong", "W: silently wrong")]],
               loc="outside upper right", ncols=3, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2)
    out = ROOT / "arxiv-v1" / "figs" / "fig_boundary_map.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main()
