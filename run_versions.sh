#!/bin/zsh
# torch の版ごとの縮小走査(Studio / MPS)。版ごとに別の run_id。1 版あたり 20〜25 分(Studio 実測からの見積もり)。
# 基本 shape のみ、fp32 / fp16、4 レイアウト、乱数入力、境界 B0-2〜B0+2 + グリッド(境界の間 2 点)、
# 全要素比較は切り替わりの前後 2 点を seed 0 のみ、index 符号化は切り替わりの異常側 1 点で rc 系のみ。
#   ./run_versions.sh [版 ...]
cd "$(dirname "$0")"
versions=(2.4.1 2.5.1 2.6.0 2.7.1 2.8.0 2.9.1 2.10.0 2.11.0 2.12.1 2.13.0)
(( $# )) && versions=("$@")
for v in $versions; do
  echo "=== torch $v $(date '+%H:%M:%S')"
  uv run python scan.py --host-tag studio --torch "$v" --python 3.12 \
    --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16 --layouts contig,bT,aT,slice \
    --grid-base 2 --idx-kinds "" --final-idx-kinds idx_a_rc,idx_b_rc --final-idx-where abnormal \
    --final-seeds 0 --budget-hours 1.0
done
echo "=== done $(date '+%H:%M:%S')"
