#!/bin/zsh
# macOS の版の軸の走査。既定は EC2 Mac の M2 Pro 32 GB 前提(fp16、基本 shape、4 レイアウト)。tmux の中で実行する。
# メモリに余裕のある機種では環境変数で広げる: DTYPES=fp32,fp16  SHAPES=all(= scan.py の既定の 6 shape)  OP=bmm_bwd
#   cd ~/mb/v2 && MPSB_GIT_REV=<rev> zsh cloud/run_macos_axis.sh <host-tag> [torch の版 ...]
#   例: MPSB_GIT_REV=f376250 zsh cloud/run_macos_axis.sh m2pro-macos14
# 既定: 2.14.0 は全走査、2.4.1 / 2.8.0 / 2.9.1 は縮小走査。途中で切れたら scan.py --resume <run_id> で続ける。
# W&B: ~/mb/.env に WANDB_API_KEY があれば --wandb を付ける。
set -e
cd "$(dirname "$0")/.."
[[ -f ~/.local/bin/env ]] && source ~/.local/bin/env
TAG=${1:?host-tag を指定する(例: m2pro-macos14)}; shift
versions=(${@:-2.14.0 2.4.1 2.8.0 2.9.1})
: ${MPSB_GIT_REV:?MPSB_GIT_REV を指定する(送った tar の revision)}
export MPSB_GIT_REV
SHAPES=${SHAPES:-out-256x64x256,in-256x256x64}; DTYPES=${DTYPES:-fp16}
COMMON=(--host-tag $TAG --dtypes $DTYPES)
[[ $SHAPES != all ]] && COMMON+=(--shapes $SHAPES)
[[ -n $OP ]] && COMMON+=(--op $OP)
ENVF=()
if [[ -f ../.env ]]; then ENVF=(--env-file ../.env); COMMON+=(--wandb --wandb-every 25); fi
REDUCED=(--grid-base 2 --idx-kinds "" --final-idx-kinds idx_a_rc,idx_b_rc --final-idx-where abnormal --final-seeds 0)

echo "=== $(sw_vers -productVersion) ($(sw_vers -buildVersion)) tag=$TAG rev=$MPSB_GIT_REV versions=$versions shapes=$SHAPES dtypes=$DTYPES op=${OP:-bmm}"
uv run $ENVF python scan.py $COMMON --dry-run | tail -12
for v in $versions; do
  echo "=== torch $v $(date '+%F %T')"
  if [[ $v == 2.14.0 ]]; then extra=(); else extra=("${REDUCED[@]}"); fi
  caffeinate -dims uv run $ENVF python scan.py $COMMON --torch $v --python 3.12 "${extra[@]}" ||  # 引用符つき: --idx-kinds "" の空文字列を落とさない
    echo "torch $v: 走査が異常終了(続きは --resume)"
done
echo "=== done $(date '+%F %T')"
ls -la results/raw/$TAG/
