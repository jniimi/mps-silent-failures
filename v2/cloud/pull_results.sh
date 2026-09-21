#!/bin/zsh
# ローカル(Studio)側で実行: クラウドの Mac から結果を一定間隔で引く。Ctrl-C で止める。
#   zsh v2/cloud/pull_results.sh <SSH の鍵> <ec2-user@HOST> [間隔(秒)=900]
#   鍵に - を渡すと鍵を指定しない(手元の別の Mac など)。リモートの置き場所は ~/mb/v2/results
cd "$(dirname "$0")/.."
KEY=${1:?鍵}; HOST=${2:?ec2-user@HOST}; INT=${3:-900}
while true; do
  date '+%F %T'
  if [[ $KEY == - ]]; then SSH="ssh -o ServerAliveInterval=30"; else SSH="ssh -i $KEY -o ServerAliveInterval=30"; fi
  rsync -az -e "$SSH" "$HOST:mb/v2/results/" results/ && ls results/raw | tr '\n' ' '; echo
  sleep $INT
done
