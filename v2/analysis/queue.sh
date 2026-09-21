#!/bin/zsh
cd /Users/USER/Documents/GitHub/mps-boundary/v2 || exit 1
S=/private/tmp/claude-501/-Users-USER-Documents-GitHub-mps-boundary/c052de50-d718-4933-a777-bc2858dc1f54/scratchpad/studio_q
UV=$(which uv)
run() { # name, args...
  name=$1; shift
  echo "=== START $name $(date '+%F %T')" >> $S/queue.status
  caffeinate -dims $UV run --env-file ../.env python scan.py --host-tag studio --wandb --wandb-every 25 "$@" > $S/$name.log 2>&1
  rc=$?
  echo "=== END $name rc=$rc $(date '+%F %T')" >> $S/queue.status
  return 0
}
run 1_all256 --budget-hours 24 --shapes all-256x256x256 --dtypes fp32,fp16 || true
run 2_bmm_bwd --budget-hours 24 --op bmm_bwd --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16 || true
run 3_offset --budget-hours 24 --layouts offset --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16 || true
run 4_far --points $S/far_points_full.json || true
echo "=== QUEUE DONE $(date '+%F %T')" >> $S/queue.status
