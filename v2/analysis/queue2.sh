#!/bin/zsh
S=/private/tmp/claude-501/-Users-USER-Documents-GitHub-mps-boundary/c052de50-d718-4933-a777-bc2858dc1f54/scratchpad/studio_q
until grep -q "QUEUE DONE" $S/queue.status; do sleep 60; done
cd /Users/USER/Documents/GitHub/mps-boundary/v2 || exit 1
echo "=== START 5_all256_supp $(date '+%F %T')" >> $S/queue.status
caffeinate -dims /opt/homebrew/bin/uv run --env-file ../.env python scan.py --host-tag studio --wandb --wandb-every 25 --points $S/all256_supp_points.json > $S/5_all256_supp.log 2>&1
echo "=== END 5_all256_supp rc=$? $(date '+%F %T')" >> $S/queue.status
echo "=== QUEUE2 DONE $(date '+%F %T')" >> $S/queue.status
