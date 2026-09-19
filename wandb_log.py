"""scan.py の 1 実行分の記録 (JSON, 標準入力) を W&B に 1 run として送る。

scan.py から背景プロセスとして起動する。ここで失敗しても走査は止まらない(jsonl には送る前に書いてある)。
"""

from __future__ import annotations

import json
import sys

import wandb


def main() -> None:
    row = json.load(sys.stdin)
    project = sys.argv[1] if len(sys.argv) > 1 else "mps-boundary"
    res_keys = {"cls", "max_rel_err", "frac_elem_bad", "n_batches_bad", "n_batches_compared", "n_elem_bad",
                "n_elem_compared", "frac_bad_elem_zero", "n_nan", "first_bad_batch", "wall_sec",
                "mps_driver_alloc_after_bmm", "mps_current_alloc_after_bmm", "max_rss", "idx_frac_nonint",
                "error", "stage", "rc", "input_mismatch_batches", "idx_delta_top", "bad_ranges", "idx_frac_invalid",
                "dev_driver_alloc", "dev_current_alloc", "hyp_match", "idx_rc_examples", "hyp_n_bad_elem"}
    config = {k: v for k, v in row.items() if k not in res_keys and k not in ("time", "hist_edges", "hist_bad",
                                                                            "hist_compared", "stderr_tail")}
    name = f"{row['shape_id']}-{row['dtype']}-{row['layout']}-{row['kind']}-B{row['B']}-{row['compare']}-s{row['seed']}"
    run = wandb.init(project=project, name=name, group=row["run_id"], job_type=row["phase"], config=config,
                     tags=[row["host_tag"], row["cls"], row["phase"], row["shape_id"], row["layout"], row["kind"]],
                     settings=wandb.Settings(silent=True))
    log = {k: row[k] for k in res_keys if k in row and isinstance(row[k], (int, float, str))}
    log.update({f"time/{k}": v for k, v in (row.get("time") or {}).items()})
    if row.get("hist_bad"):
        edges = row["hist_edges"]
        log["bad_batch_hist"] = wandb.Histogram(np_histogram=(row["hist_bad"], edges))
        log["compared_batch_hist"] = wandb.Histogram(np_histogram=(row["hist_compared"], edges))
    run.log(log)
    for h, v in (row.get("hyp_match") or {}).items():
        run.summary[f"hyp/{h}"] = v
    for k in ("input_mismatch_batches", "idx_delta_top", "idx_rc_examples", "bad_ranges", "error", "stderr_tail"):
        if k in row:
            run.summary[k] = json.dumps(row[k]) if not isinstance(row[k], str) else row[k]
    run.finish()


if __name__ == "__main__":
    main()
