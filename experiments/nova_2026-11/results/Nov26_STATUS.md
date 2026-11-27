# Nov26 experiments: status

Generated 2026-11-27 02:13 CST by `/work/flemingc/akhan1/gnn_marl_experiments/make_status.py` (rerun it to refresh).

Run directories: `/work/flemingc/akhan1/gnn_marl_experiments/Nov26_<METHOD>/<run dir>`. Logs: `/work/flemingc/akhan1/gnn_marl_experiments/Nov26_<METHOD>/logs/task_<id>.log`.

Columns: `step` is per fleet size (budget 2,000,000). `log steps/s` is the last value printed by the training
script (all fleet sizes together, between evaluations). `per-size steps/s` is step / wall time since job start,
evaluations included; the projected end of training uses it. Add up to about 1.5 h for the final zero-shot
evaluation (estimate from the smoke run, not measured at full size). `tracebacks` counts tracebacks in the log;
see the notes below the table.

| task | method | seed | job id | state, node | step | log steps/s | per-size steps/s | training ends (projected) | eval step=0 lines | config.json | metrics.csv rows | latest.pt | final.pt | tracebacks | run directory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 145 |  | finished | 1 | yes | 12908 | yes | yes | 17 | `G03.1_qmix_sector_cr_n3_s1_20261126-1616_nova` |
| 11 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 149 |  | finished | 1 | yes | 12908 | yes | yes | 6 | `G03.1_qmix_sector_cr_n3_s2_20261126-1616_nova` |
| 12 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 153 |  | finished | 1 | yes | 12908 | yes | yes | 6 | `G03.1_qmix_sector_cr_n3_s3_20261126-1616_nova` |
| 13 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 111 |  | finished | 1 | yes | 12908 | yes | yes | 21 | `G03.1_qmix_sector_cr_n6_s1_20261126-1616_nova` |
| 14 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 88 |  | finished | 1 | yes | 12908 | yes | yes | 3 | `G03.1_qmix_sector_cr_n6_s2_20261126-1616_nova` |
| 15 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 86 |  | finished | 1 | yes | 12908 | yes | yes | 4 | `G03.1_qmix_sector_cr_n6_s3_20261126-1616_nova` |
| 16 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 61 |  | finished | 1 | yes | 12908 | yes | yes | 9 | `G03.1_qmix_sector_cr_n9_s1_20261126-1616_nova` |
| 17 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 61 |  | finished | 1 | yes | 12908 | yes | yes | 16 | `G03.1_qmix_sector_cr_n9_s2_20261126-1616_nova` |
| 18 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 62 |  | finished | 1 | yes | 12908 | yes | yes | 5 | `G03.1_qmix_sector_cr_n9_s3_20261126-1616_nova` |

## Notes

- Nov26 set = QMIX tasks 10-18 rerun with the D-032 fix (airspeed state entry / 10 in the mixer, polyak 0.001),
  Slurm array `17784474`, launched 2026-11-26 16:16 CST from `/work/flemingc/akhan1/gnn_marl_nov26`
  (branch `qmix_fix`, uncommitted at launch; diff in `Nov26_QMIX/code_changes_vs_a1ddb01.patch`).
  2 CPUs, 4 GB, 24 h limit per task: killed at about 16:16 on Fri Nov 27 if unfinished (no resume, GAP-26).
- The fix was verified only at n = 6, seed 1, up to 200,000 steps. Check `loss=` in these logs: it should stay
  below about 1. In `Nov25_QMIX` it was 1e5-1e7 by step 40,000.
- Results set to aggregate: Nov25_GNN-QMIX, Nov25_VDN, Nov25_IQL, Nov25_ABL-VDN-GCN, Nov26_QMIX. Do not use Nov25_QMIX.
- Refresh: `TAG=Nov26 python3 make_status.py`.
- Committed after launch as `d1d66b1` on branch `qmix_fix` (pushed to origin 2026-11-26). The code in that commit is identical to `code_changes_vs_a1ddb01.patch` applied to `a1ddb01`.
