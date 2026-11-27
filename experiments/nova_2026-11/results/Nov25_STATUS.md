# Nov25 experiments: status

Generated 2026-11-27 02:13 CST by `/work/flemingc/akhan1/gnn_marl_experiments/make_status.py` (rerun it to refresh).

Run directories: `/work/flemingc/akhan1/gnn_marl_experiments/Nov25_<METHOD>/<run dir>`. Logs: `/work/flemingc/akhan1/gnn_marl_experiments/Nov25_<METHOD>/logs/task_<id>.log`.

Columns: `step` is per fleet size (budget 2,000,000). `log steps/s` is the last value printed by the training
script (all fleet sizes together, between evaluations). `per-size steps/s` is step / wall time since job start,
evaluations included; the projected end of training uses it. Add up to about 1.5 h for the final zero-shot
evaluation (estimate from the smoke run, not measured at full size). `tracebacks` counts tracebacks in the log;
see the notes below the table.

| task | method | seed | job id | state, node | step | log steps/s | per-size steps/s | training ends (projected) | eval step=0 lines | config.json | metrics.csv rows | latest.pt | final.pt | tracebacks | run directory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GNN-QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 73 |  | finished | 3 | yes | 32726 | yes | yes | 35 | `G05.3_gnn_qmix_sector_cr_n3-6-9_s1_20261125-1847_nova` |
| 2 | GNN-QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 68 |  | finished | 3 | yes | 32726 | yes | yes | 23 | `G05.3_gnn_qmix_sector_cr_n3-6-9_s2_20261125-1847_nova` |
| 3 | GNN-QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 71 |  | finished | 3 | yes | 32726 | yes | yes | 14 | `G05.3_gnn_qmix_sector_cr_n3-6-9_s3_20261125-1847_nova` |
| 4 | VDN | 1 | (not in queue) | ENDED  | 2,000,000 | 108 |  | finished | 3 | yes | 32726 | yes | yes | 19 | `G03.1_vdn-k3_sector_cr_n3-6-9_s1_20261125-1847_nova` |
| 5 | VDN | 2 | (not in queue) | ENDED  | 2,000,000 | 86 |  | finished | 3 | yes | 32726 | yes | yes | 17 | `G03.1_vdn-k3_sector_cr_n3-6-9_s2_20261125-1847_nova` |
| 6 | VDN | 3 | (not in queue) | ENDED  | 2,000,000 | 85 |  | finished | 3 | yes | 32726 | yes | yes | 42 | `G03.1_vdn-k3_sector_cr_n3-6-9_s3_20261125-1847_nova` |
| 7 | IQL | 1 | (not in queue) | ENDED  | 2,000,000 | 95 |  | finished | 3 | yes | 32726 | yes | yes | 110 | `G03.1_iql-k3_sector_cr_n3-6-9_s1_20261125-1847_nova` |
| 8 | IQL | 2 | (not in queue) | ENDED  | 2,000,000 | 99 |  | finished | 3 | yes | 32726 | yes | yes | 33 | `G03.1_iql-k3_sector_cr_n3-6-9_s2_20261125-1847_nova` |
| 9 | IQL | 3 | (not in queue) | ENDED  | 2,000,000 | 95 |  | finished | 3 | yes | 32726 | yes | yes | 17 | `G03.1_iql-k3_sector_cr_n3-6-9_s3_20261125-1847_nova` |
| 10 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 109 |  | finished | 1 | yes | 12908 | yes | yes | 5 | `G03.1_qmix_sector_cr_n3_s1_20261125-1847_nova` |
| 11 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 138 |  | finished | 1 | yes | 12908 | yes | yes | 4 | `G03.1_qmix_sector_cr_n3_s2_20261125-1847_nova` |
| 12 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 131 |  | finished | 1 | yes | 12908 | yes | yes | 0 | `G03.1_qmix_sector_cr_n3_s3_20261125-1847_nova` |
| 13 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 101 |  | finished | 1 | yes | 12908 | yes | yes | 9 | `G03.1_qmix_sector_cr_n6_s1_20261125-1847_nova` |
| 14 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 102 |  | finished | 1 | yes | 12908 | yes | yes | 3 | `G03.1_qmix_sector_cr_n6_s2_20261125-1847_nova` |
| 15 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 90 |  | finished | 1 | yes | 12908 | yes | yes | 3 | `G03.1_qmix_sector_cr_n6_s3_20261125-1847_nova` |
| 16 | QMIX | 1 | (not in queue) | ENDED  | 2,000,000 | 64 |  | finished | 1 | yes | 12908 | yes | yes | 7 | `G03.1_qmix_sector_cr_n9_s1_20261125-1847_nova` |
| 17 | QMIX | 2 | (not in queue) | ENDED  | 2,000,000 | 61 |  | finished | 1 | yes | 12908 | yes | yes | 8 | `G03.1_qmix_sector_cr_n9_s2_20261125-1847_nova` |
| 18 | QMIX | 3 | (not in queue) | ENDED  | 2,000,000 | 68 |  | finished | 1 | yes | 12908 | yes | yes | 3 | `G03.1_qmix_sector_cr_n9_s3_20261125-1847_nova` |
| 19 | ABL-VDN-GCN | 1 | (not in queue) | ENDED  | 2,000,000 | 69 |  | finished | 3 | yes | 32726 | yes | yes | 41 | `G03.1_vdn-gcn_sector_cr_n3-6-9_s1_20261125-1847_nova` |
| 20 | ABL-VDN-GCN | 2 | (not in queue) | ENDED  | 2,000,000 | 79 |  | finished | 3 | yes | 32726 | yes | yes | 12 | `G03.1_vdn-gcn_sector_cr_n3-6-9_s2_20261125-1847_nova` |
| 21 | ABL-VDN-GCN | 3 | (not in queue) | ENDED  | 2,000,000 | 73 |  | finished | 3 | yes | 32726 | yes | yes | 70 | `G03.1_vdn-gcn_sector_cr_n3-6-9_s3_20261125-1847_nova` |

## Notes

- Launched 2026-11-25 18:47 CST on ISU Nova, repo commit `5eb5f5f` (`gap_address`), with one uncommitted change to
  `tools/hpc/train_array.sbatch` (cluster lines only; no config or hyperparameter changed).
- Slurm arrays: GNN-QMIX `17768552` (tasks 1-3), VDN `17768553` (4-6), IQL `17768554` (7-9), QMIX `17768555`
  (10-18), ablation `17768571` (19-21). Wall-time limit 72 h, so every job is killed at Sat Nov 28 about 18:47 if
  still running. There is no resume (GAP-26).
- Tracebacks: BlueSky prints `ArgumentError: txt2hdg needs either a float ...` when it rejects a heading command.
  It is caught inside BlueSky and training continues (seen first in tasks 8 and 18). Cause not investigated.
  A traceback that ends the job shows as a state other than RUNNING before `eval_zero_shot.csv` is written.
- QMIX (tasks 10-18): the training loss rose from about 1e2 to 1e5-1e6 within the first 40,000 steps in all nine
  runs. The joint methods have losses near 0.02. Not changed (methods and hyperparameters are fixed); for the
  authors to judge.
- Measured rate is 2 to 3 times lower than the laptop estimate in docs/RUNBOOK.md.
- If a task dies, resubmit that task id (command in the header of `submit_all.sh`), keep the old run directory,
  and add a line here.
- Monitoring: `./watch.sh && tmux attach -t gnn_marl`. The tmux session created on 2026-11-25 lived on the VS Code
  tunnel's compute node and ended with that 2-hour job; `watch.sh` recreates it.
