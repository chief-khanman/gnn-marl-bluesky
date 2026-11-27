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
