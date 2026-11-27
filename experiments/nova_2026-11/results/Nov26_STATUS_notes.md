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
