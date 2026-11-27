# Experiments on ISU Nova, November 2026: data inventory

Goals served: G05.4, G06.3, G06.4, G03.1. Decisions: D-030, D-032. Gaps: GAP-30, GAP-31.

Written 2026-11-27 10:15 CST. Every Slurm task listed here ended `COMPLETED` with exit code 0. Nothing is running.
This file lists what exists on the cluster and the header and metadata of each file type, so that a transfer
format can be chosen. A machine-readable copy of the run list is `runs_index.csv` in this folder (56 rows, one per
run directory).

## 1. What the experiments are

Multi-agent reinforcement learning for aircraft conflict resolution in the BlueSky simulator (`sector_cr`
environment). Five methods, 3 seeds each, 2,000,000 environment steps per training fleet size. A "joint" method
trains one network at fleet sizes n = 3, 6 and 9 together and is then evaluated zero-shot at n = 3, 6, 9, 10, 20, 50.
QMIX has a fixed-size network, so there is one run per fleet size and it can only be evaluated at that size.

| method name in files | what it is | folder to use | runs |
|---|---|---|---|
| `gnn_qmix` | proposed: GCN agents + monotonic GAT mixer, joint | `Nov25_GNN-QMIX` | 3 (seeds 1-3) |
| `vdn-k3` | VDN baseline, MLP agent on the 3 nearest aircraft, joint | `Nov25_VDN` | 3 |
| `iql-k3` | IQL baseline, same observation, joint | `Nov25_IQL` | 3 |
| `vdn-gcn` | ablation: GCN agent with a VDN sum instead of the GAT mixer, joint | `Nov25_ABL-VDN-GCN` | 3 |
| `qmix` | QMIX baseline, MLP agent on the full observation, one run per n in 3, 6, 9 | `Nov26_QMIX` | 9 |
| `qmix` (superseded) | same, before the fix; value estimate diverged | `Nov25_QMIX` | 9, **do not report** |

## 2. Directory tree

Everything is under `experiments/nova_2026-11/` in the repository (copied on 2026-11-27 from
`/work/flemingc/akhan1/gnn_marl_experiments`, `gnn_marl_diag` and `gnn_marl_smoke` on Nova; absolute paths inside
logs, `config.json` and the launch scripts still point there).

```
experiments/nova_2026-11/
  README.md                      this file
  runs_index.csv                 one row per run directory (56), `path` relative to this folder
  results/                       the results
    Nov25_GNN-QMIX/  Nov25_VDN/  Nov25_IQL/  Nov25_ABL-VDN-GCN/  Nov26_QMIX/  Nov25_QMIX/
      logs/task_<id>.log         full stdout+stderr of one Slurm task (task ids: 1-3 GNN-QMIX, 4-6 VDN,
                                 7-9 IQL, 10-18 QMIX, 19-21 ablation)
      <run_id>/                  one per run, see section 3
    Nov26_QMIX/code_changes_vs_a1ddb01.patch   the code diff the Nov26 runs were launched with (= commit d1d66b1)
    Nov25_STATUS.md  Nov26_STATUS.md           per-task table (job id, steps, rates), plus notes
    Nov25_FINAL_REPORT.md  Nov26_FINAL_REPORT.md   Slurm end state, elapsed time and peak memory per task
  diagnostics/                   15 short diagnostic runs for the QMIX divergence. NOT results
    README.md                    findings of the first round
    train_scaled_state.py  train_variants.py   wrappers that produced the variants
    logs/  <run_id>/             round 1: 5 runs, 200,000 steps
    round2/logs/  round2/<run_id>/   round 2: 10 runs, 200,000 steps
  smoke/                         11 smoke-test runs (1,500 steps). NOT results
  launch_scripts/                the submit, status and inventory scripts as used on Nova (kept for the record)
```

Not in git: `checkpoints/*.pt` (the repository ignores weights, S-21; they stay on Nova in the paths above
and in this folder's working copy there) and the TensorBoard event files (`tb/`, redundant with `metrics.csv`).

Run id format: `<goal>_<method>_<env>_n<train sizes>_s<seed>_<YYYYMMDD-HHMM>_<machine>`, e.g.
`G05.3_gnn_qmix_sector_cr_n3-6-9_s1_20261125-1847_nova`. Machine is `nova` for results, `diag-…` for
diagnostics and `smoke` for smoke tests.

## 3. Files inside one run directory

Sizes are totals over the 30 result runs in `gnn_marl_experiments`.

| file | format | total size | content |
|---|---|---|---|
| `config.json` | JSON | 270 KB | settings and provenance, section 3.1 |
| `metrics.csv` | CSV, long format | 8.8 MB | training curve and training-time evaluation, section 3.2 |
| `eval_zero_shot.csv` | CSV, one row per evaluation episode | 1.2 MB | final evaluation of `final.pt`, section 3.3 |
| `checkpoints/final.pt` | PyTorch | 3.3 MB | weights at the end of training, section 3.4 |
| `checkpoints/latest.pt` | PyTorch | 3.3 MB | weights at the last training-time evaluation (same step as `final.pt` in every finished run) |
| `tb/events.out.tfevents.*` | TensorBoard | 15 MB | the same numbers as `metrics.csv`, section 3.5 |

### 3.1 `config.json`

Top-level keys: `run_id`, `method`, `goal_ids` (list), `config_sha` (sha256 of the settings), `repo_commit`,
`repo_dirty` (bool), `submodules`, `machine`, `created` (ISO time with offset), `args` (object).

`args` keys: `algo`, `agent_net`, `env_name`, `train_n` (list), `baseline_k_nearest`, `seed`, `max_episode_steps`
(150), `team_reward` ("mean"), `state_mode`, `spawn_min_sep_nm` (6.0), `airspace` ("fixed"), `density_ref_n` (9),
`r_detect_nm` (20.0), `pos_scale_m`, `vel_scale_mps`, `embed_dim`, `gcn_layers`, `gat_layers`, `gat_heads`,
`mix_dim`, `mixer_graph`, `mixer_agg`, `hidden_dim`, `steps_per_size` (2000000), `learning_rate` (0.0005), `gamma`
(0.99), `polyak`, `batch_size` (256), `buffer_size` (100000), `train_every` (8), `learning_starts` (5000),
`start_e`, `end_e`, `exploration_steps` (100000), `clip_gradients` (10.0), `eval_every` (20000), `num_eval_ep`
(10), `num_eval_envs`, `log_every` (2000), `device`, `runs_dir`, `machine`. Nov26 runs also have
`qmix_airspeed_div`.

Settings that differ between result sets: `polyak` is 0.001 and `qmix_airspeed_div` is 10.0 in `Nov26_QMIX`;
`polyak` is 0.01 everywhere else and `qmix_airspeed_div` is absent (equivalent to 1.0) in all Nov25 runs.

Provenance caveats: every run has `repo_dirty: true`. Nov25 runs record `repo_commit` 5eb5f5f; the only
uncommitted change then was the Slurm script. Nov26 runs record a1ddb01; the code they ran is commit d1d66b1
(branch `qmix_fix`), identical to the saved patch.

### 3.2 `metrics.csv`

Header: `step,split,n_agents,metric,value,n_episodes`

| column | meaning |
|---|---|
| `step` | environment steps **per fleet size**, 0 to 2,000,000 |
| `split` | `train` or `eval` |
| `n_agents` | fleet size the row is about (3, 6 or 9); `0` for run-wide training values |
| `metric` | see below |
| `value` | float |
| `n_episodes` | episodes averaged in this row (10 for `eval`); empty for run-wide values |

Rows: 32,725 per joint run, about 13,000 per QMIX run.
- `split=eval`: every 20,000 steps (101 points per fleet size), greedy policy, 10 episodes, the nine episode
  metrics of section 3.3.
- `split=train`, `n_agents` in 3/6/9: every 2,000 steps, the same nine metrics over the exploration episodes
  that finished in that interval.
- `split=train`, `n_agents=0`: every 2,000 steps, `loss` (TD loss, absent before step 6,000), `epsilon`,
  `steps_per_s`.

### 3.3 `eval_zero_shot.csv`

Header: `run_id,algo,agent_net,env,train_n,seed,airspace,eval_n,status,episode,ep_return_per_agent,ep_length,los_pair_steps,los_rate,los_events,los_events_per_agent,conflict_free,min_sep_nm,mean_abs_drift_deg`

| column | values |
|---|---|
| `algo` | `gnn_qmix`, `vdn-k3`, `iql-k3`, `vdn-gcn`, `qmix` |
| `agent_net` | `gcn` or `mlp` |
| `train_n` | `3-6-9` for joint runs; `3`, `6` or `9` for QMIX |
| `seed` | 1, 2, 3 |
| `airspace` | `fixed` (original sector, density rises with n) or `density` (sector area scaled by n/9, constant density). At eval_n = 9 the two are the same scenario and give identical rows |
| `eval_n` | 3, 6, 9, 10, 20, 50 |
| `status` | `ok`, or `not_evaluable` (QMIX at a fleet size other than its own; one row, metric columns empty) |
| `episode` | 0 to 99 |

Rows: 1,200 per joint run (2 airspaces x 6 sizes x 100 episodes); 210 per QMIX run (200 `ok` + 10 `not_evaluable`).

Episode metrics (definitions from `marl/metrics.py`; loss of separation, LoS, is a pair closer than 5 NM):

| metric | definition |
|---|---|
| `ep_return_per_agent` | sum over steps of the mean per-agent reward (all rewards are negative) |
| `ep_length` | control steps in the episode, at most 150 (one step = 5 s simulated) |
| `los_pair_steps` | sum over steps of the number of pairs in LoS |
| `los_rate` | `los_pair_steps / (ep_length * n(n-1)/2)` |
| `los_events` | number of times a pair goes from separated to LoS |
| `los_events_per_agent` | `los_events / n` |
| `conflict_free` | 1.0 if the episode had no LoS, else 0.0 |
| `min_sep_nm` | smallest pair separation in the episode, NM |
| `mean_abs_drift_deg` | mean over steps and agents of the absolute track deviation from the goal bearing, degrees |

### 3.4 Checkpoints

`torch.load` gives a dict with keys `agent` (state dict), `mixer` (state dict, or `None` for IQL), `step` (int),
`n_actions` (15). Example sizes: GNN-QMIX agent 4,047 parameters and mixer 10,785; QMIX n = 6 has 32,464 in total.
To re-evaluate: `python -m algorithms.gnn_qmix.evaluate --run-dir <run> --checkpoint final.pt --eval-n 3 6 9 10 20 50 --episodes 100`.
No intermediate checkpoints were kept, so the best-during-training policy cannot be recovered.

### 3.5 TensorBoard

Scalar tags `eval/<metric>/n<k>`, `rollout/<metric>/n<k>` (the nine metrics, k in 3/6/9) and `train/loss`,
`train/epsilon`, `train/steps_per_s`; 57 tags in a joint run. Redundant with `metrics.csv`.

### 3.6 Task logs (`logs/task_<id>.log`)

Text. Useful line formats:
```
job 17784474_13 on nova18-72 started Thu Nov 26 04:16:30 PM CST 2026
run_dir: <absolute path>
algo=qmix agent=mlp train_n=[6] parameters=32464
[eval] step=0 n=6 return/agent=-30.789 los_rate=0.0142 conflict_free=0.30
step=2000 eps=0.981 loss=nan steps/s=66
done: 2000000 steps per fleet size (2000000 in total) in 19581 s -> <run dir>
airspace=fixed eval_n=6: episodes=100 los_rate=0.0031 conflict_free=0.78 return/agent=-8.06
wrote <run dir>/eval_zero_shot.csv
```
The logs also contain BlueSky start-up noise and occasional caught tracebacks ending in
`bluesky.stack.argparser.ArgumentError: txt2hdg needs either a float…`; these are not failures.
Wall-clock training time per run is in the `done:` line; Slurm elapsed time and peak memory are in the
`*_FINAL_REPORT.md` files.

## 4. Result runs (30)

`loss_*` are the first, largest and last logged TD loss. `last_step` 2000000 = full budget.

| set | role | run_id | method | train_n | seed | polyak | qmix_airspeed_div | last_step | loss_first | loss_max | loss_last | zero_shot_rows | metrics_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nov25_GNN-QMIX | paper | `G05.3_gnn_qmix_sector_cr_n3-6-9_s1_20261125-1847_nova` | gnn_qmix | 3-6-9 | 1 | 0.01 | 1.0 | 2000000 | 0.0367 | 0.28 | 0.1 | 1200 | 32725 |
| Nov25_GNN-QMIX | paper | `G05.3_gnn_qmix_sector_cr_n3-6-9_s2_20261125-1847_nova` | gnn_qmix | 3-6-9 | 2 | 0.01 | 1.0 | 2000000 | 0.0572 | 1.17 | 0.362 | 1200 | 32725 |
| Nov25_GNN-QMIX | paper | `G05.3_gnn_qmix_sector_cr_n3-6-9_s3_20261125-1847_nova` | gnn_qmix | 3-6-9 | 3 | 0.01 | 1.0 | 2000000 | 0.0271 | 2.16 | 1.7 | 1200 | 32725 |
| Nov25_VDN | paper | `G03.1_vdn-k3_sector_cr_n3-6-9_s1_20261125-1847_nova` | vdn-k3 | 3-6-9 | 1 | 0.01 | 1.0 | 2000000 | 0.215 | 0.215 | 0.122 | 1200 | 32725 |
| Nov25_VDN | paper | `G03.1_vdn-k3_sector_cr_n3-6-9_s2_20261125-1847_nova` | vdn-k3 | 3-6-9 | 2 | 0.01 | 1.0 | 2000000 | 0.17 | 5.64 | 5.02 | 1200 | 32725 |
| Nov25_VDN | paper | `G03.1_vdn-k3_sector_cr_n3-6-9_s3_20261125-1847_nova` | vdn-k3 | 3-6-9 | 3 | 0.01 | 1.0 | 2000000 | 0.254 | 3.91 | 3.91 | 1200 | 32725 |
| Nov25_IQL | paper | `G03.1_iql-k3_sector_cr_n3-6-9_s1_20261125-1847_nova` | iql-k3 | 3-6-9 | 1 | 0.01 | 1.0 | 2000000 | 0.024 | 0.0591 | 0.00988 | 1200 | 32725 |
| Nov25_IQL | paper | `G03.1_iql-k3_sector_cr_n3-6-9_s2_20261125-1847_nova` | iql-k3 | 3-6-9 | 2 | 0.01 | 1.0 | 2000000 | 0.0269 | 0.0603 | 0.0168 | 1200 | 32725 |
| Nov25_IQL | paper | `G03.1_iql-k3_sector_cr_n3-6-9_s3_20261125-1847_nova` | iql-k3 | 3-6-9 | 3 | 0.01 | 1.0 | 2000000 | 0.0266 | 0.0599 | 0.0252 | 1200 | 32725 |
| Nov25_ABL-VDN-GCN | paper (optional ablation) | `G03.1_vdn-gcn_sector_cr_n3-6-9_s1_20261125-1847_nova` | vdn-gcn | 3-6-9 | 1 | 0.01 | 1.0 | 2000000 | 0.58 | 0.58 | 0.00989 | 1200 | 32725 |
| Nov25_ABL-VDN-GCN | paper (optional ablation) | `G03.1_vdn-gcn_sector_cr_n3-6-9_s2_20261125-1847_nova` | vdn-gcn | 3-6-9 | 2 | 0.01 | 1.0 | 2000000 | 0.283 | 0.283 | 0.0109 | 1200 | 32725 |
| Nov25_ABL-VDN-GCN | paper (optional ablation) | `G03.1_vdn-gcn_sector_cr_n3-6-9_s3_20261125-1847_nova` | vdn-gcn | 3-6-9 | 3 | 0.01 | 1.0 | 2000000 | 0.133 | 0.137 | 0.00718 | 1200 | 32725 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n3_s1_20261126-1616_nova` | qmix | 3 | 1 | 0.001 | 10.0 | 2000000 | 2.38 | 2.38 | 0.564 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n3_s2_20261126-1616_nova` | qmix | 3 | 2 | 0.001 | 10.0 | 2000000 | 2.49 | 2.49 | 0.592 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n3_s3_20261126-1616_nova` | qmix | 3 | 3 | 0.001 | 10.0 | 2000000 | 2.3 | 2.3 | 0.501 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n6_s1_20261126-1616_nova` | qmix | 6 | 1 | 0.001 | 10.0 | 2000000 | 5.9 | 5.9 | 0.0538 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n6_s2_20261126-1616_nova` | qmix | 6 | 2 | 0.001 | 10.0 | 2000000 | 5.95 | 5.95 | 0.0394 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n6_s3_20261126-1616_nova` | qmix | 6 | 3 | 0.001 | 10.0 | 2000000 | 11.1 | 11.1 | 0.0473 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n9_s1_20261126-1616_nova` | qmix | 9 | 1 | 0.001 | 10.0 | 2000000 | 33.7 | 33.7 | 0.0494 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n9_s2_20261126-1616_nova` | qmix | 9 | 2 | 0.001 | 10.0 | 2000000 | 10.3 | 10.3 | 0.0349 | 210 | 12907 |
| Nov26_QMIX | paper | `G03.1_qmix_sector_cr_n9_s3_20261126-1616_nova` | qmix | 9 | 3 | 0.001 | 10.0 | 2000000 | 25.8 | 25.8 | 0.0404 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n3_s1_20261125-1847_nova` | qmix | 3 | 1 | 0.01 | 1.0 | 2000000 | 398 | 8.94e+07 | 8.58e+07 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n3_s2_20261125-1847_nova` | qmix | 3 | 2 | 0.01 | 1.0 | 2000000 | 432 | 1.08e+11 | 7.15e+10 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n3_s3_20261125-1847_nova` | qmix | 3 | 3 | 0.01 | 1.0 | 2000000 | 628 | 3.93e+09 | 2.63e+09 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n6_s1_20261125-1847_nova` | qmix | 6 | 1 | 0.01 | 1.0 | 2000000 | 513 | 3.28e+07 | 8.29e+05 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n6_s2_20261125-1847_nova` | qmix | 6 | 2 | 0.01 | 1.0 | 2000000 | 832 | 2.11e+07 | 1.34e+07 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n6_s3_20261125-1847_nova` | qmix | 6 | 3 | 0.01 | 1.0 | 2000000 | 1.01e+03 | 4.49e+08 | 1.22e+08 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n9_s1_20261125-1847_nova` | qmix | 9 | 1 | 0.01 | 1.0 | 2000000 | 4.23e+03 | 4.27e+08 | 3.57e+06 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n9_s2_20261125-1847_nova` | qmix | 9 | 2 | 0.01 | 1.0 | 2000000 | 830 | 7.19e+07 | 3.9e+06 | 210 | 12907 |
| Nov25_QMIX | superseded: diverged, do not report | `G03.1_qmix_sector_cr_n9_s3_20261125-1847_nova` | qmix | 9 | 3 | 0.01 | 1.0 | 2000000 | 2.16e+03 | 3.3e+08 | 3.25e+07 | 210 | 12907 |

## 5. Diagnostic runs (15, not results)

QMIX at n = 6 or 9 for 200,000 steps with one or more changes; run-id suffix names the variant
(`div10` = airspeed state entry / 10, `dq` = double Q-learning, `polyak001` = polyak 0.001, `lr1e4`,
`every32` = one update per 32 steps, `mul03` = whole mixer state x 0.3, `div1` = unchanged control,
`vdn-dq-control` = VDN with double Q). Same file layout as section 3, without `eval_zero_shot.csv`.
They are the evidence for the QMIX fix (decision D-032 in the repo's `docs/PROJECT_RECORD.md`).

| set | run_id | method | train_n | seed | polyak | last_step | loss_max | loss_last |
|---|---|---|---|---|---|---|---|---|
| diag | `G03.1_qmix_sector_cr_n3_s1_20261126-1318_diag-div10` | qmix | 3 | 1 | 0.01 | 200000 | 65.2 | 65.2 |
| diag | `G03.1_qmix_sector_cr_n6_s1_20261126-1318_diag-div1` | qmix | 6 | 1 | 0.01 | 200000 | 3.28e+07 | 2.73e+07 |
| diag | `G03.1_qmix_sector_cr_n6_s1_20261126-1318_diag-div10` | qmix | 6 | 1 | 0.01 | 200000 | 312 | 312 |
| diag | `G03.1_qmix_sector_cr_n6_s2_20261126-1318_diag-div10` | qmix | 6 | 2 | 0.01 | 200000 | 223 | 223 |
| diag | `G03.1_qmix_sector_cr_n9_s1_20261126-1318_diag-div10` | qmix | 9 | 1 | 0.01 | 200000 | 593 | 593 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v1-div10-dq` | qmix | 6 | 1 | 0.01 | 200000 | 113 | 113 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v2-div10-polyak001` | qmix | 6 | 1 | 0.001 | 200000 | 5.9 | 0.0567 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v3-div10-lr1e4` | qmix | 6 | 1 | 0.01 | 200000 | 1.15e+03 | 1.15e+03 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v5-div10-every32` | qmix | 6 | 1 | 0.01 | 200000 | 23.1 | 1.27 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v6-div10-mul03` | qmix | 6 | 1 | 0.01 | 200000 | 18.3 | 18.3 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v7-dq-only` | qmix | 6 | 1 | 0.01 | 200000 | 4.45e+06 | 2.37e+06 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v8-div10-dq-polyak001` | qmix | 6 | 1 | 0.001 | 200000 | 8.58 | 0.0352 |
| diag/round2 | `G03.1_qmix_sector_cr_n6_s1_20261126-1543_diag-v9-div10-mul03-dq` | qmix | 6 | 1 | 0.01 | 200000 | 5.43 | 5.43 |
| diag/round2 | `G03.1_qmix_sector_cr_n9_s1_20261126-1543_diag-v4-div10-dq-n9` | qmix | 9 | 1 | 0.01 | 200000 | 157 | 155 |
| diag/round2 | `G03.1_vdn_sector_cr_n6_s1_20261126-1543_diag-v10-vdn-dq-control` | vdn | 6 | 1 | 0.01 | 200000 | 0.177 | 0.0128 |

## 6. What to know before using the numbers

- **Which sets go together:** `Nov25_GNN-QMIX`, `Nov25_VDN`, `Nov25_IQL`, `Nov25_ABL-VDN-GCN`, `Nov26_QMIX`.
  `Nov25_QMIX` is superseded (loss 1e5 to 1e11).
- **QMIX differs from the other methods in two settings** (`polyak`, `qmix_airspeed_div`); this belongs in the
  paper's baseline description and is pending author review.
- **Final checkpoint is not the best policy for some runs.** `eval_zero_shot.csv` evaluates `final.pt`. In
  `vdn-k3` seeds 2 and 3 and `gnn_qmix` seeds 2 and 3 the loss rises late in training (to 5.0, 3.9, 0.36 and 1.7)
  and the training-time return at the end is worse than its best. The curves in `metrics.csv` show this.
- **Training-time evaluation is noisy:** 10 episodes per point. The zero-shot file has 100 episodes per condition per seed.
- **Seeds:** 3 per method; per-seed values should be kept (the project's rule is that every reported number
  carries its run ids and seed count).
- The repo has an aggregation tool: `python tools/plot_results.py --runs-dir <folder of run dirs> --out <dir>`
  writes `summary_training.csv`, `summary_zero_shot.csv`, `runs_used.csv` and PNG figures. It reads one folder,
  so the five sets must first be linked into one folder. It has not been run on these results.

## 7. Size

About 11 MB of CSV, JSON and logs for the 30 result runs; diagnostics and smoke runs add about 5 MB.
Checkpoints (8 MB) and TensorBoard files (20 MB) are not in git.
