# Runbook: from a fresh machine to the paper's result tables

For any user or agent. Goals served: G05.3, G05.4, G03.1, G06.3, G06.4, G06.5, G06.9.
Design and definitions: [plans/G05.3_gnn_qmix.md](plans/G05.3_gnn_qmix.md). Standards: S-07, S-10, S-14, S-22, S-23.

Status of this runbook: every command below was run on `aadit-mbp` on 2026-11-25 with short
smoke settings (and again after the protocol decisions D-027…D-030). The Slurm script is a template that has **not** been run on a cluster yet.

## 1. Set up the machine (once)

```bash
git clone --recurse-submodules <remote> && cd gnn-marl-bluesky
uv venv -p 3.12 .venv && source .venv/bin/activate      # then the installs in README "Setup"
cp .env.example .env                                    # set GNN_MARL_MACHINE and GNN_MARL_RUNS_DIR
```

On HPC put `GNN_MARL_RUNS_DIR` on project/work storage. Do the installs on a login node
(compute nodes may have no internet). The first BlueSky import builds `~/bluesky/cache/`; do
that once on the login node too: `python -m marl.scripts.random_rollout --steps 5`.

## 2. Check that the code works (about 3 minutes)

```bash
python tools/check_docs.py
pytest                                                   # network, metric and adapter tests
EVAL_EPISODES=2 EVAL_ENVS=2 tools/hpc/run_task.sh 1 \
  --steps-per-size 1500 --learning-starts 300 --batch-size 64 --buffer-size 5000 \
  --eval-every 750 --num-eval-ep 2 --exploration-steps 1000 --runs-dir /tmp/gnn_smoke --machine smoke
python tools/plot_results.py --runs-dir /tmp/gnn_smoke --out /tmp/gnn_smoke/results
```

The third command trains GNN-QMIX on n ∈ {3, 6, 9} for 1500 steps per fleet size, evaluates the
checkpoint at n ∈ {3, 6, 9, 10, 20, 50} in both airspaces and must end with
`wrote …/eval_zero_shot.csv`. Smoke numbers mean nothing.

## 3. The run matrix

[`tools/hpc/matrix.txt`](../tools/hpc/matrix.txt), one training run per line. Task ids:

| tasks | method | config | trained at | zero-shot | seeds |
|---|---|---|---|---|---|
| 1–3 | GNN-QMIX (proposed) | `configs/gnn_qmix_sector_cr.json` | n ∈ {3, 6, 9} jointly | yes | 1, 2, 3 |
| 4–6 | VDN, 3-nearest observation | `configs/vdn_sector_cr.json` | n ∈ {3, 6, 9} jointly | yes | 1, 2, 3 |
| 7–9 | IQL, 3-nearest observation | `configs/iql_sector_cr.json` | n ∈ {3, 6, 9} jointly | yes | 1, 2, 3 |
| 10–18 | QMIX, full observation | `configs/qmix_sector_cr.json` | n = 3, 6, 9 separately | no (`not_evaluable`) | 1, 2, 3 each |
| 19–21 | ablation: GCN agent + VDN sum | `configs/ablation_vdn_gcn_sector_cr.json` | n ∈ {3, 6, 9} jointly | yes | 1, 2, 3 |

**Budget (D-030):** 2 000 000 environment steps **per training fleet size** in every run, so a
joint run collects 6M steps and a QMIX run 2M. Every method sees the same experience at every
size. The `step` in the logs is the per-fleet-size count. No pair of aircraft starts closer than
6 NM (D-028). Before launching, check `artifacts/registry.toml` for a run with the same
`config_sha` (S-23).

Resources per task: a jointly trained run starts 10 processes (1 learner, 3 collectors,
6 evaluators); a QMIX run starts 4. Measured on Nova: the processes mostly wait on each other, so a joint run
uses about 1.1 CPU cores and 4.2 GB (request 4 CPUs, 8 GB) and a QMIX run 1.0 core and 2.0 GB (request 2 CPUs, 4 GB). No GPU is needed (the networks have 4 000 to 33 000
parameters). Measured on `aadit-mbp`: about 200 environment steps per second in total for a
joint run, so roughly 8 to 9 hours of collection for 6M steps, plus evaluation; a QMIX run
is about a third of that. The 24-hour limit in the Slurm template is an estimate with margin;
correct it after the first real run. There is no resume from a checkpoint (GAP-26): a job that
is killed must be started again.

## 4. Launch

One task on any machine:

```bash
tools/hpc/run_task.sh 1          # trains matrix line 1, then evaluates it at every fleet size, both airspaces
```

All core tasks on a Slurm cluster (first fill in the `TODO(cluster)` lines of the script and
record them in [HANDOFF.md](HANDOFF.md) "Cluster specifics"):

```bash
mkdir -p slurm_logs
sbatch --array=1-18 tools/hpc/train_array.sbatch         # add 19-21 for the ablation
```

To change a setting for every run, edit the config file and commit it before launching, so the
run's `config.json` and commit hash describe what ran (S-10). One-off overrides go on the
command line, e.g. `tools/hpc/run_task.sh 1 --r-detect-nm 30`; they are recorded in `config.json`.

### GAP-31 rerun (stabilized settings, kept checkpoints)

Plan: [plans/GAP-31_training_stability.md](plans/GAP-31_training_stability.md). 30 tasks in
[`tools/hpc/matrix_gap31.txt`](../tools/hpc/matrix_gap31.txt): four GNN-QMIX arms (R = the 2026-11-25
settings, A = shared stabilizers, B = A + detached embeddings, C = B + LayerNorm in the mixer), VDN, IQL, the
GCN + VDN ablation and QMIX, all with configs in `configs/gap31/`.

```bash
MATRIX=tools/hpc/matrix_gap31.txt tools/hpc/run_task.sh 4        # one task on any machine
tools/hpc/submit_gap31.sh Nov27                                  # all 30 on Nova, from the dedicated clone
SUFFIX=VDN TASKS=14 tools/hpc/submit_gap31.sh Nov27              # resubmit one task (no resume, GAP-26)
```

Each run keeps `checkpoints/step_<step>.pt` every 20 000 steps and writes the diagnostics `lr`, `grad_norm`,
`q_taken_mean`, `q_tot_mean`, `td_target_mean`, `mixer_gain_mean`, `mixer_gain_max` to `metrics.csv`
(`split=train`, `n_agents=0`). After the usual evaluation of `final.pt`, `run_task.sh` runs
`tools/select_checkpoint.py` (centred 5-evaluation mean of the return at the training sizes, steps ≥ 100 000)
and evaluates that checkpoint too: `best_checkpoint.json`, `eval_zero_shot_best.csv`. `eval_zero_shot.csv`
still means "final checkpoint". The four GNN-QMIX arms share the method name `gnn_qmix`; they are told apart
by parent folder and `config.json`, so run `tools/plot_results.py` on one arm's folder at a time.
A task that Slurm requeues after a node failure starts again from step 0 in a **new** run directory
(the log is appended to); the unfinished directory stays and must be left out of the aggregation.

## 5. Watch a run

```bash
tail -f slurm_logs/gnn-qmix_<job>_<task>.out             # "[eval] step=… n=… los_rate=…"
tensorboard --logdir $GNN_MARL_RUNS_DIR                  # rollout/, eval/, train/ tags
```

Each run directory `$GNN_MARL_RUNS_DIR/<goal>_<algo>_<env>_n<train-n>_s<seed>_<time>_<machine>/` holds:

| file | content |
|---|---|
| `config.json` | resolved settings, `config_sha`, repo commit, submodule hashes |
| `metrics.csv` | long format: `step, split, n_agents, metric, value, n_episodes` (`split` = train or eval) |
| `tb/` | the same numbers for TensorBoard |
| `checkpoints/latest.pt`, `final.pt` | weights (`agent`, `mixer`); never commit them (S-21) |
| `eval_zero_shot.csv` | one row per evaluation episode per fleet size and airspace (`fixed`: density rises with n; `density`: sector grows with n); `status = not_evaluable` when a fixed-n network is asked for another n |

To evaluate again (other sizes, more episodes): 
`python -m algorithms.gnn_qmix.evaluate --run-dir <run> --eval-n 3 6 9 10 12 20 50 --episodes 100`.

## 6. Aggregate and plot

```bash
python tools/plot_results.py --runs-dir $GNN_MARL_RUNS_DIR --out results/
```

Writes `summary_training.csv`, `summary_zero_shot.csv`, `runs_used.csv`, `curves_<metric>.png`
and `zero_shot_<metric>_<airspace>.png`. It warns about any method with fewer than 3 seeds.

## 7. Put results into the paper

`paper/sections/06b_results.tex` has empty tables and figure slots marked `% TODO(results)`,
each naming the file its numbers come from. Rules (S-14): a number enters the paper only with
the run IDs from `runs_used.csv`; the seed count is stated next to it; Table 1 stays labeled
"hypotheses" until measured values replace it. Changes to claims go to the authors (S-04).

## 8. Finish the session

Publish the checkpoints and result files that back a reported number and add their registry
entries ([ARTIFACTS.md](ARTIFACTS.md); the store is still pending Q-07). Add a work-log entry
with the job IDs, `State:` and `Next:` (S-22).
