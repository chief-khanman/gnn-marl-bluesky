# Working Across Machines: state handoff protocol

> Standard **S-22**. Applies to every contributor and agent, on every machine (personal
> laptop, lab/work machine, HPC).

## Principle

**Only two things are shared: the git remote and the artifact store.** Anything that exists
on just one machine is invisible to your collaborators, and to you on your other machines.
That includes uncommitted code, local `runs/`, agent memory, chat history and shell history.
Before you leave a machine, everything another person (or you, elsewhere) would need must be
pushed or published.

## What state exists and how it travels

| state | lives in | travels by | notes |
|---|---|---|---|
| code, docs, configs, the record | git | push / pull | the only channel for text |
| third-party code | git submodules (pinned) | `git submodule update --init` | never edit (S-06) |
| Python environment | `pyproject.toml` + lock file (INF-7) | **recreated** per machine | never copy a venv between machines |
| machine settings (paths, device, machine tag) | `.env` (gitignored), from `.env.example` | set once per machine | see below |
| trained weights, eval results, logs that back a result | artifact store + `artifacts/registry.toml` | publish / pull ([ARTIFACTS.md](ARTIFACTS.md)) | registry entry is committed to git |
| exploratory runs (`runs/`) | local disk | not shared | publish them as artifacts if anyone else may need them |
| work in progress and "what's next" | your branch + the work-log handoff fields | push | coordination board: pending decision Q-08 |
| agent memory / session context (`~/.claude/…` etc.) | **machine-local** | **does not travel** | anything an agent must know goes into repo docs |
| secrets (HF token, etc.) | per-machine login or env var | never | never commit; `.env` is gitignored |
| BlueSky navdata cache (`~/bluesky/`) | per machine | regenerated on first import | |

## Per-machine setup (once)

1. Clone with submodules: `git clone --recurse-submodules <remote>`.
2. Create the env per README "Setup" (from the lock file once INF-7 exists).
3. `cp .env.example .env` and fill in:
   - `GNN_MARL_MACHINE`: short unique tag used in run IDs, e.g. `aadit-mbp`, `lab-ws1`, `hpc`.
   - `GNN_MARL_RUNS_DIR`: where `runs/` go (on HPC: project/work storage, **not** `$HOME`).
   - `GNN_MARL_ARTIFACTS`: local artifact cache (same storage advice).
4. Log in to the artifact store ([ARTIFACTS.md](ARTIFACTS.md) §3).
5. Run the smoke tests (S-20 Tier 2) and record the machine in the work log the first time.

## Session start

1. `git fetch && git pull` on your branch; `git submodule update --init`.
2. Read `docs/PROJECT_RECORD.md` §2, §3, §7 and the latest work-log entries, especially their **Next:** lines.
3. Check that your env still matches the lock file (reinstall if `pyproject.toml` or the lock changed).
4. Before launching any non-trivial training run, look for an artifact in
   `artifacts/registry.toml` with the same `config_sha`. If one exists, pull it instead of re-running (S-23).
5. Run the S-20 baseline tests so you know which failures were already there before your change.

## Session end (handoff)

Do all of these before you walk away from a machine or end an agent session:

1. **No unpushed work.** Commit to your own branch and push. Commits marked `wip:` are allowed
   only on your own branch and are squashed or cleaned before merge (S-21).
2. **Publish shared results.** Any checkpoint or evaluation someone else may need goes to the
   artifact store, and its registry entry is committed ([ARTIFACTS.md](ARTIFACTS.md)).
3. **Write the handoff** in the work log (`PROJECT_RECORD.md` §8). Each entry ends with:
   - `State:` branch name, last commit, anything still running (e.g. HPC job IDs and where they write)
   - `Next:` the concrete next step, so the next session (on any machine) starts there
4. If a long job is still running (HPC), say who will collect its outputs and publish them.

## Run identity across machines

Run IDs must be unique across all machines. The run directory name is
`<goal-id>_<algo>_<env>_n<train-n>_s<seed>_<YYYYMMDD-HHMM>_<machine>` (S-10, D-014).
`<machine>` is `GNN_MARL_MACHINE`.

## HPC notes

- Compute nodes often have **no internet**. Do installs and artifact pulls on a login or
  transfer node, then point jobs at the local cache.
- Training is headless. Never pass `--render`. Set `SDL_VIDEODRIVER=dummy` if pygame complains.
- One BlueSky env per process (S-07) fits naturally with job arrays: one seed/config per task.
- Put `runs/` and the artifact cache on project/work storage, not the home quota.
- Cluster-specific details (cluster name, `module load` lines, partition, GPU flags) go in the
  section below, written by whoever sets it up first. **Don't guess them.**

### Cluster specifics

Filled in on 2026-11-25 for the first cluster used. Verified there unless marked otherwise.

- Cluster: **ISU Nova** (Slurm). Machine tag in run IDs: `nova`. Repo clone: `/work/flemingc/akhan1/gnn_marl`.
- Access used: VS Code tunnel, which is itself a 2-hour Slurm job on a compute node
  (submit host `nova-vscode.its.iastate.edu`). That compute node had internet, so the installs
  were done there. A tmux session started inside the tunnel job ends with that job; batch jobs do not.
- Partition / account: `nova` / `flemingc`. Partition wall-time limit 31 days; default 4 h, so always pass `--time`.
- Modules / Python setup: no `module load` lines. Python 3.12.4 from `/work/flemingc/akhan1/miniconda`,
  venv at `<repo>/.venv` (found by `tools/hpc/run_task.sh` without `PYTHON=`), pip installs from README
  "Setup" with `torch==2.9.0` from `https://download.pytorch.org/whl/cpu` (CPU-only build, D-005 pins otherwise).
- Storage paths: experiments in `/work/flemingc/akhan1/gnn_marl_experiments/<Mon><DD>_<METHOD>/`
  (`logs/task_<id>.log` plus one run directory per task); smoke runs in `/work/flemingc/akhan1/gnn_marl_smoke`.
  BlueSky cache in `~/bluesky` (home is shared with the compute nodes).
- Job script: `tools/hpc/train_array.sbatch` (10 CPUs, 72 h). One array per method, launched by
  `/work/flemingc/akhan1/gnn_marl_experiments/submit_all.sh` (outside git), which adds `--mem=32G` for joint
  runs and `--cpus-per-task=4 --mem=16G` for QMIX. Measured in the smoke job: peak memory 3.4 GB for a joint run.
- Monitoring: `/work/flemingc/akhan1/gnn_marl_experiments/watch.sh` recreates the tmux session `gnn_marl`
  (one `squeue` window, one log window per method) on the node you are logged in to.
