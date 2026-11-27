# PLAN-GAP-31: Stop the late-training drift (all methods) and remove the GAT mixer's amplifiers, then rerun

| field | value |
|---|---|
| goals | G05.3, G05.4 (GNN-QMIX training), G03.1, G06.4 (baselines under the same settings), G06.7 (ablation), G06.3 (evaluation of a selected checkpoint) |
| paper basis | §05.1 mixer equations and training paragraph; §06 protocol. `mixer_norm` (arm C) adds a LayerNorm that is not in the paper's mixer equation: if arm C is the one reported, the equation changes and that goes to the authors (S-04) |
| status | `approved` (Aadit-Farhan Khan, 2026-11-27, in the implementing session on Nova: "I approve the plan. Implement the changes, and run the smoke test. After the smoke tests pass, run the jobs"). Deviations agreed in that session are listed in the work log (S-05) |
| author(s) / agent | Aadit-Farhan Khan with Claude (Claude Fable 5.2, Claude app), from `experiments/nova_2026-11/` and the code at `4f6d321` |
| gaps addressed | GAP-31 (late drift; shared stabilizers + kept checkpoints), GAP-32 (GAT mixer amplifiers). Does not fix GAP-26 (no resume) |
| decisions needed | Q-17 (which checkpoint is reported), Q-18 (which GNN-QMIX arm is "the proposed method"; whether these runs supersede the 2026-11-25/26 sets). The runs do not wait for these answers |

**For the implementing agent.** This plan is meant to be executed top to bottom in one session on Nova:
§3 code → §5.1 tests → one commit confirmation from the user (S-21) → §5.2 smoke → §4.3 launch.
Every flag defaults to today's behavior, so nothing already measured changes meaning. Do not redesign;
where this plan and the code disagree, stop and ask.

## 0. Why (measured on the 2026-11-25 runs, training-time evaluation, 10 episodes per evaluation)

Return per agent at n = 9, mean of the three seeds, in 250 000-step windows of `metrics.csv`:

| window | GNN-QMIX | GCN + VDN ablation | GNN-QMIX loss (s1 / s2 / s3) |
|---|---|---|---|
| 1.00M–1.50M | about -5.8 | about -7.0 | 0.03 / 0.03 / 0.13 |
| 1.75M–2.00M | about -9.9 | about -5.2 | 0.10 / 0.28 / 0.95 |

- GNN-QMIX learns as fast as or faster than the ablation until about 1.5M steps, then its loss rises in
  all three seeds and the return collapses in seeds 2 and 3. Only `final.pt` was evaluated.
- VDN (3-nearest MLP) drifts the same way and earlier (GAP-31). IQL drifts mildly. The ablation does not
  drift within 2M steps. So the instability is in the shared TD setup, and the GAT mixer amplifies it.
- **Not established:** which single setting causes the drift. No run has isolated one. This plan therefore
  applies the standard stabilizers together to every method, and separates only the mixer-specific
  changes into arms (§4.2). All expectations in §5.3 are hypotheses (S-14).

Causes read from the code (`algorithms/gnn_qmix/`):

| # | where | what | scope |
|---|---|---|---|
| 1 | `train.py::td_loss` | target = max over the target network's Q (no double Q): overestimation | all methods |
| 2 | `train.py::main` | constant `learning_rate` 5e-4 for all 2M steps | all methods |
| 3 | `Args.polyak` | 0.01; QMIX needed 0.001 (D-032) | all but QMIX |
| 4 | `networks.py` agents | own airspeed `(tas-150)/6` enters unscaled (std 4.3, max 13.7; other entries within ±3, GAP-30) | all methods |
| 5 | `GATMixer.forward` | the same unscaled airspeed is in the mixer state; `qmix_airspeed_div` reaches only `QMixer` | GNN-QMIX |
| 6 | `td_loss` | `h` goes to the mixer with gradients, so the mixer loss reshapes the GCN trunk that selects actions (plan G05.3 §3.3 chose this) | GNN-QMIX |
| 7 | `GATMixer.forward` | `w1`, `w2` = `abs(Linear(g))` on residual GAT features with no normalization: the gain dQ_tot/dQ_i is unbounded | GNN-QMIX |

## 1. Upstream inputs (what this plan consumes)
- `algorithms/gnn_qmix/{common,networks,train,evaluate}.py` and `tools/hpc/{run_task.sh,train_array.sbatch}` as of `4f6d321`.
- Observation layout `[cos(drift), sin(drift), airspeed | 7 per other agent]`: airspeed is index `N_OWN - 1` = 2.
- Entity state, `n × N_STATE` rows `[x, y, vx, vy, cos(drift), sin(drift), airspeed]`: airspeed is index `N_STATE - 1` = 6.
- `metrics.csv` long format and `evaluate.py --checkpoint/--out` (plan G05.3 §3.6).

## 2. Downstream outputs (what this plan guarantees)
- New `Args` fields (§3.1), all defaulting to current behavior. `config_sha` of every config changes
  because `Args` gains fields; say so in the work log.
- Run directory additions: `checkpoints/step_<7-digit step>.pt` (same keys as `final.pt`),
  `best_checkpoint.json`, `eval_zero_shot_best.csv` (same columns as `eval_zero_shot.csv`).
- New `metrics.csv` rows with `split=train, n_agents=0`: `lr`, `grad_norm`, `q_taken_mean`, `q_tot_mean`,
  `td_target_mean`, and when the mixer has parameters `mixer_gain_mean`, `mixer_gain_max`.
- `eval_zero_shot.csv` keeps meaning "final checkpoint". Consumer: `tools/plot_results.py` (unchanged).

## 3. Design

Three changes, three commits (S-18, S-19). Change A and B are `algorithms/`, change C is `configs/` + `tools/`.

### 3.1 Change A — GAP-31: shared stabilizers, kept checkpoints, diagnostics (`algorithms/gnn_qmix/`)

Add to `Args` (section "Training" unless noted), with these docstrings' meaning:

| field | default | behavior |
|---|---|---|
| `double_q: bool` | `False` | next action = argmax of the **online** agent on `next_obs`; its value is read from the **target** agent |
| `lr_final_frac: float` | `1.0` | learning rate falls linearly from `learning_rate` at step 0 to `learning_rate * lr_final_frac` at `steps_per_size` |
| `obs_airspeed_div: float` | `1.0` | every agent network divides observation entry 2 (own airspeed) by this before anything else |
| `checkpoint_every: int` | `0` | > 0: also save `checkpoints/step_{step:07d}.pt` at every evaluation whose step is a multiple of it. `validate()` requires a multiple of `eval_every` |
| `log_diagnostics: bool` | `False` | write the §2 diagnostic rows at every log interval |

Add `checkpoint_every` and `log_diagnostics` to `NON_SEMANTIC`.

Code, exactly:
1. `networks.py`: `GCNAgent.__init__(..., obs_airspeed_div: float = 1.0)` and
   `MLPAgent.__init__(..., obs_airspeed_div: float = 1.0)`. First line of both `forward`s, when the
   value is not 1.0: `obs = torch.cat([obs[..., :2], obs[..., 2:3] / div, obs[..., 3:]], dim=-1)`
   (no in-place edit of the replay tensor). `build_networks` passes `args.obs_airspeed_div`. Because
   `evaluate.py` builds networks through `build_networks`, evaluation gets the same scaling with no change there.
2. `train.py`: new helper
   `next_values(args, agent, target_agent, next_obs_flat) -> (q_next_value (B·n,), h_next)` under
   `torch.no_grad()`: `double_q=False` returns `target_q.max(-1).values`; `True` returns
   `target_q.gather(-1, agent(next_obs_flat)[0].argmax(-1, keepdim=True)).squeeze(-1)`. `td_loss` uses it
   for IQL and for the mixed methods. `h_next` always comes from the target agent.
3. `train.py`: `lr_at(args, step) -> float` = `learning_rate * (1 - (1 - lr_final_frac) * min(step, steps_per_size) / steps_per_size)`;
   before each `optimizer.step()` set every param group's `lr` to `lr_at(args, step)`.
4. `train.py::run_eval`: after `save("latest.pt", step)`, if `checkpoint_every > 0 and step % checkpoint_every == 0`:
   `save(f"step_{step:07d}.pt", step)`. The last `run_eval` (step = `steps_per_size`) is included.
5. `train.py`: `td_diagnostics(args, agent, mixer, target_agent, target_mixer, batch) -> dict[str, float]`,
   no optimizer step, called once per log interval on one fresh batch per fleet size and averaged over
   sizes: `q_taken_mean`, `q_tot_mean` (IQL: same as `q_taken_mean`), `td_target_mean`, and for a mixer
   with parameters `mixer_gain_mean` / `mixer_gain_max` = mean / max over batch and agents of
   `torch.autograd.grad(q_tot.sum(), q_taken)[0]` (for `mixer_agg="mean"` multiply by n so that VDN-like
   mixing reads 1.0 at every n). `grad_norm` = mean of the values returned by `clip_grad_norm_` since the
   last log. `lr` = current rate. Write them next to `loss` (CSV and `train/` TensorBoard tags, S-11).

### 3.2 Change B — GAP-32: GAT mixer options (`algorithms/gnn_qmix/`)

| field (section "Networks") | default | behavior |
|---|---|---|
| `mixer_airspeed_div: float` | `1.0` | `GATMixer` divides entity-state entry 6 (airspeed) by this before `self.embed`. Positions used for the `radius` graph are untouched |
| `mixer_detach_h: bool` | `False` | `td_loss` passes `h.detach()` to the online mixer (the target path is already without gradients) |
| `mixer_norm: bool` | `False` | `GATMixer` applies `nn.LayerNorm(embed_dim)` to every node embedding `g_i` after the GAT stack, before `pooled`, `w1`, `b1`, `w2`, `b2`. `False` uses `nn.Identity()`, so parameter names and old checkpoints are unchanged |

`build_networks` passes `mixer_airspeed_div` and `mixer_norm` to `GATMixer`. `validate()` rejects any of
the three being non-default when `algo != "gnn_qmix"`. LayerNorm acts per node with shared parameters:
no n-sized layer, permutation invariance and `W1, W2 ≥ 0` (monotonicity, IGM) are kept.
`qmix_airspeed_div` and `QMixer` are not touched (D-008, D-032).

### 3.3 Change C — configs, matrix, launch and checkpoint selection (`configs/`, `tools/`)

1. `tools/select_checkpoint.py --run-dir <run> [--window 5] [--min-step 100000]` (stdlib + csv only):
   reads `metrics.csv` rows `split=eval, metric=ep_return_per_agent` for the run's training sizes;
   per evaluation step takes the mean over training sizes; score(step) = mean of that value over the
   `window` evaluations centred on the step; candidates = steps ≥ `min-step` that have a
   `checkpoints/step_*.pt` file and a full window; picks the highest score (tie: later step). Writes
   `best_checkpoint.json` = `{checkpoint, step, score, window, min_step, rule: "centred mean of training-size eval return"}`
   and prints the file name. It never reads zero-shot results, so selection uses training sizes only.
2. `tools/hpc/run_task.sh`: matrix file from `MATRIX="${MATRIX:-tools/hpc/matrix.txt}"`. After the existing
   evaluation, if any `checkpoints/step_*.pt` exists: `BEST=$("$PY" tools/select_checkpoint.py --run-dir "$RUN_DIR")`
   (plus `${SELECT_ARGS:-}` from the environment, used only by the smoke run), then the same `evaluate`
   call with `--checkpoint "$BEST" --out eval_zero_shot_best.csv`.
3. Configs in `configs/gap31/` (§4.1), matrix `tools/hpc/matrix_gap31.txt` (§4.2), launcher
   `tools/hpc/submit_gap31.sh` (§4.3). Add the rows to `configs/README.md` and a short "GAP-31 rerun"
   section to `docs/RUNBOOK.md`. The existing `configs/*_sector_cr.json` and `matrix.txt` are **not edited**
   (an existing test asserts their content).

## 4. Configuration

### 4.1 Config files (`configs/gap31/`)

Every file starts from the matching `configs/<name>_sector_cr.json` (same `env_name`, `steps_per_size`,
`max_episode_steps`, `r_detect_nm`, `spawn_min_sep_nm`, `algo`, `train_n`, and `agent_net` /
`baseline_k_nearest` where present), gets a `_goals` line naming GAP-31 (and GAP-32 for arms A–C), and adds:

`KEEP` = `"checkpoint_every": 20000, "log_diagnostics": true`
`STAB` = `KEEP` + `"double_q": true, "polyak": 0.001, "lr_final_frac": 0.1, "obs_airspeed_div": 10.0`

| file | based on | adds | role |
|---|---|---|---|
| `gnn_qmix_repro.json` | `gnn_qmix` | `KEEP` | arm R: the 2026-11-25 settings, with checkpoints and diagnostics |
| `gnn_qmix_stab.json` | `gnn_qmix` | `STAB`, `"mixer_airspeed_div": 10.0` | arm A: shared stabilizers only |
| `gnn_qmix_stab_detach.json` | `gnn_qmix` | arm A + `"mixer_detach_h": true` | arm B |
| `gnn_qmix_stab_detach_norm.json` | `gnn_qmix` | arm B + `"mixer_norm": true` | arm C |
| `vdn_stab.json` | `vdn` | `STAB` | baseline |
| `iql_stab.json` | `iql` | `STAB` | baseline |
| `ablation_vdn_gcn_stab.json` | `ablation_vdn_gcn` | `STAB` | ablation |
| `qmix_stab.json` | `qmix` | `STAB` (keeps `"qmix_airspeed_div": 10.0`; `polyak` is already 0.001) | baseline |

With `STAB` every method has the same target rule, target rate, learning-rate schedule and airspeed
scaling, which removes the QMIX-only difference recorded in D-032.

### 4.2 Run matrix (`tools/hpc/matrix_gap31.txt`, same line format as `matrix.txt`)

| tasks | config | train-n | seeds | folder suffix |
|---|---|---|---|---|
| 1–3 | `configs/gap31/gnn_qmix_repro.json` | 3,6,9 | 1, 2, 3 | `GNNQMIX-R` |
| 4–6 | `configs/gap31/gnn_qmix_stab.json` | 3,6,9 | 1, 2, 3 | `GNNQMIX-A` |
| 7–9 | `configs/gap31/gnn_qmix_stab_detach.json` | 3,6,9 | 1, 2, 3 | `GNNQMIX-B` |
| 10–12 | `configs/gap31/gnn_qmix_stab_detach_norm.json` | 3,6,9 | 1, 2, 3 | `GNNQMIX-C` |
| 13–15 | `configs/gap31/vdn_stab.json` | 3,6,9 | 1, 2, 3 | `VDN` |
| 16–18 | `configs/gap31/iql_stab.json` | 3,6,9 | 1, 2, 3 | `IQL` |
| 19–21 | `configs/gap31/ablation_vdn_gcn_stab.json` | 3,6,9 | 1, 2, 3 | `ABL-VDN-GCN` |
| 22–30 | `configs/gap31/qmix_stab.json` | 3 (22–24), 6 (25–27), 9 (28–30) | 1, 2, 3 each | `QMIX` |

30 runs, 2 000 000 steps per training fleet size each (D-030). The four GNN-QMIX arms share the run-ID
method name `gnn_qmix`; they are told apart by parent folder and `config.json`. Run
`tools/plot_results.py` on one arm's folder at a time, never on a folder holding two arms.

### 4.3 Launch on Nova (`tools/hpc/submit_gap31.sh`)

Use a **dedicated clone** at the launch commit with its own `.venv` (as was done for
`gnn_marl_nov26`), and do not edit it until every task has finished: each task runs its evaluations
from the clone hours after it starts.

```bash
#!/usr/bin/env bash
# GAP-31 rerun (docs/plans/GAP-31_training_stability.md §4). Usage: submit_gap31.sh [TAG]   e.g. Nov27
set -euo pipefail
EXP=/work/flemingc/akhan1/gnn_marl_experiments; TAG=${1:?give a date tag, e.g. Nov27}
cd /work/flemingc/akhan1/gnn_marl_gap31
sub() {  # sub <folder suffix> <array range> <extra sbatch flags...>
  local s=$1 r=$2; shift 2
  local d=$EXP/${TAG}_GAP31_$s; mkdir -p "$d/logs"
  sbatch --parsable --array="$r" --job-name="g31-$s" "$@" \
    --export=ALL,RUNS_DIR="$d",MATRIX=tools/hpc/matrix_gap31.txt \
    --output="$d/logs/task_%a.log" tools/hpc/train_array.sbatch | sed "s/^/$s $r job /"
}
JOINT="--cpus-per-task=4 --mem=16G --time=72:00:00"
sub GNNQMIX-R 1-3 $JOINT;   sub GNNQMIX-A 4-6 $JOINT
sub GNNQMIX-B 7-9 $JOINT;   sub GNNQMIX-C 10-12 $JOINT
sub VDN 13-15 $JOINT;       sub IQL 16-18 $JOINT;   sub ABL-VDN-GCN 19-21 $JOINT
sub QMIX 22-30 --cpus-per-task=2 --mem=4G --time=24:00:00
```

Time limits: 72 h is the template's value for joint runs; set it from the measured 2026-11-25 elapsed
times in `sacct` (double Q adds one agent forward pass per update) and write what was used in the work log.
Record the job IDs. Disk: 101 checkpoints per run, well under 1 MB each.

## 5. Acceptance tests and evidence

### 5.1 Unit tests (add to `algorithms/tests/test_gnn_qmix_networks.py`; all existing tests stay green)
1. Defaults: `Args()` has every new field at its "off" value, and `GATMixer()` has exactly the parameter names it has today.
2. `next_values`: with `double_q=True` the value is ≤ the `double_q=False` value elementwise, and equal when `agent is target_agent`.
3. `lr_at`: equals `learning_rate` at step 0, `learning_rate * lr_final_frac` at `steps_per_size`, constant when the fraction is 1.0.
4. `obs_airspeed_div`: `agent(obs)` with div 10 equals a div-1 agent with the same weights on `obs` whose entry 2 was divided by 10 (GCN and MLP).
5. `mixer_airspeed_div`: the same equivalence for `GATMixer` on state entry 6.
6. `mixer_detach_h=True`: with `q_taken` detached, `q_tot.backward()` leaves every agent parameter's gradient `None` or zero; with `False` the GCN trunk gets a nonzero gradient.
7. `mixer_norm=True`: the existing permutation-invariance, monotonicity and any-n tests, parametrized over `mixer_norm`.
8. `validate()` rejects mixer options with a non-GNN-QMIX `algo`, and a `checkpoint_every` that is not a multiple of `eval_every`.
9. Every file in `configs/gap31/` loads through `args_from_dict(...).validate()` and differs from its base config only by the keys in §4.1.

S-20: Tier 0, Tier 1 (`pytest algorithms/tests`), Tier 2 before and after each change. Then show the diff
summary and results and ask the user **once** for the commits (S-21): branch
`fix/GAP-31-training-stability`, commits `fix(algorithms): …GAP-31`, `fix(algorithms): …GAP-32`,
`exp(configs): …`, `chore(tools): …`, merged to `main` and pushed before the clone in §4.3 is made.
Runs must come from a clean commit (`repo_dirty = false` in `config.json`, S-10).

### 5.2 Smoke (about 10 minutes, in the dedicated clone, before §4.3)
```bash
for t in 1 10 22; do
  SELECT_ARGS="--min-step 0 --window 1" MATRIX=tools/hpc/matrix_gap31.txt EVAL_EPISODES=2 EVAL_ENVS=2 tools/hpc/run_task.sh $t \
    --steps-per-size 1500 --learning-starts 300 --batch-size 64 --buffer-size 5000 --eval-every 750 \
    --checkpoint-every 750 --num-eval-ep 2 --exploration-steps 1000 --log-every 500 \
    --runs-dir /work/flemingc/akhan1/gnn_marl_smoke/gap31 --machine smoke
done
```
Pass = each run ends with two `wrote …` lines, and has `checkpoints/step_0000750.pt`, `step_0001500.pt`,
`best_checkpoint.json`, `eval_zero_shot.csv`, `eval_zero_shot_best.csv`,
and `metrics.csv` rows for `lr`, `grad_norm`, `q_tot_mean`, `td_target_mean` (tasks 1, 10, 22) and
`mixer_gain_mean` (all three have a parametric mixer). `tools/plot_results.py` runs on each smoke folder. Smoke numbers mean nothing.

### 5.3 Evidence from the full runs (report these; do not put numbers in the paper, S-14)
Per run, from `metrics.csv` in 250 000-step windows: mean `loss`, `mixer_gain_mean`, `q_tot_mean`, and
training-time return at n = 3, 6, 9. Per folder: `tools/plot_results.py`, plus the same zero-shot table
built from `eval_zero_shot_best.csv`.

Hypotheses to check (written before the runs):
- H1 (drift): in arms A–C and the stabilized baselines, the last window's mean loss is at most 3× the
  lowest window's, and the last window's n = 9 return is within 1.0 of the best window's. Arm R drifts again.
- H2 (mixer): the simplest of arms A, B, C that meets H1 in all three seeds is at least as good as the
  stabilized GCN + VDN ablation on training-size return over the last 250 000 steps.
- H3 (diagnosis): where drift occurs, `q_tot_mean` rises above the return actually obtained and/or
  `mixer_gain_mean` grows before the loss does.

Any outcome is a result. If no arm meets H2, record that the GAT mixer gives no benefit over the sum on
this task and pass it to the authors (S-04); do not add further fixes under this plan.

## 6. Risks and alternatives
- The stabilizers are applied as a bundle, so these runs cannot say which one matters. Arm R against
  arm A measures the bundle; separating it needs more runs and its own plan.
- `polyak = 0.001` slowed early learning in the QMIX diagnostics (record, 2026-11-26). With
  `lr_final_frac = 0.1` on top, a method may still be improving at 2M steps. The windows in §5.3 show this.
- H1 and H2 are judged on 10-episode training evaluations; the 100-episode zero-shot files are the
  check, but they include unseen sizes and must not be used to pick an arm or a checkpoint.
- Best-checkpoint reporting favors noisy methods slightly even with a centred window. Report final and
  selected side by side until Q-17 is answered.
- Not tried here, in order of expected value if H2 fails: GATv2-style attention with `mixer_graph="radius"`
  (with the full graph the current attention ranks neighbors identically for every agent), a mixer of
  the form sum + bounded correction, a larger replay buffer.

## 7. Out of scope
Resume from a checkpoint (GAP-26). Any change to the paper text or tables (authors, S-04). Aggregation
across arms and the final figures (after Q-17 and Q-18). Changing observation or state scaling inside the
environment adapter (the scaling here lives in the networks so the replay data and evaluation stay unchanged).
