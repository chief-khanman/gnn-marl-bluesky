# PLAN — Graph-Encoded MARL for Zero-Shot Scalable Conflict Resolution (BlueSky)

> **Audience:** Claude Code (VS Code). Work through the phases **in order**. Each phase ends with
> acceptance checks: run them, and only move on when they pass. Commit at the end of every phase.
>
> **Starter files already provided in this folder** (tested 2026-11-07 on Python 3.12, Linux):
> `paper/` (the extended abstract split into sections), `simulator/bluesky_marl/`,
> `marl/interfaces/bluesky_interface.py`, `marl/scripts/random_rollout.py`,
> `marl/tests/test_bluesky_interface.py`, `algorithms/qmix_bluesky.py`.
> Keep them. Don't regenerate them from scratch.

---

## 0. Context (read once)

**Research goal.** CTDE value-based MARL (QMIX) has a mixer whose input size is fixed by the
number of agents *n*, so changing *n* means retraining. We put a permutation-equivariant **GNN
encoder + fixed-size readout** in front of the centralized Q-network, so the Q-network sees a
latent `z ∈ R^d` whatever *n* is. That lets a model trained at n ∈ {3,6,9} run zero-shot at
n ∈ {10,20,50}. The full method also has a nominal waypoint controller ("PID") and a learned
conflict-resolution override, plus a latency-δ broadcast channel fed through a GRU. The paper
source is in `paper/`.

**Simulator.** BlueSky ATC simulator via the multi-agent (PettingZoo) fork of BlueSky-Gym:
`https://github.com/jangroter/bluesky-gym`, branch **`petting_zoo_sac`**, pinned at commit
**`3a536c0d23ae8340ef106620afd168c3c7681507`**. The official repo
(`TUDelft-CNS-ATM/bluesky-gym`) is single-agent only.
The useful envs live in `bluesky_zoo/`. Both are `pettingzoo.ParallelEnv`:

| key in our registry | class | goal structure | obs/agent | action/agent |
|---|---|---|---|---|
| `sector_cr` | `bluesky_zoo.sector_cr.sector_cr.SectorCR` | each aircraft has its **own** exit waypoint | `Box(24,)` (own state + 3 nearest neighbours) | `Box(-1,1,(2,))` = (Δheading×22.5°, Δspeed) |
| `merge` | `bluesky_zoo.merge.mergev3.MergeEnv` | all aircraft go to a **shared** merge point | `Box(24,)` | `Box(-1,1,(2,))` |

**MARL library.** CleanMARL: `https://github.com/AmineAndam04/cleanmarl`, pinned at commit
**`5cf837eec651e3e4c238d47770c1845eab5763e6`**. It has single-file VDN, QMIX, COMA, MADDPG,
FACMAC, IPPO and MAPPO. Envs plug in through `marlbench` (`MARLEnv` interface: `reset`,
`step`, `get_state`, `get_avail_actions`, `get_obs_size`, `get_state_size`,
`get_action_size`, `sample`, `close`).

### Verified facts and gotchas (all reproduced; don't "fix" them away)

1. **BlueSky is a process-wide singleton.** Two env objects in one process share `bs.traf`.
   The second `reset()` silently takes over the first one's aircraft (same IDs `KL001…`),
   with no error. Rule: **one BlueSky env per process.** `simulator/bluesky_marl/make_env`
   raises if you try to make a second one. Parallel or eval envs must use
   `marlbench.vec_envs.SubprocVectorEnv` (spawn).
2. **Dependency conflict.** The fork's `pyproject.toml` pins `numpy==1.24` and `torch==2.2.0`.
   CleanMARL needs Python ≥ 3.12, `numpy==2.3.4` and `torch==2.9.0`. Fix: **don't pip-install
   the fork**. Add it as a git submodule; `make_env` puts it on `sys.path`. BlueSky runs fine
   on numpy 2.3.4 (verified).
3. The fork's `pyproject` doesn't package `bluesky_zoo`, which is another reason for the
   path approach.
4. Importing `bluesky_gym` pulls in `stable_baselines3` (through `bluesky_gym/utils/logger.py`),
   so `stable-baselines3` must be installed even though we don't use it.
5. The envs **ignore `reset(seed=...)`** and use the global `np.random`. Seed with
   `np.random.seed(seed)`; the adapter already does this.
6. When an episode ends, the env sets `env.agents = []`. Reset before stepping again.
   `SectorCR` has **no time limit**. Always wrap it in `marlbench.wrappers.TimeLimit`.
7. Termination is all-or-nothing: Merge ends for everyone when any aircraft finishes;
   SectorCR truncates when all aircraft have left. So there's no per-agent death masking yet.
8. `PathPlanningEnv` (older `petting_zoo` branch) crashes in `__init__`. Don't use it.
9. `reset()` prints `episode: N, avg rew: …` every episode. Harmless; silence it later if noisy.
10. CleanMARL's `MixingNetwork(n_agents, s_dim)` is hard-wired to *n*. That's exactly the
    baseline limitation the paper argues against, so keep it unchanged as the QMIX baseline.

---

## 1. Initialize the repository (git)

Target layout (✅ = provided, 🔨 = you create):

```
gnn-marl-bluesky/
├── PLAN.md                         ✅ this file
├── CLAUDE.md                       🔨 short rules (see §1.4)
├── README.md                       🔨
├── pyproject.toml                  🔨 (see §1.3)
├── .gitignore                      🔨
├── paper/                          ✅ LaTeX (AIAA template), one file per section
├── simulator/
│   ├── __init__.py                 ✅
│   ├── third_party/bluesky-gym/    🔨 git submodule (pinned)
│   └── bluesky_marl/               ✅ make_env(name, n_agents) + env registry
├── marl/                           "MARL calling the sim": adapters, runners, tests
│   ├── interfaces/bluesky_interface.py   ✅ marlbench adapter (discrete actions, team reward, state)
│   ├── scripts/random_rollout.py         ✅ steps 6–8 smoke test
│   └── tests/test_bluesky_interface.py   ✅
├── algorithms/
│   ├── third_party/cleanmarl/      🔨 git submodule (pinned)
│   ├── qmix_bluesky.py             ✅ CleanMARL qmix.py + `bluesky` env branch
│   └── qmix_gnn/                   🔨 later (Phase 6)
├── configs/                        🔨 yaml/json run configs
└── runs/                           (gitignored) tensorboard logs, checkpoints
```

### 1.1 Commands

```bash
cd gnn-marl-bluesky
git init -b main
git submodule add -b petting_zoo_sac https://github.com/jangroter/bluesky-gym.git simulator/third_party/bluesky-gym
git -C simulator/third_party/bluesky-gym checkout 3a536c0d23ae8340ef106620afd168c3c7681507
git submodule add https://github.com/AmineAndam04/cleanmarl.git algorithms/third_party/cleanmarl
git -C algorithms/third_party/cleanmarl checkout 5cf837eec651e3e4c238d47770c1845eab5763e6
```

### 1.2 `.gitignore`
`runs/`, `wandb/`, `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`, `paper/*.aux`,
`paper/*.bbl`, `paper/*.blg`, `paper/*.log`, `paper/*.out`, `paper/*.fdb_latexmk`,
`paper/*.fls`, `paper/*.synctex.gz`, `paper/main.pdf`.

### 1.3 Python environment (Python **3.12**, uv recommended)

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install "numpy==2.3.4" "gymnasium==1.2.1" "pettingzoo==1.25.0" "marlbench==0.0.5" \
               bluesky-simulator pygame stable-baselines3 \
               "torch==2.9.0" "tyro==0.9.35" tensorboard pytest
uv pip install -e .
```
Tested versions: bluesky-simulator 1.1.1, numpy 2.3.4, pettingzoo 1.25.0, marlbench 0.0.5,
torch 2.9.0. **Don't** `pip install` CleanMARL's full pyproject: it pulls jax/flax/smaclite,
which we don't need.

`pyproject.toml`: a setuptools project named `gnn-marl-bluesky`, `requires-python = ">=3.12"`,
`[tool.setuptools.packages.find] include = ["simulator*", "marl*", "algorithms*"]` and
`exclude = ["*third_party*"]`, the deps above, and
`[tool.pytest.ini_options] testpaths = ["marl/tests"]`. Until the editable install exists,
`export PYTHONPATH=$PWD` works too.

The first BlueSky import downloads or builds a navdata cache in `~/bluesky/`. The warnings
"RTree could not be loaded" and "Failed to load BADA performance model" are expected and
harmless (it falls back to OpenAP).

### 1.4 `CLAUDE.md` (create it with these rules)
- One BlueSky env per process; parallel or eval envs go in subprocesses.
- Never edit files under `*/third_party/`. Copy files out and modify the copies (as with `qmix_bluesky.py`).
- Seed with `np.random.seed`; envs ignore `reset(seed)`.
- Run `python -m marl.scripts.random_rollout` and `pytest` before every commit.
- Paper: one section per file in `paper/sections/`; don't change claims in the paper without asking the authors.

**Acceptance:** `git status` is clean after the first commit; `git submodule status` shows both
pinned hashes.
Commit message: `chore: initialize repo, submodules, paper scaffold`.

---

## 2. Paper scaffold (`paper/`)

Provided: `main.tex` (preamble, title and authors copied from the extended abstract, then
`\input{sections/..}` for each section), `references.bib`, `figures/`, and `sections/`:

| file | contents (verbatim from the extended abstract) |
|---|---|
| `00_abstract.tex` | abstract |
| `01_nomenclature.tex` | nomenclature table |
| `02_introduction.tex` | introduction, theoretical claims, methodological constructions, experimental program |
| `03_background.tex` | MARL, GNNs for MAS, safety-critical / scale-invariant control (GCBF) |
| `04_problem_formulation.tex` | Markov game, broadcast communication model (GRU) |
| `05_methodology.tex` | GNN encoder, hybrid PID + conditional MARL, policy learning |
| `06_experiments.tex` | setup, hypothesized outcomes table, planned evaluation |
| `07_conclusion.tex` | conclusion |
| `08_acknowledgments.tex` | acknowledgments |

To do:
1. Download the AIAA LaTeX template (Overleaf: "AIAA Conference and Journal Template") and put
   `new-aiaa.cls` in `paper/`. It's not provided.
2. Build with `cd paper && latexmk -pdf main.tex`. Acceptance: `main.pdf` builds and has no undefined references.
3. `references.bib`: the entries marked **TODO-VERIFY** (`qmix-gnn`, `gnn_marl`, GCBF authors,
   GraphMIX venue) must be checked by the authors. **Don't invent bibliographic details.**
   `bluesky`, `bluesky-gym` and `cleanmarl` are added for the new implementation.
4. Add a short section `paper/sections/06a_simulation_environment.tex` (an `\input` placeholder
   is fine) that describes the BlueSky setup once Phase 4 is done. Don't edit the authors'
   claims. A `% TODO(authors)` note in `06_experiments.tex` flags that the "validation
   environments" paragraph needs updating to mention BlueSky.

Commit: `docs(paper): split extended abstract into sections`.

---

## 3. Simulator layer (`simulator/`)

Provided: `simulator/bluesky_marl/envs.py` with
`make_env(name="sector_cr"|"merge", n_agents=5, render_mode=None)`. It adds the submodule to
`sys.path`, returns the raw PettingZoo `ParallelEnv`, and refuses a second instance in the
same process.

To do:
1. Add `simulator/tests/test_api.py`. It runs PettingZoo's `parallel_api_test(env, num_cycles=200)`
   on `sector_cr` with `n_agents=3`, in a **separate process** (`subprocess` or
   `pytest-forked`) because of the singleton.
2. Later, but planned here: a `SimConfig` dataclass (n_agents, intruders *m*, detection
   radius, separation, `ACTION_FREQUENCY`) so experiments can vary them without editing
   third-party code. Do it by subclassing `SectorCR` / `MergeEnv` in
   `simulator/bluesky_marl/custom_envs.py`, not by editing the submodule.

**Acceptance:** `parallel_api_test` passes (or any failures are documented in `simulator/README.md`; the fork is research code).

---

## 4. MARL ↔ simulator: random-action smoke test (plan steps 6–8)

Provided: `marl/scripts/random_rollout.py`. It creates one multi-agent env, then at every step
builds **one joint action dict with a random action for every active agent**
(`env.action_space(agent).sample()`) and calls `env.step(actions)` once. That's
PettingZoo's Parallel API, so all agents act simultaneously. It runs 100 steps, resets
automatically if the episode ends, and logs team reward every 20 steps.

```bash
python -m marl.scripts.random_rollout --env sector_cr --n-agents 5 --steps 100
python -m marl.scripts.random_rollout --env merge     --n-agents 5 --steps 100
python -m marl.scripts.random_rollout --env sector_cr --n-agents 5 --steps 100 --render   # pygame window, local only
```

Reference output (verified):
```
env=sector_cr agents=['KL001', ..., 'KL005'] obs_dim=(24,)
done: 100 steps, resets=0, team_return≈-134, wall≈0.4s
env=merge ... done: 100 steps, resets=0, team_return≈-89, wall≈0.5s
```

To do:
1. Add `--n-agents` sweeps {3, 5, 10, 20} and check obs_dim stays 24 (fixed K=3 neighbours).
   That fixed-K encoding is what the GNN will replace.
2. Optional "parallel envs" check: 2 envs in `marlbench.vec_envs.SubprocVectorEnv`,
   each stepping with random actions for 100 steps, to confirm the multiprocess pattern works.

**Acceptance:** both commands finish 100 steps with no exception, and every acting agent gets an observation.
Commit: `feat(marl): random-action multi-agent rollout on BlueSky`.

---

## 5. Algorithms: CleanMARL QMIX baseline on BlueSky (plan step 9)

### 5.1 Adapter (provided: `marl/interfaces/bluesky_interface.py`)
`BlueSkyInterface(env_name, n_agents, team_reward="sum")` implements marlbench `MARLEnv`:
- **Discrete actions:** 15 = 5 heading levels {-1,-.5,0,.5,1} × 3 speed levels {-1,0,1}, mapped to the env's continuous `[-1,1]^2`.
- **Team reward:** the sum (or mean) of per-agent rewards, returned as a scalar, which is what QMIX expects.
- **State v0:** the concatenated agent observations (`n*24`). This is a placeholder. Phase 6 replaces it with simulator-truth state and the graph.
- `get_avail_actions` returns all ones; `sample()` gives uniform random indices.

Test: `pytest marl/tests` reproduces exactly what `qmix.py` does with an env, wrapped in
`TimeLimit → RecordEpisodeStatistics → AddAgentID` (verified passing).

### 5.2 QMIX (provided: `algorithms/qmix_bluesky.py`)
This is a copy of CleanMARL `qmix.py` with three changes: an `env_type == "bluesky"` branch
in `make_env` (adapter + `TimeLimit`), an `--n_agents` arg, and **`SubprocVectorEnv` for the
eval envs** when `env_type == "bluesky"` (singleton rule).

```bash
# smoke test (verified: ~1 min CPU, logs rollout/ eval/ train/ scalars to tensorboard)
python algorithms/qmix_bluesky.py --env_type bluesky --env_name sector_cr --n_agents 5 \
  --max_episode_steps 100 --total_timesteps 3000 --eval_steps 10 --num_eval_ep 2 \
  --log_every 5 --work_dir runs/qmix_smoke
tensorboard --logdir runs
```
The smoke run was verified: `train/loss` came out around 1e3 because rewards are
unnormalized. For real runs add `--normalize_reward` and tune `max_episode_steps` (150–300).

### 5.3 To do
1. Sanity-check CleanMARL itself on a standard task first, so algorithm bugs and env bugs can
   be told apart: `pip install "pettingzoo[mpe]"`, then
   `python algorithms/third_party/cleanmarl/cleanmarl/qmix.py --env_type pz --env_family mpe --env_name simple_spread_v3 --total_timesteps 200000`.
2. A baseline QMIX run on `sector_cr`, n=5, ~1–2M steps, 3 seeds. Log the metrics the paper
   needs: team return, **intrusions per episode** (`info[...total_intrusions]`), and
   average drift. Add these to tensorboard in `qmix_bluesky.py`.
3. Show the baseline limitation: load the n=5 QMIX checkpoint with an n=10 env and record
   that it **can't be evaluated** (mixer input mismatch). This becomes the "Not transferable"
   row of Table 1.
4. Add more baselines in the same way when needed: copy `mappo.py` → `algorithms/mappo_bluesky.py`, same `bluesky` branch.

**Acceptance:** the smoke test runs end-to-end, the simple_spread sanity run learns (return
goes up), and the BlueSky QMIX baseline produces curves for 3 seeds.
Commit: `feat(algorithms): CleanMARL QMIX baseline on BlueSky`.

---

## 6. Next phases (outline; plan each in detail before coding)

| # | Work | Paper section |
|---|---|---|
| 6.1 | **Graph builder** `marl/graph.py`: from `bs.traf`, nodes = aircraft (own state, goal-relative bearing/dist, PID action, later GRU hidden); edges within `r_detect` with features (distance, relative heading, relative velocity, time to CPA). Output `torch_geometric.data.Data` or padded tensors plus a mask. Add a `get_graph()` to the adapter. | 05 §GNN encoder |
| 6.2 | **QMIX-GNN** `algorithms/qmix_gnn/`: shared per-agent Q-network (already n-agnostic) plus a GNN encoder (L message-passing layers) → readout [sum‖mean‖max] → `z ∈ R^d` → hypernetwork mixer that conditions on `z`. Mixing over agents has to stay n-agnostic (e.g., monotonic attention / sum over agents of w_i(z,h_i)·Q_i), not fixed n slots. Write this down precisely before coding; it's the paper's core contribution. | 05, claim 1 |
| 6.3 | **Variable-n replay buffer and mixed-n training**: one process per n ∈ {3,6,9} (SubprocVectorEnv), with episodes padded to n_max plus an agent mask in a shared buffer. | 05 parallel training protocol |
| 6.4 | **Zero-shot evaluation** at n ∈ {10,20,50}: success rate, LoS rate, path efficiency, intervention count. Scaling curve and fitted degradation exponent. | 06 |
| 6.5 | **Nominal controller + conditional override**: BlueSky's autopilot already steers to waypoints (LNAV/HDG). Decide whether "PID" means that autopilot or an explicit PID on heading error that outputs HDG/SPD commands. Override only when an intruder is inside `r_detect` (BlueSky ASAS/CD can provide detection), plus an intervention cost in the reward. | 05 hybrid architecture, claim 2 |
| 6.6 | **Broadcast channel with latency δ** and a GRU belief state (start from CleanMARL `qmix_lstm.py`). | 04, claim 3 |
| 6.7 | **Non-cooperative intruders (m)**: spawn extra aircraft with `bs.traf.cre` that aren't in `env.agents`. | 04 |
| 6.8 | Baselines: MAPPO (CleanMARL), GCBF (external), PID-only ablation, and optionally BlueSky's built-in MVP conflict resolution as a classical reference. | 06 |

### Open decisions for the authors (ask; don't decide)
1. **UAS vs. airliners:** BlueSky's default traffic is A320s at FL350 with 5 NM separation.
   Keep the "UAS" framing with scaled separation and speeds, or reframe as ATM/AAM?
2. Which env is primary: `sector_cr` (individual goals) or `merge` (shared goal), or both?
3. Discretization granularity (15 actions now), and whether to move to a continuous method (FACMAC / MAPPO-continuous).
4. Does BlueSky replace the "2D navigation benchmark" in the paper, or come in addition to it?
