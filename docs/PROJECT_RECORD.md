# Project Record: status, standards, decisions, gaps

> **Single source of truth** for everyone working on this project, human or agent.
> **Read it before you start. Update it before you finish.** (S-01, S-02)
>
> Last updated: 2026-11-25 · Current phase: **Stage 1: Framework** (see §2)

Contents: [1 How to use](#1-how-to-use-this-record) · [2 Workflow & phase](#2-workflow-stages-and-current-phase) ·
[3 Status board](#3-status-board) · [4 Standards](#4-standards-binding) · [5 Decisions](#5-decision-log) ·
[6 Gap register](#6-gap-register) · [7 Open questions](#7-open-questions-for-the-authors) · [8 Work log](#8-work-log)

---

## 1. How to use this record

| You are about to… | Read | Then update |
|---|---|---|
| start any session | §2 phase, §3 status, §7 open questions | — |
| work on a goal | its file in `docs/goals/`, relevant D-/GAP- entries | goal status (§3 and goal file) |
| make a choice others must follow | §5 to check it isn't already decided | add a **D-** entry |
| find an inconsistency or weakness | §6 to check it isn't already logged | add a **GAP-** entry |
| need an author decision | §7 | add a **Q-** entry; don't decide it yourself |
| make a code or doc change | §4 "Change discipline" (S-18…S-21) | commit only after tests pass and confirmation |
| switch machines (laptop / lab / HPC) | [HANDOFF.md](HANDOFF.md) | push branch, publish artifacts, write `State:`/`Next:` |
| launch experiments (laptop or HPC) | [RUNBOOK.md](RUNBOOK.md) | registry entry, work log with job IDs |
| share or reuse trained weights | [ARTIFACTS.md](ARTIFACTS.md), `artifacts/registry.toml` | registry entry in the same PR |
| finish a session / open a PR | — | add a §8 work-log entry (who, which agent, what, IDs, `State:`, `Next:`) |

Related files: [README.md](../README.md) (overview) · [docs/goals/](goals/README.md) (paper → goals) ·
[HANDOFF.md](HANDOFF.md) (multi-machine) · [ARTIFACTS.md](ARTIFACTS.md) (weights) ·
[docs/plans/](plans/) (detailed plans) · [PLAN.md](../PLAN.md) (original bootstrap plan) ·
[CLAUDE.md](../CLAUDE.md) / [AGENTS.md](../AGENTS.md) (agent rules).

---

## 2. Workflow stages and current phase

The project moves through four gated stages. A stage is entered only when the previous
stage's exit criterion is met, and entering a stage is recorded as a D- entry.

| stage | activity | exit criterion |
|---|---|---|
| **1. Framework** ← *current* | repo scaffold, paper → goals, standards, seed gap register | goals cover every paper promise. Record and goals reviewed by the authors |
| **2. Gap analysis** | analyze each GAP-xx; resolve or escalate to Q-xx; authors answer Q-xx | every GAP is `resolved`, `accepted-risk`, or `deferred` with reason |
| **3. Planning** | one plan per goal or goal group in `docs/plans/` using the template; plans chained by interfaces | every plan `approved`; the interfaces between consecutive plans match (checked in a joint review) |
| **4. Implementation & evidence** | build in dependency order; produce evidence; update paper with measured results | G02.4 claim gate passes |

Exception: infrastructure items `INF-*` that PLAN.md already specifies in detail
(phases 1–5) may proceed during Stage 1–2. They exist so the gap analysis can be grounded in
a running simulator.

---

## 3. Status board

### Infrastructure (from PLAN.md phases 1–5)
| ID | item | status | notes |
|---|---|---|---|
| INF-1 | repo init, submodules pinned, `.gitignore`, `pyproject.toml`, `CLAUDE.md` | **done** 2026-11-07 | both submodule hashes verified |
| INF-2 | paper build (`new-aiaa.cls` + `paper/render.sh`) | **done** 2026-11-25 (aadit-mbp) | `paper/render.sh` + VS Code task (no `latexmk` needed). Paper compiles with no undefined references. Other machines: install the TeX packages listed in README. `new-aiaa.cls` / `.bst` are committed in `paper/` (author's decision 2026-11-25: publicly available template files) |
| INF-3 | `simulator/tests/test_api.py` (`parallel_api_test` in a subprocess) | todo | |
| INF-4 | random-rollout sweeps n∈{3,5,10,20}; SubprocVectorEnv check | todo | |
| INF-5 | QMIX smoke run, simple_spread sanity run | **in-progress** | QMIX smoke run passes in `.venv` (2026-11-09). simple_spread sanity run not done (needs `pettingzoo[mpe]`) |
| INF-6 | Python 3.12 venv per PLAN.md §1.3 | **done** 2026-11-09 (aadit-mbp) | `uv venv -p 3.12 .venv` + PLAN.md §1.3 installs + `-e .`. Versions match the tested set (bluesky-simulator 1.1.1, numpy 2.3.4, torch 2.9.0). Other machines: repeat the steps until INF-7 adds a lock file |
| INF-7 | Lock file (`uv pip compile` → `requirements.lock`, platform note for CUDA torch) | todo | needed so every machine gets the same env (HANDOFF.md) |
| INF-8 | `tools/artifacts.py` (publish / pull / verify / find) + add `huggingface_hub` to deps | todo | after Q-07 |
| INF-9 | Git remote (GitHub) + branch protection on `main` | **in-progress** | remote `origin` exists (seen 2026-11-25). Branch protection not checked |
| INF-10 | `tools/check_docs.py`: ID and link consistency check (S-20 Tier 0) | **done** 2026-11-07 | |

### Paper goals (see `docs/goals/`)
All `G*`, `T*` and `D*` goals are **defined** (Stage 1), except: G01.1 **done** (naming
contract), G00.3 **in-progress** (logging convention active), G06.10 **blocked** (Q-04),
G03.2 **deferred** (MAPPO is not a baseline for now, D-024; the smoke-tested script stays),
G03.3 **deferred** (D-024), G06.8 **retired** (D-025).
As of 2026-11-25 (plan `docs/plans/G05.3_gnn_qmix.md`, approved): G05.1, G05.2, G05.3, G05.4,
G04.5, G03.1, G06.2, G06.3 are **in-progress**: code written and smoke-tested, 3-seed HPC results
outstanding ([RUNBOOK.md](RUNBOOK.md)). G06.10 **unblocked** (D-023). G04.1 **in-progress** (configurable `sector_cr`: spawn separation,
area scaling). Work order (D-022):
(1) GNN-QMIX, (2) PID + GNN-QMIX (G05.5–G05.9, needs its own plan), then the deferred items.

---

## 4. Standards (binding)

Standards are numbered so reviews can cite them ("violates S-07"). Changing a standard needs
a D- entry. **Enforcement:** agents are pointed here by `CLAUDE.md` / `AGENTS.md`. Every PR
uses the checklist in `.github/pull_request_template.md`. Reviewers reject PRs that violate a
standard without a recorded exception. Automated checks are planned (INF, later).

### Process
- **S-01 Read first.** Read §2, §3, §7 and the relevant goal file before starting work.
- **S-02 Record after.** Every session or PR adds a §8 entry and updates goal/INF status. New
  decisions → §5. New weaknesses → §6. Questions for authors → §7. A PR without a record update is incomplete.
- **S-03 Paper is the spec.** Every module, config, experiment and plan names the goal ID(s)
  it serves (module docstring, config header, plan header). Work that maps to no goal needs a
  new goal (with paper basis) or an INF item first.
- **S-04 Paper claims belong to the authors.** Code-side contributors may add `% TODO(authors)`
  notes, the factual `06a_simulation_environment.tex`, and bibliography entries for software
  used. Anything that changes a claim, number, or framing goes to §7 as a question.
- **S-05 Stage gate.** Don't write implementation code for a goal unless its plan in
  `docs/plans/` has status `approved` (INF items excepted, see §2). Every plan uses
  `docs/plans/_TEMPLATE.md` and declares its **upstream inputs** and **downstream outputs**
  so consecutive plans connect.

### Code
- **S-06 Third-party is read-only.** Never edit `*/third_party/`. To change behavior, copy
  the file out (like `algorithms/qmix_bluesky.py`) or subclass (like `custom_envs.py`). The copy's
  header states the origin path, the pinned commit, and a list of changes.
- **S-07 Simulator rules.** One BlueSky env per process. Parallel/eval envs use
  `marlbench.vec_envs.SubprocVectorEnv`. Seed with `np.random.seed` (envs ignore `reset(seed)`).
  Always wrap `sector_cr` in `TimeLimit`. Reset after `env.agents == []`.
- **S-08 Naming follows the paper.** Use the symbol ↔ code names in
  [goals/01_nomenclature.md](goals/01_nomenclature.md). Physical quantities carry units in
  the name when ambiguous (`latency_s`, `r_detect_nm`). New symbols get added to that table first.
- **S-09 Style.** Python ≥ 3.12. Type hints on public functions. Module docstring states purpose,
  goal IDs, and paper equation numbers implemented. Run configs are `@dataclass` + `tyro` (as
  CleanMARL does). Values that an experiment may vary live in config (`SimConfig`, `configs/`),
  never as constants in code. Prefer small single-purpose modules over editing large copied scripts.
- **S-10 Experiments are reproducible.** Every run has a config file in `configs/`. The run dir
  is `$GNN_MARL_RUNS_DIR/<goal-id>_<algo>_<env>_n<train-n>_s<seed>_<YYYYMMDD-HHMM>_<machine>/`
  (D-014) and contains the resolved config, its `config_sha`, the repo commit hash and both
  submodule hashes. Minimum **3 seeds** for any reported result; the seed count is always stated
  alongside the result.
- **S-11 Metrics have one definition.** Paper metrics are implemented once (G06.2) and
  referenced by canonical name. Tensorboard tags keep CleanMARL's prefixes (`rollout/`,
  `eval/`, `train/`) followed by the canonical metric name.
- **S-12 Tests.** BlueSky tests run one per process (subprocess or `pytest --forked`). Which
  tests to run before a commit is set by S-20. Equivariance and variable-n property tests are
  mandatory for G05.2/G05.3. Tests must not require a display. Every new test directory is
  added to `testpaths` in `pyproject.toml`.

### Git
- **S-13 Commits and branches.** Conventional commits `<type>(<scope>): <summary>` with type
  ∈ {feat, fix, docs, chore, refactor, test, exp} and scope = top-level dir or `paper`. Reference
  goal IDs in the body. Branches: `<type>/<goal-id>-<slug>` (e.g. `feat/G05.2-encoder`). One
  goal per PR where possible. Never commit `runs/` or large artifacts.

### Change discipline (D-012)
- **S-18 One issue at a time.** A change addresses exactly **one** item: a goal (G…), an
  infrastructure item (INF-…), or a gap (GAP-…). Don't bundle unrelated fixes. If you notice
  another problem along the way, record it (GAP- entry or a work-log note) and leave it for its
  own change. Don't fix it in passing.
- **S-19 Stay local.** Touch only the area that the issue belongs to:

  | area | paths | typical goals |
  |---|---|---|
  | simulator | `simulator/` (not `third_party/`) | G04.1, INF-3 |
  | marl | `marl/` (interfaces, graph, metrics, scripts, tests) | G04.3–G04.5, G05.1, G06.2 |
  | algorithms | `algorithms/` (not `third_party/`) | G03.x, G05.2–G05.11 |
  | experiments | `configs/`, `artifacts/registry.toml` | G06.x |
  | paper | `paper/` | G06.11, bibliography |
  | docs / process | `docs/`, `README.md`, `CLAUDE.md`, `AGENTS.md`, `.github/`, `tools/` | — |

  A change that must cross areas goes through an interface declared in an approved plan.
  Where possible, split it into one commit per area. No drive-by reformatting, renames or
  dependency bumps. Every change updates the record (S-02), and that update doesn't count as
  leaving the area.
- **S-20 Prove nothing broke.** Before starting, run the relevant tiers on the clean tree so
  failures that were already there are known. After the change, run them again. A change
  must not turn any passing check red. The tiers:

  | tier | when | command |
  |---|---|---|
  | 0 | always (incl. docs-only changes) | `python tools/check_docs.py` |
  | 1 | code in an area changed | `pytest <area>/tests` (that area's tests) |
  | 2 | any code change | `python -m marl.scripts.random_rollout --env sector_cr` and `--env merge`, then `pytest` (all) |

  The commit body states which tiers ran and their result (e.g. `Tests: T0 pass, T2 pass`).
  A tier that couldn't run is stated as such, with the reason. Never claim a test passed if it didn't run.
- **S-21 Add, commit, push: after confirmation.**
  1. Stage **explicit paths** (`git add <paths>`, never `git add -A` / `git add .`), then
     review `git diff --staged` for anything out of scope or secret.
  2. One logical change per commit, conventional message (S-13). The body lists goal IDs and tests (S-20).
  3. **Agents never commit or push without explicit confirmation from their human in the
     current session.** Show the diff summary and test results, then wait. An approval covers
     only that change; it is not standing permission.
  4. Humans push only after S-20 passes. Never force-push `main` or a branch someone else uses.
     `wip:` commits only on your own branch, squashed or cleaned before merge.
  5. Never commit `.env`, tokens, `runs/`, checkpoints (`*.pt`) or other artifacts (they go to the store, S-23).

### Multi-machine and shared artifacts (D-013, D-015)
- **S-22 Leave no state on one machine.** Follow [HANDOFF.md](HANDOFF.md): at session end, push
  your branch, publish needed artifacts, and end the work-log entry with `State:` and `Next:`.
  Agent memory is machine-local and does not count as shared state. Anything an agent must
  know goes into repo docs.
- **S-23 Artifacts are registered and immutable.** Follow [ARTIFACTS.md](ARTIFACTS.md):
  weights-only checkpoints, manifest with checksums and `config_sha`, an entry in
  `artifacts/registry.toml`, never overwritten. Check the registry for a matching
  `config_sha` before launching a run so nobody repeats one.

### Research integrity
- **S-14 Hypothesized ≠ measured.** No number enters the paper without a run ID that produced
  it. Table 1 stays labeled "hypotheses" until G06.9 is done. Negative results are recorded in §8, not dropped.
- **S-15 No invented references.** Bibliography entries marked TODO-VERIFY stay so until an
  author checks the publisher page. Agents never fill in authors, venues or years from memory.
- **S-16 AI-use disclosure.** Each §8 entry names who did the work and which AI agent/tool
  (if any) assisted, so §08 Acknowledgments can be written accurately (G00.3).
- **S-17 Record hygiene.** §5 and §8 are append-only. Change a decision with a new entry that
  says `supersedes D-xxx`. Use absolute ISO dates. IDs are never reused.

---

## 5. Decision log

Format: **D-xxx** · date · status (`adopted` / `proposed` / `superseded by D-yyy`) · decision · reason · source.

| ID | date | status | decision | reason / source |
|---|---|---|---|---|
| D-001 | 2026-11-07 | adopted (role pending Q-04) | Simulator: BlueSky via `jangroter/bluesky-gym` branch `petting_zoo_sac` @ `3a536c0`. Envs `sector_cr` (individual goals) and `merge` (shared goal) | only multi-agent PettingZoo fork. Official repo is single-agent. PLAN.md §0 |
| D-002 | 2026-11-07 | adopted | MARL library: CleanMARL @ `5cf837e` | single-file VDN/QMIX/MAPPO/… easy to copy and modify. PLAN.md §0 |
| D-003 | 2026-11-07 | adopted | Third-party code as git submodules, **not** pip-installed. `make_env` adds bluesky-gym to `sys.path` | fork pins numpy 1.24 / torch 2.2 and doesn't package `bluesky_zoo`. PLAN.md fact 2–3 |
| D-004 | 2026-11-07 | adopted | One BlueSky env per process; `make_env` raises on a second instance | `bs.traf` is a process-wide singleton. PLAN.md fact 1 |
| D-005 | 2026-11-07 | adopted | Python 3.12; numpy 2.3.4, torch 2.9.0, pettingzoo 1.25.0, marlbench 0.0.5, gymnasium 1.2.1, tyro 0.9.35 | CleanMARL requirements; BlueSky verified on numpy 2.3.4. PLAN.md §1.3 |
| D-006 | 2026-11-07 | adopted | Seed with `np.random.seed`; adapter does it in `reset` | envs ignore `reset(seed)`. PLAN.md fact 5 |
| D-007 | 2026-11-07 | adopted (**v0, placeholder**) | Adapter: 15 discrete actions (5 heading × 3 speed), team reward = sum, state = concatenated obs | QMIX needs Discrete + scalar team reward. State replaced in G04.5. Discretization open (Q-03) |
| D-008 | 2026-11-07 | adopted | CleanMARL's fixed-n `MixingNetwork` is kept unmodified as the QMIX baseline | it is exactly the limitation the paper argues against. PLAN.md fact 10 |
| D-009 | 2026-11-07 | adopted | Framework-first workflow: Framework → Gap analysis → Planning → Implementation (§2) | user direction 2026-11-07: plan everything coherently before implementing |
| D-010 | 2026-11-07 | adopted | Documentation structure: this record is the single source of truth. `docs/goals/` mirrors `paper/sections/`. Detailed plans in `docs/plans/`. IDs per `docs/goals/README.md` | so collaborators and their agents share one status and standard set |
| D-011 | 2026-11-07 | adopted | `PLAN.md` is the canonical bootstrap plan. `PLAN_bluesky_marl_repo.md` is a byte-identical copy, kept until an author removes it | avoid two diverging copies. Edit only `PLAN.md` |
| D-012 | 2026-11-07 | adopted | Change discipline S-18…S-21: one issue per change, area-local edits, tiered regression tests, explicit staging, commit and push only after confirmation | user direction 2026-11-07 (item 4) |
| D-013 | 2026-11-07 | **proposed** (pending Q-07) | Artifact store = one private Hugging Face Hub model repo under a project org. Git-tracked `artifacts/registry.toml` as the index. Bundle/manifest format per ARTIFACTS.md | user direction 2026-11-07 (item 2). Rationale and alternatives in ARTIFACTS.md §2 |
| D-014 | 2026-11-07 | adopted | Run IDs gain a `_<machine>` suffix. Every run records `config_sha` (amends S-10) | unique IDs across machines; detect duplicate runs |
| D-015 | 2026-11-07 | adopted | Multi-machine handoff protocol (HANDOFF.md, S-22). Work-log entries end with `State:` and `Next:` | user direction 2026-11-07 (item 1) |
| D-016 | 2026-11-17 | adopted (amends D-007) | Adapter observation = own builder that sees **all** other agents: `obs_dim = 3 + 7·(n−1)` (17, 38, 59 for n = 3, 6, 9). Default `obs_mode="all_agents"`; `"env"` keeps the raw observation. State stays "concatenated obs" (now `n·obs_dim`) | raw env obs is capped at 3 neighbors and is the wrong shape for n ≤ 3 (GAP-21). User direction 2026-11-17. Plan `docs/plans/GAP-01_observation_builder.md` |
| D-017 | 2026-11-17 | adopted | Training fleet sizes are **n ∈ {3, 6, 9}**, trained jointly (n = MARL-controlled agents). The paper is the source of truth for this. Zero-shot evaluation sizes stay {10, 20, 50} | author decision 2026-11-17. Resolves GAP-01, answers Q-05 |
| D-018 | 2026-11-25 | adopted | Execution model: centralized training, **decentralized execution**, following the TransfQMix principle (arXiv:2301.05334v1). Each vehicle takes the argmax of its own Q_i from its own observation graph. The mixer is used in training only | author decision 2026-11-25. Resolves GAP-02, answers Q-06 |
| D-019 | 2026-11-25 | adopted | Proposed method = **GNN-QMIX**: a shared GCN in each agent encodes the agent's observation graph; a GAT mixer over the agent graph generates the monotonic mixing weights from per-agent node embeddings (W1 row per agent, mean‖max readout for b1, W2, b2, absolute values). No layer depends on n. Replaces "fixed latent z into a centralized Q-network" | author decision 2026-11-25 ("double GNN"). Resolves GAP-03 and GAP-20. Plan `docs/plans/G05.3_gnn_qmix.md` |
| D-020 | 2026-11-25 | adopted | The sensor radius r_detect decides who is in a vehicle's observation graph: another vehicle is added when it is inside the radius. The same per-vehicle rule is the activation trigger of the hybrid variant. The learned safety score is not the switching rule | author decision 2026-11-25. Resolves GAP-04. GAP-05 deferred |
| D-021 | 2026-11-25 | adopted | The PID controller and the learned policy both act through BlueSky's heading and speed commands (HDG/SPD). The paper's thrust/pitch/roll/yaw-rate and 6-DOF statements are removed | author decision 2026-11-25. Resolves GAP-06. Answers the dynamics part of Q-01; the UAS vs ATM framing stays open |
| D-022 | 2026-11-25 | adopted | Work order: (1) GNN-QMIX, (2) PID + GNN-QMIX. On hold until both work: latency / broadcast / GRU (GAP-07), intruders and obstacles (GAP-09), action design and continuous methods (GAP-12; the 15 discrete actions of D-007 stay) | author decision 2026-11-25 |
| D-023 | 2026-11-25 | adopted | BlueSky **replaces** both the 2D navigation benchmark and the custom 3D UAV environment | author decision 2026-11-25. Resolves GAP-11, answers Q-04. GAP-10 becomes accepted-risk |
| D-024 | 2026-11-25 | adopted | Baselines are **QMIX, VDN and IQL** only. MAPPO and GCBF are not compared for now (G03.2, G03.3 deferred). QMIX-GNN (MDPI) is not compared. Reported baseline results come from `algorithms/gnn_qmix/train.py` (same loop, state and metrics for every method; CleanMARL's mixer logic unchanged, D-008); `qmix_bluesky.py` and `mappo_bluesky.py` stay as smoke scripts | author decision 2026-11-25. Resolves GAP-13 |
| D-025 | 2026-11-25 | adopted | Emergence analysis (heading-direction entropy, trajectory mutual information, deadlock count) is removed from the paper. G06.8 retired | author decision 2026-11-25. GAP-14 deferred |
| D-026 | 2026-11-25 | adopted, amended by D-027…D-030 (r_detect still pending Q-10) | Experiment protocol: 2 000 000 environment steps per run (BlueSky-Gym's examples use `total_timesteps=2e6`; TransfQMix trains 2M), 150 steps per episode, 3 seeds, team reward = mean over agents, global state = n × 7 entity matrix (`state_mode="entities"`), r_detect = 20 NM, metrics and result files per plan §3.5–3.6, final evaluation 100 episodes per fleet size | plan `docs/plans/G05.3_gnn_qmix.md` §4. The entity state resolves GAP-24(b) |
| D-027 | 2026-11-25 | adopted | Baseline observation: **VDN and IQL read a fixed k-nearest observation** (own features + the k = 3 nearest other agents, width 24 for any n), are trained jointly at n ∈ {3, 6, 9} and are evaluated zero-shot. QMIX keeps the full observation and its fixed-n mixer, one run per fleet size. Node features follow TransfQMix: no reward and no action features | author decisions 2026-11-25. Answers Q-12 and Q-09; resolves GAP-08 and GAP-24(a) |
| D-028 | 2026-11-25 | adopted | No pair of aircraft starts in conflict: `ConfigurableSectorCR` (`simulator/bluesky_marl/custom_envs.py`, a subclass, S-06) spawns aircraft at least `spawn_min_sep_nm` = 6 NM apart, for every method in training and evaluation. Metrics then count only conflicts that arise under the policy. Reported safety metrics: LoS rate, LoS events, conflict-free episode rate | author decision 2026-11-25. Answers Q-14; resolves GAP-25. The 6 NM value (1 NM above the minimum) is proposed |
| D-029 | 2026-11-25 | adopted | Zero-shot evaluation uses **both** airspace rules: `fixed` (original sector, density rises with n) and `density` (sector area × n / 9, constant density). Training uses `fixed` | author decision 2026-11-25. Answers Q-11; resolves GAP-19. The reference size 9 (largest training size) is proposed |
| D-030 | 2026-11-25 | adopted (amends D-026) | Training budget = **2 000 000 environment steps per training fleet size** for every method (joint runs 6M in total, each QMIX run 2M). Reason: a per-run budget would give a single-size model three times the experience at its size; per-size, every method sees the same data at every size, the same total, and the same number of updates. All step counts in configs and logs are per fleet size | author direction 2026-11-25 ("budget should properly reflect experiment and have valid reasoning"). Answers Q-15 |
| D-031 | 2026-11-25 | adopted | The paper scope changes of 2026-11-25 are approved: claim 3 (latency) removed, claim 1 restated as two propositions, the "no prior work" novelty sentence removed, safety score / CBF as future work, MAPPO and GCBF in related work only. TransfQMix as a baseline is on hold until the first results exist; it is then added to compare parameter efficiency and performance | author decision 2026-11-25. Answers Q-16; Q-13 on hold |

---

## 6. Gap register

Seeded during Stage 1 by reading the paper against the simulator and PLAN.md. **Entries
are observations, not yet analyzed** (Stage 2 does that). Status: `open` → `resolved` (D-xxx)
| `accepted-risk` | `deferred` | `escalated` (Q-xx).

Severity: **H** blocks a core claim or goal · **M** affects an experiment or its fairness · **L** wording/consistency.

| ID | sev | observation | affects | status |
|---|---|---|---|---|
| GAP-01 | M | Training fleet sizes are inconsistent: §02 and Table 1 say "training at n=5"; §05 says n∈{2,4,8,16,…,n_max}; §06 says n∈{3,5,8}. | G05.4, G06.1, G06.9 | **resolved** (D-017): paper §02, §05, §06 and Table 1 now all say n ∈ {3, 6, 9} |
| GAP-02 | H | Execution model is unclear: §05 says that at activation the policy builds G and "queries the centralized Q-network to select the optimal joint corrective action". That is centralized execution, which conflicts with the CTDE framing and the decentralized broadcast channel (in QMIX the mixer is used only in training). | G05.3, G04.5, C1 | **resolved** (D-018): decentralized execution, mixer in training only |
| GAP-03 | H | A centralized Q-network on a fixed z can't by itself output per-agent actions for variable n. The paper doesn't say how per-agent utilities, z and $Q_{tot}$ combine, or how monotonicity/IGM is kept. | G05.3, T1 | **resolved** (D-019): per-agent utilities from the GCN agent, monotonic GAT mixer; monotonicity and any-n forward/backward are tested |
| GAP-04 | M | Two activation triggers: "when any other agent enters $r_{\text{detect}}$" vs the per-vehicle rule $s^i<\theta$ (Eq. 4). Which governs, and is activation per-vehicle or fleet-wide? | G05.6 | **resolved** (D-020): the sensor-radius rule governs, per vehicle |
| GAP-05 | M | Training of $s^i$ is under-specified: §03 says it approximates the CBF condition (needs a supervised/auxiliary loss and a defined $h^{ij}$); §05 gives no loss. Also, $h^{ij}$ (CBF) collides with $h^i_t$ (GRU) in notation. | G03.4, G05.6, G01.1 | deferred (D-020): the safety score is no longer the switch; kept as a possible extension. Notation clash still to fix if it returns |
| GAP-06 | H | Dynamics mismatch: the paper's PID outputs thrust/pitch/roll/yaw rate (6-DOF UAS). BlueSky is a point-mass airliner model with HDG/SPD/ALT commands (A320, FL350, 5 NM separation, 150 kt). | G05.5, G03.3, G04.1, G00.2 | **resolved** (D-021): HDG/SPD commands; paper updated. Framing part stays in Q-01 |
| GAP-07 | M | Latency scale: δ∈{0,0.1,0.5,1.0} s, but a `sector_cr` control step is `ACTION_FREQUENCY`=5 × `DT`=1 s = **5 s** of sim time. All proposed δ fall inside one step unless step timing changes. | G04.3, G06.6, T3 | deferred (D-022) |
| GAP-08 | L | Node features include "received reward" and "current action". Reward isn't available at decentralized execution time, and current action is circular at decision time. | G05.1 | **resolved** (D-027): both features dropped, as in TransfQMix |
| GAP-09 | M | Static obstacles and non-cooperative intruders (m) don't exist in the BlueSky envs. They must be added (e.g. `bs.traf.cre` aircraft not in `env.agents`). | G04.1, G04.3 | deferred (D-022) |
| GAP-10 | M | The motivation (and the §06 custom 3D env) relies on vehicles entering and leaving mid-episode. The BlueSky envs have fixed n per episode and all-or-nothing termination (PLAN.md fact 7), and there's no per-agent death masking. | G04.1, G05.4, G06.10 | accepted-risk (D-023): fleet size is fixed within an episode and varies between episodes; the observation graph still changes as vehicles enter and leave the sensor radius. Stated as a limitation in the paper |
| GAP-11 | M | Validation environments: the paper names a 2D navigation benchmark + a custom 3D 6-DOF env. The implementation uses BlueSky. Replace or add? (`% TODO(authors)` in `06_experiments.tex`) | G06.10, G00.2 | **resolved** (D-023) |
| GAP-12 | M | Action design: 15 discrete absolute actions (D-007) vs "corrective commands relative to the PID baseline". MAPPO/GCBF are naturally continuous. | G05.7, G03.2 | deferred (D-022) |
| GAP-13 | H | Baseline fairness: VDN/IQL and parameter-shared IPPO/MAPPO-with-local-critic *are* n-transferable. The "QMIX/MAPPO not transferable" row may be challenged, and InforMARL/GraphMIX/QMIX-GNN are discussed but not compared against. | G03.1, G03.2, G06.4, G06.9 | **resolved** (D-024, D-027): baselines QMIX, VDN, IQL; VDN and IQL on the fixed k-nearest observation are evaluated zero-shot |
| GAP-14 | M | Emergence metrics (heading-direction entropy, trajectory MI, deadlock) have no operational definitions yet. | G06.2, G06.8 | deferred (D-025): concepts removed from the paper |
| GAP-15 | M | C2 needs a defined "end-to-end MARL" comparator and a rule for transitions where no agent is overridden (do they train the Q-net?). | G05.11, G06.7, T2 | half resolved: the end-to-end comparator is GNN-QMIX without the PID layer (D-022). The rule for transitions with no active vehicle is open (PID + GNN-QMIX plan) |
| GAP-16 | L | Bibliography TODO-VERIFY: `qmix-gnn`, `gnn_marl` (placeholders), GCBF author list, GraphMIX year/venue. | paper | open (authors) |
| GAP-17 | L | Missing nomenclature symbols: $L$ (GNN layers), $h^{ij}$, $\alpha$, $d$ readout composition. | G01.1 | open: the paper's nomenclature was rewritten on 2026-11-25; `docs/goals/01_nomenclature.md` still has to follow |
| GAP-18 | M | "Mission success" per env: `merge` ends for everyone when any aircraft finishes; `sector_cr` truncates when all have left. Per-agent success needs a definition. | G04.2, G06.2 | open → Q-02. No mission-success rate for now; safety metrics per D-028 |
| GAP-19 | M | Scaling rule for zero-shot tests: does airspace size grow with n (constant density) or stay fixed (rising density)? This changes what "generalization" means. | G06.1, G04.1, D1 | **resolved** (D-029): both rules are evaluated |
| GAP-20 | M | Sum pooling in the readout scales with n, so z's distribution shifts at unseen n. That may hurt zero-shot even though the dimension is fixed (C1 is about dimension/invariance, not distribution). | G05.2, T1, D1 | **resolved** (D-019): mean and max readouts, mean aggregation in the mixer, mean team reward |
| GAP-21 | M | Env observations hold only the `NUM_AC_STATE`=3 nearest *agents*. For n≤3 the obs is shorter than the declared `Box(24,)` (17 at n=3, 10 at n=2), which breaks fixed-shape code for the n=3 sweep (INF-4) and the {3,5,8} / {2,4,…} training sets. For n>4, agents beyond the 3rd neighbor are invisible to the policy. | INF-4, D-007, G04.1, G05.1 | **resolved** for the adapter (D-016). The raw env (`obs_mode="env"`) is unchanged |
| GAP-22 | L | Env-wrapper unit/doc mismatches: the speed action adds a "kt" delta (20/3) to CAS in m/s before converting, so the effective step is ≈±13 kt, not ±6.7 kt. `merge` compares `DISTANCE_MARGIN` (labeled km) with a distance in NM. The obs docstrings say "heading difference", but the code computes the relative-velocity track angle. The `merge` docstring lists FAF distance and a FAF-reached flag that the obs doesn't contain. | G04.1, G05.1, G05.7 | open |
| GAP-24 | M | With D-016 the per-agent obs width depends on n. (a) Any network that reads `obs` directly (QMIX/MAPPO agents, not only the mixer/critic) is now n-specific, so the baselines' "not transferable" result is partly a product of our observation design; a reviewer may ask for a fixed-width (k-nearest or pooled) baseline. This changes the GAP-13 picture. (b) The concatenated state grows as n·(7n−4): 51, 228, 531 for n = 3, 6, 9 and 17 300 at n = 50. | G03.1, G03.2, G04.5, G05.1, G06.4 | **resolved**: (a) D-027 (k-nearest observation for VDN and IQL; QMIX stays n-specific by design), (b) D-026 (entity state) |
| GAP-23 | L | `algorithms/qmix_bluesky.py` has no origin header (source path, pinned commit, list of changes), which S-06 requires. `mappo_bluesky.py` has one to copy from. | G03.1 | open |
| GAP-25 | M | `sector_cr` spawns aircraft uniformly at random, so pairs can start inside the 5 NM minimum. Such pairs count in `los_pair_steps` and make `conflict_free` 0 whatever the policy does (seen in smoke runs at n = 9; expected for almost every episode at n = 20 and 50), but they are not counted in `los_events`. The conflict-free rate cannot be a primary metric at large n unless spawning or the metric changes. | G06.2, G06.5, G06.9, G04.1 | **resolved** (D-028): 6 NM minimum spawn separation; tested at n = 9 and n = 50 |
| GAP-26 | L | `algorithms/gnn_qmix/train.py` cannot resume from a checkpoint. A killed HPC job has to start again. | G05.4 | open |
| GAP-27 | L | The paper was rewritten on 2026-11-25 (new sections `06a`, `06b`, new claims and contribution list). `docs/goals/*` still quote the old text in their "Paper basis" lines, and the paper has not been compiled (INF-2). | all goals, INF-2 | open |
| GAP-28 | L | In the fixed airspace at n = 50 with 6 NM spawn separation the sector is nearly full; sectors that cannot hold the fleet are redrawn, so the n = 50 fixed-airspace scenarios are biased toward the larger sectors (2400–3750 NM² range). | G06.5, D-028 | open |
| GAP-29 | L | The k-nearest baselines see the 3 nearest aircraft at any distance, while GNN-QMIX sees only aircraft inside r_detect = 20 NM. The two observations differ in range as well as in structure. | G03.1, G06.4, D-027 | open |

---

## 7. Open questions for the authors

Agents and code contributors **don't decide these**. Record the answer as a D- entry when given.

| ID | question | from | status |
|---|---|---|---|
| Q-01 | UAS vs airliners: keep the "UAS" framing with scaled separation/speeds in BlueSky, or reframe as ATM/AAM? What dynamics level does the PID act on? | PLAN.md §6; GAP-06 | **partly answered** 2026-11-25 (D-021: HDG/SPD commands). Still open: keep the UAS / AAM framing and title, or reframe as ATM? |
| Q-02 | Primary env: `sector_cr` (individual goals), `merge` (shared goal), or both? | PLAN.md §6; GAP-18 | open |
| Q-03 | Action discretization (15 now) vs a continuous method (FACMAC / MAPPO-continuous); absolute vs residual-on-PID actions | PLAN.md §6; GAP-12 | deferred (D-022) |
| Q-04 | Does BlueSky replace the "2D navigation benchmark", or come in addition? What happens to the custom 3D env? | PLAN.md §6; GAP-11 | **answered** 2026-11-25: BlueSky replaces both (D-023) |
| Q-05 | Which training fleet sizes are canonical: n=5, {3,5,8}, or {2,4,8,16,…}? | GAP-01 | **answered** 2026-11-17: {3, 6, 9} (D-017) |
| Q-06 | Is execution meant to be decentralized (CTDE) or centralized-on-activation? | GAP-02 | **answered** 2026-11-25: decentralized, TransfQMix principle (D-018) |
| Q-07 | Confirm the artifact backend (HF Hub proposed, D-013). Who creates the org, and what is it named? | 2026-11-07 item 2 | open |
| Q-08 | Work-coordination mechanism (who is working on what, claims, "next"): under discussion, not implemented | 2026-11-07 item 3 | open (discussion) |
| Q-09 | GAP-08 in plain words: the old paper put each vehicle's "received reward" and "current action" on its graph node. A vehicle decides from its node features, but the reward only exists after it has acted, and the current action is the thing being decided, so neither can be an input at decision time. Options: (a) drop both (proposed, and what the code does; TransfQMix also drops last actions), (b) use the previous step's action, (c) previous action and previous reward. Which? | GAP-08 | **answered** 2026-11-25: drop both, follow TransfQMix (D-027) |
| Q-10 | Value of the sensor radius r_detect. Proposed 20 NM (sector diameter ≈ 55–70 NM, separation minimum 5 NM, head-on closing ≈ 100 s before a 5 NM loss at 20 NM). | D-020, D-026 | open |
| Q-11 | Zero-shot scaling rule: keep the sector size fixed (density rises with n, current behavior) or grow the area with n (constant density)? Also: are the evaluation sizes {10, 20, 50} final (is n = 12 wanted)? | GAP-19 | **answered** 2026-11-25: both rules (D-029). Evaluation sizes stay {10, 20, 50} |
| Q-12 | Baseline observation. Now QMIX/VDN/IQL read the all-agents vector (width depends on n), so they cannot be evaluated at another n at all. Alternative: give them a fixed k-nearest observation so VDN and IQL can be evaluated zero-shot, which is the stronger comparison. Which goes in the paper? | GAP-24a, GAP-13 | **answered** 2026-11-25: fixed k-nearest observation for VDN and IQL (D-027) |
| Q-13 | Should TransfQMix itself be a baseline? It is the closest prior method and reviewers are likely to ask. Not implemented. | D-018, D-024 | on hold (D-031): add after the first results |
| Q-14 | Success metric. `sector_cr` has no "mission complete" event, and aircraft can spawn already in conflict (GAP-25). Options: change spawning so no pair starts inside 5 NM (needs an env subclass), ignore pairs in conflict at t = 0, or report LoS rate and LoS events only. Is a mission-success rate still wanted? | GAP-18, GAP-25 | **answered** 2026-11-25: change spawning so no pair starts in conflict (D-028) |
| Q-15 | Step budget. A jointly trained GNN-QMIX run spends 2M steps over three fleet sizes; each baseline spends 2M at one size. Keep equal budget per run (current), or equal budget per fleet size? | D-026 | **answered** 2026-11-25: budget per fleet size (D-030) |
| Q-16 | Paper scope changes made on 2026-11-25 need author sign-off (S-04): theoretical claim 3 (latency) removed; claim 1 restated as two propositions; the novelty sentence "no prior work resolves the agent-count coupling" removed because TransfQMix does; safety score / CBF threshold moved to future work; "PID action as input to the policy" left undecided; MAPPO and GCBF only in related work. | D-018…D-025 | **answered** 2026-11-25: approved (D-031) |

---

## 8. Work log

Append-only, newest last. Format: date · who / agent · machine · what · tests (S-20) · IDs
touched, then `State:` (branch, last commit, running jobs) and `Next:` (concrete next step), per S-22.

- **2026-11-07** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · Stage 1 started.
  Initialized git repo (`main`). Added submodules `simulator/third_party/bluesky-gym` @
  `3a536c0` and `algorithms/third_party/cleanmarl` @ `5cf837e`. Created `.gitignore`,
  `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, `README.md`, this record, `docs/goals/*`
  (paper → goals), `docs/plans/_TEMPLATE.md`, `configs/README.md`, PR template. Seeded GAP-01…20
  and Q-05/Q-06. Not done: no venv (dev machine lacks Python 3.12/`uv`), so `random_rollout` and
  `pytest` were **not run** before the initial commit (S-12 exception for the scaffold commit
  only). Paper not built (no `new-aiaa.cls` / `latexmk`).
  IDs: INF-1, D-001…D-011, G01.1, G00.3, GAP-01…20, Q-05, Q-06.
- **2026-11-07 (2)** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · aadit-mbp ·
  Added the multi-machine and collaboration standards: change discipline S-18…S-21 (D-012),
  `docs/HANDOFF.md` + `.env.example` (S-22, D-015), `docs/ARTIFACTS.md` +
  `artifacts/registry.toml` (S-23, D-013 *proposed*), run-ID machine suffix + `config_sha`
  (D-014, amends S-10), `tools/check_docs.py` (INF-10). Updated CLAUDE.md, AGENTS.md, PR template,
  README, `.gitignore`. Item 3 (who-works-on-what board) is **not implemented**; it's under
  discussion as Q-08. Tests: T0 pass (also checked that it catches planted errors); T1/T2 not
  applicable (no code touched in an area with tests).
  IDs: D-012…D-015, S-18…S-23, INF-7…INF-10, Q-07, Q-08.
  State: `main`, uncommitted, awaiting confirmation to commit (S-21); no git remote yet (INF-9).
  Next: decide Q-08 (coordination board) and Q-07 (artifact backend); create the GitHub remote; then Stage 2 gap analysis.
- **2026-11-09** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · aadit-mbp ·
  Read-only inventory: compared the states, actions and reward variables the paper proposes with
  what `sector_cr` / `merge` (mergev3) expose (answer given in chat, not saved as a doc). BlueSky
  isn't installed here, so `bs.traf` fields weren't introspected. Logged GAP-21 (obs length depends
  on n for n≤3; only 3 neighbors visible) and GAP-22 (unit/doc mismatches in the env wrappers).
  No code changed. Tests: T0 pass; T1/T2 not applicable.
  IDs: GAP-21, GAP-22.
  State: `main`, uncommitted (on top of the 2026-11-07 (2) changes), no remote.
  Next: same as the previous entry. GAP-21 should be considered before INF-4.
- **2026-11-09 (2)** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · aadit-mbp ·
  INF-6: installed `uv` 0.13.7 (Homebrew), created `.venv` (CPython 3.12.16, gitignored), ran the
  PLAN.md §1.3 installs and `uv pip install -e .`. Resolved: bluesky-simulator 1.1.1,
  bluesky-navdata 1.0.0, numpy 2.3.4, torch 2.9.0, pettingzoo 1.25.0, marlbench 0.0.5,
  gymnasium 1.2.1, tyro 0.9.35, stable-baselines3 2.9.2, pytest 9.2.0. First BlueSky import
  built `~/bluesky/cache/navdata.p`. Smoke test: `make_env` for `sector_cr` and `merge` (n=5),
  50 random steps each, obs (24,), in separate processes. No repo code changed.
  Tests: T0 pass; T2 pass (`random_rollout --env sector_cr` and `--env merge` 100 steps each; `pytest` 1 passed).
  IDs: INF-6.
  State: `main`, uncommitted, no remote. `.venv` is local to this machine.
  Next: INF-7 (lock file, so other machines get the same env), then INF-3/INF-4 (mind GAP-21 at n=3).
- **2026-11-09 (3)** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · aadit-mbp ·
  Baselines. INF-5 (part): QMIX smoke run (`qmix_bluesky.py`, sector_cr, n=5, 3000 steps) passes
  in `.venv` and logs rollout/eval/train scalars; `train/loss` ≈1e3 as before (rewards unnormalized).
  G03.2: wrote `docs/plans/G03.2_mappo_baseline.md`, **approved by Aadit-Farhan Khan 2026-11-09**,
  then created `algorithms/mappo_bluesky.py` (CleanMARL `mappo.py` @ `5cf837e` + origin header + the
  same 3 edits as the QMIX copy). Smoke runs on `sector_cr` and `merge` (n=5, 3000 steps) wrote all
  plan §2 tags; entropy ≈2.69 ≈ ln 15. Logged GAP-23 (QMIX copy lacks its S-06 header). The plan's
  §6 notes that a MAPPO actor with `agent_ids=False` is n-transferable, which bears on Table 1 (GAP-13).
  Tests: T0 pass; T2 pass (`random_rollout` both envs; `pytest` 1 passed). No T1 tests exist for `algorithms/`.
  IDs: INF-5, G03.2, GAP-23, GAP-13.
  State: `main`, uncommitted, no remote. Smoke-run outputs only in the session scratchpad (not kept).
  Next: authors decide how Table 1 treats MAPPO transfer (GAP-13); INF-5 simple_spread sanity run;
  INF-7 lock file.
- **2026-11-17** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) · aadit-mbp · branch `gap_address` ·
  GAP-01. Probed the raw envs first: per-agent obs is 17 / 24 / 24 for n = 3 / 6 / 9 (not linear; n = 3
  mismatches the declared 24), and extra non-agent aircraft are accepted by BlueSky but invisible to
  obs and reward. Author decision: training sizes n ∈ {3, 6, 9} agents (D-017). Built the observation
  builder (`marl/interfaces/observation.py`, adapter `obs_mode`, D-016; plan
  `docs/plans/GAP-01_observation_builder.md`). **Measured** in both envs: obs width 17 / 38 / 59 / 80
  for n = 3 / 6 / 9 / 12 (declared = actual, +21 per +3 agents); concatenated state 51 / 228 / 531 / 960
  (quadratic, GAP-24). QMIX and MAPPO smoke runs pass at n = 3 and n = 9 (n = 3 could not run before).
  Paper: §02, §05, §06 and Table 1 now state n ∈ {3, 6, 9}; evaluation sizes {10, 20, 50} untouched.
  Updated README, PLAN.md, ARTIFACTS.md and goals 00/03/05/06 to match.
  Tests: T0 pass; T1 pass (`pytest marl/tests`, 6 passed; 1 passed before the change); T2 pass
  (`random_rollout` both envs; `pytest` 6 passed).
  IDs: GAP-01, GAP-21, GAP-24, D-016, D-017, Q-05, G04.1, G05.1, G06.1.
  State: `gap_address`, uncommitted, no remote.
  Next: GAP-02 (execution model). Authors: confirm evaluation sizes (is n = 12 wanted?) and see GAP-24.
- **2026-11-25** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2; one sub-agent rewrote `paper/`) ·
  aadit-mbp · branch `gap_address` ·
  Author decisions on the gap register (D-018…D-025): TransfQMix-style decentralized execution; GNN-QMIX =
  GCN agents + monotonic GAT mixer; sensor-radius graph and trigger; HDG/SPD actions; BlueSky replaces the
  other environments; baselines QMIX, VDN, IQL; latency, intruders/obstacles and action design on hold;
  emergence analysis removed. By author direction this session covered several gaps and areas at once
  (exception to S-18/S-19). Wrote and got approved `docs/plans/G05.3_gnn_qmix.md`, then built:
  `algorithms/gnn_qmix/` (networks, train, evaluate, common), `marl/metrics.py`, adapter `state_mode="entities"`
  plus per-step safety metrics, `configs/*.json`, `tools/plot_results.py`, `tools/hpc/` (matrix, task
  launcher, Slurm template), `docs/RUNBOOK.md`. Step limit: 2 000 000 per run (BlueSky-Gym's examples use
  2e6). Paper: all sections rewritten to the new scope following TransfQMix's structure; results section
  `06b_results.tex` is an empty skeleton with `% TODO(results)` markers; no numbers entered (S-14);
  bibliography entries added only from the TransfQMix PDF, `gcn`, `gat`, `iql` are TODO-VERIFY (S-15).
  The paper was **not compiled** (INF-2). **Smoke runs only** (3000 steps, not results): all four algorithms
  and the GCN + VDN ablation train; GNN-QMIX trained on {3, 6, 9} evaluates at n = 10, 20, 50 with the same
  weights; a QMIX n = 3 checkpoint reports `not_evaluable` at n = 10; the plotting tool produced tables and
  figures. About 200 env steps/s for the joint run on this laptop. Slurm template not run on a cluster.
  Found: aircraft can spawn in conflict (GAP-25); no resume (GAP-26); goal files lag the paper (GAP-27).
  Tests: T0 pass; T1 pass (`pytest marl/tests` 10 passed, `pytest algorithms/tests` 17 passed; 6 passed before
  the change); T2 pass (`random_rollout` both envs; `pytest` 27 passed).
  IDs: D-018…D-026, GAP-02…GAP-15, GAP-17…GAP-20, GAP-24…GAP-27, Q-01, Q-03, Q-04, Q-06, Q-09…Q-16, G03.1,
  G04.5, G05.1…G05.4, G06.2, G06.3, G06.8, G06.10.
  State: `gap_address`, uncommitted (includes the 2026-11-17 changes), no remote. Nothing running.
  Next: authors answer Q-09…Q-16 (Q-12 and Q-14 change what is run). Fill the cluster lines in
  `tools/hpc/train_array.sbatch`, then launch tasks 1–30 per `docs/RUNBOOK.md`. Then the PID + GNN-QMIX plan.
- **2026-11-25 (2)** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) · aadit-mbp · branch `gap_address` ·
  Applied the author's answers to Q-09…Q-16 (D-027…D-031). (1) VDN and IQL read a fixed 3-nearest
  observation, train jointly at {3, 6, 9} and are evaluated zero-shot; QMIX stays full-observation and
  fixed-n. (2) `simulator/bluesky_marl/custom_envs.py`: `ConfigurableSectorCR` with 6 NM minimum spawn
  separation and sector-area scaling; **measured** minimum initial separation > 5 NM over 8 seeds each at
  n = 9 and n = 50. (3) Zero-shot evaluation runs every fleet size in both airspaces (`fixed`, `density`).
  (4) Budget is 2 000 000 steps per training fleet size; all step counts are per fleet size. (5) Paper
  updated for all of this; signed-off items marked. Run matrix is now 18 core tasks + 3 ablation tasks.
  Table 1 gained a VDN / IQL row that only restates the central hypothesis (authors to review).
  **Smoke runs only** (1500 steps per size, not results): tasks 1, 4, 7, 10, 19 ran through
  `tools/hpc/run_task.sh` (train, evaluate at n = 3…50 in both airspaces, aggregate). Logged GAP-28, GAP-29.
  Still open: Q-10 (r_detect = 20 NM is a proposed value), Q-01 framing/title, Q-02, Q-07, Q-08.
  Tests: T0 pass; T1 pass (`simulator/tests` 4, `marl/tests` 10, `algorithms/tests` 22); T2 pass
  (`random_rollout` both envs; `pytest` 36 passed).
  IDs: D-026…D-031, GAP-08, GAP-13, GAP-18, GAP-19, GAP-24, GAP-25, GAP-28, GAP-29, Q-09…Q-16, G03.1, G04.1,
  G06.1, G06.3.
  State: `gap_address`, committed and pushed to `origin` (see `git log`). Nothing running.
  Next: confirm Q-10; fill the cluster lines in `tools/hpc/train_array.sbatch` and launch tasks 1–18
  (`docs/RUNBOOK.md`). Then the PID + GNN-QMIX plan (G05.5–G05.9).
- **2026-11-25 (3)** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) · nova (ISU Nova HPC, first use) · branch `gap_address` ·
  Set the project up on the cluster and launched the run matrix. Clone at `/work/flemingc/akhan1/gnn_marl`
  (commit `5eb5f5f`), venv `.venv` with the D-005 pins and CPU-only `torch==2.9.0`. Filled the cluster lines of
  `tools/hpc/train_array.sbatch` (partition `nova`, account `flemingc`, 72 h, optional `RUNS_DIR` and extra
  training flags) and the "Cluster specifics" of `docs/HANDOFF.md`. No config, method or hyperparameter changed;
  `artifacts/registry.toml` has no entry, so nothing to reuse (S-23).
  Launched 18:47 CST, one Slurm array per method, results under
  `/work/flemingc/akhan1/gnn_marl_experiments/Nov25_<METHOD>/`: GNN-QMIX tasks 1–3 job `17768552`, VDN 4–6 job
  `17768553`, IQL 7–9 job `17768554`, QMIX 10–18 job `17768555`, ablation 19–21 job `17768571`. All 21 started;
  status table in `Nov25_STATUS.md` in that folder (refresh with `make_status.py`).
  **Measured** in the first 10 minutes (not results): 18 to 29 steps/s per fleet size for the joint runs, 48 to
  121 for QMIX, which is 2 to 3 times slower than the `aadit-mbp` estimate in the runbook; projected end of
  training Thu Nov 26 for most runs, about 02:00 Fri Nov 27 for the slowest joint runs, inside the 72 h limit.
  Observed, not investigated: (a) BlueSky prints a caught `ArgumentError: txt2hdg needs either a float`
  traceback now and then (tasks 8, 18) and training continues; (b) the QMIX training loss grew to 1e5–1e6 in
  all nine QMIX runs within 40 000 steps while the joint methods stay near 0.02. Neither has a GAP entry yet.
  Tests: T0 pass (`check_docs` 0 problems, before and after); T2 pass (`random_rollout`; `pytest` 36 passed);
  smoke task 1 through the Slurm script (job `17768517`, 1500 steps per size) ended with `wrote …/eval_zero_shot.csv`.
  IDs: G05.4, G06.3, G06.4, G03.1, D-005, D-030, GAP-26, S-22, S-23.
  State: `gap_address`, committed and pushed to `origin` (see `git log`); the jobs run the code of `5eb5f5f`.
  Running: Slurm jobs `17768552`, `17768553`, `17768554`, `17768555`, `17768571` on Nova, writing to
  `/work/flemingc/akhan1/gnn_marl_experiments/Nov25_*`; they are killed at about 18:47 on Sat Nov 28 if unfinished.
  Next: after the jobs end, rerun `make_status.py` and check that all 21 run directories have `final.pt` and
  `eval_zero_shot.csv`; resubmit any dead task id (no resume, GAP-26). Authors to look at the QMIX loss before
  its numbers are used. Then aggregate per RUNBOOK §6 over a folder of symlinks to the run directories, and
  publish the checkpoints (Q-07 still open).
- **2026-11-25 (4)** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) · aadit-mbp · branch `gap_address` ·
  INF-2 (part): added `paper/render.sh` (pdflatex → bibtex → pdflatex ×2; by-products go to
  `$TMPDIR/gnn-marl-paper-build`, only `paper/main.pdf` is written to the repo and it is gitignored; opens
  the PDF in a VS Code tab) and `.vscode/tasks.json` (default build task "Render paper", Cmd+Shift+B).
  README build instructions updated. This machine has BasicTeX 2026 without `latexmk`, so the script does
  not use it. **The paper itself was not compiled**: `new-aiaa.cls` is still missing, the script stops with
  a message saying so. The script was exercised on a toy `article` document only (success, undefined
  citation warning, and LaTeX error paths). Unknown until the class is present: whether BasicTeX has every
  package the class needs, and whether `new-aiaa.bst` ships with the template.
  Tests: T0 pass (before and after). T1/T2 not run (no Python code changed).
  IDs: INF-2.
  State: `gap_address`, uncommitted (committed with entry (5)). Nothing running.
  Next: copy `new-aiaa.cls` (and `.bst`) into `paper/`, run `paper/render.sh`, install any missing
  packages with `tlmgr`, then close INF-2. Otherwise unchanged from entry 2026-11-25 (3).
- **2026-11-25 (5)** · Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) · aadit-mbp · branch `gap_address` ·
  INF-2: the author added `new-aiaa.cls` and `new-aiaa.bst` to `paper/` (untracked). `paper/render.sh` then
  failed on `newtxtext.sty` not found: BasicTeX lacks packages the class loads. Installed into the user tree
  (`tlmgr --usermode install`, no sudo): abstract, preprint (authblk), enumitem, footmisc, lettrine, newtx,
  quoting, titlesec, xstring, kastrup, fontaxes, tex-gyre, txfonts. The paper now **compiles**: no LaTeX
  errors, no undefined citations or references. Remaining output, not fixed here (S-18): one overfull hbox
  (8.6 pt) in `03_background.tex`; BibTeX warnings for empty `pages` (10 entries) and empty `year`
  (`cleanmarl`, `gat`, `gcn`, `gnn_marl`, `iql`, `qmix-gnn`), left as is under S-15. No paper text was
  changed. README lists the package install line; the script's hint now points to it.
  Tests: T0 pass (before and after). T1/T2 not run (no Python code changed).
  IDs: INF-2, GAP-27 (the "not compiled" part is now met).
  The author approved committing `new-aiaa.cls` / `.bst` (publicly available template files) with the
  render script.
  State: `gap_address`, committed and pushed to `origin` (see `git log`). Nothing running.
  Next: fill the missing bibliography years from the publisher pages (S-15). Otherwise unchanged from entry 2026-11-25 (3) (Nova jobs still running).
