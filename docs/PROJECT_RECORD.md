# Project Record: status, standards, decisions, gaps

> **Single source of truth** for everyone working on this project, human or agent.
> **Read it before you start. Update it before you finish.** (S-01, S-02)
>
> Last updated: 2026-11-07 · Current phase: **Stage 1: Framework** (see §2)

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
| finish a session / open a PR | — | add a §8 work-log entry (who, which agent, what, IDs) |

Related files: [README.md](../README.md) (overview) · [docs/goals/](goals/README.md) (paper → goals) ·
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
| INF-2 | paper build (`new-aiaa.cls` + `latexmk`) | todo | class file not in repo. `latexmk` not installed on the dev machine |
| INF-3 | `simulator/tests/test_api.py` (`parallel_api_test` in a subprocess) | todo | |
| INF-4 | random-rollout sweeps n∈{3,5,10,20}; SubprocVectorEnv check | todo | |
| INF-5 | QMIX smoke run, simple_spread sanity run | todo | needs the Python 3.12 venv |
| INF-6 | Python 3.12 venv per PLAN.md §1.3 | todo | dev machine has no 3.12 or `uv` yet (conda env is 3.11) |

### Paper goals (see `docs/goals/`)
All `G*`, `T*` and `D*` goals are **defined** (Stage 1), except: G01.1 **done** (naming
contract), G00.3 **in-progress** (logging convention active), G06.10 **blocked** (Q-04).
G05.3 is the critical-path item and is blocked by GAP-03 until a design is approved.

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
  is `runs/<goal-id>_<algo>_<env>_n<train-n>_s<seed>_<YYYYMMDD-HHMM>/` and contains the resolved
  config, the repo commit hash and both submodule hashes. Minimum **3 seeds** for any reported
  result; the seed count is always stated alongside the result.
- **S-11 Metrics have one definition.** Paper metrics are implemented once (G06.2) and
  referenced by canonical name. Tensorboard tags keep CleanMARL's prefixes (`rollout/`,
  `eval/`, `train/`) followed by the canonical metric name.
- **S-12 Tests.** BlueSky tests run one per process (subprocess or `pytest --forked`). Run
  `python -m marl.scripts.random_rollout` and `pytest` before every commit. Equivariance and
  variable-n property tests are mandatory for G05.2/G05.3. Tests must not require a display.

### Git
- **S-13 Commits and branches.** Conventional commits `<type>(<scope>): <summary>` with type
  ∈ {feat, fix, docs, chore, refactor, test, exp} and scope = top-level dir or `paper`. Reference
  goal IDs in the body. Branches: `<type>/<goal-id>-<slug>` (e.g. `feat/G05.2-encoder`). One
  goal per PR where possible. Never commit `runs/` or large artifacts.

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

---

## 6. Gap register

Seeded during Stage 1 by reading the paper against the simulator and PLAN.md. **Entries
are observations, not yet analyzed** (Stage 2 does that). Status: `open` → `resolved` (D-xxx)
| `accepted-risk` | `deferred` | `escalated` (Q-xx).

Severity: **H** blocks a core claim or goal · **M** affects an experiment or its fairness · **L** wording/consistency.

| ID | sev | observation | affects | status |
|---|---|---|---|---|
| GAP-01 | M | Training fleet sizes are inconsistent: §02 and Table 1 say "training at n=5"; §05 says n∈{2,4,8,16,…,n_max}; §06 says n∈{3,5,8}. | G05.4, G06.1, G06.9 | open |
| GAP-02 | H | Execution model is unclear: §05 says that at activation the policy builds G and "queries the centralized Q-network to select the optimal joint corrective action". That is centralized execution, which conflicts with the CTDE framing and the decentralized broadcast channel (in QMIX the mixer is used only in training). | G05.3, G04.5, C1 | open |
| GAP-03 | H | A centralized Q-network on a fixed z can't by itself output per-agent actions for variable n. The paper doesn't say how per-agent utilities, z and $Q_{tot}$ combine, or how monotonicity/IGM is kept. | G05.3, T1 | open (blocking) |
| GAP-04 | M | Two activation triggers: "when any other agent enters $r_{\text{detect}}$" vs the per-vehicle rule $s^i<\theta$ (Eq. 4). Which governs, and is activation per-vehicle or fleet-wide? | G05.6 | open |
| GAP-05 | M | Training of $s^i$ is under-specified: §03 says it approximates the CBF condition (needs a supervised/auxiliary loss and a defined $h^{ij}$); §05 gives no loss. Also, $h^{ij}$ (CBF) collides with $h^i_t$ (GRU) in notation. | G03.4, G05.6, G01.1 | open |
| GAP-06 | H | Dynamics mismatch: the paper's PID outputs thrust/pitch/roll/yaw rate (6-DOF UAS). BlueSky is a point-mass airliner model with HDG/SPD/ALT commands (A320, FL350, 5 NM separation, 150 kt). | G05.5, G03.3, G04.1, G00.2 | open → Q-01 |
| GAP-07 | M | Latency scale: δ∈{0,0.1,0.5,1.0} s, but a `sector_cr` control step is `ACTION_FREQUENCY`=5 × `DT`=1 s = **5 s** of sim time. All proposed δ fall inside one step unless step timing changes. | G04.3, G06.6, T3 | open |
| GAP-08 | L | Node features include "received reward" and "current action". Reward isn't available at decentralized execution time, and current action is circular at decision time. | G05.1 | open |
| GAP-09 | M | Static obstacles and non-cooperative intruders (m) don't exist in the BlueSky envs. They must be added (e.g. `bs.traf.cre` aircraft not in `env.agents`). | G04.1, G04.3 | open |
| GAP-10 | M | The motivation (and the §06 custom 3D env) relies on vehicles entering and leaving mid-episode. The BlueSky envs have fixed n per episode and all-or-nothing termination (PLAN.md fact 7), and there's no per-agent death masking. | G04.1, G05.4, G06.10 | open |
| GAP-11 | M | Validation environments: the paper names a 2D navigation benchmark + a custom 3D 6-DOF env. The implementation uses BlueSky. Replace or add? (`% TODO(authors)` in `06_experiments.tex`) | G06.10, G00.2 | open → Q-04 |
| GAP-12 | M | Action design: 15 discrete absolute actions (D-007) vs "corrective commands relative to the PID baseline". MAPPO/GCBF are naturally continuous. | G05.7, G03.2 | open → Q-03 |
| GAP-13 | H | Baseline fairness: VDN/IQL and parameter-shared IPPO/MAPPO-with-local-critic *are* n-transferable. The "QMIX/MAPPO not transferable" row may be challenged, and InforMARL/GraphMIX/QMIX-GNN are discussed but not compared against. | G03.1, G03.2, G06.4, G06.9 | open |
| GAP-14 | M | Emergence metrics (heading-direction entropy, trajectory MI, deadlock) have no operational definitions yet. | G06.2, G06.8 | open |
| GAP-15 | M | C2 needs a defined "end-to-end MARL" comparator and a rule for transitions where no agent is overridden (do they train the Q-net?). | G05.11, G06.7, T2 | open |
| GAP-16 | L | Bibliography TODO-VERIFY: `qmix-gnn`, `gnn_marl` (placeholders), GCBF author list, GraphMIX year/venue. | paper | open (authors) |
| GAP-17 | L | Missing nomenclature symbols: $L$ (GNN layers), $h^{ij}$, $\alpha$, $d$ readout composition. | G01.1 | open |
| GAP-18 | M | "Mission success" per env: `merge` ends for everyone when any aircraft finishes; `sector_cr` truncates when all have left. Per-agent success needs a definition. | G04.2, G06.2 | open → Q-02 |
| GAP-19 | M | Scaling rule for zero-shot tests: does airspace size grow with n (constant density) or stay fixed (rising density)? This changes what "generalization" means. | G06.1, G04.1, D1 | open |
| GAP-20 | M | Sum pooling in the readout scales with n, so z's distribution shifts at unseen n. That may hurt zero-shot even though the dimension is fixed (C1 is about dimension/invariance, not distribution). | G05.2, T1, D1 | open |

---

## 7. Open questions for the authors

Agents and code contributors **don't decide these**. Record the answer as a D- entry when given.

| ID | question | from | status |
|---|---|---|---|
| Q-01 | UAS vs airliners: keep the "UAS" framing with scaled separation/speeds in BlueSky, or reframe as ATM/AAM? What dynamics level does the PID act on? | PLAN.md §6; GAP-06 | open |
| Q-02 | Primary env: `sector_cr` (individual goals), `merge` (shared goal), or both? | PLAN.md §6; GAP-18 | open |
| Q-03 | Action discretization (15 now) vs a continuous method (FACMAC / MAPPO-continuous); absolute vs residual-on-PID actions | PLAN.md §6; GAP-12 | open |
| Q-04 | Does BlueSky replace the "2D navigation benchmark", or come in addition? What happens to the custom 3D env? | PLAN.md §6; GAP-11 | open |
| Q-05 | Which training fleet sizes are canonical: n=5, {3,5,8}, or {2,4,8,16,…}? | GAP-01 | open |
| Q-06 | Is execution meant to be decentralized (CTDE) or centralized-on-activation? | GAP-02 | open |

---

## 8. Work log

Append-only, newest last. Format: date · who / agent · what · IDs touched.

- **2026-11-07** · Aadit-Farhan Khan with Claude Code (Claude Opus 5.5) · Stage 1 started.
  Initialized git repo (`main`). Added submodules `simulator/third_party/bluesky-gym` @
  `3a536c0` and `algorithms/third_party/cleanmarl` @ `5cf837e`. Created `.gitignore`,
  `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, `README.md`, this record, `docs/goals/*`
  (paper → goals), `docs/plans/_TEMPLATE.md`, `configs/README.md`, PR template. Seeded GAP-01…20
  and Q-05/Q-06. Not done: no venv (dev machine lacks Python 3.12/`uv`), so `random_rollout` and
  `pytest` were **not run** before the initial commit (S-12 exception for the scaffold commit
  only). Paper not built (no `new-aiaa.cls` / `latexmk`).
  IDs: INF-1, D-001…D-011, G01.1, G00.3, GAP-01…20, Q-05, Q-06.
