# Graph-Encoded MARL for Zero-Shot Scalable UAS Conflict Resolution

Research code and paper for **"Graph-Encoded Multi-Agent Reinforcement Learning for Zero-Shot
Scalable UAS Conflict Resolution"**: Aadit-Farhan Khan\*, Adam Haroon\*, Cody Fleming
(Iowa State University, Dept. of Mechanical Engineering; \*equal contribution). AIAA format.

> **Status: Stage 1 (Framework).** The repository structure, paper → goal breakdown and
> project standards are in place. Method code isn't implemented yet. Next come gap
> analysis and planning. See [docs/PROJECT_RECORD.md](docs/PROJECT_RECORD.md).

## The problem

Centralized-training/decentralized-execution (CTDE) MARL methods such as QMIX and MAPPO
have a mixer or critic whose input size is fixed by the number of agents *n*. A policy
trained for one fleet size can't run at another without retraining. In Advanced Air Mobility
(AAM), the fleet size changes all the time: vehicles rotate out for battery swaps, join as
demand grows, or drop out.

## The proposed method (paper §05)

1. **GNN encoder + fixed readout.** The fleet is a graph *G=(V,E)* (nodes = vehicles; edges
   within the detection radius *r_detect*, with distance, relative heading, relative velocity
   and time-to-CPA). Permutation-equivariant message passing followed by a
   sum‖max‖mean readout gives a latent *z ∈ ℝ^d* whose size doesn't depend on *n*. The
   centralized Q-network sees only *z*, so it is decoupled from fleet size.
2. **Hybrid control.** A fixed PID controller tracks waypoints. A learned conflict-resolution
   policy overrides it only when a learned safety score *sⁱ* falls below a threshold *θ*
   (grounded in a control-barrier-function condition), with a cost per intervention.
3. **Broadcast channel with latency δ.** Intruder detections reach the fleet after δ.
   Each vehicle integrates messages through a GRU whose hidden state becomes a node feature.
4. **Mixed-n training.** Trained jointly at several fleet sizes (n∈{3,6,9}) and evaluated
   zero-shot at n∈{10,20,50} against QMIX, MAPPO, GCBF and a PID-only ablation.

Three theoretical claims accompany the method: fleet-size invariance (C1), the
sample-efficiency gain from hybrid decomposition (C2), and graceful degradation with latency (C3).
[docs/goals/02_introduction.md](docs/goals/02_introduction.md) maps each claim to the code and the evidence.

## Stack

| component | source | pinned |
|---|---|---|
| Simulator | [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) ATC simulator via the multi-agent PettingZoo fork of BlueSky-Gym, [`jangroter/bluesky-gym`](https://github.com/jangroter/bluesky-gym) branch `petting_zoo_sac` (envs `sector_cr`, `merge`) | `3a536c0` (submodule) |
| MARL algorithms | [CleanMARL](https://github.com/AmineAndam04/cleanmarl) single-file QMIX/VDN/MAPPO/… | `5cf837e` (submodule) |
| Env interface | `marlbench` `MARLEnv` | 0.0.5 |
| Python / DL | Python 3.12, numpy 2.3.4, torch 2.9.0 | see `pyproject.toml` |

## Repository layout

```
gnn-marl-bluesky/
├── README.md                  this file
├── CLAUDE.md, AGENTS.md       rules for coding agents (point to the project record)
├── PLAN.md                    original bootstrap plan (phases 1–6)
├── docs/
│   ├── PROJECT_RECORD.md      ★ status, binding standards, decisions, gaps, open questions, work log
│   ├── goals/                 programming goals derived from paper/, one file per paper section
│   ├── plans/                 detailed per-goal plans (Stage 3); _TEMPLATE.md
│   ├── HANDOFF.md             carrying state between machines (laptop / lab / HPC)
│   └── ARTIFACTS.md           publishing and pulling trained weights
├── artifacts/registry.toml    index of shared, immutable run artifacts (bytes live in the store)
├── tools/check_docs.py        Tier-0 check: IDs and links in the docs resolve
├── paper/                     LaTeX (AIAA), main.tex + sections/ (one file per section) + references.bib
├── simulator/
│   ├── bluesky_marl/          make_env(name, n_agents): env registry, singleton guard
│   └── third_party/bluesky-gym/   git submodule (read-only)
├── marl/                      the MARL ↔ simulator layer
│   ├── interfaces/bluesky_interface.py   marlbench adapter (discrete actions, team reward, state)
│   ├── scripts/random_rollout.py         random-action multi-agent smoke test
│   └── tests/
├── algorithms/
│   ├── qmix_bluesky.py        CleanMARL QMIX + BlueSky env branch (fixed-n baseline)
│   └── third_party/cleanmarl/ git submodule (read-only)
├── configs/                   run / scenario configs (filled in Stage 3–4)
└── runs/                      (gitignored) tensorboard logs, checkpoints
```

## Setup

```bash
git clone --recurse-submodules <repo-url> && cd gnn-marl-bluesky
# or, in an existing clone:  git submodule update --init

uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install "numpy==2.3.4" "gymnasium==1.2.1" "pettingzoo==1.25.0" "marlbench==0.0.5" \
               bluesky-simulator pygame stable-baselines3 \
               "torch==2.9.0" "tyro==0.9.35" tensorboard pytest
uv pip install -e .
```

Don't `pip install` the bluesky-gym fork or CleanMARL's full `pyproject` (conflicting pins and
unneeded jax/flax). The first BlueSky import builds a navdata cache in `~/bluesky/`. The warnings
"RTree could not be loaded" and "Failed to load BADA performance model" are expected.

## Smoke tests (from PLAN.md, verified by the plan author)

```bash
python -m marl.scripts.random_rollout --env sector_cr --n-agents 5 --steps 100
python -m marl.scripts.random_rollout --env merge     --n-agents 5 --steps 100
pytest
python algorithms/qmix_bluesky.py --env_type bluesky --env_name sector_cr --n_agents 5 \
  --max_episode_steps 100 --total_timesteps 3000 --eval_steps 10 --num_eval_ep 2 \
  --log_every 5 --work_dir runs/qmix_smoke
```

**Critical:** BlueSky is a process-wide singleton. Only **one env per Python process**. Use
`marlbench.vec_envs.SubprocVectorEnv` for parallel or eval envs.

## Building the paper

Put `new-aiaa.cls` and `new-aiaa.bst` (from the Overleaf "AIAA Conference and Journal Template") in
`paper/`, then run `paper/render.sh` (in VS Code: Cmd+Shift+B, task "Render paper"). It writes
`paper/main.pdf` and opens it in a VS Code tab; all other LaTeX output goes to a scratch directory
outside the repo. Needs only `pdflatex` and `bibtex` (BasicTeX is enough, no `latexmk`).

On BasicTeX the AIAA class needs extra packages. Install them once per machine (no `sudo`; they go to
`~/Library/texmf`; run `tlmgr init-usertree` first if that tree does not exist):
`tlmgr --usermode install abstract preprint enumitem footmisc lettrine newtx quoting titlesec xstring kastrup fontaxes tex-gyre txfonts`

## Contributing (humans and agents)

1. Read [docs/PROJECT_RECORD.md](docs/PROJECT_RECORD.md). Its standards (S-01…S-23) are binding.
2. Pick a goal ID from [docs/goals/](docs/goals/README.md). Paper goals need an approved plan in
   `docs/plans/` before implementation.
3. One issue per change, stay in its area, run the test tiers, and commit or push only after
   confirmation (S-18…S-21). Use the PR checklist. Update the record's work log in the same PR.
4. Working on several machines (laptop / lab / HPC)? Follow [docs/HANDOFF.md](docs/HANDOFF.md).
   Sharing or pulling trained weights? Follow [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Acknowledgments

BlueSky (TU Delft CNS/ATM), BlueSky-Gym (Groot et al.) and its PettingZoo fork
(`jangroter/bluesky-gym`), and CleanMARL (A. Andam). Citations are in `paper/references.bib`. Third-party code keeps its own licenses inside `*/third_party/`.
