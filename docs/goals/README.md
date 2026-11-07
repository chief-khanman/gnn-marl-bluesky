# Programming Goals: derived from `paper/`

The paper is the specification. Every programming goal in this folder comes from a specific
statement in `paper/sections/*.tex`, and each goal file mirrors one paper section file:

| paper section file | goal file | what it yields |
|---|---|---|
| `00_abstract.tex`, `07_conclusion.tex`, `08_acknowledgments.tex` | [00_framing.md](00_framing.md) | scope, stated limitations, AI-use disclosure |
| `01_nomenclature.tex` | [01_nomenclature.md](01_nomenclature.md) | the symbol ↔ code naming contract |
| `02_introduction.tex` | [02_introduction.md](02_introduction.md) | claims C1–C3, constructions M1–M4, theory deliverables T1–T3, traceability |
| `03_background.tex` | [03_background.md](03_background.md) | baselines (QMIX, MAPPO, GCBF) and the CBF-based threshold |
| `04_problem_formulation.tex` | [04_problem_formulation.md](04_problem_formulation.md) | the Markov game, objective, broadcast channel, GRU belief |
| `05_methodology.tex` | [05_methodology.md](05_methodology.md) | graph, GNN encoder, n-agnostic Q-network, mixed-n training, PID + switching |
| `06_experiments.tex` | [06_experiments.md](06_experiments.md) | scenarios, metrics, evaluation harness, deliverables D1–D4, Table 1 |

Engineering infrastructure that the paper does not ask for directly (repo, env adapter, smoke
tests) is tracked as `INF-*` in `docs/PROJECT_RECORD.md` §3.

## ID scheme

- `Gss.k`: programming goal *k* from paper section *ss* (e.g. `G05.3`).
- `C1–C3`: theoretical claims (§02). `M1–M4`: methodological constructions (§02).
- `T1–T3`: proof deliverables for C1–C3. `D1–D4`: planned evaluation deliverables (§06).
- `GAP-xx`, `Q-xx`, `D-xxx`, `S-xx`: gaps, open questions, decisions, standards (all in `PROJECT_RECORD.md`).

IDs are permanent. Don't renumber them. A retired goal is marked `retired`, not deleted.

## Goal entry format (required)

Every goal has: **Paper basis** (section and equation, with a short quote) · **Objective** ·
**Interface** (inputs → outputs) · **Depends on / Feeds** · **Acceptance (evidence)** ·
**Planned location** · **Status** · **Gaps**.

Status values: `defined` → `planned` (approved plan in `docs/plans/`) → `in-progress` →
`done` (acceptance evidence linked) | `blocked` (names the Q/GAP that blocks it) | `retired`.

## Traceability: paper claim → goals → evidence

| paper item | realized by | evidence that will support it in the paper |
|---|---|---|
| **C1** fleet-size invariance (equivariant GNN + fixed readout) | G05.1, G05.2, G05.3, G05.4, T1 | equivariance/invariance property tests; D1 scaling curve; Table 1 |
| **C2** hybrid decomposition improves sample efficiency | G05.5, G05.6, G05.7, G05.8, T2 | D3 sample-efficiency comparison vs end-to-end MARL; intervention count E[I] ≪ T |
| **C3** graceful degradation with latency δ | G04.3, G04.4, G05.10, T3 | D2 success-vs-δ curve vs theoretical characterization |
| **M1** GNN encoder before centralized Q-network | G05.2, G05.3 | architecture section and code; C1 evidence |
| **M2** conditional, sparse-intervention policy (s^i < θ) | G05.6, G03.4, G05.8 | intervention frequency metric; handoff analysis G05.9 |
| **M3** GNN spatial + GRU temporal integration | G04.4, G05.10 | D2; latency ablation |
| **M4** single-level training with PID action as input | G05.5, G05.7 | D3 "PID-input" ablation |
| Exp. program: baselines | G03.1, G03.2, G03.3, G06.4 | Table 1, D1 |
| Exp. program: zero-shot n=5 → {10,20,50} | G05.4, G06.1, G06.3, G06.5 | D1, Table 1 |
| Exp. program: emergent coordination | G06.2, G06.8 | D4 |

## Dependency order (high level)

```
INF (repo, env, adapter, QMIX smoke)                    ← PLAN.md phases 1–5
  └─ G04.1 env/SimConfig ─┬─ G04.2 reward ─┐
                          ├─ G04.5 state ──┤
                          └─ G05.5 PID ────┼─ G05.1 graph ─ G05.2 encoder ─ G05.3 n-agnostic Q ─ G05.4 mixed-n training
G04.3 broadcast ─ G04.4 GRU ─ G05.10 ──────┘                                         │
G03.4 CBF threshold ─ G05.6 switching ─ G05.7/G05.8 ─────────────────────────────────┤
G06.2 metrics ─ G06.3 eval harness ──────────────────────────────────────────────────┴─ G06.4–G06.9 deliverables
```

The detailed plans (`docs/plans/`) must agree with this order. A plan that changes it must
update this diagram in the same PR.
