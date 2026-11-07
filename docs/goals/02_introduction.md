# Goals from §02 Introduction

§02 defines what the project must **prove** (claims C1–C3), **build** (constructions M1–M4)
and **run** (experimental program). The build and run items are realized by goals in
§03–§06 (see the traceability table in [README.md](README.md)). The goals here are the
**theory deliverables** and the **claim-level acceptance** that ties everything together.

## Claims (verbatim intent)
- **C1:** "Fleet-size invariance follows from permutation equivariance of GNN message passing
  combined with a fixed-size global readout."
- **C2:** "Hybrid decomposition removes the reactive control layer from the learning problem
  … reducing the effective state–action complexity and improving sample efficiency."
- **C3:** "Coordination quality degrades gracefully with communication latency δ … relating
  it to the decay of mutual information … and derive a quantitative bound where the
  analysis admits one."

## Constructions
- **M1** GNN encoder → fixed $z\in\mathbb{R}^d$ before the centralized Q-network → G05.2, G05.3
- **M2** conditionally activated, sparse-intervention policy ($s^i<\theta$) → G05.6, G05.8, G03.4
- **M3** GNN spatial + GRU temporal integration → G04.4, G05.10
- **M4** single-level training, PID action as policy input → G05.5, G05.7

### G02.1 (T1): Formal statement and proof of C1
- **Objective:** a formal statement that $z$ is invariant to node permutation and has fixed
  dimension for every $n$, and that the downstream Q-network (as actually built in G05.3)
  is therefore defined for every $n$. Saying the *whole* $Q_{tot}$ is well-defined at any $n$
  depends on the G05.3 design. The proof must match the code.
- **Depends on:** G05.3 design being final.
- **Acceptance:** written proof (paper appendix) + property tests in code (G05.2) that check
  the stated invariances numerically.
- **Status:** defined. **Gaps:** GAP-03, GAP-20.

### G02.2 (T2): Formal argument for C2
- **Objective:** formalize "reduced effective state–action complexity" (e.g. the learner acts
  only on intervention steps; relate to $\mathbb{E}[I]\ll T$) and state what sample-efficiency
  result is predicted, so D3 can test it.
- **Depends on:** G05.6 (what exactly the learner controls), G05.7.
- **Acceptance:** statement + proof or explicit argument; a matching D3 experiment.
- **Status:** defined. **Gaps:** GAP-15.

### G02.3 (T3): Latency degradation characterization for C3
- **Objective:** a mutual-information-decay characterization of coordination quality vs δ,
  with a bound if one can be derived. Its prediction must be in the same units as D2's x-axis.
- **Depends on:** G04.3 (channel model), G04.4.
- **Acceptance:** derivation + overlay against measured D2 curve.
- **Status:** defined. **Gaps:** GAP-07.

### G02.4: Claim-level acceptance gate
- **Objective:** before the manuscript is finalized, each of C1–C3 and M1–M4 has (a) its
  theory or architecture text, (b) code at the goals listed above with status `done`, and (c)
  the evidence artifact in the traceability table, linked to run IDs (S-14).
- **Status:** defined.
