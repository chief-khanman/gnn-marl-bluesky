# Goals from §06 Experimental Design and Expected Results

### G06.1: Scenario and configuration matrix
- **Paper basis:** "fleet sizes n∈{5,10,20,50}, m non-cooperative intruders, static obstacles
  of varying density, and communication latencies δ∈{0,0.1,0.5,1.0} s. … trained jointly across
  n∈{3,5,8} … evaluated zero-shot at n∈{10,20,50}. Loss-of-separation events are deliberately
  induced during training … a separate, more realistic scenario is used to evaluate the handoff."
- **Objective:** versioned config files in `configs/` for every cell of the matrix, including
  (a) a *conflict-inducing training* scenario generator and (b) a *realistic handoff* scenario.
- **Depends on:** G04.1. **Feeds:** G06.3–G06.9.
- **Acceptance:** each config resolves to a runnable env. The matrix is listed in a single index file.
- **Status:** defined. **Gaps:** GAP-01, GAP-07, GAP-09, GAP-19.

### G06.2: Metrics library
- **Paper basis:** "mission success rate, loss-of-separation rate, path efficiency,
  intervention frequency, and emergence measures: heading-direction entropy, trajectory mutual
  information, and deadlock count." Plus handoff discontinuity (G05.9).
- **Objective:** `marl/metrics.py`: one function per metric, computed from logged
  trajectories, with a written operational definition. Canonical names per S-11.
- **Acceptance:** unit tests on synthetic trajectories with known answers.
- **Status:** defined. **Gaps:** GAP-14, GAP-18.

### G06.3: Evaluation harness (zero-shot across n)
- **Objective:** load any checkpoint, evaluate it on a list of (n, m, δ, scenario) configs in
  subprocess envs (S-07), with N episodes × seeds, and write results in a standard
  table format with run provenance (S-10). Includes the "cannot evaluate" outcome for fixed-n baselines.
- **Depends on:** G06.1, G06.2. **Feeds:** G06.5–G06.9.
- **Status:** defined.

### G06.4: Baseline and ablation runs
- **Paper basis:** "Baselines are QMIX, MAPPO, GCBF, and a local-only ablation in which the
  conflict-resolution policy is removed and the PID controller operates alone."
- **Objective:** training and evaluation for G03.1, G03.2, G03.3 and PID-only (G05.5) on the G06.1 matrix.
- **Status:** defined. **Gaps:** GAP-13.

### G06.5 (D1): Scaling curve and fitted degradation exponent
- **Paper basis:** "(i) a quantitative scaling curve of mission success rate against fleet
  size n, with the empirically fitted degradation exponent for the proposed method and for the CTDE baselines."
- **Objective:** success vs n plot (train n marked), fit procedure stated in advance
  (functional form, CI method).
- **Depends on:** G05.4, G06.3, G06.4. **Status:** defined.

### G06.6 (D2): Latency sensitivity
- **Paper basis:** "(ii) the effect of communication latency δ on success rate, compared
  against the theoretical degradation characterization."
- **Depends on:** G04.3, G05.10, G02.3. **Status:** defined. **Gaps:** GAP-07.

### G06.7 (D3): Ablations and sample efficiency
- **Paper basis:** "(iii) ablations isolating the conflict-resolution policy and the PID input,
  including a comparison of sample efficiency against end-to-end MARL."
- **Objective:** variants: no CR policy (PID-only), no PID input, end-to-end MARL (no PID
  layer). Learning curves vs env steps on equal budgets.
- **Depends on:** G05.7, G05.11, G02.2. **Status:** defined. **Gaps:** GAP-15.

### G06.8 (D4): Emergence analysis
- **Paper basis:** "(iv) … lane formation through heading-direction entropy, leader–follower
  structure through trajectory mutual information, and deadlock frequency relative to baselines."
- **Depends on:** G06.2, G06.3. **Status:** defined. **Gaps:** GAP-14.

### G06.9: Measured Table 1
- **Paper basis:** Table 1 (hypothesized zero-shot success after training at n=5).
- **Objective:** the same table filled with measured values linked to run IDs. Until then the
  paper's table stays labeled as hypotheses (S-14).
- **Depends on:** G06.5, G06.4. **Status:** defined. **Gaps:** GAP-01, GAP-13.

### G06.10: Validation environments
- **Paper basis:** "Validation is conducted in two environments … a standard 2D multi-agent
  navigation benchmark … a custom 3D UAV simulation environment … six-degree-of-freedom
  kinematics … stochastic agent entry and exit events." The `% TODO(authors)` note says BlueSky is now used.
- **Objective:** an author decision on the role of BlueSky (replace or add), then adapters for
  every environment the paper keeps.
- **Status:** blocked on Q-04. **Gaps:** GAP-10, GAP-11.

### G06.11: Paper section `06a_simulation_environment.tex`
- **Paper basis:** PLAN.md §2.4 (an implementation-side addition, not an author claim).
- **Objective:** a factual description of the BlueSky setup (version pins, dynamics, step timing,
  separation, scenario generation) once G04.1 is done. No claim changes.
- **Depends on:** G04.1, INF-4. **Status:** defined.
