# Goals from §03 Background and Related Work

§03 fixes which prior methods are **baselines** and gives GCBF a second role: it **grounds
the intervention threshold θ**.

### G03.1: QMIX baseline (fixed-n mixer, unmodified)
- **Paper basis:** "The QMIX mixing network … take a joint input whose dimensionality is fixed
  by the training team size, so changing n requires architectural modification and retraining."
- **Objective:** a QMIX baseline on the BlueSky env with CleanMARL's `MixingNetwork(n_agents,
  s_dim)` kept as is (D-008), plus a recorded demonstration that an n=5 checkpoint
  **cannot be evaluated** at n=10 (mixer input mismatch). That demonstration is the Table 1
  "Not transferable" row.
- **Interface:** `algorithms/qmix_bluesky.py` (exists) → checkpoints + tensorboard in `runs/`.
- **Depends on:** INF-5. **Feeds:** G06.4, G06.9.
- **Acceptance:** curves for 3 seeds at n=5 with team return, intrusions/episode and drift
  logged; a logged failure on n=10 load.
- **Status:** defined (smoke run exists per PLAN.md §5). **Gaps:** GAP-13.

### G03.2: MAPPO baseline
- **Paper basis:** "MAPPO centralized critic take[s] a joint input whose dimensionality is fixed."
- **Objective:** copy CleanMARL `mappo.py` → `algorithms/mappo_bluesky.py` with the same
  `bluesky` env branch as QMIX. Centralized critic kept fixed-n.
- **Depends on:** INF-5. **Feeds:** G06.4.
- **Status:** in-progress. Plan `docs/plans/G03.2_mappo_baseline.md` approved 2026-11-09;
  `algorithms/mappo_bluesky.py` smoke-tested on both envs. **Gaps:** GAP-12 (continuous vs discrete), GAP-13.

### G03.3: GCBF baseline
- **Paper basis:** "Neural Graph Control Barrier Functions provide distributed collision
  avoidance with a formal safety certificate … GCBF serves … as a baseline for empirical comparison."
- **Objective:** run GCBF (external implementation; not in CleanMARL) on the same scenarios
  and metrics. This needs an adapter from GCBF's dynamics/action interface to BlueSky's.
- **Depends on:** G04.1, G06.2. **Feeds:** G06.4.
- **Status:** defined. **Gaps:** GAP-06 (dynamics model mismatch), GAP-16 (citation).

### G03.4: CBF-grounded intervention threshold θ
- **Paper basis:** "the CBF condition $\dot h^{ij}+\alpha(h^{ij})\ge 0$ … provides a
  computable, certificate-backed criterion … The learned safety score $s^i$ is trained to
  approximate this barrier condition, and θ is calibrated such that $s^i<\theta$ coincides
  with the barrier condition being at risk of violation."
- **Objective:** (1) define $h^{ij}$ for BlueSky geometry (e.g. a distance-to-separation-minimum
  barrier), (2) compute the CBF condition per vehicle pair from simulator truth as a training
  label, (3) define the calibration procedure for θ.
- **Interface:** sim state → per-vehicle CBF label / margin → used by G05.6 loss and θ calibration.
- **Depends on:** G04.1, G04.5. **Feeds:** G05.6.
- **Status:** defined. **Gaps:** GAP-04, GAP-05.
