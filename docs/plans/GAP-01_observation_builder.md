# PLAN-GAP-01: Observation builder that sees all agents; training sizes n ∈ {3, 6, 9}

| field | value |
|---|---|
| goals | G04.1, G05.1 (input side), G06.1; closes GAP-01, fixes the adapter half of GAP-21 |
| paper basis | §02 experimental program, §05.1 parallel training protocol, §06.1 setup, Table 1 |
| status | `approved` (Aadit-Farhan Khan, 2026-11-17: "create our own observation builder and complete gap 01") |
| author(s) / agent | Aadit-Farhan Khan with Claude Code (Claude Fable 5.2) |
| gaps addressed | GAP-01 (resolved, D-017). GAP-21 (adapter fixed, D-016). Opens GAP-24 |
| decisions needed | none. Evaluation sizes stay {10, 20, 50} unless the authors change them |

## 1. Upstream inputs (what this plan consumes)
- Raw env observation `obs[agent][0:3]` = cos(drift), sin(drift), normalized airspeed. These three
  entries are present for every n in both `sector_cr` and `merge`.
- Simulator truth per agent: `bs.traf.lat`, `bs.traf.lon` (deg), `bs.traf.gseast`, `bs.traf.gsnorth` (m/s).

## 2. Downstream outputs (what this plan guarantees)
- `marl/interfaces/observation.py`: `ObsConfig`, `obs_dim(n_agents) -> int`,
  `relative_features(lat_deg, lon_deg, gs_east_mps, gs_north_mps, cfg) -> float32 (n, n-1, 7)`,
  `build_observations(own, rel) -> float32 (n, obs_dim(n))`.
- `BlueSkyInterface(..., obs_mode="all_agents" | "env", obs_config=None)`. Default `all_agents`.
  - `obs`: `float32 (n_agents, 3 + 7 * (n_agents - 1))` → 17, 38, 59 for n = 3, 6, 9.
  - `state` (v0, unchanged rule from D-007): concatenated obs, `n_agents * obs_dim`.
- Consumers: `qmix_bluesky.py`, `mappo_bluesky.py` (no code change needed), later G05.1.

## 3. Design
One agent's observation is `[own (3) | other_1 (7) | … | other_{n-1} (7)]`, others sorted nearest
first. Each block: dx_east, dy_north, dvx_east, dvy_north, cos ψ, sin ψ, distance, where ψ is the
direction of the relative velocity. Positions use a local flat-earth frame (x east, y north);
velocities are ground-speed components in the same frame. This differs from the raw env on
purpose: it has no 3-neighbor cap, one frame for positions and velocities, and one set of
normalization scales for both envs (GAP-22).

## 4. Configuration
`ObsConfig(pos_scale_m=50_000.0, vel_scale_mps=150.0)`. Training fleet sizes: n ∈ {3, 6, 9} (D-017).

## 5. Acceptance tests and evidence
`marl/tests/test_observation.py`: geometry, ordering and antisymmetry unit tests; for each env
and each n ∈ {3, 6, 9}, a subprocess records declared and actual obs width and asserts they are
equal, equal to `obs_dim(n)`, and linear in n. QMIX and MAPPO smoke runs at n = 3 and n = 9.

## 6. Risks and alternatives
- The obs width now depends on n, so every network that reads `obs` directly is n-specific,
  and the concatenated state grows as n·(7n − 4) (GAP-24). The proposed method must read the
  graph (G05.1), not this flat vector.
- Alternative kept available: `obs_mode="env"` (fixed 24 wide for n ≥ 4, broken for n ≤ 3).

## 7. Out of scope
Non-cooperative intruders and obstacles in the observation (GAP-09). Graph tensors (G05.1).
Simulator-truth global state (G04.5). Fixing the raw env (third-party, S-06).
