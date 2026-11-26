# Goals from §04 Problem Formulation

### G04.1: Configurable Markov-game environment (`SimConfig`)
- **Paper basis:** "a fleet of n cooperating UAS, each assigned a sequence of mission
  waypoints, operating in a shared airspace volume that also contains static obstacles and m
  non-cooperative intruders whose trajectories are not controlled and not known in advance."
- **Objective:** a `SimConfig` dataclass (`n_agents`, `n_intruders`, waypoint sequence spec,
  obstacle density, `r_detect_nm`, separation minimum, airspace size/scaling rule, step timing)
  and subclasses of `SectorCR` / `MergeEnv` in `simulator/bluesky_marl/custom_envs.py`.
  Never edit the submodule (S-06).
- **Interface:** `make_env(name, config: SimConfig)` → PettingZoo `ParallelEnv`.
- **Depends on:** INF-3. **Feeds:** nearly everything.
- **Acceptance:** PettingZoo `parallel_api_test` passes for each config. Each paper scenario
  axis (§06 setup) can be set from config alone.
- **Status:** in-progress (2026-11-25): `simulator/bluesky_marl/custom_envs.py::ConfigurableSectorCR`
  (spawn separation D-028, area scaling D-029) through `make_env(..., **env_kwargs)`. No `SimConfig`
  dataclass yet. **Gaps:** GAP-09, GAP-10, GAP-28.

### G04.2: Objective and reward specification
- **Paper basis:** "The learning objective combines three terms: drive the loss-of-separation
  count toward zero, maximize the mission-completion rate, and minimize path length and time."
  Plus §05: "a positive reward for reducing loss-of-separation events and a cost for each intervention."
- **Objective:** a written per-agent reward $r^i$ (LoS term, completion term, path/time term,
  intervention cost) and the team aggregation used by value decomposition. The terms are
  configurable weights, logged separately.
- **Depends on:** G04.1. **Feeds:** G05.8, all training.
- **Acceptance:** each reward term is logged separately in tensorboard. The definition is in the plan.
- **Status:** defined. **Gaps:** GAP-18 (what counts as mission success per env).

### G04.3: Decentralized broadcast channel with latency δ
- **Paper basis:** "When any cooperating vehicle detects an intruder at time t, that detection
  is relayed to every other fleet member, but arrives subject to a latency δ … broadcast
  rather than point-to-point … δ is treated as an experimental variable."
- **Objective:** a channel module: detection events → message $m^i_t$ delivered to all fleet
  members after `latency_s`; message content/format defined; δ=0 is the no-latency limit.
- **Interface:** per-step detections → per-agent `msg` tensor (fixed size, n-agnostic).
- **Depends on:** G04.1 (intruders, detection). **Feeds:** G04.4, G02.3, D2.
- **Acceptance:** unit test that a detection at t arrives at exactly t+δ (in sim time) for all agents.
- **Status:** defined. **Gaps:** GAP-07, GAP-09.

### G04.4: GRU belief state
- **Paper basis:** Eq. (1) $h^i_t=\mathrm{GRU}(h^i_{t-1},[o^i_t,m^i_t])$.
- **Objective:** per-agent shared-weight GRU over `[obs, msg]`; training with sequence replay
  (starting point: CleanMARL `qmix_lstm.py`).
- **Depends on:** G04.3. **Feeds:** G05.10.
- **Status:** defined.

### G04.5: Global (training-time) state
- **Paper basis:** the Markov game's global state space $\mathcal S$; CTDE uses it during training only.
- **Objective:** replace adapter state v0 (concatenated observations, D-007) with
  simulator-truth state and the graph. The state must also be n-agnostic for the proposed method.
- **Depends on:** G04.1. **Feeds:** G05.1, G05.3, G03.4.
- **Status:** in-progress (2026-11-25): adapter `state_mode="entities"`, an n × 7 entity matrix (D-026).
