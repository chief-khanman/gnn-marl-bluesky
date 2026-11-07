# Goals from §05 Methodology

This is the paper's core contribution. G05.3 in particular must be specified precisely
before any code is written (PLAN.md §6.2).

## A. GNN encoder for an agent-count-agnostic Q-network (§05.1)

### G05.1: Fleet interaction graph builder
- **Paper basis:** "Each node $v^i$ carries the observation $o^i_t$, current action, received
  reward, velocity, and heading … Edges … between vehicles within $r_{\text{detect}}$ … carry
  … Euclidean distance, relative heading angle, relative velocity, and time-to-closest-point-of-approach."
- **Objective:** `marl/graph.py`: from simulator truth (`bs.traf`) or observations, build node
  features (+ `a_pid` per G05.7, + `belief` per G05.10) and radius-based edges with the four
  edge features. Output padded tensors + `node_mask` (and/or `torch_geometric.data.Data`). Exposed
  through `get_graph()` on the adapter.
- **Depends on:** G04.1, G04.5, G05.5. **Feeds:** G05.2.
- **Acceptance:** unit tests on hand-built geometries (TCPA, relative heading wrap-around,
  edge count vs radius). Works for n ∈ {2…50}.
- **Status:** defined. **Gaps:** GAP-08 (reward as a node feature at execution time).

### G05.2: Equivariant GNN encoder + fixed-size readout → z
- **Paper basis:** Eq. (2) $z=\mathrm{Readout}(\mathrm{GNN}(G))$, $d$ independent of $n$;
  Eq. (3) $\mathrm{GNN}(P\cdot G)=P\cdot\mathrm{GNN}(G)$; readout = sum ‖ max ‖ mean.
- **Objective:** `algorithms/qmix_gnn/encoder.py`: L edge-aware message-passing layers;
  readout concatenating sum, max and mean pooling over masked nodes → `z ∈ R^latent_dim`.
  It must also expose per-node embeddings (needed by G05.3/G05.6).
- **Acceptance:** property tests: (i) node embeddings permute with P, (ii) z unchanged under
  P, (iii) z shape identical for n ∈ {2, 5, 10, 50}, (iv) padding/masking doesn't change z.
- **Depends on:** G05.1. **Feeds:** G05.3, G05.6, G02.1.
- **Status:** defined. **Gaps:** GAP-20 (sum pooling scales with n).

### G05.3: n-agnostic centralized Q-network / mixer conditioned on z ★ core
- **Paper basis:** "The latent vector z is passed directly into the centralized Q-network …
  constant dimension d regardless of how many vehicles are active. … trained jointly end-to-end."
  Contrast: "$Q_{tot}=f(Q_1,\dots,Q_n;s)$ has n fixed input slots."
- **Objective:** a precise, written design (before code) for how per-agent utilities and z
  combine into $Q_{tot}$ for **any n**, so that (a) per-agent greedy actions remain
  decentralized-executable or the execution model is stated explicitly, (b) the IGM/monotonicity
  property holds if value decomposition is kept, (c) no layer has n-sized input. Candidate
  form from PLAN.md: $Q_{tot}=\sum_i w_i(z,h_i)\,Q_i + b(z)$ with $w_i\ge 0$.
- **Depends on:** G05.2. **Feeds:** G05.4, G05.6, G02.1.
- **Acceptance:** design doc approved by authors. Code loads a checkpoint trained at one n and
  runs forward/backward at any other n. C1 property tests pass end-to-end.
- **Status:** defined. **Gaps:** GAP-02, GAP-03 (blocking).

### G05.4: Mixed-n parallel training protocol and variable-n replay buffer
- **Paper basis:** "environments with different fleet sizes … contribute to a shared replay
  buffer. Gradient updates train the encoder jointly across all fleet sizes."
- **Objective:** one subprocess per training n (S-07); a shared buffer storing episodes
  padded to `n_max` with an agent mask; batch sampling mixing n.
- **Depends on:** G05.3, INF-4. **Feeds:** G06.5.
- **Acceptance:** a training run that draws from all training n with logged per-n sample
  share; masked agents provably contribute zero gradient (test).
- **Status:** defined. **Gaps:** GAP-01 (which n to train on).

## B. Hybrid control architecture and conditional activation (§05.2)

### G05.5: Nominal waypoint controller π_PID
- **Paper basis:** "a nominal PID controller $\pi^i_{\text{PID}}$ that computes control actions
  — thrust, pitch, roll, and yaw rate — to track the assigned waypoint."
- **Objective:** a fixed (non-learned) waypoint-tracking controller producing `a_pid` in the
  env's action space. Decide between BlueSky's own autopilot (LNAV/HDG) and an explicit PID
  on heading/speed error (PLAN.md §6.5).
- **Depends on:** G04.1. **Feeds:** G05.1, G05.6, G05.7, G06.4 (PID-only ablation).
- **Acceptance:** PID-only rollouts reach their waypoints with no traffic. Gains documented.
- **Status:** defined. **Gaps:** GAP-06 (blocking: the action space doesn't match the paper's).

### G05.6: Safety score sⁱ, threshold θ and switching law
- **Paper basis:** Eq. (4): $a^i_{final}=\pi_{MARL}(G,\mathbf a_{PID})^i$ if $s^i<\theta$, else
  $a^i_{PID}$; "$s^i=f_{GNN}(G)^i$ is a learned safety score derived from the GNN node embedding."
  Also: the policy "activates" "when any other agent enters $r_{\text{detect}}$".
- **Objective:** a safety-score head on node embeddings, trained toward the G03.4 CBF label;
  the per-agent switch; logging of which agents are overridden each step.
- **Depends on:** G05.2, G05.3, G03.4, G05.5. **Feeds:** G05.8, G05.9, G06.2.
- **Acceptance:** with θ = −∞ the system equals PID-only (test). Intervention events logged.
- **Status:** defined. **Gaps:** GAP-04, GAP-05.

### G05.7: PID action as policy input (corrective commands)
- **Paper basis:** "π_MARL takes both the graph G and the vector of locally proposed PID
  actions … so the MARL controller learns to issue corrective commands relative to the reactive baseline."
- **Objective:** `a_pid` included as a node feature, plus a decision on whether the learned
  action is absolute or a residual on `a_pid`.
- **Depends on:** G05.5, G05.1. **Feeds:** D3 PID-input ablation.
- **Status:** defined. **Gaps:** GAP-12.

### G05.8: Intervention cost and LoS reward
- **Paper basis:** "reinforced with a positive reward for reducing loss-of-separation events
  and a cost for each intervention."
- **Objective:** the reward terms inside G04.2's spec, with configurable intervention cost;
  `n_interventions` per episode logged. This makes $\mathbb E[I]\ll T$ measurable.
- **Depends on:** G04.2, G05.6. **Status:** defined.

### G05.9: Handoff continuity measurement
- **Paper basis:** "whether the transition from MARL-issued corrective actions to PID-issued
  waypoint-tracking actions produces discontinuous commands — is characterized as part of the experimental program."
- **Objective:** a metric for command discontinuity at switch events (e.g. |Δheading cmd|,
  |Δspeed cmd| at the switch step), evaluated in the "realistic" scenario of §06.
- **Depends on:** G05.6, G06.2. **Status:** defined.

## C. Conflict-resolution policy learning (§05.3)

### G05.10: GRU belief as node feature (spatial + temporal)
- **Paper basis:** "The GRU hidden state $h^i_t$ … is included as a node feature in G at each
  timestep, so the GNN encoder aggregates both instantaneous spatial relationships and
  temporally accumulated broadcast detection information."
- **Objective:** wire G04.4 output into G05.1 node features; recurrent training through the
  graph pipeline (sequence batches, hidden-state burn-in policy).
- **Depends on:** G04.4, G05.1, G05.2. **Feeds:** D2, C3 evidence.
- **Status:** defined.

### G05.11: Single-level training scheme (only π_MARL is learned)
- **Paper basis:** "the only learned component is the conflict-resolution policy π_MARL, and
  training is a single-level reinforcement learning problem."
- **Objective:** a training loop where PID is frozen, and transitions where no agent is
  overridden are handled per a stated rule (stored? masked? only intervention steps train the Q-net?).
- **Depends on:** G05.5, G05.6, G05.4. **Status:** defined. **Gaps:** GAP-15.
