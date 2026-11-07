# Goals from §01 Nomenclature

### G01.1: Symbol ↔ code naming contract
- **Paper basis:** the nomenclature table in `01_nomenclature.tex`.
- **Objective:** one canonical code name per paper symbol, used in configs, function
  arguments, tensorboard tags and result files. Standard **S-08** makes this binding.
- **Acceptance:** new code and configs use these names. A reviewer can grep a symbol's
  code name and find every place it is used.
- **Status:** done (contract defined below). New symbols are added here before they are used in code.
- **Gaps:** GAP-05 (notation collision), GAP-17 (symbols missing from the paper's table).

| symbol | meaning (paper) | code name | notes |
|---|---|---|---|
| $n$ | number of cooperating UAS | `n_agents` | already used by adapter and `qmix_bluesky.py` |
| $m$ | number of non-cooperative intruders | `n_intruders` | not yet in the envs (GAP-09) |
| $G=(V,E)$ | fleet interaction graph | `graph` (`nodes`, `edges`, `edge_index`, `edge_attr`, `node_mask`) | G05.1 |
| $\delta$ | broadcast latency, s | `latency_s` | always in **seconds**; converted to steps in one place only (GAP-07) |
| $o^i_t$ | local observation | `obs` | shape `(n_agents, obs_dim)` |
| $m^i_t$ | broadcast message received | `msg` | G04.3 |
| $h^i_t$ | GRU hidden (belief) state | `belief` / `h` | G04.4 |
| $\pi^i_{\text{PID}}$ | local PID policy | `pid_policy` | G05.5 |
| $\pi_{\text{MARL}}$ | conflict-resolution policy | `cr_policy` | G05.3/G05.6 |
| $\mathbf{a}_{\text{PID}}$ | proposed PID actions | `a_pid` | G05.7 |
| $\mathbf{a}_{\text{final}}$ | actions after override | `a_final` | G05.6 |
| $z$ | GNN latent | `z` | G05.2 |
| $d$ | latent dimension | `latent_dim` | independent of $n$ |
| $s^i$ | learned safety score | `safety_score` | G05.6 |
| $\theta$ | intervention threshold | `theta` | G03.4 |
| $r_{\text{detect}}$ | detection radius | `r_detect_nm` | unit in the name (BlueSky uses NM) |
| $P$ | permutation matrix | `perm` | tests only |
| $T$ | episode length (control steps) | `episode_len` / `max_episode_steps` | CleanMARL uses `max_episode_steps` |
| $I$ | interventions per episode | `n_interventions` | metric, G06.2 |
| $\gamma$ | discount | `gamma` | |
| $L$ | number of message-passing layers (§05, not in table) | `n_gnn_layers` | **not in the paper's nomenclature table**: suggest to authors |
| $h^{ij}$, $\alpha$ | CBF and class-K function (§03, not in table) | `cbf_h`, `cbf_alpha` | **not in table**; naming collides with $h^i_t$ (GAP-05) |
