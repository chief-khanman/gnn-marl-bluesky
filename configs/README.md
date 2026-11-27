# configs/

Run and scenario configurations (S-10). Every experiment that produces a reported result
starts from a file here.

Rules:
- One file per scenario cell or run family. The header comment names the goal ID(s) (S-03).
- Field names follow the nomenclature contract (`docs/goals/01_nomenclature.md`, S-08).
- The full scenario matrix (G06.1) is indexed in `configs/INDEX.md` once it exists.

Files are JSON read by `python -m algorithms.gnn_qmix.train --config <file>`; the `_goals` key is
the header comment. Command-line flags override the file. How to launch: `docs/RUNBOOK.md`.

| file | goals | what |
|---|---|---|
| `gnn_qmix_sector_cr.json` | G05.3, G05.4 | proposed method, trained jointly at n ∈ {3, 6, 9} |
| `vdn_sector_cr.json`, `iql_sector_cr.json` | G03.1, G06.4 | baselines on the fixed 3-nearest observation (D-027), trained jointly at n ∈ {3, 6, 9} |
| `qmix_sector_cr.json` | G03.1, G06.4 | QMIX baseline, full observation and fixed-n mixer, one run per fleet size (`--train-n`) |
| `ablation_vdn_gcn_sector_cr.json` | G06.7 | GCN agent with a VDN sum instead of the GAT mixer |
| `smoke.json` | INF-5 | short smoke test, not for results |
| `gap31/gnn_qmix_repro.json` | G05.4, GAP-31 | rerun arm R: `gnn_qmix_sector_cr.json` plus kept checkpoints and diagnostics |
| `gap31/gnn_qmix_stab.json` | G05.4, GAP-31, GAP-32 | arm A: shared stabilizers (double Q, polyak 0.001, learning rate falling to 10 %, airspeed ÷ 10 in agents and mixer state) |
| `gap31/gnn_qmix_stab_detach.json` | G05.4, GAP-32 | arm B: arm A plus agent embeddings detached from the mixer loss |
| `gap31/gnn_qmix_stab_detach_norm.json` | G05.4, GAP-32 | arm C: arm B plus LayerNorm in the GAT mixer (not in the paper's mixer equation) |
| `gap31/vdn_stab.json`, `gap31/iql_stab.json`, `gap31/qmix_stab.json` | G03.1, G06.4, GAP-31 | baselines with the shared stabilizers |
| `gap31/ablation_vdn_gcn_stab.json` | G06.7, GAP-31 | ablation with the shared stabilizers |

All configs: 2 000 000 steps per training fleet size (D-030), 6 NM minimum spawn separation (D-028).
The `gap31/` configs are the rerun of `docs/plans/GAP-31_training_stability.md`; their matrix is `tools/hpc/matrix_gap31.txt`.
