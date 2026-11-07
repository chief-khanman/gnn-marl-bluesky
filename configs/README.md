# configs/

Run and scenario configurations (S-10). Every experiment that produces a reported result
starts from a file here.

Rules:
- One file per scenario cell or run family. The header comment names the goal ID(s) (S-03).
- Field names follow the nomenclature contract (`docs/goals/01_nomenclature.md`, S-08).
- The full scenario matrix (G06.1) is indexed in `configs/INDEX.md` once it exists.

Empty during Stage 1 (Framework). It gets filled during Stage 3–4.
