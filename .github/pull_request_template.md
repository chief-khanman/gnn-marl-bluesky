## What and why
<!-- One paragraph. -->

**Goal IDs:** <!-- e.g. G05.2, INF-3 -->
**Plan:** <!-- docs/plans/... (required for paper goals, S-05) -->

**Tests (S-20):** <!-- e.g. T0 pass · T1 marl pass · T2 pass (baseline before change: T2 pass) -->

## Checklist (docs/PROJECT_RECORD.md §4)
- [ ] Addresses exactly one G-/INF-/GAP- item; edits stay in that item's area; no drive-by changes (S-18, S-19)
- [ ] Explicit paths staged; no `.env`, tokens, `runs/` or `*.pt` (S-21)
- [ ] Shared weights published and registered in `artifacts/registry.toml` (S-23)
- [ ] Work maps to a goal ID / INF item, and the module docstring or config header cites it (S-03)
- [ ] For a paper goal: the plan in `docs/plans/` is `approved` (S-05)
- [ ] Nothing under `*/third_party/` was edited (S-06)
- [ ] One BlueSky env per process; seeding via `np.random.seed` (S-07)
- [ ] Names follow `docs/goals/01_nomenclature.md` (S-08)
- [ ] Required S-20 tiers ran before and after; nothing that passed now fails (S-12, S-20)
- [ ] Results come with a config in `configs/`, run IDs, and ≥3 seeds (S-10, S-14)
- [ ] No paper claims changed without author sign-off; no invented references (S-04, S-15)
- [ ] `docs/PROJECT_RECORD.md` updated: work log (names the AI agent used, if any; ends with `State:`/`Next:`), status, new D-/GAP-/Q- entries (S-02, S-16, S-22)
