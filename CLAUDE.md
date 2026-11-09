# CLAUDE.md

**Before any work, read `docs/PROJECT_RECORD.md`.** It holds the current phase, the binding
standards (S-xx), the decisions (D-xxx), the gap register (GAP-xx) and the open questions
(Q-xx). Then read the goal file in `docs/goals/` for the paper section you are touching.
**After any work, update `docs/PROJECT_RECORD.md`:** add a work-log entry, record any new
decision or gap, and update goal status.

Current phase: **framework / planning** (see the record). Do not write implementation code
for a goal until it has an approved plan in `docs/plans/` (S-05).

Hard rules:
- One BlueSky env per process; parallel or eval envs go in subprocesses (`marlbench.vec_envs.SubprocVectorEnv`).
- Never edit files under `*/third_party/`. Copy files out and modify the copies (as with `algorithms/qmix_bluesky.py`).
- Seed with `np.random.seed`; envs ignore `reset(seed)`. Wrap `sector_cr` in `TimeLimit` (no built-in limit).
- Run `python -m marl.scripts.random_rollout` and `pytest` before every commit.
- Paper: one section per file in `paper/sections/`; don't change claims in the paper without asking the authors.
- Never invent bibliographic details or results. Hypothesized ≠ measured (S-14, S-15).
- Every code module, config and experiment cites the goal ID(s) it serves (S-03).

Change discipline (S-18…S-21):
- One issue (one G-/INF-/GAP- item) per change. Edit only that area. Log anything else you notice instead of fixing it.
- Run the S-20 test tiers before and after (Tier 0 `python tools/check_docs.py` always). Report results truthfully.
- Stage explicit paths only. **Never commit or push without the user's explicit OK in this session.** Show the diff summary and test results first.

Across machines (S-22, S-23): your memory doesn't travel to the user's other machines.
Write what matters into the repo. End sessions with `State:`/`Next:` in the work log.
Trained weights go through `docs/ARTIFACTS.md`, never into git.
