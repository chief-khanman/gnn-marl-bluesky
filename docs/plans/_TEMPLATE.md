# PLAN-<goal-id(s)>: <title>

> Copy this file to `docs/plans/<goal-id>_<slug>.md`. A plan is required before implementing
> any paper goal (S-05). Keep it short and concrete: it is a contract between plans.

| field | value |
|---|---|
| goals | e.g. G05.2 (and the G-IDs it partially serves) |
| paper basis | section + equation numbers |
| status | `draft` → `in-review` → `approved` (by whom, date) → `superseded` |
| author(s) / agent | |
| gaps addressed | GAP-xx (resolved how) |
| decisions needed | Q-xx / proposed D-xxx |

## 1. Upstream inputs (what this plan consumes)
For each input: the producing plan/goal, exact type/shape/units, and where it lives in code.
These must match the **downstream outputs** section of the producing plan, word for word.

## 2. Downstream outputs (what this plan guarantees)
Exact interface: function/class signatures, tensor shapes (with `n_agents` symbolic),
units, config fields, and logged metric names (S-11). Consumers are listed.

## 3. Design
The method, written precisely enough that the paper text and proofs can be checked against
it. Equations reference the paper numbering.

## 4. Configuration
New `SimConfig` / run-config fields with defaults and the experiment ranges (G06.1).

## 5. Acceptance tests and evidence
Tests (unit and property) and the run or artifact that marks the goal `done`.

## 6. Risks and alternatives
What could fail, what we'd do instead, and which gap it reopens.

## 7. Out of scope
What this plan deliberately does not do, and which plan does.
