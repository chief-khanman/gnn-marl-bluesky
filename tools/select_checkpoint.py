#!/usr/bin/env python3
"""Pick the checkpoint of one run by a fixed rule on training-size evaluations only (GAP-31, Q-17).

Goals: G06.3, G06.4. Plan: docs/plans/GAP-31_training_stability.md §3.3. Stdlib only.

    python tools/select_checkpoint.py --run-dir <run> [--window 5] [--min-step 100000]

Reads `metrics.csv` (rows split=eval, metric=ep_return_per_agent, the run's training fleet sizes).
Per evaluation step: the mean return over the training sizes. Score of a step: the mean of that
value over the `window` evaluations centred on it. Candidates: steps >= `min-step` that have a
`checkpoints/step_<step>.pt` file and a full window. The highest score wins; on a tie the later
step. Writes `best_checkpoint.json` into the run directory and prints the checkpoint file name.
It never reads zero-shot results, so the choice uses the training fleet sizes only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RULE = "centred mean of training-size eval return"


def eval_curve(run_dir: Path, sizes: list[int]) -> dict[int, float]:
    """step -> mean over the training sizes of the evaluation return (steps that have every size)."""
    per_step: dict[int, dict[int, float]] = {}
    with open(run_dir / "metrics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "eval" and row["metric"] == "ep_return_per_agent" and int(row["n_agents"]) in sizes:
                per_step.setdefault(int(row["step"]), {})[int(row["n_agents"])] = float(row["value"])
    return {step: sum(v.values()) / len(sizes) for step, v in per_step.items() if len(v) == len(sizes)}


def select(curve: dict[int, float], available: set[int], window: int, min_step: int) -> tuple[int, float]:
    """(step, score) of the best candidate. Raises ValueError if there is none."""
    steps = sorted(curve)
    half = window // 2
    best: tuple[float, int] | None = None
    for i, step in enumerate(steps):
        if step < min_step or step not in available or i - half < 0 or i + half >= len(steps):
            continue
        score = sum(curve[s] for s in steps[i - half:i + half + 1]) / (2 * half + 1)
        if best is None or (score, step) > best:
            best = (score, step)
    if best is None:
        raise ValueError("no checkpoint satisfies the rule (min-step, window and step_*.pt files)")
    return best[1], best[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--window", type=int, default=5, help="odd number of evaluations in the centred mean")
    p.add_argument("--min-step", type=int, default=100_000)
    a = p.parse_args(argv)
    if a.window < 1 or a.window % 2 == 0:
        p.error("--window must be a positive odd number")
    run_dir = Path(a.run_dir)
    sizes = [int(n) for n in json.loads((run_dir / "config.json").read_text())["args"]["train_n"]]
    available = {int(f.stem.split("_")[1]) for f in (run_dir / "checkpoints").glob("step_*.pt")}
    step, score = select(eval_curve(run_dir, sizes), available, a.window, a.min_step)
    name = f"step_{step:07d}.pt"
    (run_dir / "best_checkpoint.json").write_text(json.dumps(
        {"checkpoint": name, "step": step, "score": score, "window": a.window, "min_step": a.min_step,
         "rule": RULE}, indent=2) + "\n")
    print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
