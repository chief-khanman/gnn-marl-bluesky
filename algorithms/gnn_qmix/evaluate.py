"""Evaluate a trained run at any list of fleet sizes, including sizes never trained on.

Goals: G06.3 (evaluation harness, zero-shot across n), G06.5, G06.9. Plan: docs/plans/G05.3_gnn_qmix.md §3.6.

    python -m algorithms.gnn_qmix.evaluate --run-dir runs/<run_id> --eval-n 3 6 9 10 20 50 --episodes 100

Writes <run-dir>/eval_zero_shot.csv with one row per episode. A network with fixed-n layers
(an MLP agent that reads all other agents) asked for a fleet size it was not trained on gets
one row with status = not_evaluable and no metrics. GCN agents and k-nearest MLP agents run at
any fleet size. Each fleet size is evaluated in both airspaces: `fixed` (original sector, so
density rises with n) and `density` (sector area scaled by n / density_ref_n) (D-029).
Execution is decentralized: only the agent network is loaded; the mixer is not used.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from algorithms.gnn_qmix.common import args_from_dict, build_networks, evaluate_policy, make_vec_env, method_name
from marl.interfaces.observation import obs_dim
from marl.metrics import EPISODE_METRICS


@dataclass
class EvalArgs:
    run_dir: str
    """Run directory written by algorithms.gnn_qmix.train"""
    eval_n: tuple[int, ...] = (3, 6, 9, 10, 20, 50)
    """Fleet sizes to evaluate at"""
    airspace: tuple[str, ...] = ("fixed", "density")
    """fixed: original sector size, density rises with n. density: sector area scaled by n / density_ref_n (D-029)"""
    episodes: int = 100
    """Episodes per fleet size"""
    num_envs: int = 4
    """Parallel evaluation processes"""
    seed: int = 20_000
    """Scenario seed; the same seed gives every run the same scenarios"""
    checkpoint: str = "final.pt"
    device: str = "cpu"
    out: str = "eval_zero_shot.csv"


def main(ev: EvalArgs) -> Path:
    run_dir = Path(ev.run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    args = args_from_dict(meta["args"])
    device = torch.device(ev.device)
    ckpt = torch.load(run_dir / "checkpoints" / ev.checkpoint, map_location=device, weights_only=True)
    train_n = "-".join(map(str, args.train_n))
    head = [meta["run_id"], method_name(args), args.agent_kind(), args.env_name, train_n, args.seed]

    out_path = run_dir / ev.out
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "algo", "agent_net", "env", "train_n", "seed", "airspace", "eval_n", "status",
                         "episode", *EPISODE_METRICS])
        for airspace, n in [(a, n) for a in ev.airspace for n in ev.eval_n]:
            if not args.agent_runs_at_any_n() and n not in args.train_n:
                writer.writerow([*head, airspace, n, "not_evaluable", "", *[""] * len(EPISODE_METRICS)])
                print(f"eval_n={n}: not_evaluable (network input is {obs_dim(args.train_n[0])} wide, "
                      f"n={n} needs {obs_dim(n)})", flush=True)
                continue
            agent, _ = build_networks(args, obs_dim(n), 0, n, ckpt["n_actions"])
            agent.load_state_dict(ckpt["agent"])
            agent.to(device).eval()
            env = make_vec_env(args, n, min(ev.num_envs, ev.episodes), auto_reset=False, airspace=airspace)
            try:
                episodes = evaluate_policy(agent, env, ev.episodes, ev.seed, device)
            finally:
                env.close()
            for i, ep in enumerate(episodes):
                writer.writerow([*head, airspace, n, "ok", i, *[f"{ep[m]:.6g}" for m in EPISODE_METRICS]])
            f.flush()
            mean = {m: sum(e[m] for e in episodes) / len(episodes) for m in ("los_rate", "conflict_free",
                                                                             "ep_return_per_agent")}
            print(f"airspace={airspace} eval_n={n}: episodes={len(episodes)} los_rate={mean['los_rate']:.4f} "
                  f"conflict_free={mean['conflict_free']:.2f} return/agent={mean['ep_return_per_agent']:.2f}",
                  flush=True)
    print(f"wrote {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
