"""Smoke test (plan steps 6-8): create one multi-agent BlueSky env, send a random action to
every agent simultaneously (one joint-action dict per step, PettingZoo Parallel API), run 100 steps.

    python -m marl.scripts.random_rollout --env sector_cr --n-agents 5 --steps 100
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from simulator.bluesky_marl import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="sector_cr", choices=["sector_cr", "merge"])
    p.add_argument("--n-agents", type=int, default=5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()

    np.random.seed(a.seed)  # env ignores reset(seed=...); seed the global RNG it uses
    env = make_env(a.env, n_agents=a.n_agents, render_mode="human" if a.render else None)
    obs, _ = env.reset(seed=a.seed)
    for ag in env.agents:
        env.action_space(ag).seed(a.seed + hash(ag) % 1000)
    print(f"env={a.env} agents={env.agents} obs_dim={next(iter(obs.values())).shape}")

    t0, resets, team_return = time.time(), 0, 0.0
    for t in range(a.steps):
        if not env.agents:  # episode ended (env empties .agents on done/truncation)
            obs, _ = env.reset()
            resets += 1
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}  # random, all agents at once
        obs, rew, term, trunc, info = env.step(actions)
        team_return += float(sum(rew.values()))
        assert set(obs) == set(actions), "every acting agent must get an observation"
        if t % 20 == 0:
            print(f"t={t:3d} team_reward={sum(rew.values()):8.3f} any_term={any(term.values())} "
                  f"all_trunc={all(trunc.values()) if trunc else None}")
    print(f"done: {a.steps} steps, resets={resets}, team_return={team_return:.2f}, "
          f"wall={time.time() - t0:.2f}s")
    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
