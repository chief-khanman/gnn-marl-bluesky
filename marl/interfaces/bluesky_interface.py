"""marlbench MARLEnv adapter so CleanMARL algorithms (QMIX, VDN, MAPPO ...) can drive BlueSky.

What it adds on top of the raw PettingZoo env:
  * discrete actions: index -> (heading delta, speed delta) in [-1, 1]^2 (QMIX needs Discrete)
  * team reward: sum (or mean) of per-agent rewards -> scalar
  * a flat global state for the mixer (v0: concatenated agent observations;
    replace with true simulator state / graph in the QMIX-GNN phase)
  * fixed-agent-set arrays of shape (n_agents, ...) as CleanMARL expects
"""
from __future__ import annotations

import itertools

import numpy as np

from marlbench.interfaces.common import MARLEnv
from simulator.bluesky_marl import make_env

HEADING_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # x D_HEADING (22.5 deg) inside the env
SPEED_LEVELS = (-1.0, 0.0, 1.0)                # x D_VELOCITY inside the env


class BlueSkyInterface(MARLEnv):
    def __init__(self, env_name="sector_cr", n_agents=5, team_reward="sum", render_mode=None, **_):
        self.env = make_env(env_name, n_agents=n_agents, render_mode=render_mode)
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        self.action_table = np.array(list(itertools.product(HEADING_LEVELS, SPEED_LEVELS)), dtype=np.float64)
        self.n_actions = len(self.action_table)
        self.obs_dim = int(np.prod(self.env.observation_space(self.agents[0]).shape))
        self.team_reward = team_reward
        self._last_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

    def _stack(self, obs_dict):
        return np.stack([np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in self.agents])

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # the underlying env uses the global numpy RNG
        obs, info = self.env.reset(seed=seed)
        self._last_obs = self._stack(obs)
        return self._last_obs, info

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1).astype(int)
        joint = {a: self.action_table[actions[i]] for i, a in enumerate(self.agents)}
        obs, rew, term, trunc, info = self.env.step(joint)
        if obs:
            self._last_obs = self._stack(obs)
        r = np.array([float(rew.get(a, 0.0)) for a in self.agents])
        reward = float(r.sum() if self.team_reward == "sum" else r.mean())
        done = bool(any(term.values())) if term else False
        truncated = bool(all(trunc.values())) if trunc else False
        flat_info = {f"{a}_{k}": v for a, d in info.items() for k, v in d.items()
                     if np.isscalar(v)}
        return self._last_obs, reward, done, truncated, flat_info

    def get_state(self):
        return self._last_obs.reshape(-1)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=bool)

    def get_obs_size(self):
        return self.obs_dim

    def get_state_size(self):
        return self.obs_dim * self.n_agents

    def get_action_size(self):
        return self.n_actions

    def sample(self):
        return np.random.randint(0, self.n_actions, size=self.n_agents)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
