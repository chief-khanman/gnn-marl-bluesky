"""marlbench MARLEnv adapter so CleanMARL algorithms (QMIX, VDN, MAPPO ...) can drive BlueSky.

What it adds on top of the raw PettingZoo env:
  * discrete actions: index -> (heading delta, speed delta) in [-1, 1]^2 (QMIX needs Discrete)
  * team reward: sum (or mean) of per-agent rewards -> scalar
  * a flat global state for the mixer (v0: concatenated agent observations;
    replace with true simulator state / graph in the QMIX-GNN phase)
  * fixed-agent-set arrays of shape (n_agents, ...) as CleanMARL expects
  * obs_mode="all_agents" (default): our own observation that sees every other agent, width
    3 + 7 * (n_agents - 1) (marl/interfaces/observation.py; GAP-01, GAP-21, D-016).
    obs_mode="env" keeps the raw bluesky-gym observation (3 nearest agents; broken for n <= 3).
  * state_mode="entities": the global state as an entity matrix, one row of N_STATE = 7 values per
    agent (x, y, vx, vy, cos(drift), sin(drift), airspeed), flattened to n_agents * 7. Positions
    are relative to the fleet centroid. Used by the GNN-QMIX mixer and the baselines
    (G04.5, G05.3; plan docs/plans/G05.3_gnn_qmix.md). Default "concat_obs" is the v0 state.
  * env_kwargs: passed to the configurable env subclass (spawn_min_sep_nm, area_scale;
    simulator/bluesky_marl/custom_envs.py). None keeps the original bluesky-gym env.
  * safety metrics in every step info: los_pairs, los_onsets, min_sep_nm, mean_abs_drift_rad
    (G06.2; definitions in marl/metrics.py).
"""
from __future__ import annotations

import itertools

import numpy as np

from marlbench.interfaces.common import MARLEnv
from marl.interfaces.observation import (
    EARTH_RADIUS_M,
    N_OWN,
    ObsConfig,
    build_observations,
    obs_dim,
    relative_features,
)
from marl.metrics import LOS_NM, pairwise_distance_nm, step_safety_metrics
from simulator.bluesky_marl import make_env

OBS_MODES = ("all_agents", "env")
STATE_MODES = ("concat_obs", "entities")
N_STATE = 7  # x, y, vx, vy, cos(drift), sin(drift), airspeed

HEADING_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # x D_HEADING (22.5 deg) inside the env
SPEED_LEVELS = (-1.0, 0.0, 1.0)                # x D_VELOCITY inside the env


class BlueSkyInterface(MARLEnv):
    def __init__(self, env_name="sector_cr", n_agents=5, team_reward="sum", render_mode=None,
                 obs_mode="all_agents", obs_config: ObsConfig | None = None, state_mode="concat_obs",
                 los_nm: float = LOS_NM, env_kwargs: dict | None = None, **_):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {OBS_MODES}")
        if state_mode not in STATE_MODES:
            raise ValueError(f"unknown state_mode {state_mode!r}; choose from {STATE_MODES}")
        self.obs_mode = obs_mode
        self.state_mode = state_mode
        self.los_nm = los_nm
        self.obs_config = obs_config or ObsConfig()
        self.env = make_env(env_name, n_agents=n_agents, render_mode=render_mode, **(env_kwargs or {}))
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)
        self.action_table = np.array(list(itertools.product(HEADING_LEVELS, SPEED_LEVELS)), dtype=np.float64)
        self.n_actions = len(self.action_table)
        if obs_mode == "all_agents":
            self.obs_dim = obs_dim(self.n_agents)
        else:
            self.obs_dim = int(np.prod(self.env.observation_space(self.agents[0]).shape))
        self.team_reward = team_reward
        self._last_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        self._entities = np.zeros((self.n_agents, N_STATE), dtype=np.float32)
        self._dist_nm = np.zeros((self.n_agents, self.n_agents))
        self._mean_abs_drift_rad = 0.0
        self._prev_los = None

    def _stack(self, obs_dict):
        import bluesky as bs  # noqa: PLC0415  (already initialized by the env)

        raw = [np.asarray(obs_dict[a], dtype=np.float32).reshape(-1) for a in self.agents]
        own = np.stack([r[:N_OWN] for r in raw])
        idx = [bs.traf.id2idx(a) for a in self.agents]
        lat, lon = np.asarray(bs.traf.lat[idx], dtype=np.float64), np.asarray(bs.traf.lon[idx], dtype=np.float64)
        ve, vn = np.asarray(bs.traf.gseast[idx], dtype=np.float64), np.asarray(bs.traf.gsnorth[idx], dtype=np.float64)
        self._update_truth(own, lat, lon, ve, vn)
        if self.obs_mode == "env":
            return np.stack(raw)
        return build_observations(own, relative_features(lat, lon, ve, vn, self.obs_config))

    def _update_truth(self, own, lat, lon, ve, vn):
        """Simulator-truth quantities behind the entity state and the safety metrics."""
        cfg = self.obs_config
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        x = (lon_r - lon_r.mean()) * EARTH_RADIUS_M * np.cos(lat_r.mean())
        y = (lat_r - lat_r.mean()) * EARTH_RADIUS_M
        self._entities = np.concatenate(
            [np.stack([x / cfg.pos_scale_m, y / cfg.pos_scale_m, ve / cfg.vel_scale_mps, vn / cfg.vel_scale_mps],
                      axis=1), own], axis=1).astype(np.float32)
        self._dist_nm = pairwise_distance_nm(lat, lon)
        self._mean_abs_drift_rad = float(np.abs(np.arctan2(own[:, 1], own[:, 0])).mean())

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # the underlying env uses the global numpy RNG
        obs, info = self.env.reset(seed=seed)
        self._last_obs = self._stack(obs)
        _, self._prev_los = step_safety_metrics(self._dist_nm, None, self.los_nm)
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
        safety, self._prev_los = step_safety_metrics(self._dist_nm, self._prev_los, self.los_nm)
        flat_info.update(safety, mean_abs_drift_rad=self._mean_abs_drift_rad)
        return self._last_obs, reward, done, truncated, flat_info

    def get_state(self):
        if self.state_mode == "entities":
            return self._entities.reshape(-1)
        return self._last_obs.reshape(-1)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=bool)

    def get_obs_size(self):
        return self.obs_dim

    def get_state_size(self):
        if self.state_mode == "entities":
            return N_STATE * self.n_agents
        return self.obs_dim * self.n_agents

    def get_action_size(self):
        return self.n_actions

    def sample(self):
        return np.random.randint(0, self.n_actions, size=self.n_agents)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
