"""Paper metrics, one definition each (goal G06.2; standard S-11).

Plan: docs/plans/G05.3_gnn_qmix.md §3.5. Canonical per-episode names:

    ep_return_per_agent   sum over steps of the mean per-agent reward
    ep_length             number of control steps T
    los_pair_steps        sum over steps of the number of pairs in loss of separation (LoS)
    los_rate              los_pair_steps / (T * n(n-1)/2)
    los_events            number of times a pair goes from separated to LoS
    los_events_per_agent  los_events / n
    conflict_free         1.0 if los_pair_steps == 0 else 0.0
    min_sep_nm            smallest pair separation in the episode, NM
    mean_abs_drift_deg    mean over steps and agents of |track deviation from the goal bearing|

LoS = pair separation below `los_nm` (5 NM, the separation minimum of the BlueSky envs).
The pure functions need neither BlueSky nor marlbench.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
NM_TO_M = 1852.0
LOS_NM = 5.0

EPISODE_METRICS = (
    "ep_return_per_agent",
    "ep_length",
    "los_pair_steps",
    "los_rate",
    "los_events",
    "los_events_per_agent",
    "conflict_free",
    "min_sep_nm",
    "mean_abs_drift_deg",
)


def pairwise_distance_nm(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Pair separations in NM, shape (n, n), local flat-earth frame (same as observation.py)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    dy = (lat[None, :] - lat[:, None]) * EARTH_RADIUS_M
    dx = (lon[None, :] - lon[:, None]) * EARTH_RADIUS_M * np.cos(0.5 * (lat[None, :] + lat[:, None]))
    return np.hypot(dx, dy) / NM_TO_M


def step_safety_metrics(
    dist_nm: np.ndarray, prev_los: np.ndarray | None, los_nm: float = LOS_NM
) -> tuple[dict[str, float], np.ndarray]:
    """LoS counts for one step.

    `dist_nm` is (n, n). `prev_los` is the boolean LoS matrix of the previous step (None on the
    first step, where every pair in LoS counts as a new event). Returns the step metrics and
    the current LoS matrix.
    """
    n = dist_nm.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    los = (dist_nm < los_nm) & upper
    onsets = los if prev_los is None else los & ~prev_los
    min_sep = float(dist_nm[upper].min()) if n > 1 else float("inf")
    return {"los_pairs": float(los.sum()), "los_onsets": float(onsets.sum()), "min_sep_nm": min_sep}, los


class EpisodeAccumulator:
    """Turns per-step values into the canonical per-episode metrics."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.return_per_agent = 0.0
        self.los_pair_steps = 0.0
        self.los_events = 0.0
        self.min_sep_nm = float("inf")
        self.drift_sum_rad = 0.0

    def add(self, reward_per_agent: float, los_pairs: float, los_onsets: float, min_sep_nm: float,
            mean_abs_drift_rad: float) -> None:
        self.steps += 1
        self.return_per_agent += float(reward_per_agent)
        self.los_pair_steps += float(los_pairs)
        self.los_events += float(los_onsets)
        self.min_sep_nm = min(self.min_sep_nm, float(min_sep_nm))
        self.drift_sum_rad += float(mean_abs_drift_rad)

    def result(self) -> dict[str, float]:
        n, t = self.n_agents, max(self.steps, 1)
        pairs = max(n * (n - 1) / 2, 1)
        return {
            "ep_return_per_agent": self.return_per_agent,
            "ep_length": float(self.steps),
            "los_pair_steps": self.los_pair_steps,
            "los_rate": self.los_pair_steps / (t * pairs),
            "los_events": self.los_events,
            "los_events_per_agent": self.los_events / n,
            "conflict_free": 1.0 if self.los_pair_steps == 0 else 0.0,
            "min_sep_nm": self.min_sep_nm,
            "mean_abs_drift_deg": float(np.degrees(self.drift_sum_rad / t)),
        }


def make_episode_metrics_wrapper(env):
    """Wrap a BlueSkyInterface (possibly inside other marlbench wrappers) so that the info of the
    last step of every episode carries `info["episode_metrics"]` (dict of EPISODE_METRICS)."""
    from marlbench.wrappers.core import Wrapper  # noqa: PLC0415

    class EpisodeMetrics(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._acc = EpisodeAccumulator(env.n_agents)
            self._mean_reward = env.get_wrapper_attr("team_reward") == "mean"

        def reset(self, seed=None):
            self._acc.reset()
            return self.env.reset(seed=seed)

        def step(self, actions):
            obs, reward, terminated, truncated, info = self.env.step(actions)
            per_agent = reward if self._mean_reward else reward / self._acc.n_agents
            self._acc.add(per_agent, info["los_pairs"], info["los_onsets"], info["min_sep_nm"],
                          info["mean_abs_drift_rad"])
            if terminated or truncated:
                info["episode_metrics"] = self._acc.result()
            return obs, reward, terminated, truncated, info

    return EpisodeMetrics(env)
