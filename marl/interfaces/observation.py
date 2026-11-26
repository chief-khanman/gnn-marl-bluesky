"""Per-agent observation builder that sees ALL other agents (goals G04.1, G05.1; GAP-01, GAP-21).

Plan: docs/plans/GAP-01_observation_builder.md.

The bluesky-gym envs observe only the 3 nearest agents, so their observation is 24 wide for
n >= 4 and silently shorter for n <= 3. This builder reads simulator truth and gives every
agent one 7-value block per other agent, so the width is linear in n:

    obs_dim(n) = N_OWN + N_PER_OTHER * (n - 1)        # 17, 38, 59 for n = 3, 6, 9

Layout of one agent's observation:
    [ own (3) | other_1 (7) | other_2 (7) | ... ]     others sorted nearest first
    own      = cos(drift), sin(drift), normalized airspeed   (taken from the env's own obs[0:3])
    other_j  = dx_east, dy_north, dvx_east, dvy_north, cos(psi), sin(psi), distance
               psi = direction of the relative velocity, atan2(dvy_north, dvx_east)

Positions use a local flat-earth frame (x east, y north) and velocities are the simulator's
ground-speed components in the same frame. The pure functions here need no BlueSky import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
N_OWN = 3
N_PER_OTHER = 7


@dataclass(frozen=True)
class ObsConfig:
    """Normalization scales. Experiments may vary them, so they live in config (S-09)."""

    pos_scale_m: float = 50_000.0
    vel_scale_mps: float = 150.0


def obs_dim(n_agents: int) -> int:
    """Width of one agent's observation for a fleet of `n_agents`."""
    return N_OWN + N_PER_OTHER * (n_agents - 1)


def relative_features(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    gs_east_mps: np.ndarray,
    gs_north_mps: np.ndarray,
    cfg: ObsConfig = ObsConfig(),
) -> np.ndarray:
    """Relative state of every other agent, as seen from each agent.

    Inputs are arrays of shape (n,). Returns float32 of shape (n, n - 1, N_PER_OTHER); row i
    lists the other agents sorted by distance from agent i, nearest first.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    ve = np.asarray(gs_east_mps, dtype=np.float64)
    vn = np.asarray(gs_north_mps, dtype=np.float64)
    n = lat.shape[0]

    # [i, j] = quantity of agent j relative to agent i
    dy = (lat[None, :] - lat[:, None]) * EARTH_RADIUS_M
    dx = (lon[None, :] - lon[:, None]) * EARTH_RADIUS_M * np.cos(0.5 * (lat[None, :] + lat[:, None]))
    dvx = ve[None, :] - ve[:, None]
    dvy = vn[None, :] - vn[:, None]
    dist = np.hypot(dx, dy)
    psi = np.arctan2(dvy, dvx)

    feats = np.stack(
        [
            dx / cfg.pos_scale_m,
            dy / cfg.pos_scale_m,
            dvx / cfg.vel_scale_mps,
            dvy / cfg.vel_scale_mps,
            np.cos(psi),
            np.sin(psi),
            dist / cfg.pos_scale_m,
        ],
        axis=-1,
    )  # (n, n, 7)

    out = np.zeros((n, max(n - 1, 0), N_PER_OTHER), dtype=np.float32)
    for i in range(n):
        others = np.array([j for j in range(n) if j != i], dtype=int)
        order = others[np.argsort(dist[i, others], kind="stable")]
        out[i] = feats[i, order]
    return out


def build_observations(own: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Join own features (n, N_OWN) and relative features (n, n-1, N_PER_OTHER) -> (n, obs_dim(n))."""
    n = own.shape[0]
    return np.concatenate([own.astype(np.float32), rel.reshape(n, -1)], axis=1)
