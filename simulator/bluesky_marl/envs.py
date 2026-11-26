"""Single entry point for creating BlueSky multi-agent (PettingZoo ParallelEnv) environments.

The environments live in the pinned git submodule simulator/third_party/bluesky-gym
(jangroter/bluesky-gym, branch petting_zoo_sac). That repo does not package `bluesky_zoo`,
so the submodule root is put on sys.path here instead of pip-installing it (its
pyproject also pins numpy==1.24, which conflicts with CleanMARL's numpy 2.x).

WARNING: BlueSky is a process-wide singleton (bs.traf). Only ONE env instance may exist
per Python process. A second instance silently takes over the first one's aircraft.
Use subprocesses (marlbench SubprocVectorEnv) for parallel/eval envs.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SUBMODULE = Path(__file__).resolve().parents[1] / "third_party" / "bluesky-gym"
if str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

_INSTANCES = 0


def _sector_cr(**kw):
    from bluesky_zoo.sector_cr.sector_cr import SectorCR
    return SectorCR(**kw)


def _merge(**kw):
    from bluesky_zoo.merge.mergev3 import MergeEnv
    return MergeEnv(**kw)


# sector_cr: every aircraft has its OWN exit waypoint (individual goals)
# merge:     all aircraft head to a common merge point (shared goal)
ENV_REGISTRY = {"sector_cr": _sector_cr, "merge": _merge}


def _sector_cr_configurable(**kw):
    from simulator.bluesky_marl.custom_envs import ConfigurableSectorCR
    return ConfigurableSectorCR(**kw)


# Envs that accept extra keyword arguments (spawn_min_sep_nm, area_scale; see custom_envs.py)
CONFIGURABLE = {"sector_cr": _sector_cr_configurable}


def make_env(name: str = "sector_cr", n_agents: int = 5, render_mode: str | None = None, **env_kwargs):
    """`env_kwargs` selects the configurable subclass of the env; without them the original env is used."""
    global _INSTANCES
    if env_kwargs and name not in CONFIGURABLE:
        raise ValueError(f"env {name!r} takes no extra arguments (got {sorted(env_kwargs)})")
    if name not in ENV_REGISTRY:
        raise ValueError(f"unknown env {name!r}; choose from {list(ENV_REGISTRY)}")
    _INSTANCES += 1
    if _INSTANCES > 1:
        raise RuntimeError(
            "A BlueSky env already exists in this process. BlueSky shares one global "
            "simulator, so a second instance would corrupt the first. Use one process per env."
        )
    if env_kwargs:
        return CONFIGURABLE[name](render_mode=render_mode, n_agents=n_agents, **env_kwargs)
    return ENV_REGISTRY[name](render_mode=render_mode, n_agents=n_agents)
