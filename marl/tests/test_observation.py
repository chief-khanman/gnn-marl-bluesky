"""Observation builder tests (GAP-01, GAP-21; plan docs/plans/GAP-01_observation_builder.md).

The BlueSky checks run in subprocesses, one per fleet size (S-07, S-12).
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from marl.interfaces.observation import N_PER_OTHER, ObsConfig, build_observations, obs_dim, relative_features

TRAIN_N = (3, 6, 9)  # paper training fleet sizes


def test_obs_dim_is_linear_in_n():
    dims = [obs_dim(n) for n in TRAIN_N]
    assert dims == [17, 38, 59]
    assert len({b - a for a, b in zip(dims, dims[1:])}) == 1  # equal steps for equal steps in n


def test_relative_features_geometry_and_order():
    # agent 0 at origin; agent 1 is 0.1 deg north; agent 2 is 1 deg north
    lat = np.array([0.0, 0.1, 1.0])
    lon = np.zeros(3)
    ve = np.array([0.0, 10.0, 0.0])
    vn = np.array([100.0, 100.0, 50.0])
    cfg = ObsConfig(pos_scale_m=1.0, vel_scale_mps=1.0)
    rel = relative_features(lat, lon, ve, vn, cfg)
    assert rel.shape == (3, 2, N_PER_OTHER)
    near, far = rel[0]
    assert near[6] < far[6]                                 # nearest first
    assert abs(far[1] - 111_195) < 100 and abs(far[0]) < 1  # 1 deg north = ~111.2 km
    assert np.allclose(near[2:4], [10.0, 0.0])              # relative velocity of agent 1
    assert np.allclose(near[4:6], [1.0, 0.0])               # psi = 0 (relative velocity points east)
    # antisymmetry: what 1 sees of 0 is the negative of what 0 sees of 1
    assert np.allclose(rel[1][0][:4], -near[:4])


def test_build_observations_shape():
    n = 6
    own = np.zeros((n, 3))
    rel = np.zeros((n, n - 1, N_PER_OTHER), dtype=np.float32)
    assert build_observations(own, rel).shape == (n, obs_dim(n))


_PROBE = """
import json, sys, numpy as np
np.random.seed(0)
from marl.interfaces.bluesky_interface import BlueSkyInterface
env_name, n = sys.argv[1], int(sys.argv[2])
it = BlueSkyInterface(env_name, n_agents=n)
obs, _ = it.reset(seed=0)
shapes = {tuple(obs.shape)}
for _ in range(10):
    obs, *_ = it.step(it.sample())
    shapes.add(tuple(obs.shape))
print("RESULT " + json.dumps(dict(declared=it.get_obs_size(), shapes=sorted(shapes),
      state=int(it.get_state().shape[0]), declared_state=it.get_state_size(),
      finite=bool(np.isfinite(obs).all()))))
"""


def _probe(env_name, n):
    out = subprocess.run([sys.executable, "-c", _PROBE, env_name, str(n)], capture_output=True, text=True, timeout=180)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, out.stderr[-2000:]
    return json.loads(line[-1][7:])


@pytest.mark.parametrize("env_name", ["sector_cr", "merge"])
def test_simulator_obs_dim_recorded_for_training_sizes(env_name):
    record = {n: _probe(env_name, n) for n in TRAIN_N}
    for n, r in record.items():
        assert r["declared"] == obs_dim(n)
        assert r["shapes"] == [[n, obs_dim(n)]]            # actual == declared, every step
        assert r["state"] == r["declared_state"] == n * obs_dim(n)
        assert r["finite"]
    dims = [record[n]["declared"] for n in TRAIN_N]
    assert len({b - a for a, b in zip(dims, dims[1:])}) == 1  # linear in n
