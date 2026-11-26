"""ConfigurableSectorCR: spawn separation and airspace scaling (G04.1; D-028, D-029; S-07, S-12).

Each BlueSky check runs in its own subprocess.
"""
import json
import subprocess
import sys

import pytest

_PROBE = """
import json, sys, numpy as np
import bluesky as bs
from simulator.bluesky_marl import make_env
from marl.metrics import pairwise_distance_nm
n, sep, scale = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
env = make_env("sector_cr", n_agents=n, spawn_min_sep_nm=sep, area_scale=scale)
mins, areas = [], []
for seed in range(8):
    np.random.seed(seed)
    obs, _ = env.reset()
    d = pairwise_distance_nm(bs.traf.lat[:n], bs.traf.lon[:n])
    d[np.diag_indices(n)] = np.inf
    mins.append(float(d.min())); areas.append(float(env.poly_area))
    assert len(obs) == n
print(json.dumps({"min_sep": min(mins), "area_min": min(areas), "area_max": max(areas)}))
"""


def probe(n, sep, scale):
    out = subprocess.run([sys.executable, "-c", _PROBE, str(n), str(sep), str(scale)], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize("n", [9, 50])
def test_no_pair_spawns_in_loss_of_separation(n):
    r = probe(n, 6.0, 1.0)
    assert r["min_sep"] > 5.0                      # separation minimum is 5 NM
    assert 2400 <= r["area_min"] and r["area_max"] <= 3750  # fixed airspace: original area range


def test_density_airspace_scales_the_sector_area():
    r = probe(18, 6.0, 2.0)                        # n / n_ref = 18 / 9
    assert 4800 <= r["area_min"] and r["area_max"] <= 7500 and r["min_sep"] > 5.0


def test_make_env_rejects_extra_arguments_for_other_envs():
    code = "from simulator.bluesky_marl import make_env; make_env('merge', n_agents=3, area_scale=2.0)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode != 0
