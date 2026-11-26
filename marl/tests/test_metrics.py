"""Known-answer tests for the paper metrics (G06.2; plan docs/plans/G05.3_gnn_qmix.md §3.5).

The BlueSky check runs in a subprocess (S-07, S-12).
"""
import json
import subprocess
import sys

import numpy as np

from marl.metrics import EPISODE_METRICS, EpisodeAccumulator, pairwise_distance_nm, step_safety_metrics


def test_pairwise_distance_one_degree_of_latitude_is_60_nm():
    d = pairwise_distance_nm(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert d.shape == (2, 2) and d[0, 0] == 0
    assert abs(d[0, 1] - 60.04) < 0.05 and d[0, 1] == d[1, 0]


def test_step_metrics_count_pairs_once_and_onsets_only_on_entry():
    far = np.array([[0.0, 10.0, 20.0], [10.0, 0.0, 10.0], [20.0, 10.0, 0.0]])
    close = far.copy()
    close[0, 1] = close[1, 0] = 4.0
    m0, los0 = step_safety_metrics(far, None)
    assert m0 == {"los_pairs": 0.0, "los_onsets": 0.0, "min_sep_nm": 10.0}
    m1, los1 = step_safety_metrics(close, los0)
    assert m1 == {"los_pairs": 1.0, "los_onsets": 1.0, "min_sep_nm": 4.0}
    m2, _ = step_safety_metrics(close, los1)  # still in LoS: counted as a pair-step, not a new event
    assert m2["los_pairs"] == 1.0 and m2["los_onsets"] == 0.0


def test_episode_accumulator_known_answers():
    acc = EpisodeAccumulator(n_agents=3)  # 3 pairs
    acc.add(-1.0, 0, 0, 12.0, np.radians(10))
    acc.add(-2.0, 1, 1, 4.0, np.radians(30))
    acc.add(-1.0, 2, 1, 3.0, np.radians(20))
    acc.add(0.0, 0, 0, 8.0, np.radians(20))
    r = acc.result()
    assert set(r) == set(EPISODE_METRICS)
    assert r["ep_return_per_agent"] == -4.0 and r["ep_length"] == 4
    assert r["los_pair_steps"] == 3 and abs(r["los_rate"] - 3 / (4 * 3)) < 1e-12
    assert r["los_events"] == 2 and abs(r["los_events_per_agent"] - 2 / 3) < 1e-12
    assert r["conflict_free"] == 0.0 and r["min_sep_nm"] == 3.0
    assert abs(r["mean_abs_drift_deg"] - 20.0) < 1e-9
    acc.reset()
    acc.add(-1.0, 0, 0, 9.0, 0.0)
    assert acc.result()["conflict_free"] == 1.0


_PROBE = """
import json, numpy as np
from marlbench.wrappers import TimeLimit
from marl.interfaces.bluesky_interface import BlueSkyInterface, N_STATE
from marl.metrics import make_episode_metrics_wrapper
n = 6
env = make_episode_metrics_wrapper(TimeLimit(env=BlueSkyInterface("sector_cr", n_agents=n, team_reward="mean",
                                                                  state_mode="entities"), max_episode_steps=12))
env.reset(seed=3)
state = env.get_state()
steps, info, done = 0, {}, False
while not done:
    _, _, term, trunc, info = env.step(env.sample())
    steps += 1
    done = term or trunc
ent = state.reshape(n, N_STATE)
print(json.dumps({"state_size": env.get_state_size(), "state_len": int(state.size), "steps": steps,
                  "centroid": float(np.abs(ent[:, :2].mean(axis=0)).max()),
                  "step_keys": sorted(k for k in info if k in ("los_pairs", "los_onsets", "min_sep_nm", "mean_abs_drift_rad")),
                  "episode": info["episode_metrics"]}))
"""


def test_adapter_entity_state_and_episode_metrics_in_bluesky():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    r = json.loads(out.strip().splitlines()[-1])
    assert r["state_size"] == r["state_len"] == 6 * 7
    assert r["centroid"] < 1e-5  # positions are relative to the fleet centroid
    assert r["step_keys"] == ["los_onsets", "los_pairs", "mean_abs_drift_rad", "min_sep_nm"]
    assert r["steps"] == 12 and r["episode"]["ep_length"] == 12
    assert set(r["episode"]) == set(EPISODE_METRICS)
    assert 0.0 <= r["episode"]["los_rate"] <= 1.0 and r["episode"]["min_sep_nm"] >= 0.0
