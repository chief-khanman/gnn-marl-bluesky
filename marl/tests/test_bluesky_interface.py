"""Mimics exactly what cleanmarl/qmix.py does with an env, using marlbench's own wrappers.
Run each test in its own process (BlueSky singleton):  pytest --forked  or  python -m pytest -p no:cacheprovider marl/tests -k <name>
"""
import numpy as np


def test_qmix_style_episode():
    from marlbench.wrappers import AddAgentID, RecordEpisodeStatistics, TimeLimit
    from marl.interfaces.bluesky_interface import BlueSkyInterface

    env = AddAgentID(RecordEpisodeStatistics(TimeLimit(BlueSkyInterface("sector_cr", n_agents=5), 100)))
    obs, _ = env.reset(seed=1)
    n, s = env.n_agents, env.get_state_size()
    assert obs.shape[0] == n and env.get_avail_actions().shape == (n, env.get_action_size())
    done = trunc = False
    steps = 0
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(env.sample())
        assert obs.shape[0] == n and env.get_state().shape == (s,) and isinstance(r, float)
        steps += 1
    assert steps <= 100 and "episode_stats" in info
