"""Property tests for the GNN-QMIX networks (G05.2, G05.3; S-12; plan docs/plans/G05.3_gnn_qmix.md §5).

Pure torch: no BlueSky needed.
"""
import pytest
import torch

from algorithms.gnn_qmix.common import Args, build_networks
from algorithms.gnn_qmix.networks import (
    DIST_INDEX,
    N_OWN,
    N_PER_OTHER,
    N_STATE,
    PAD_DISTANCE,
    GATMixer,
    GCNAgent,
    MLPAgent,
    QMixer,
    k_nearest_observation,
)
from algorithms.gnn_qmix.train import td_loss

N_ACTIONS = 15
R = 0.74  # normalized sensor radius


def make_obs(batch: int, n: int, max_dist: float = 0.7) -> torch.Tensor:
    """Random observations whose distance entry agrees with (dx, dy); all neighbors within `max_dist`."""
    others = torch.randn(batch, n - 1, N_PER_OTHER)
    pos = torch.rand(batch, n - 1, 2) * 2 - 1
    pos = pos / pos.norm(dim=-1, keepdim=True) * torch.rand(batch, n - 1, 1) * max_dist
    others[..., :2] = pos
    others[..., 6] = pos.norm(dim=-1)
    return torch.cat([torch.randn(batch, N_OWN), others.reshape(batch, -1)], dim=1)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_agent_q_is_invariant_to_neighbor_order():
    agent = GCNAgent(N_ACTIONS, r_detect_norm=R)
    obs = make_obs(4, 6)
    others = obs[:, N_OWN:].reshape(4, 5, N_PER_OTHER)
    perm = torch.tensor([3, 0, 4, 1, 2])
    obs_perm = torch.cat([obs[:, :N_OWN], others[:, perm].reshape(4, -1)], dim=1)
    q, h = agent(obs)
    q_perm, h_perm = agent(obs_perm)
    assert torch.allclose(q, q_perm, atol=1e-5) and torch.allclose(h, h_perm, atol=1e-5)


def test_agents_outside_sensor_radius_do_not_change_q():
    agent = GCNAgent(N_ACTIONS, r_detect_norm=R)
    obs = make_obs(4, 4)
    far = torch.randn(4, 2, N_PER_OTHER)
    far[..., :2] = torch.tensor([3.0, 4.0])
    far[..., 6] = 5.0  # well outside R
    q_small, _ = agent(obs)
    q_big, _ = agent(torch.cat([obs, far.reshape(4, -1)], dim=1))  # n = 4 -> n = 6
    assert torch.allclose(q_small, q_big, atol=1e-5)


def test_neighbor_inside_radius_changes_q():
    agent = GCNAgent(N_ACTIONS, r_detect_norm=R)
    obs = make_obs(4, 3)
    other = obs.clone()
    other[:, N_OWN + 2] += 1.0  # relative velocity of the nearest neighbor
    assert not torch.allclose(agent(obs)[0], agent(other)[0], atol=1e-5)


@pytest.mark.parametrize("n", [3, 9, 50])
def test_one_parameter_set_runs_at_any_fleet_size(n):
    args = Args(algo="gnn_qmix")
    agent, mixer = build_networks(args, obs_dim=0, state_dim=0, n_agents=0, n_actions=N_ACTIONS)
    b = 5
    q, h = agent(make_obs(b * n, n))
    assert q.shape == (b * n, N_ACTIONS)
    q_tot = mixer(q.max(-1).values.reshape(b, n), torch.randn(b, n * N_STATE), h.reshape(b, n, -1))
    assert q_tot.shape == (b,)
    q_tot.sum().backward()  # backward works at this n


def test_parameter_shapes_do_not_depend_on_n():
    args = Args(algo="gnn_qmix")
    shapes = []
    for n in (3, 50):
        agent, mixer = build_networks(args, obs_dim=N_OWN + N_PER_OTHER * (n - 1), state_dim=n * N_STATE,
                                      n_agents=n, n_actions=N_ACTIONS)
        shapes.append({k: tuple(v.shape) for net in (agent, mixer) for k, v in net.state_dict().items()})
    assert shapes[0] == shapes[1]


@pytest.mark.parametrize("graph", ["full", "radius"])
def test_mixer_is_invariant_to_agent_permutation(graph):
    mixer = GATMixer(mixer_graph=graph, r_detect_norm=R)
    b, n = 4, 6
    q, state, h = torch.randn(b, n), torch.randn(b, n, N_STATE), torch.randn(b, n, 32)
    perm = torch.randperm(n)
    out = mixer(q, state.reshape(b, -1), h)
    out_perm = mixer(q[:, perm], state[:, perm].reshape(b, -1), h[:, perm])
    assert torch.allclose(out, out_perm, atol=1e-5)


@pytest.mark.parametrize("agg", ["mean", "sum"])
def test_mixer_is_monotonic_in_every_agent_utility(agg):
    mixer = GATMixer(mixer_agg=agg, r_detect_norm=R)
    for n in (3, 9, 20):
        q = torch.randn(8, n, requires_grad=True)
        q_tot = mixer(q, torch.randn(8, n * N_STATE), torch.randn(8, n, 32))
        (grad,) = torch.autograd.grad(q_tot.sum(), q)
        assert (grad >= 0).all()


def test_qmix_baseline_is_tied_to_its_fleet_size():
    mixer = QMixer(n_agents=3, s_dim=3 * N_STATE)
    assert mixer(torch.randn(2, 3), torch.randn(2, 3 * N_STATE)).shape == (2,)
    with pytest.raises(RuntimeError):
        mixer(torch.randn(2, 6), torch.randn(2, 6 * N_STATE))


def test_qmix_airspeed_div_rescales_only_the_airspeed_state_entry():
    """GAP-30 / D-032: airspeed_div = k gives the mixer the state with every entity's last entry divided by k."""
    n, b = 3, 5
    torch.manual_seed(0)
    plain = QMixer(n_agents=n, s_dim=n * N_STATE)
    torch.manual_seed(0)
    scaled = QMixer(n_agents=n, s_dim=n * N_STATE, airspeed_div=10.0)
    assert set(plain.state_dict()) == set(scaled.state_dict())  # same checkpoint format
    q, state = torch.randn(b, n), torch.randn(b, n * N_STATE)
    by_hand = state.reshape(b, n, N_STATE).clone()
    by_hand[..., -1] /= 10.0
    assert torch.allclose(scaled(q, state), plain(q, by_hand.reshape(b, -1)), atol=1e-6)
    assert not torch.allclose(scaled(q, state), plain(q, state))
    with pytest.raises(ValueError):
        QMixer(n_agents=n, s_dim=n * N_STATE + 1, airspeed_div=10.0)


def test_qmix_config_sets_the_stabilizing_options_and_other_configs_do_not():
    import json
    from pathlib import Path

    configs = Path(__file__).resolve().parents[2] / "configs"
    qmix = json.loads((configs / "qmix_sector_cr.json").read_text())
    assert qmix["qmix_airspeed_div"] == 10.0 and qmix["polyak"] == 0.001
    for name in ("gnn_qmix", "vdn", "iql", "ablation_vdn_gcn"):
        other = json.loads((configs / f"{name}_sector_cr.json").read_text())
        assert "qmix_airspeed_div" not in other and "polyak" not in other
    assert Args().qmix_airspeed_div == 1.0 and Args().polyak == 0.01


@pytest.mark.parametrize("algo", ["gnn_qmix", "qmix", "vdn", "iql"])
def test_td_loss_backpropagates_for_every_algorithm(algo):
    import copy

    n, b = 3, 16
    args = Args(algo=algo, train_n=(n,))
    args.validate()
    agent, mixer = build_networks(args, N_OWN + N_PER_OTHER * (n - 1), n * N_STATE, n, N_ACTIONS)
    batch = (make_obs(b * n, n).reshape(b, n, -1), torch.randn(b, n * N_STATE),
             torch.randint(0, N_ACTIONS, (b, n)), torch.randn(b), make_obs(b * n, n).reshape(b, n, -1),
             torch.randn(b, n * N_STATE), torch.zeros(b))
    loss = td_loss(args, agent, mixer, copy.deepcopy(agent), copy.deepcopy(mixer), batch)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in agent.parameters())


def test_k_nearest_observation_has_fixed_width_and_pads_missing_neighbors():
    k, width = 3, N_OWN + N_PER_OTHER * 3
    big = make_obs(4, 9)
    assert torch.equal(k_nearest_observation(big, k), big[:, :width])  # nearest-first: keep the first k blocks
    small = make_obs(4, 3)  # only 2 other agents
    out = k_nearest_observation(small, k)
    assert out.shape == (4, width) and torch.equal(out[:, :small.shape[1]], small)
    pad = out[:, small.shape[1]:]
    assert (pad[:, DIST_INDEX] == PAD_DISTANCE).all() and pad.abs().sum() == 4 * PAD_DISTANCE


@pytest.mark.parametrize("n", [3, 9, 50])
def test_k_nearest_mlp_agent_runs_at_any_fleet_size(n):
    agent = MLPAgent(obs_dim=999, n_actions=N_ACTIONS, k_nearest=3)  # obs_dim is ignored with k_nearest
    q, _ = agent(make_obs(5 * n, n))
    assert q.shape == (5 * n, N_ACTIONS)


def test_which_methods_train_jointly_and_run_zero_shot():
    vdn_k = Args(algo="vdn", baseline_k_nearest=3, train_n=(3, 6, 9))
    vdn_k.validate()
    assert vdn_k.agent_runs_at_any_n()
    qmix_k = Args(algo="qmix", baseline_k_nearest=3, train_n=(3,))
    qmix_k.validate()
    assert qmix_k.agent_runs_at_any_n() and not qmix_k.is_n_agnostic()  # mixer is fixed-n, agent is not
    assert not Args(algo="vdn", train_n=(3,)).agent_runs_at_any_n()
    with pytest.raises(ValueError):
        Args(algo="vdn", train_n=(3, 6, 9)).validate()  # full-observation MLP is tied to one n


def test_fixed_n_algorithms_reject_several_train_sizes():
    with pytest.raises(ValueError):
        Args(algo="qmix", train_n=(3, 6, 9)).validate()
    Args(algo="vdn", agent_net="gcn", train_n=(3, 6, 9)).validate()  # GCN agent + VDN is n-agnostic
