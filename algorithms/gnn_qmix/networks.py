"""GNN-QMIX networks: GCN agent and monotonic GAT mixer, plus the baseline networks.

Goals: G05.1 (observation graph), G05.2 (GNN encoder), G05.3 (n-agnostic monotonic mixer),
G03.1 (QMIX / VDN / IQL baselines). Plan: docs/plans/G05.3_gnn_qmix.md §3.2-3.4.
Principle from TransfQMix (arXiv:2301.05334v1): entity-matrix inputs, graph networks whose
parameter count does not depend on the number of entities, mixing weights generated from
per-agent node embeddings. Paper equations: §05.1 Eq. (2)-(3) and the mixer equations.

No layer in GCNAgent or GATMixer has a dimension that depends on the number of agents.
`QMixer` does (CleanMARL's `MixingNetwork` @ 5cf837e, logic unchanged, D-008). `MLPAgent` does
when it reads the whole observation; with `k_nearest > 0` it reads only the own features and
the k nearest other agents, so its input width is fixed and it can be run at any fleet size
(the fixed k-nearest baseline observation, D-027).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OWN = 3        # cos(drift), sin(drift), airspeed            (marl/interfaces/observation.py)
N_PER_OTHER = 7  # dx, dy, dvx, dvy, cos(psi), sin(psi), distance
DIST_INDEX = 6   # position of the distance inside one "other" block
N_NODE = N_OWN + N_PER_OTHER + 1  # + IS_SELF
N_STATE = 7      # x, y, vx, vy, cos(drift), sin(drift), airspeed  (state_mode="entities")


def observation_graph(obs: torch.Tensor, r_detect_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat observation -> one ego graph per agent.

    obs: (B, N_OWN + N_PER_OTHER * (n - 1)). Returns node features (B, n, N_NODE) with the
    self node at index 0, and the adjacency (B, n, n) with self-loops. A neighbor is in the
    graph only if it is within the sensor radius of the observing agent; two neighbors are
    joined if they are within the radius of each other. Nodes outside the radius are isolated
    (zero features, no edges to the self node).
    """
    b = obs.shape[0]
    own = obs[:, :N_OWN]
    others = obs[:, N_OWN:].reshape(b, -1, N_PER_OTHER)
    k = others.shape[1]
    in_range = others[..., DIST_INDEX] <= r_detect_norm  # (B, k)

    nodes = obs.new_zeros(b, k + 1, N_NODE)
    nodes[:, 0, :N_OWN] = own
    nodes[:, 0, -1] = 1.0
    nodes[:, 1:, N_OWN:N_OWN + N_PER_OTHER] = others * in_range.unsqueeze(-1)

    pos = torch.cat([obs.new_zeros(b, 1, 2), others[..., :2]], dim=1)  # self at the origin
    present = torch.cat([in_range.new_ones(b, 1), in_range], dim=1)
    close = torch.cdist(pos, pos) <= r_detect_norm
    adj = close & present.unsqueeze(1) & present.unsqueeze(2)
    adj = adj | torch.eye(k + 1, dtype=torch.bool, device=obs.device)
    return nodes, adj.float()


class GCNAgent(nn.Module):
    """Shared per-agent utility network on the observation graph.

    forward(obs) -> (q, h): q (B, n_actions), h (B, embed_dim) = embedding of the self node.
    """

    def __init__(self, n_actions: int, embed_dim: int = 32, gcn_layers: int = 2, r_detect_norm: float = 0.74):
        super().__init__()
        self.r_detect_norm = r_detect_norm
        self.embed = nn.Linear(N_NODE, embed_dim)
        self.gcn = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(gcn_layers))
        self.q_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_actions))

    def forward(self, obs: torch.Tensor, avail_action: torch.Tensor | None = None):
        nodes, adj = observation_graph(obs, self.r_detect_norm)
        deg_inv_sqrt = adj.sum(-1).clamp(min=1.0).pow(-0.5)
        a_hat = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)  # D^-1/2 (A + I) D^-1/2
        x = F.relu(self.embed(nodes))
        for layer in self.gcn:
            x = F.relu(layer(a_hat @ x))
        h = x[:, 0]
        q = self.q_head(h)
        if avail_action is not None:
            q = q.masked_fill(~avail_action, -1e8)
        return q, h


class GATLayer(nn.Module):
    """Multi-head graph attention on a dense, masked adjacency (Velickovic et al. form)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by gat_heads"
        self.heads, self.d = heads, dim // heads
        self.w = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.d))
        self.a_dst = nn.Parameter(torch.empty(heads, self.d))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        wx = self.w(x).view(b, n, self.heads, self.d)
        e = (wx * self.a_dst).sum(-1).unsqueeze(2) + (wx * self.a_src).sum(-1).unsqueeze(1)  # (B, i, j, H)
        e = F.leaky_relu(e, 0.2).masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", alpha, wx)
        return out.reshape(b, n, self.heads * self.d)


class GATMixer(nn.Module):
    """Monotonic mixer whose weights come from a GAT over the agent graph.

    forward(q, state, h) -> q_tot (B,). q: (B, n) chosen utilities, state: (B, n * N_STATE)
    entity state, h: (B, n, embed_dim) agent embeddings.
    """

    def __init__(self, embed_dim: int = 32, mix_dim: int = 32, gat_layers: int = 2, gat_heads: int = 4,
                 mixer_graph: str = "full", mixer_agg: str = "mean", r_detect_norm: float = 0.74):
        super().__init__()
        if mixer_graph not in ("full", "radius") or mixer_agg not in ("mean", "sum"):
            raise ValueError("mixer_graph must be full|radius and mixer_agg mean|sum")
        self.mixer_graph, self.mixer_agg, self.r_detect_norm = mixer_graph, mixer_agg, r_detect_norm
        self.embed = nn.Linear(N_STATE + embed_dim, embed_dim)
        self.gat = nn.ModuleList(GATLayer(embed_dim, gat_heads) for _ in range(gat_layers))
        self.w1 = nn.Linear(embed_dim, mix_dim)
        self.b1 = nn.Linear(2 * embed_dim, mix_dim)
        self.w2 = nn.Linear(2 * embed_dim, mix_dim)
        self.b2 = nn.Sequential(nn.Linear(2 * embed_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, 1))

    def forward(self, q: torch.Tensor, state: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        b, n = q.shape
        ent = state.reshape(b, n, N_STATE)
        if self.mixer_graph == "full":
            adj = q.new_ones(b, n, n)
        else:
            adj = (torch.cdist(ent[..., :2], ent[..., :2]) <= self.r_detect_norm).float()
        g = F.relu(self.embed(torch.cat([ent, h], dim=-1)))
        for layer in self.gat:
            g = F.elu(layer(g, adj)) + g
        pooled = torch.cat([g.mean(dim=1), g.max(dim=1).values], dim=-1)

        w1 = torch.abs(self.w1(g))                         # (B, n, mix_dim), >= 0
        mixed = (q.unsqueeze(-1) * w1)
        mixed = mixed.mean(dim=1) if self.mixer_agg == "mean" else mixed.sum(dim=1)
        hidden = F.elu(mixed + self.b1(pooled))
        w2 = torch.abs(self.w2(pooled))                    # (B, mix_dim), >= 0
        return (hidden * w2).sum(-1) + self.b2(pooled).squeeze(-1)


# ---------------------------------------------------------------- fixed-n baselines (G03.1)

PAD_DISTANCE = 10.0  # distance entry of a missing neighbor (10 x pos_scale_m = 500 km: far away)


def k_nearest_observation(obs: torch.Tensor, k: int) -> torch.Tensor:
    """Own features + the k nearest other agents, width N_OWN + N_PER_OTHER * k for any n.

    The observation lists the other agents nearest first, so this keeps the first k blocks.
    With fewer than k other agents the missing blocks are zero with distance = PAD_DISTANCE.
    """
    width = N_OWN + N_PER_OTHER * k
    if obs.shape[-1] >= width:
        return obs[..., :width]
    missing = (width - obs.shape[-1]) // N_PER_OTHER
    pad = obs.new_zeros(*obs.shape[:-1], missing, N_PER_OTHER)
    pad[..., DIST_INDEX] = PAD_DISTANCE
    return torch.cat([obs, pad.flatten(-2)], dim=-1)


class MLPAgent(nn.Module):
    """CleanMARL utility network on the flat observation.

    k_nearest = 0: reads the whole observation (width depends on n, so the network is tied to n).
    k_nearest > 0: reads the fixed k-nearest observation and runs at any fleet size.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64, num_layers: int = 1, k_nearest: int = 0):
        super().__init__()
        self.k_nearest = k_nearest
        if k_nearest > 0:
            obs_dim = N_OWN + N_PER_OTHER * k_nearest
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor, avail_action: torch.Tensor | None = None):
        if self.k_nearest > 0:
            obs = k_nearest_observation(obs, self.k_nearest)
        h = self.body(obs)
        q = self.q_head(h)
        if avail_action is not None:
            q = q.masked_fill(~avail_action, -1e8)
        return q, h


class QMixer(nn.Module):
    """CleanMARL `MixingNetwork` (cleanmarl/qmix.py @ 5cf837e). Fixed n_agents and state size."""

    def __init__(self, n_agents: int, s_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.hypernet_weight_1 = nn.Linear(s_dim, n_agents * hidden_dim)
        self.hypernet_bias_1 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_weight_2 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_bias_2 = nn.Sequential(nn.Linear(s_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, q: torch.Tensor, state: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        q = q.reshape(-1, 1, self.n_agents)
        w1 = torch.abs(self.hypernet_weight_1(state)).reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.hypernet_bias_1(state).reshape(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)
        w2 = torch.abs(self.hypernet_weight_2(state)).reshape(-1, self.hidden_dim, 1)
        b2 = self.hypernet_bias_2(state).reshape(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).reshape(-1)


class VDNMixer(nn.Module):
    """Q_tot = sum of the agents' utilities (no parameters)."""

    def forward(self, q: torch.Tensor, state: torch.Tensor | None = None, h: torch.Tensor | None = None) -> torch.Tensor:
        return q.sum(dim=-1)
