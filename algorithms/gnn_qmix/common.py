"""Shared pieces of the GNN-QMIX training and evaluation scripts (goals G05.4, G06.3; S-10).

Config dataclass, env factories (one BlueSky env per process, S-07), network construction,
run-directory provenance, CSV logging and the greedy evaluation loop.
Plan: docs/plans/G05.3_gnn_qmix.md.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from algorithms.gnn_qmix.networks import GATMixer, GCNAgent, MLPAgent, QMixer, VDNMixer
from marl.metrics import EPISODE_METRICS, NM_TO_M

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGOS = ("gnn_qmix", "qmix", "vdn", "iql")
AIRSPACES = ("fixed", "density")
GOAL_ID = {"gnn_qmix": "G05.3", "qmix": "G03.1", "vdn": "G03.1", "iql": "G03.1"}
# Fields that do not change what a run computes; left out of config_sha.
NON_SEMANTIC = ("device", "runs_dir", "machine", "num_eval_envs", "log_every", "checkpoint_every", "log_diagnostics")


@dataclass
class Args:
    # What to run
    algo: str = "gnn_qmix"
    """gnn_qmix (GCN agent + GAT mixer) or a fixed-n baseline: qmix, vdn, iql"""
    agent_net: str = "auto"
    """auto (gcn for gnn_qmix, mlp for baselines), gcn, or mlp. gcn + vdn/iql is an ablation"""
    env_name: str = "sector_cr"
    """sector_cr or merge"""
    train_n: tuple[int, ...] = (3, 6, 9)
    """Training fleet sizes, trained jointly (D-017). The QMIX mixer and an MLP agent without k-nearest take exactly one"""
    baseline_k_nearest: int = 0
    """MLP agent input: 0 = all other agents (width depends on n), k > 0 = the k nearest (fixed width, D-027)"""
    seed: int = 1
    # Environment
    max_episode_steps: int = 150
    """Control steps per episode (one step = 5 s of simulated time)"""
    team_reward: str = "mean"
    """mean or sum of the per-agent rewards"""
    state_mode: str = "entities"
    """entities (n x 7 entity matrix) or concat_obs"""
    spawn_min_sep_nm: float = 6.0
    """sector_cr: no two aircraft are created closer than this (D-028). 0 = original spawning"""
    airspace: str = "fixed"
    """sector_cr training airspace: fixed (original sector size) or density (area grows with n, D-029)"""
    density_ref_n: int = 9
    """In the density airspace the sector area is the original area times n / density_ref_n"""
    r_detect_nm: float = 20.0
    """Sensor radius: who is in an agent's observation graph (Q-10)"""
    pos_scale_m: float = 50_000.0
    vel_scale_mps: float = 150.0
    # Networks
    embed_dim: int = 32
    gcn_layers: int = 2
    gat_layers: int = 2
    gat_heads: int = 4
    mix_dim: int = 32
    mixer_graph: str = "full"
    """full or radius"""
    mixer_agg: str = "mean"
    """mean or sum over agents inside the mixer"""
    hidden_dim: int = 64
    """Width of the baseline MLP agent and of the QMIX mixer"""
    qmix_airspeed_div: float = 1.0
    """QMIX mixer only: divide the airspeed entry of the entity state by this (GAP-30, D-032). 1.0 = unchanged"""
    # Training
    steps_per_size: int = 2_000_000
    """Environment steps PER training fleet size (D-030). Every step count below is also per fleet size"""
    learning_rate: float = 5e-4
    gamma: float = 0.99
    polyak: float = 0.01
    batch_size: int = 256
    """Transitions per fleet size per update"""
    buffer_size: int = 100_000
    """Transitions kept per fleet size"""
    train_every: int = 8
    """One update (one batch per fleet size) every this many steps"""
    learning_starts: int = 5_000
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_steps: int = 100_000
    clip_gradients: float = 10.0
    double_q: bool = False
    """Next action = argmax of the online agent, valued by the target agent (GAP-31). False = max of the target agent"""
    lr_final_frac: float = 1.0
    """The learning rate falls linearly from learning_rate at step 0 to learning_rate * this at steps_per_size (GAP-31)"""
    obs_airspeed_div: float = 1.0
    """Every agent network divides its own-airspeed observation entry by this (GAP-31, GAP-30). 1.0 = unchanged"""
    # Evaluation and logging
    eval_every: int = 20_000
    num_eval_ep: int = 10
    """Greedy evaluation episodes per fleet size"""
    num_eval_envs: int = 2
    """Evaluation processes per fleet size"""
    log_every: int = 2_000
    checkpoint_every: int = 0
    """> 0: also keep checkpoints/step_<step>.pt at every evaluation whose step is a multiple of this (GAP-31)"""
    log_diagnostics: bool = False
    """Write lr, grad_norm, q_taken_mean, q_tot_mean, td_target_mean and the mixer gain at every log interval (GAP-31)"""
    device: str = "cpu"
    runs_dir: str = ""
    """Default: $GNN_MARL_RUNS_DIR, else ./runs"""
    machine: str = ""
    """Default: $GNN_MARL_MACHINE, else the host name"""

    def validate(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        if self.agent_kind() not in ("gcn", "mlp"):
            raise ValueError("agent_net must be auto, gcn or mlp")
        if self.airspace not in AIRSPACES:
            raise ValueError(f"airspace must be one of {AIRSPACES}")
        if not self.is_n_agnostic() and len(self.train_n) != 1:
            raise ValueError(f"{self.algo} with this agent has fixed-n networks: give one train_n")
        if self.checkpoint_every < 0 or (self.checkpoint_every and self.checkpoint_every % self.eval_every):
            raise ValueError("checkpoint_every must be 0 or a multiple of eval_every")

    def agent_kind(self) -> str:
        if self.agent_net == "auto":
            return "gcn" if self.algo == "gnn_qmix" else "mlp"
        return self.agent_net

    def agent_runs_at_any_n(self) -> bool:
        """Decentralized execution at an unseen fleet size needs only the agent network."""
        return self.agent_kind() == "gcn" or self.baseline_k_nearest > 0

    def is_n_agnostic(self) -> bool:
        """Can be TRAINED on several fleet sizes at once (the QMIX mixer cannot)."""
        return self.agent_runs_at_any_n() and self.algo != "qmix"

    def r_detect_norm(self) -> float:
        return self.r_detect_nm * NM_TO_M / self.pos_scale_m


def load_dotenv() -> None:
    """Read KEY=VALUE lines of the repo's .env (docs/HANDOFF.md) without overriding the environment."""
    path = REPO_ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if value.strip():
                    os.environ.setdefault(key.strip(), value.strip())


def args_from_dict(d: dict) -> Args:
    known = {k: v for k, v in d.items() if k in Args.__dataclass_fields__}
    if "train_n" in known:
        known["train_n"] = tuple(known["train_n"])
    return Args(**known)


# ------------------------------------------------------------------ environments

def env_kwargs_for(args: Args, n_agents: int, airspace: str | None = None) -> dict | None:
    """Spawn separation and airspace scaling for sector_cr (simulator/bluesky_marl/custom_envs.py)."""
    if args.env_name != "sector_cr":
        return None
    airspace = airspace or args.airspace
    scale = n_agents / args.density_ref_n if airspace == "density" else 1.0
    return {"spawn_min_sep_nm": args.spawn_min_sep_nm, "area_scale": scale}


def make_env_fn(args: Args, n_agents: int, airspace: str | None = None):
    """Factory for one BlueSky env. Call it inside a fresh process only (S-07)."""
    cfg = asdict(args)
    env_kwargs = env_kwargs_for(args, n_agents, airspace)

    def env_fn():
        from marlbench.wrappers import TimeLimit  # noqa: PLC0415

        from marl.interfaces.bluesky_interface import BlueSkyInterface  # noqa: PLC0415
        from marl.interfaces.observation import ObsConfig  # noqa: PLC0415
        from marl.metrics import make_episode_metrics_wrapper  # noqa: PLC0415

        env = BlueSkyInterface(
            env_name=cfg["env_name"], n_agents=n_agents, team_reward=cfg["team_reward"], obs_mode="all_agents",
            obs_config=ObsConfig(pos_scale_m=cfg["pos_scale_m"], vel_scale_mps=cfg["vel_scale_mps"]),
            state_mode=cfg["state_mode"], env_kwargs=env_kwargs,
        )
        return make_episode_metrics_wrapper(TimeLimit(env=env, max_episode_steps=cfg["max_episode_steps"]))

    return env_fn


def make_vec_env(args: Args, n_agents: int, n_envs: int, auto_reset: bool, airspace: str | None = None):
    from marlbench.vec_envs import SubprocVectorEnv  # noqa: PLC0415

    return SubprocVectorEnv([make_env_fn(args, n_agents, airspace) for _ in range(n_envs)], auto_reset=auto_reset)


# ------------------------------------------------------------------ networks

def build_networks(args: Args, obs_dim: int, state_dim: int, n_agents: int, n_actions: int):
    """Returns (agent, mixer). mixer is None for IQL. obs_dim/state_dim/n_agents matter only for fixed-n networks."""
    if args.agent_kind() == "gcn":
        agent = GCNAgent(n_actions, args.embed_dim, args.gcn_layers, args.r_detect_norm(), args.obs_airspeed_div)
    else:
        agent = MLPAgent(obs_dim, n_actions, args.hidden_dim, k_nearest=args.baseline_k_nearest,
                         obs_airspeed_div=args.obs_airspeed_div)
    if args.algo == "gnn_qmix":
        if args.state_mode != "entities":
            raise ValueError("the GAT mixer needs state_mode='entities'")
        mixer = GATMixer(args.embed_dim, args.mix_dim, args.gat_layers, args.gat_heads, args.mixer_graph,
                         args.mixer_agg, args.r_detect_norm())
    elif args.algo == "qmix":
        mixer = QMixer(n_agents, state_dim, args.hidden_dim, args.qmix_airspeed_div)
    elif args.algo == "vdn":
        mixer = VDNMixer()
    else:
        mixer = None
    return agent, mixer


# ------------------------------------------------------------------ provenance and logging

def method_name(args: Args) -> str:
    """Algorithm plus the non-default agent input, e.g. gnn_qmix, vdn-k3, vdn-gcn, qmix."""
    if args.agent_net != "auto":
        return f"{args.algo}-{args.agent_kind()}"
    if args.agent_kind() == "mlp" and args.baseline_k_nearest > 0:
        return f"{args.algo}-k{args.baseline_k_nearest}"
    return args.algo


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_sha(args: Args) -> str:
    semantic = {k: v for k, v in asdict(args).items() if k not in NON_SEMANTIC}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def create_run_dir(args: Args) -> Path:
    """<runs>/<goal>_<algo>_<env>_n<train-n>_s<seed>_<YYYYMMDD-HHMM>_<machine>/ with config.json (S-10, D-014)."""
    load_dotenv()
    runs = Path(args.runs_dir or os.environ.get("GNN_MARL_RUNS_DIR") or "runs")
    machine = args.machine or os.environ.get("GNN_MARL_MACHINE") or socket.gethostname().split(".")[0]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    algo = method_name(args)
    run_id = (f"{GOAL_ID[args.algo]}_{algo}_{args.env_name}_n{'-'.join(map(str, args.train_n))}"
              f"_s{args.seed}_{stamp}_{machine}")
    run_dir = runs / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    meta = {
        "run_id": run_id,
        "method": algo,
        "goal_ids": [GOAL_ID[args.algo], "G05.4", "G06.4"],
        "config_sha": config_sha(args),
        "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        "submodules": _git("submodule", "status"),
        "machine": machine,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "args": asdict(args),
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    return run_dir


class MetricsCsv:
    """Long-format metrics file: step, split, n_agents, metric, value, n_episodes."""

    FIELDS = ("step", "split", "n_agents", "metric", "value", "n_episodes")

    def __init__(self, path: Path):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def write(self, step: int, split: str, n_agents: int, values: dict[str, float], n_episodes: int = 0) -> None:
        for metric, value in values.items():
            self._writer.writerow([step, split, n_agents, metric, f"{value:.6g}", n_episodes])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def mean_metrics(episodes: list[dict[str, float]]) -> dict[str, float]:
    return {m: float(np.mean([e[m] for e in episodes])) for m in EPISODE_METRICS}


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate_policy(agent, vec_env, episodes: int, seed: int, device: torch.device) -> list[dict[str, float]]:
    """Greedy, decentralized rollouts. Returns one metrics dict per finished episode."""
    results: list[dict[str, float]] = []
    round_idx = 0
    while len(results) < episodes:
        obs, _ = vec_env.reset(seed=seed + 1000 * round_idx)
        round_idx += 1
        while vec_env.get_env_mask().any():
            active = vec_env.get_env_mask()
            b, n, d = obs.shape
            q, _ = agent(torch.from_numpy(obs).float().to(device).reshape(b * n, d))
            actions = q.argmax(dim=-1).reshape(b, n).cpu().numpy()
            obs, _, done, truncated, infos = vec_env.step(actions)
            for i in np.nonzero(active & (done | truncated))[0]:
                results.append(infos[i]["episode_metrics"])
    return results[:episodes]
