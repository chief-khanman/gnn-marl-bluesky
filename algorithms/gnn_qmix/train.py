"""Train GNN-QMIX or a baseline (QMIX, VDN, IQL) on BlueSky.

Goals: G05.3, G05.4 (joint training over fleet sizes), G03.1, G06.4. Plan: docs/plans/G05.3_gnn_qmix.md.
Runbook: docs/RUNBOOK.md.

    python -m algorithms.gnn_qmix.train --config configs/gnn_qmix_sector_cr.json --seed 1
    python -m algorithms.gnn_qmix.train --config configs/qmix_sector_cr.json --train-n 6 --seed 1

Budget (D-030): `steps_per_size` environment steps for EACH training fleet size, so a jointly
trained run and a single-size run see the same amount of experience at every fleet size. The
`step` written to the logs is the per-fleet-size step count.

Every flag of `Args` (algorithms/gnn_qmix/common.py) can be given on the command line and
overrides the config file. One BlueSky env per process (S-07): one collector process per
training fleet size, plus `num_eval_envs` evaluation processes per fleet size.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from algorithms.gnn_qmix.common import (
    Args,
    MetricsCsv,
    args_from_dict,
    build_networks,
    create_run_dir,
    evaluate_policy,
    make_vec_env,
    mean_metrics,
)


class TransitionBuffer:
    """Uniform replay of single transitions for one fleet size."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, n_agents), dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, state, action, reward, next_obs, next_state, terminated) -> None:
        i = self.pos
        self.obs[i], self.state[i], self.action[i], self.reward[i] = obs, state, action, reward
        self.next_obs[i], self.next_state[i], self.terminated[i] = next_obs, next_state, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.from_numpy(a[idx]).to(device) for a in
                     (self.obs, self.state, self.action, self.reward, self.next_obs, self.next_state, self.terminated))


@torch.no_grad()
def next_values(args: Args, agent, target_agent, next_obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap value of every agent's next observation, shape (B * n,), and the target agent's embedding.

    double_q = False: max over the target agent's Q. True: the online agent picks the action and the
    target agent values it (GAP-31). The embedding always comes from the target agent.
    """
    target_q, h_next = target_agent(next_obs_flat)
    if not args.double_q:
        return target_q.max(dim=-1).values, h_next
    best = agent(next_obs_flat)[0].argmax(dim=-1, keepdim=True)
    return target_q.gather(-1, best).squeeze(-1), h_next


def _td_terms(args: Args, agent, mixer, target_agent, target_mixer, batch):
    """(prediction, target, q_taken): what the TD loss compares, plus the chosen utilities (B, n)."""
    obs, state, action, reward, next_obs, next_state, terminated = batch
    b, n, d = obs.shape
    q, h = agent(obs.reshape(b * n, d))
    q_taken = q.gather(-1, action.reshape(b * n, 1)).reshape(b, n)
    q_next, h_next = next_values(args, agent, target_agent, next_obs.reshape(b * n, d))
    q_next = q_next.reshape(b, n)
    if mixer is None:  # IQL: every agent regresses on the team reward
        target = reward.unsqueeze(-1) + args.gamma * (1 - terminated).unsqueeze(-1) * q_next
        return q_taken, target, q_taken
    with torch.no_grad():
        target_tot = target_mixer(q_next, next_state, h_next.reshape(b, n, -1))
        target = reward + args.gamma * (1 - terminated) * target_tot
    h = h.reshape(b, n, -1)
    q_tot = mixer(q_taken, state, h.detach() if args.mixer_detach_h else h)
    return q_tot, target, q_taken


def td_loss(args: Args, agent, mixer, target_agent, target_mixer, batch) -> torch.Tensor:
    prediction, target, _ = _td_terms(args, agent, mixer, target_agent, target_mixer, batch)
    return F.mse_loss(prediction, target)


def lr_at(args: Args, step: int) -> float:
    """Learning rate at a per-fleet-size step: linear from learning_rate to learning_rate * lr_final_frac."""
    progress = min(step, args.steps_per_size) / args.steps_per_size
    return args.learning_rate * (1 - (1 - args.lr_final_frac) * progress)


def td_diagnostics(args: Args, agent, mixer, target_agent, target_mixer, batch) -> dict[str, float]:
    """Value scales on one batch, no optimizer step (GAP-31).

    mixer_gain_* = dQ_tot / dQ_i over batch and agents, for a mixer with parameters. With
    mixer_agg = "mean" it is multiplied by n, so mixing that behaves like a plain sum reads 1.0 at every n.
    """
    with torch.no_grad():
        prediction, target, q_taken = _td_terms(args, agent, mixer, target_agent, target_mixer, batch)
    out = {"q_taken_mean": float(q_taken.mean()), "q_tot_mean": float(prediction.mean()),
           "td_target_mean": float(target.mean())}
    if mixer is not None and any(True for _ in mixer.parameters()):
        obs, state = batch[0], batch[1]
        b, n, d = obs.shape
        with torch.no_grad():
            h = agent(obs.reshape(b * n, d))[1].reshape(b, n, -1)
        q_in = q_taken.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            gain = torch.autograd.grad(mixer(q_in, state, h).sum(), q_in)[0]
        if args.algo == "gnn_qmix" and args.mixer_agg == "mean":
            gain = gain * n
        out["mixer_gain_mean"], out["mixer_gain_max"] = float(gain.mean()), float(gain.max())
    return out


def soft_update(target, source, polyak: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - polyak).add_(polyak * p)


def parse_args(argv: list[str] | None = None) -> Args:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, rest = pre.parse_known_args(argv)
    default = Args()
    if known.config:
        default = args_from_dict(json.loads(Path(known.config).read_text()))
    args = tyro.cli(Args, default=default, args=rest)
    args.validate()
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = create_run_dir(args)
    print(f"run_dir: {run_dir}", flush=True)

    sizes = list(args.train_n)
    train_envs = {n: make_vec_env(args, n, 1, auto_reset=True) for n in sizes}
    eval_envs = {n: make_vec_env(args, n, args.num_eval_envs, auto_reset=False) for n in sizes}
    n0 = sizes[0]
    n_actions = train_envs[n0].get_action_size()
    agent, mixer = build_networks(args, train_envs[n0].get_obs_size(), train_envs[n0].get_state_size(), n0, n_actions)
    agent.to(device)
    target_agent = copy.deepcopy(agent)
    params = list(agent.parameters())
    target_mixer = None
    if mixer is not None:
        mixer.to(device)
        target_mixer = copy.deepcopy(mixer)
        params += list(mixer.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    buffers = {n: TransitionBuffer(args.buffer_size, n, train_envs[n].get_obs_size(), train_envs[n].get_state_size())
               for n in sizes}

    csv_log = MetricsCsv(run_dir / "metrics.csv")
    writer = SummaryWriter(str(run_dir / "tb"))
    n_params = sum(p.numel() for p in params)
    print(f"algo={args.algo} agent={args.agent_kind()} train_n={sizes} parameters={n_params}", flush=True)

    def save(name: str, step: int) -> None:
        torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict() if mixer is not None else None,
                    "step": step, "n_actions": n_actions}, run_dir / "checkpoints" / name)

    def run_eval(step: int) -> None:
        for n in sizes:
            episodes = evaluate_policy(agent, eval_envs[n], args.num_eval_ep, 100_000 + args.seed, device)
            values = mean_metrics(episodes)
            csv_log.write(step, "eval", n, values, len(episodes))
            for metric, value in values.items():
                writer.add_scalar(f"eval/{metric}/n{n}", value, step)
            print(f"[eval] step={step} n={n} return/agent={values['ep_return_per_agent']:.3f} "
                  f"los_rate={values['los_rate']:.4f} conflict_free={values['conflict_free']:.2f}", flush=True)
        save("latest.pt", step)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save(f"step_{step:07d}.pt", step)

    obs, state = {}, {}
    for i, n in enumerate(sizes):
        o, _ = train_envs[n].reset(seed=args.seed + 10_000 * i)
        obs[n], state[n] = o[0], train_envs[n].get_state()[0]
    finished = {n: [] for n in sizes}
    losses: list[float] = []
    grad_norms: list[float] = []
    step, next_eval, next_log, next_train = 0, 0, args.log_every, args.learning_starts
    t_start = t_log = time.time()
    steps_at_log = 0

    while step < args.steps_per_size:  # `step` counts environment steps per fleet size
        if step >= next_eval:
            run_eval(step)
            next_eval += args.eval_every
        epsilon = max(args.end_e, args.start_e + (args.end_e - args.start_e) * step / max(args.exploration_steps, 1))
        for n in sizes:
            with torch.no_grad():
                q, _ = agent(torch.from_numpy(obs[n]).float().to(device))
            action = q.argmax(dim=-1).cpu().numpy()
            explore = np.random.random(n) < epsilon
            action = np.where(explore, np.random.randint(0, n_actions, size=n), action)
            next_obs, reward, done, truncated, infos = train_envs[n].step(action[None])
            info = infos[0]
            ended = bool(done[0] or truncated[0])
            next_o = info["final_obs"] if ended else next_obs[0]
            next_s = info["final_state"] if ended else train_envs[n].get_state()[0]
            buffers[n].add(obs[n], state[n], action, reward[0], next_o, next_s, bool(done[0]))
            if ended:
                finished[n].append(info["final_info"]["episode_metrics"])
            obs[n], state[n] = next_obs[0], train_envs[n].get_state()[0]
        step += 1

        if step >= next_train and all(buffers[n].size >= args.batch_size for n in sizes):
            next_train += args.train_every
            loss = sum(td_loss(args, agent, mixer, target_agent, target_mixer,
                               buffers[n].sample(args.batch_size, device)) for n in sizes) / len(sizes)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_gradients > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_gradients)
            elif args.log_diagnostics:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))  # measures, does not clip
            if args.log_diagnostics:
                grad_norms.append(float(grad_norm))
            for group in optimizer.param_groups:
                group["lr"] = lr_at(args, step)
            optimizer.step()
            soft_update(target_agent, agent, args.polyak)
            if mixer is not None:
                soft_update(target_mixer, mixer, args.polyak)
            losses.append(float(loss))

        if step >= next_log:
            next_log += args.log_every
            now = time.time()
            sps = (step - steps_at_log) * len(sizes) / max(now - t_log, 1e-9)
            t_log, steps_at_log = now, step
            for n in sizes:
                if finished[n]:
                    values = mean_metrics(finished[n])
                    csv_log.write(step, "train", n, values, len(finished[n]))
                    for metric, value in values.items():
                        writer.add_scalar(f"rollout/{metric}/n{n}", value, step)
                    finished[n] = []
            train_values = {"epsilon": epsilon, "steps_per_s": sps}
            if losses:
                train_values["loss"] = float(np.mean(losses))
                losses = []
            if args.log_diagnostics:
                train_values["lr"] = lr_at(args, step)
                if grad_norms:
                    train_values["grad_norm"] = float(np.mean(grad_norms))
                    grad_norms = []
                if all(buffers[n].size >= args.batch_size for n in sizes):
                    per_size = [td_diagnostics(args, agent, mixer, target_agent, target_mixer,
                                               buffers[n].sample(args.batch_size, device)) for n in sizes]
                    train_values.update({k: float(np.mean([d[k] for d in per_size])) for k in per_size[0]})
            csv_log.write(step, "train", 0, train_values)
            for metric, value in train_values.items():
                writer.add_scalar(f"train/{metric}", value, step)
            print(f"step={step} eps={epsilon:.3f} loss={train_values.get('loss', float('nan')):.4f} "
                  f"steps/s={sps:.0f}", flush=True)

    run_eval(step)
    save("final.pt", step)
    csv_log.close()
    writer.close()
    for env in list(train_envs.values()) + list(eval_envs.values()):
        env.close()
    print(f"done: {step} steps per fleet size ({step * len(sizes)} in total) in {time.time() - t_start:.0f} s -> {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1:])
