"""DIAGNOSTIC ONLY (not a result). QMIX training variants selected by environment variables; repo untouched.
SPEED_DIV: divide the airspeed entry of the mixer state. STATE_MUL: multiply the whole mixer state.
DOUBLE_Q=1: next action chosen by the online agent, valued by the target agent."""
import os, sys, torch, torch.nn.functional as F
from algorithms.gnn_qmix import networks, train
DIV = float(os.environ.get("SPEED_DIV", "1")); MUL = float(os.environ.get("STATE_MUL", "1")); DQ = os.environ.get("DOUBLE_Q", "0") == "1"
_orig = networks.QMixer.forward
def forward(self, q, state, h=None):
    s = state.reshape(state.shape[0], self.n_agents, networks.N_STATE).clone()
    s[..., 6] = s[..., 6] / DIV
    return _orig(self, q, s.reshape(state.shape[0], -1) * MUL, h)
networks.QMixer.forward = forward
def td_loss(args, agent, mixer, target_agent, target_mixer, batch):
    obs, state, action, reward, next_obs, next_state, terminated = batch
    b, n, d = obs.shape
    q, h = agent(obs.reshape(b * n, d))
    q_taken = q.gather(-1, action.reshape(b * n, 1)).reshape(b, n)
    with torch.no_grad():
        q_next, h_next = target_agent(next_obs.reshape(b * n, d))
        a_star = agent(next_obs.reshape(b * n, d))[0].argmax(-1, keepdim=True)
        q_next_max = q_next.gather(-1, a_star).reshape(b, n)
    if mixer is None:
        target = reward.unsqueeze(-1) + args.gamma * (1 - terminated).unsqueeze(-1) * q_next_max
        return F.mse_loss(q_taken, target)
    with torch.no_grad():
        target = reward + args.gamma * (1 - terminated) * target_mixer(q_next_max, next_state, h_next.reshape(b, n, -1))
    return F.mse_loss(mixer(q_taken, state, h.reshape(b, n, -1)), target)
if DQ:
    train.td_loss = td_loss
if __name__ == "__main__":
    print(f"DIAGNOSTIC: SPEED_DIV={DIV} STATE_MUL={MUL} DOUBLE_Q={DQ}", flush=True)
    train.main(sys.argv[1:])
