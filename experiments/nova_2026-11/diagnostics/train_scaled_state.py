"""DIAGNOSTIC ONLY (not a result): QMIX training with the airspeed entry of the mixer's entity state divided by
SPEED_DIV. Monkeypatches QMixer.forward in the learner process; nothing in the repo is changed."""
import os, sys, torch
from algorithms.gnn_qmix import networks, train
DIV = float(os.environ.get("SPEED_DIV", "10"))
_orig = networks.QMixer.forward
def forward(self, q, state, h=None):
    s = state.reshape(state.shape[0], self.n_agents, networks.N_STATE).clone()
    s[..., 6] = s[..., 6] / DIV
    return _orig(self, q, s.reshape(state.shape[0], -1), h)
networks.QMixer.forward = forward
if __name__ == "__main__":
    print(f"DIAGNOSTIC: mixer airspeed state entry divided by {DIV}", flush=True)
    train.main(sys.argv[1:])
