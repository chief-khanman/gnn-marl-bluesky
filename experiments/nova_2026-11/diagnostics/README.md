# QMIX divergence diagnostics (2026-11-26, Nova). NOT results; nothing here goes into the paper.

Question: why does the QMIX loss reach 1e5-1e11 in tasks 10-18 (Nov25_QMIX)?

Finding 1 (measured): the mixer's entity state contains the env's "normalized airspeed" = (tas - 150) / 6
(simulator/third_party/bluesky-gym/bluesky_zoo/sector_cr/sector_cr.py line 246), which has mean -2.2, std 4.3,
max 13.65 while every other state entry is within about +-3. QMixer (CleanMARL) builds its mixing weights as
|Linear(state)|, so they scale with that entry: at initialization dQ_tot/dq_i is about 53 and Q_tot is about +100
(the true value is about -14, all rewards are negative). The trained n=6 final.pt checkpoints give Q_tot of
7.6e4, 4.3e5 and 9.9e5 on random-policy states.
Finding 2 (measured): dividing that one state entry by 10 inside the mixer (train_scaled_state.py, SPEED_DIV=10)
gives loss 21 instead of 1.75e7 at step 72,000 (n=6, seed 1; control run with SPEED_DIV=1 reproduces task 13 exactly).
Finding 3 (measured): with the entry rescaled the loss still rises steadily (n=6 seed 1: 1.8, 10, 21, 42, 68 at
steps 24k...120k), so a second, weaker instability remains. Its cause is not established.
Offline replays (fixed buffer of 6,000 random transitions): learning rate 1e-4 and hard target updates every 200
updates did not stop the growth.

Files: train_scaled_state.py (wrapper, repo untouched), logs/<n>_<seed>_div<k>.log, run directories *_diag-div<k>
(200,000 steps each, Slurm jobs 17779275-17779279).
