#!/usr/bin/env bash
# Train one line of tools/hpc/matrix.txt, then evaluate the result at every fleet size in both airspaces.
# Works on any machine; the Slurm script calls it with the array task id.
#   tools/hpc/run_task.sh <task-id> [extra train flags, e.g. --steps-per-size 1500]
# Goals: G05.4, G06.3, G06.4. Runbook: docs/RUNBOOK.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK_ID="$1"; shift
MATRIX="${MATRIX:-tools/hpc/matrix.txt}"   # e.g. MATRIX=tools/hpc/matrix_gap31.txt
LINE=$(grep -v '^#' "$MATRIX" | sed -n "${TASK_ID}p")
[ -n "$LINE" ] || { echo "no task ${TASK_ID} in ${MATRIX}" >&2; exit 2; }
read -r CONFIG TRAIN_N SEED <<<"$LINE"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PYTHON:-.venv/bin/python}"
LOG=$(mktemp)
# shellcheck disable=SC2086
"$PY" -m algorithms.gnn_qmix.train --config "$CONFIG" --train-n ${TRAIN_N//,/ } --seed "$SEED" "$@" | tee "$LOG"
RUN_DIR=$(grep '^run_dir: ' "$LOG" | head -1 | cut -d' ' -f2-)
rm -f "$LOG"
"$PY" -m algorithms.gnn_qmix.evaluate --run-dir "$RUN_DIR" --eval-n 3 6 9 10 20 50 \
  --episodes "${EVAL_EPISODES:-100}" --num-envs "${EVAL_ENVS:-4}"
# Runs that kept step checkpoints (checkpoint_every > 0, GAP-31): also evaluate the checkpoint chosen by the
# fixed rule of tools/select_checkpoint.py. eval_zero_shot.csv stays "final checkpoint".
if compgen -G "$RUN_DIR/checkpoints/step_*.pt" > /dev/null; then
  # shellcheck disable=SC2086
  BEST=$("$PY" tools/select_checkpoint.py --run-dir "$RUN_DIR" ${SELECT_ARGS:-})
  "$PY" -m algorithms.gnn_qmix.evaluate --run-dir "$RUN_DIR" --checkpoint "$BEST" --out eval_zero_shot_best.csv \
    --eval-n 3 6 9 10 20 50 --episodes "${EVAL_EPISODES:-100}" --num-envs "${EVAL_ENVS:-4}"
fi
