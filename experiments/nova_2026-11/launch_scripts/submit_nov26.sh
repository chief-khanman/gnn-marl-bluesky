#!/usr/bin/env bash
# Nov26 set: QMIX reruns (tasks 10-18) with the D-032 fix, from the clone /work/flemingc/akhan1/gnn_marl_nov26.
# Sized from the Nov25 measurements: 2 CPUs, 4 GB (peak was 2.0 GB), 24 h (longest Nov25 QMIX run: 10 h).
# Resubmit one task: change --array below to that id.
set -euo pipefail
EXP=/work/flemingc/akhan1/gnn_marl_experiments; TAG=Nov26
cd /work/flemingc/akhan1/gnn_marl_nov26
d=$EXP/${TAG}_QMIX; mkdir -p "$d/logs"
sbatch --parsable --array="${1:-10-18}" --job-name=gm26-QMIX --cpus-per-task=2 --mem=4G --time=24:00:00 \
  --export=ALL,RUNS_DIR="$d" --output="$d/logs/task_%a.log" tools/hpc/train_array.sbatch
