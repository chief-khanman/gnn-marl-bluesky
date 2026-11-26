#!/usr/bin/env bash
# Slurm job array: one training run + its zero-shot evaluation per task (docs/RUNBOOK.md §4).
# Cluster lines are set for ISU Nova (docs/HANDOFF.md "Cluster specifics"); change them elsewhere.
# Submit one array per method so each writes into its own parent folder, e.g.
#   D=/work/flemingc/akhan1/gnn_marl_experiments/Nov25_GNN-QMIX; mkdir -p $D/logs
#   sbatch --array=1-3 --export=ALL,RUNS_DIR=$D --output=$D/logs/task_%a.log tools/hpc/train_array.sbatch
# Measured on Nova (2026-11-25 runs): a joint run used about 1.1 CPU cores on average and 4.2 GB at peak
# (10 processes, mostly waiting on each other); a QMIX run about 1.0 core and 2.0 GB (4 processes).
# The defaults below fit a joint run. For QMIX tasks (10-18) add --cpus-per-task=2 --mem=4G --time=24:00:00.
# Extra training flags can be passed after the script name (used for smoke runs only).
#SBATCH --job-name=gnn-qmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --partition=nova
#SBATCH --account=flemingc
set -euo pipefail
# Nova: no module lines needed; Python comes from the repo's .venv (tools/hpc/run_task.sh default).
cd "${SLURM_SUBMIT_DIR}"
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) started $(date)"
tools/hpc/run_task.sh "${SLURM_ARRAY_TASK_ID}" ${RUNS_DIR:+--runs-dir "$RUNS_DIR"} "$@"
