#!/usr/bin/env bash
# Slurm job array: one training run + its zero-shot evaluation per task (docs/RUNBOOK.md §4).
# Cluster lines are set for ISU Nova (docs/HANDOFF.md "Cluster specifics"); change them elsewhere.
# Submit one array per method so each writes into its own parent folder, e.g.
#   D=/work/flemingc/akhan1/gnn_marl_experiments/Nov25_GNN-QMIX; mkdir -p $D/logs
#   sbatch --array=1-3 --export=ALL,RUNS_DIR=$D --output=$D/logs/task_%a.log tools/hpc/train_array.sbatch
# QMIX tasks (10-18) start 4 processes: add --cpus-per-task=4 --mem=8G.
# Extra training flags can be passed after the script name (used for smoke runs only).
#SBATCH --job-name=gnn-qmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --partition=nova
#SBATCH --account=flemingc
set -euo pipefail
# Nova: no module lines needed; Python comes from the repo's .venv (tools/hpc/run_task.sh default).
cd "${SLURM_SUBMIT_DIR}"
echo "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) started $(date)"
tools/hpc/run_task.sh "${SLURM_ARRAY_TASK_ID}" ${RUNS_DIR:+--runs-dir "$RUNS_DIR"} "$@"
