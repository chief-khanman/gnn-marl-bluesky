#!/usr/bin/env bash
# Slurm job array: one training run + its zero-shot evaluation per task (docs/RUNBOOK.md §4).
# TEMPLATE: the lines marked TODO(cluster) must be set for your cluster (docs/HANDOFF.md "Cluster specifics").
#   mkdir -p slurm_logs && sbatch --array=1-18 tools/hpc/train_array.sbatch
#SBATCH --job-name=gnn-qmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
##SBATCH --partition=TODO(cluster)
##SBATCH --account=TODO(cluster)
set -euo pipefail
# TODO(cluster): module load lines, if the cluster needs them
cd "${SLURM_SUBMIT_DIR}"
tools/hpc/run_task.sh "${SLURM_ARRAY_TASK_ID}"
