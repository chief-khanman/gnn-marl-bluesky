#!/usr/bin/env bash
# Submits the gnn-marl-bluesky run matrix on ISU Nova, one Slurm array per method.
#   submit_all.sh            -> tasks 1-18 (required)
#   submit_all.sh ablation   -> tasks 19-21 only
# Resubmit one dead task, e.g. task 5 (VDN):
#   D=$EXP/${TAG}_VDN; sbatch --array=5 --export=ALL,RUNS_DIR=$D --output=$D/logs/task_%a.log tools/hpc/train_array.sbatch
set -euo pipefail
EXP=/work/flemingc/akhan1/gnn_marl_experiments
TAG=${TAG:-Nov25}
cd /work/flemingc/akhan1/gnn_marl
sub() {  # sub <method> <array range> [extra sbatch flags]
  local method=$1 range=$2; shift 2
  local d=$EXP/${TAG}_${method}
  mkdir -p "$d/logs"
  sbatch --parsable --array="$range" --job-name="gm-$method" --export=ALL,RUNS_DIR="$d" \
    --output="$d/logs/task_%a.log" "$@" tools/hpc/train_array.sbatch | sed "s/^/$method $range job /"
}
if [ "${1:-}" = ablation ]; then
  sub ABL-VDN-GCN 19-21 --mem=32G
else
  sub GNN-QMIX 1-3 --mem=32G
  sub VDN 4-6 --mem=32G
  sub IQL 7-9 --mem=32G
  sub QMIX 10-18 --cpus-per-task=4 --mem=16G
fi
