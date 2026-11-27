#!/usr/bin/env bash
# GAP-31 rerun on ISU Nova (docs/plans/GAP-31_training_stability.md §4). Usage: submit_gap31.sh [TAG]   e.g. Nov27
# Run it from the dedicated clone and do not edit that clone until every task has finished: each task runs
# its evaluations from the clone hours after it starts.
# Resources are the measured need (2026-11-25/26 runs): joint run about 1.1 cores and 4.2 GB at peak, QMIX
# about 1.0 core and 2.0 GB. Elapsed then: joint 16.5 to 26.4 h, QMIX 4.2 to 10 h. This rerun adds a second
# zero-shot evaluation and one more agent forward pass per update.
# Resubmit one task (there is no resume, GAP-26): SUFFIX=VDN TASKS=14 submit_gap31.sh Nov27
set -euo pipefail
EXP=/work/flemingc/akhan1/gnn_marl_experiments; TAG=${1:?give a date tag, e.g. Nov27}
cd "${GAP31_CLONE:-/work/flemingc/akhan1/gnn_marl_gap31}"
sub() {  # sub <folder suffix> <array range> <extra sbatch flags...>
  local s=$1 r=$2; shift 2
  local d=$EXP/${TAG}_GAP31_$s; mkdir -p "$d/logs"
  sbatch --parsable --array="$r" --job-name="g31-$s" "$@" \
    --export=ALL,RUNS_DIR="$d",MATRIX=tools/hpc/matrix_gap31.txt \
    --output="$d/logs/task_%a.log" --open-mode=append tools/hpc/train_array.sbatch | sed "s/^/$s $r job /"
}
JOINT="--cpus-per-task=4 --mem=8G --time=72:00:00"
QMIX="--cpus-per-task=2 --mem=4G --time=24:00:00"
if [ -n "${SUFFIX:-}" ]; then   # one folder, chosen tasks
  if [ "$SUFFIX" = QMIX ]; then sub QMIX "${TASKS:?}" $QMIX; else sub "$SUFFIX" "${TASKS:?}" $JOINT; fi
  exit 0
fi
sub GNNQMIX-R 1-3 $JOINT;   sub GNNQMIX-A 4-6 $JOINT
sub GNNQMIX-B 7-9 $JOINT;   sub GNNQMIX-C 10-12 $JOINT
sub VDN 13-15 $JOINT;       sub IQL 16-18 $JOINT;   sub ABL-VDN-GCN 19-21 $JOINT
sub QMIX 22-30 $QMIX
