#!/usr/bin/env bash
# Runs once, after every training job has ended (any outcome). Records how each one ended.
#SBATCH --job-name=gm-report
#SBATCH --partition=nova
#SBATCH --account=flemingc
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=/work/flemingc/akhan1/gnn_marl_experiments/Nov25_FINAL_REPORT.log
cd /work/flemingc/akhan1/gnn_marl_experiments
JOBS=17768552,17768553,17768554,17768555,17768571
{
  echo "# Nov25 final report, written $(date)"
  echo; echo "## How each Slurm task ended (anything other than COMPLETED 0:0 needs a look)"
  sacct -j $JOBS -X --format=JobID%16,JobName%16,State%14,ExitCode,Elapsed,Start,End,NodeList%16
  echo; echo "## Tasks that did not complete"
  sacct -j $JOBS -X -n --format=JobID%16,State%14,ExitCode | grep -v "COMPLETED" || echo "none"
  echo; echo "## Run directories without eval_zero_shot.csv"
  for d in Nov25_*/*_nova/; do [ -f "$d/eval_zero_shot.csv" ] || echo "$d"; done
  echo; echo "## Peak memory per task"
  sacct -j $JOBS --format=JobID%20,MaxRSS,TotalCPU -n | grep batch
} > Nov25_FINAL_REPORT.md 2>&1
/work/flemingc/akhan1/gnn_marl/.venv/bin/python make_status.py > /dev/null
