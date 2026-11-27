#!/usr/bin/env bash
# Recreates the monitoring tmux session `gnn_marl` on whatever node you are logged in to
# (the Slurm jobs do not depend on it). Usage: watch.sh && tmux attach -t gnn_marl
EXP=/work/flemingc/akhan1/gnn_marl_experiments; TAG=${TAG:-Nov25}
tmux has-session -t gnn_marl 2>/dev/null && { echo "session exists: tmux attach -t gnn_marl"; exit 0; }
tmux new -d -s gnn_marl -n squeue "watch -n 30 'squeue -u \$USER -o \"%.16i %.12j %.2t %.11M %.4C %R\"'"
for d in $EXP/${TAG}_*/; do
  [ -d "$d/logs" ] || continue
  m=$(basename "$d"); m=${m#${TAG}_}
  tmux new-window -d -t gnn_marl -n "$m" "tail -n 3 -F $d/logs/task_*.log | grep --line-buffered -E '==>|run_dir|steps/s|\\[eval\\]|Traceback|Error|done:|wrote'"
done
echo "tmux attach -t gnn_marl"
