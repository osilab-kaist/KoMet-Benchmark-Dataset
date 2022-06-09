# Start multiple tmux sessions with different parameters for run.sh (`INDEX`)
for INDEX in $(seq 0 4); do
  CMD="INDEX=$INDEX source scripts/lr_experiments/run.sh"
  echo "$CMD"
  tmux new-session -d -s "p$(printf "%02d" $INDEX)" "$CMD"
done
