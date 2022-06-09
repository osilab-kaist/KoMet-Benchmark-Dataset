# Start multiple tmux sessions with different parameters for run.sh (`INDEX`)
for INDEX in $(seq 0 7); do
  CMD="INDEX=$INDEX source scripts/window_size_experiments/run.sh"
  echo "$CMD"
  tmux new-session -d -s "p$(printf "%02d" $INDEX)" "$CMD"
done
