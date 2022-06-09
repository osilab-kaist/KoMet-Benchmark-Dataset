# Start multiple tmux sessions with different parameters for run.sh (`INDEX`)
for INDEX in $(seq 0 8); do
  CMD="INDEX=$INDEX source scripts/sampling_method_experiments/run.sh"
  echo "$CMD"
  tmux new-session -d -s "v$(printf "%02d" $INDEX)" "$CMD"
done
