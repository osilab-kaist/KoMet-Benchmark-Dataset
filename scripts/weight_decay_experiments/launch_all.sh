# Start multiple tmux sessions with different parameters for run.sh (`INDEX`)
for INDEX in $(seq 0 4); do
  CMD="INDEX=$INDEX source scripts/weight_decay_experiments/run.sh"
  echo "$CMD"
  tmux new-session -d -s "d$(printf "%02d" $INDEX)" "$CMD"
done
