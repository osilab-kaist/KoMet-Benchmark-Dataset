#!/bin/bash

WINDOW_SIZES=( 1 2 3 4 5 6 9 12 )
GPU_INDICES=( 0 1 2 3 )
GPU_INDEX=${GPU_INDICES[INDEX % 4]}

export CUDA_VISIBLE_DEVICES=$GPU_INDEX

WS=${WINDOW_SIZES[INDEX]}
for MODEL in "unet" "convlstm" "metnet"; do
  if [[ WS -lt 6 ]]; then
    python train.py --model=$MODEL --device="$GPU_INDEX" --seed=0 --input_data="gdaps_kim" \
                    --reference=aws --num_epochs=20 --normalization \
                    --window_size="$WS" --start_lead_time 6 --end_lead_time 88 \
                    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
                    --rain_thresholds 0.1 10.0 \
                    --interpolate_aws \
                    --intermediate_test \
                    --custom_name="ws$(printf "%02d" $WS)_interpolated"
  else
    python train.py --model=$MODEL --device="$GPU_INDEX" --seed=0 --input_data="gdaps_kim" \
                    --reference=aws --num_epochs=20 --normalization \
                    --window_size="$WS" --start_lead_time "$WS" --end_lead_time 88 \
                    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
                    --rain_thresholds 0.1 10.0 \
                    --interpolate_aws \
                    --intermediate_test \
                    --custom_name="ws$(printf "%02d" $WS)_interpolated"
  fi
done