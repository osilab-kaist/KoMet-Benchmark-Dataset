#!/bin/bash

LRS=("0.01" "0.003" "0.001" "0.0003" "0.0001")
GPU_INDICES=(0 1 2 3 4)
GPU_INDEX=${GPU_INDICES[INDEX]}

export CUDA_VISIBLE_DEVICES=$GPU_INDEX

for MODEL in "unet" "convlstm" "metnet"; do
  LR=${LRS[INDEX]};
  python train.py --model=$MODEL --device="$GPU_INDEX" --seed=0 --input_data="gdaps_kim" \
    --reference=aws --num_epochs=20 --normalization \
    --start_lead_time 6 --end_lead_time 88 \
    --date_intervals "2020-07-01" "2020-08-31" "2021-07-01" "2021-08-31" \
    --lr "$LR" \
    --rain_thresholds 0.1 10.0 \
    --interpolate_aws \
    --intermediate_test \
    --custom_name="lr${LR}_interpolated"
done
